"""
Non-IID data partitioning utilities for federated learning.

Provides functions to partition a dataset across agents with controlled
heterogeneity. The main function ``partition_non_iid`` assigns each agent
a random subset of N classes and distributes samples for each class among
the agents assigned to it with random (Dirichlet-drawn) proportions to
introduce statistical skew.
"""

import torch


def partition_non_iid(
    labels: list[int],
    n_agents: int,
    classes_per_agent: int,
    seed: int = 42,
    alpha: float = 0.5,
) -> dict[int, list[int]]:
    """Partition dataset indices into Non-IID shards with statistical skew.

    Each agent is randomly assigned exactly ``classes_per_agent`` classes.
    For every class, its samples are distributed among the agents assigned
    to that class with **random proportions** drawn from a symmetric
    Dirichlet(alpha) distribution, introducing statistical skew (different
    agents receive different amounts of data per class).

    Parameters
    ----------
    labels : list[int]
        List of integer class labels, one per sample (same length as the
        dataset). Used only to determine which samples belong to which class.
    n_agents : int
        Number of agents (clients) to partition into.
    classes_per_agent : int
        Number of classes assigned to each agent. Must satisfy
        ``1 <= classes_per_agent <= num_unique_classes``.
    seed : int, optional
        Random seed for reproducibility (default: 42).
    alpha : float, optional
        Concentration parameter of the Dirichlet distribution that controls
        the statistical skew (default: 0.5).
        - alpha -> 0: extreme skew (one agent gets almost all samples)
        - alpha = 1.0: uniformly random proportions
        - alpha -> inf: near-equal split (approaches uniform)

    Returns
    -------
    agent_indices : dict[int, list[int]]
        Mapping from agent index to the list of dataset sample indices that
        agent receives.

    Raises
    ------
    ValueError
        If ``classes_per_agent`` exceeds the number of unique classes or is
        less than 1.
    """
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    unique_classes = torch.unique(labels_tensor).tolist()
    num_classes = len(unique_classes)

    if not 1 <= classes_per_agent <= num_classes:
        raise ValueError(
            f'classes_per_agent={classes_per_agent} must be in '
            f'[1, {num_classes}] (number of unique classes in the dataset)'
        )

    generator = torch.Generator().manual_seed(seed)

    # assign each agent a random subset of classes
    agent_class_sets: dict[int, list[int]] = {}
    for i in range(n_agents):
        perm = torch.randperm(num_classes, generator=generator)
        agent_class_sets[i] = [
            unique_classes[j] for j in perm[:classes_per_agent].tolist()
        ]

    class_to_agents: dict[int, list[int]] = {c: [] for c in unique_classes}
    for agent_id, class_list in agent_class_sets.items():
        for c in class_list:
            class_to_agents[c].append(agent_id)

    # for each class, distribute its sample indices among assigned
    # agents with Dirichlet-drawn random proportions
    agent_indices: dict[int, list[int]] = {i: [] for i in range(n_agents)}

    dirichlet = torch.distributions.Dirichlet(
        torch.ones(1)  # placeholder, resized per class below
    )

    for c in unique_classes:
        # All sample indices belonging to this class
        c_indices = torch.where(labels_tensor == c)[0]
        # Shuffle deterministically
        c_perm = torch.randperm(len(c_indices), generator=generator)
        c_indices = c_indices[c_perm]

        assigned_agents = class_to_agents[c]
        if not assigned_agents:
            # Edge case: no agent was assigned this class
            continue

        n_assigned = len(assigned_agents)
        n_samples = len(c_indices)

        if n_assigned == 1:
            # Only one agent gets this class
            agent_indices[assigned_agents[0]].extend(c_indices.tolist())
            continue

        # Draw random proportions from Dirichlet(alpha, alpha, ..., alpha)
        # Ensure alpha is a float, as torch._standard_gamma requires a float tensor
        concentration = torch.full((n_assigned,), float(alpha), dtype=torch.float32)
        gamma_samples = torch.zeros(n_assigned)
        for k in range(n_assigned):
            # Gamma(alpha, 1) via torch 
            gamma_samples[k] = torch._standard_gamma(
                concentration[k:k+1], generator=generator
            ).item()

        # Normalise to get Dirichlet proportions
        proportions = gamma_samples / gamma_samples.sum()

        # Convert proportions to integer counts
        counts = (proportions * n_samples).long()
        # Distribute rounding remainder to random agents
        remainder = n_samples - counts.sum().item()
        if remainder > 0:
            bonus_idx = torch.randperm(n_assigned, generator=generator)[:remainder]
            counts[bonus_idx] += 1
        elif remainder < 0:
            # Over-allocated due to rounding 
            trim_idx = torch.argsort(counts, descending=True)[:abs(remainder)]
            counts[trim_idx] -= 1

        # Ensure no negative counts 
        counts = counts.clamp(min=0)

        # Assign sample chunks according to the random counts
        offset = 0
        for k, agent_id in enumerate(assigned_agents):
            count = counts[k].item()
            agent_indices[agent_id].extend(
                c_indices[offset:offset + count].tolist()
            )
            offset += count

    return agent_indices
