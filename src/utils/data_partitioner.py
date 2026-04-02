"""
Non-IID data partitioning utilities for federated learning.

``partition_non_iid`` assigns each agent a random set
of classes and then allocates each class to its assigned agents according to a
Dirichlet distribution. The implementation guarantees that every sample is
assigned exactly once and that every agent receives at least one sample.
"""

from collections import defaultdict

import torch


def _draw_dirichlet_proportions(
    n_parts: int,
    alpha: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Draw symmetric Dirichlet proportions with a deterministic generator."""
    concentration = torch.full((n_parts,), float(alpha), dtype=torch.float32)
    gamma_samples = torch._standard_gamma(concentration, generator=generator)
    gamma_sum = gamma_samples.sum()
    if gamma_sum <= 0:
        return torch.full((n_parts,), 1 / n_parts, dtype=torch.float32)
    return gamma_samples / gamma_sum


def _sample_class_assignments(
    unique_classes: list[int],
    n_agents: int,
    classes_per_agent: int,
    generator: torch.Generator,
) -> dict[int, list[int]]:
    """Assign random classes to agents ensuring every class is covered."""
    agent_class_sets = {agent_id: set() for agent_id in range(n_agents)}
    shuffled_classes = [
        unique_classes[idx]
        for idx in torch.randperm(
            len(unique_classes), generator=generator
        ).tolist()
    ]

    # First guarantee that every class is available to at least one agent.
    for class_label in shuffled_classes:
        loads = torch.tensor(
            [len(agent_class_sets[agent_id]) for agent_id in range(n_agents)],
            dtype=torch.long,
        )
        candidate_agents = torch.where(loads == loads.min())[0]
        chosen_idx = candidate_agents[
            torch.randint(
                len(candidate_agents), (1,), generator=generator
            ).item()
        ].item()
        agent_class_sets[chosen_idx].add(class_label)

    # Then top up each agent to the requested class count when possible.
    shuffled_agents = torch.randperm(n_agents, generator=generator).tolist()
    for agent_id in shuffled_agents:
        target_count = max(classes_per_agent, len(agent_class_sets[agent_id]))
        if len(agent_class_sets[agent_id]) >= target_count:
            continue

        available_classes = [
            class_label
            for class_label in shuffled_classes
            if class_label not in agent_class_sets[agent_id]
        ]
        extra_needed = min(
            target_count - len(agent_class_sets[agent_id]),
            len(available_classes),
        )
        for class_idx in torch.randperm(
            len(available_classes), generator=generator
        )[:extra_needed].tolist():
            agent_class_sets[agent_id].add(available_classes[class_idx])

    return {
        agent_id: sorted(class_labels)
        for agent_id, class_labels in agent_class_sets.items()
    }


def partition_non_iid(
    labels: list[int],
    n_agents: int,
    classes_per_agent: int,
    seed: int = 42,
    alpha: float = 0.5,
    agent_classes: dict[int, list[int]] | None = None,
    return_agent_classes: bool = False,
) -> dict[int, list[int]] | tuple[dict[int, list[int]], dict[int, list[int]]]:
    """Partition dataset indices into non-IID shards with Dirichlet skew.

    Parameters
    ----------
    labels : list[int]
        Integer class labels, one per sample.
    n_agents : int
        Number of agents.
    classes_per_agent : int
        Target number of classes per agent. When the dataset has more classes
        than total available class slots, some agents may receive more than
        this target so that every class remains assigned.
    seed : int, optional
        Random seed for reproducibility.
    alpha : float, optional
        Symmetric Dirichlet concentration parameter. Lower values increase
        skew, higher values approach a more even class-wise split.
    agent_classes : dict[int, list[int]] | None, optional
        Explicit class assignments to reuse across multiple dataset splits.
        When omitted, assignments are sampled randomly.
    return_agent_classes : bool, optional
        If ``True``, also return the sampled/reused agent-to-class mapping.

    Returns
    -------
    dict[int, list[int]] or tuple[dict[int, list[int]], dict[int, list[int]]]
        Per-agent dataset indices and, optionally, the class assignments.
    """
    if n_agents < 1:
        raise ValueError('n_agents must be at least 1')
    if classes_per_agent < 1:
        raise ValueError('classes_per_agent must be at least 1')
    if alpha <= 0:
        raise ValueError('alpha must be strictly positive')

    labels_tensor = torch.tensor(labels, dtype=torch.long)
    if labels_tensor.numel() == 0:
        empty_partition = {agent_id: [] for agent_id in range(n_agents)}
        if return_agent_classes:
            resolved_classes = agent_classes or {
                agent_id: [] for agent_id in range(n_agents)
            }
            return empty_partition, resolved_classes
        return empty_partition

    unique_classes = torch.unique(labels_tensor).tolist()
    num_classes = len(unique_classes)
    unique_class_set = set(unique_classes)

    if classes_per_agent > num_classes:
        raise ValueError(
            f'classes_per_agent={classes_per_agent} must be in '
            f'[1, {num_classes}] (number of unique classes in the dataset)'
        )

    generator = torch.Generator().manual_seed(seed)

    if agent_classes is None:
        resolved_agent_classes = _sample_class_assignments(
            unique_classes=unique_classes,
            n_agents=n_agents,
            classes_per_agent=classes_per_agent,
            generator=generator,
        )
    else:
        resolved_agent_classes = {}
        for agent_id in range(n_agents):
            assigned_classes = sorted(set(agent_classes.get(agent_id, [])))
            if not assigned_classes:
                raise ValueError(
                    f'agent {agent_id} has no assigned classes'
                    f' in agent_classes'
                )
            unknown_classes = set(assigned_classes) - unique_class_set
            if unknown_classes:
                raise ValueError(
                    f'agent {agent_id} was assigned classes not'
                    f' present in the '
                    f'dataset split: {sorted(unknown_classes)}'
                )
            resolved_agent_classes[agent_id] = assigned_classes

    class_to_agents: dict[int, list[int]] = defaultdict(list)
    for agent_id, class_labels in resolved_agent_classes.items():
        for class_label in class_labels:
            class_to_agents[class_label].append(agent_id)

    uncovered_classes = sorted(unique_class_set - set(class_to_agents))
    if uncovered_classes:
        raise ValueError(
            'Every class in the split must be assigned to at least one agent. '
            f'Uncovered classes: {uncovered_classes}'
        )

    agent_indices: dict[int, list[int]] = {
        agent_id: [] for agent_id in range(n_agents)
    }
    agent_indices_by_class: dict[int, dict[int, list[int]]] = {
        agent_id: defaultdict(list) for agent_id in range(n_agents)
    }

    for class_label in unique_classes:
        class_indices = torch.where(labels_tensor == class_label)[0]
        class_indices = class_indices[
            torch.randperm(len(class_indices), generator=generator)
        ]

        assigned_agents = class_to_agents[class_label]
        if len(assigned_agents) == 1:
            agent_id = assigned_agents[0]
            selected_indices = class_indices.tolist()
            agent_indices[agent_id].extend(selected_indices)
            agent_indices_by_class[agent_id][class_label].extend(
                selected_indices
            )
            continue

        proportions = _draw_dirichlet_proportions(
            n_parts=len(assigned_agents),
            alpha=alpha,
            generator=generator,
        )
        draws = torch.multinomial(
            proportions,
            num_samples=len(class_indices),
            replacement=True,
            generator=generator,
        )
        counts = torch.bincount(draws, minlength=len(assigned_agents))

        offset = 0
        for position, agent_id in enumerate(assigned_agents):
            count = counts[position].item()
            selected_indices = class_indices[offset : offset + count].tolist()
            agent_indices[agent_id].extend(selected_indices)
            agent_indices_by_class[agent_id][class_label].extend(
                selected_indices
            )
            offset += count

    if n_agents == 1:
        result: (
            dict[int, list[int]]
            | tuple[dict[int, list[int]], dict[int, list[int]]]
        )
        result = agent_indices
        if return_agent_classes:
            result = (agent_indices, resolved_agent_classes)
        return result

    empty_agents = [
        agent_id for agent_id, indices in agent_indices.items() if not indices
    ]
    for agent_id in empty_agents:
        donor_found = False
        candidate_classes = resolved_agent_classes[agent_id]
        for class_label in candidate_classes:
            donors = [
                donor_id
                for donor_id in class_to_agents[class_label]
                if len(agent_indices_by_class[donor_id][class_label]) > 1
            ]
            if donors:
                donor_id = max(
                    donors,
                    key=lambda current_agent: len(
                        agent_indices_by_class[current_agent][class_label]
                    ),
                )
                moved_index = agent_indices_by_class[donor_id][
                    class_label
                ].pop()
                agent_indices[donor_id].remove(moved_index)
                agent_indices[agent_id].append(moved_index)
                agent_indices_by_class[agent_id][class_label].append(
                    moved_index
                )
                donor_found = True
                break

        if donor_found:
            continue

        donors = [
            donor_id
            for donor_id, indices in agent_indices.items()
            if len(indices) > 1
        ]
        if not donors:
            raise RuntimeError(
                'Could not allocate at least one sample to every agent.'
            )

        donor_id = max(
            donors, key=lambda current_agent: len(agent_indices[current_agent])
        )
        moved_index = agent_indices[donor_id].pop()
        moved_class = labels[moved_index]
        agent_indices[agent_id].append(moved_index)
        agent_indices_by_class[donor_id][moved_class].remove(moved_index)
        agent_indices_by_class[agent_id][moved_class].append(moved_index)
        if moved_class not in resolved_agent_classes[agent_id]:
            resolved_agent_classes[agent_id] = sorted(
                resolved_agent_classes[agent_id] + [moved_class]
            )

    result = agent_indices
    if return_agent_classes:
        result = (agent_indices, resolved_agent_classes)
    return result
