"""
Non-IID data partitioning utilities for federated learning.

``partition_non_iid`` assigns each agent a random set
of classes and then allocates each class to its assigned agents according to a
Dirichlet distribution. The implementation guarantees that every sample is
assigned exactly once and that every agent receives at least one sample.
"""

from collections import defaultdict

import torch
import random
import numpy as np
import torch


def build_shared_class_partition(
    unique_classes: list[int],
    n_agents: int,
    shared_classes: int,
    seed: int = 42,
) -> dict[int, list[int]]:
    """Build class assignments with an exact globally shared-label count.

    The resulting mapping is intended for ``split_strategy='class_partition'``:
    ``shared_classes`` labels are visible to every agent, while the remaining
    labels are distributed approximately evenly across agents.
    """
    if n_agents < 1:
        raise ValueError('n_agents must be at least 1')

    resolved_classes = sorted(set(unique_classes))
    num_classes = len(resolved_classes)
    if num_classes == 0:
        return {agent_id: [] for agent_id in range(n_agents)}

    if shared_classes < 0 or shared_classes > num_classes:
        raise ValueError(
            f'shared_classes={shared_classes} must be in [0, {num_classes}]'
        )
    if shared_classes == 0 and n_agents > num_classes:
        raise ValueError(
            'shared_classes=0 cannot guarantee a non-empty class set for '
            'each agent when n_agents exceeds the number of classes'
        )

    generator = torch.Generator().manual_seed(seed)
    shuffled_indices = torch.randperm(
        num_classes, generator=generator
    ).tolist()
    shuffled_classes = [resolved_classes[idx] for idx in shuffled_indices]

    shared = shuffled_classes[:shared_classes]
    residual = shuffled_classes[shared_classes:]

    assignments = {
        agent_id: list(shared) for agent_id in range(n_agents)
    }
    agent_order = torch.randperm(n_agents, generator=generator).tolist()

    for offset, class_label in enumerate(residual):
        agent_id = agent_order[offset % n_agents]
        assignments[agent_id].append(class_label)

    return {
        agent_id: sorted(class_labels)
        for agent_id, class_labels in assignments.items()
    }


def partition_by_agent_classes(
    labels: list[int],
    agent_classes: dict[int, list[int]],
    n_agents: int,
    seed: int = 42,
) -> dict[int, list[int]]:
    """Partition indices disjointly while respecting per-agent class sets.

    Each sample index is assigned to exactly one agent. For a class label
    available to multiple agents, its examples are split approximately evenly
    among those agents, preserving label overlap without duplicating samples.
    """
    if n_agents < 1:
        raise ValueError('n_agents must be at least 1')

    labels_tensor = torch.tensor(labels, dtype=torch.long)
    if labels_tensor.numel() == 0:
        return {agent_id: [] for agent_id in range(n_agents)}

    unique_classes = torch.unique(labels_tensor).tolist()
    label_to_agents: dict[int, list[int]] = {}
    for class_label in unique_classes:
        assigned_agents = [
            agent_id
            for agent_id in range(n_agents)
            if class_label in set(agent_classes.get(agent_id, []))
        ]
        if not assigned_agents:
            raise ValueError(
                f'class {class_label} is not assigned to any agent'
            )
        label_to_agents[int(class_label)] = sorted(assigned_agents)

    partition = {agent_id: [] for agent_id in range(n_agents)}
    generator = torch.Generator().manual_seed(seed)

    for class_label in unique_classes:
        class_indices = torch.where(labels_tensor == int(class_label))[0]
        shuffled = class_indices[
            torch.randperm(len(class_indices), generator=generator)
        ].tolist()
        assigned_agents = label_to_agents[int(class_label)]

        for offset, sample_idx in enumerate(shuffled):
            agent_id = assigned_agents[offset % len(assigned_agents)]
            partition[agent_id].append(int(sample_idx))

    for agent_id in range(n_agents):
        partition[agent_id].sort()

    return partition


def _draw_dirichlet_proportions(
    n_parts: int,
    alpha: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Draw class-allocation proportions with a deterministic generator.

    Non-negative ``alpha`` uses a symmetric Dirichlet. Negative ``alpha``
    switches to the bounded uniform heuristic used by the heterogeneity
    experiments.
    """

    # If alpha is negative, use the exact pFedHN logic
    # to guarantee the tight sample bounds [1325, 2000] 
    if alpha < 0:
        probs = torch.rand(n_parts, generator=generator, dtype=torch.float32) * 0.2 + 0.4
        probs_sum = probs.sum()
        if probs_sum <= 0:
            return torch.full((n_parts,), 1 / n_parts, dtype=torch.float32)
        return probs / probs_sum

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


def _sample_exact_k_class_assignments(
    unique_classes: list[int],
    n_agents: int,
    classes_per_agent: int,
    generator: torch.Generator,
) -> dict[int, list[int]]:
    """Assign exactly ``classes_per_agent`` classes to every agent.

    The assignment is deterministic under ``generator`` and guarantees that
    every dataset class is assigned to at least one agent.
    """
    if n_agents < 1:
        raise ValueError('n_agents must be at least 1')
    if classes_per_agent < 1:
        raise ValueError('classes_per_agent must be at least 1')

    resolved_classes = sorted(set(unique_classes))
    num_classes = len(resolved_classes)
    if num_classes == 0:
        return {agent_id: [] for agent_id in range(n_agents)}

    if classes_per_agent > num_classes:
        raise ValueError(
            f'classes_per_agent={classes_per_agent} must be in '
            f'[1, {num_classes}] (number of unique classes in the dataset)'
        )
    if n_agents * classes_per_agent < num_classes:
        raise ValueError(
            'Cannot assign every class at least once while giving each agent '
            f'exactly {classes_per_agent} classes: need at least '
            f'{num_classes} total class slots, got '
            f'{n_agents * classes_per_agent}.'
        )

    shuffled_classes = [
        resolved_classes[idx]
        for idx in torch.randperm(
            num_classes, generator=generator
        ).tolist()
    ]
    shuffled_agents = torch.randperm(n_agents, generator=generator).tolist()

    assignments = {agent_id: set() for agent_id in range(n_agents)}

    # Coverage pass: place every class at least once using a deterministic
    # round-robin over a shuffled agent order.
    for offset, class_label in enumerate(shuffled_classes):
        agent_id = shuffled_agents[offset % n_agents]
        assignments[agent_id].add(class_label)

    # Top-up pass: give every agent exactly K classes without duplicates.
    for rank, agent_id in enumerate(shuffled_agents):
        cursor = rank % num_classes
        while len(assignments[agent_id]) < classes_per_agent:
            class_label = shuffled_classes[cursor % num_classes]
            assignments[agent_id].add(class_label)
            cursor += 1

    return {
        agent_id: sorted(class_labels)
        for agent_id, class_labels in assignments.items()
    }


def _generate_skewed_proportions(
    n_parts: int,
    alpha: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Route to Dirichlet or extreme Log-Normal class proportions."""
    if n_parts < 1:
        raise ValueError('n_parts must be at least 1')
    if alpha == 0:
        raise ValueError('alpha must be non-zero')
    if n_parts == 1:
        return torch.ones(1, dtype=torch.float32)

    if alpha > 0:
        concentration = torch.full(
            (n_parts,),
            float(alpha),
            dtype=torch.float32,
        )
        weights = torch._standard_gamma(concentration, generator=generator)
    else:
        # Match the aggressive skew regime with a high-variance Log-Normal.
        weights = torch.exp(
            torch.randn(n_parts, generator=generator, dtype=torch.float32)
            * 2.0
        )

    total_weight = weights.sum()
    if total_weight <= 0:
        return torch.full((n_parts,), 1 / n_parts, dtype=torch.float32)
    return weights / total_weight


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
        Allocation concentration parameter. Positive values use a symmetric
        Dirichlet, where lower values increase skew and higher values
        approach a more even class-wise split. Negative values switch to a
        bounded-uniform sampling scheme before normalization.
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
    if alpha == 0:
        raise ValueError('alpha must be non-zero')

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


def partition_non_iid_with_margin(
    labels: list[int],
    n_agents: int,
    classes_per_agent: int,
    seed: int = 42,
    alpha: float = 0.5,
    agent_classes: dict[int, list[int]] | None = None,
    return_agent_classes: bool = False,
    safety_margin: int = 10,
) -> dict[int, list[int]] | tuple[dict[int, list[int]], dict[int, list[int]]]:
    """Partition with exact-K class assignment and per-class safety_margin samples.

    For every class assigned to an agent, ``safety_margin`` examples are
    reserved first to guarantee survival under later 80% starvation. The
    remaining class pool is then distributed with a skew controlled by
    ``alpha``:
    positive values use Dirichlet draws and negative values use an extreme
    Log-Normal router.
    """
    if n_agents < 1:
        raise ValueError('n_agents must be at least 1')
    if classes_per_agent < 1:
        raise ValueError('classes_per_agent must be at least 1')
    if safety_margin < 0:
        raise ValueError('safety_margin must be non-negative')
    if alpha == 0:
        raise ValueError('alpha must be non-zero')

    labels_tensor = torch.tensor(labels, dtype=torch.long)
    if labels_tensor.numel() == 0:
        empty_partition = {agent_id: [] for agent_id in range(n_agents)}
        if return_agent_classes:
            resolved_classes = agent_classes or {
                agent_id: [] for agent_id in range(n_agents)
            }
            return empty_partition, resolved_classes
        return empty_partition

    unique_classes = sorted(torch.unique(labels_tensor).tolist())
    unique_class_set = set(unique_classes)
    generator = torch.Generator().manual_seed(seed)

    if agent_classes is None:
        resolved_agent_classes = _sample_exact_k_class_assignments(
            unique_classes=unique_classes,
            n_agents=n_agents,
            classes_per_agent=classes_per_agent,
            generator=generator,
        )
    else:
        resolved_agent_classes = {}
        for agent_id in range(n_agents):
            assigned_classes = sorted(set(agent_classes.get(agent_id, [])))
            if len(assigned_classes) != classes_per_agent:
                raise ValueError(
                    f'agent {agent_id} must have exactly '
                    f'{classes_per_agent} classes, got '
                    f'{len(assigned_classes)}'
                )
            unknown_classes = set(assigned_classes) - unique_class_set
            if unknown_classes:
                raise ValueError(
                    f'agent {agent_id} was assigned classes not present in '
                    f'the dataset split: {sorted(unknown_classes)}'
                )
            resolved_agent_classes[agent_id] = assigned_classes

        covered_classes = set().union(*resolved_agent_classes.values())
        uncovered_classes = sorted(unique_class_set - covered_classes)
        if uncovered_classes:
            raise ValueError(
                'Every class in the split must be assigned to at least one '
                f'agent. Uncovered classes: {uncovered_classes}'
            )

    class_to_agents: dict[int, list[int]] = defaultdict(list)
    for agent_id in range(n_agents):
        for class_label in resolved_agent_classes[agent_id]:
            class_to_agents[int(class_label)].append(agent_id)

    agent_indices: dict[int, list[int]] = {
        agent_id: [] for agent_id in range(n_agents)
    }

    # Index the global data once, class by class, using shuffled global row ids.
    for class_label in unique_classes:
        class_indices = torch.where(labels_tensor == int(class_label))[0]
        shuffled_indices = class_indices[
            torch.randperm(len(class_indices), generator=generator)
        ].tolist()
        assigned_agents = class_to_agents[int(class_label)]

        required_safety_margin = safety_margin * len(assigned_agents)
        if len(shuffled_indices) < required_safety_margin:
            raise ValueError(
                f'class {class_label} has {len(shuffled_indices)} samples, '
                f'but needs at least {required_safety_margin} to allocate '
                f'{safety_margin} safety margin samples to each of the '
                f'{len(assigned_agents)} assigned agents.'
            )

        cursor = 0
        for agent_id in assigned_agents:
            safety_indices = shuffled_indices[
                cursor : cursor + safety_margin
            ]
            agent_indices[agent_id].extend(safety_indices)
            cursor += safety_margin

        remaining_indices = shuffled_indices[cursor:]
        if not remaining_indices:
            continue

        proportions = _generate_skewed_proportions(
            n_parts=len(assigned_agents),
            alpha=alpha,
            generator=generator,
        )
        draws = torch.multinomial(
            proportions,
            num_samples=len(remaining_indices),
            replacement=True,
            generator=generator,
        )
        counts = torch.bincount(draws, minlength=len(assigned_agents))

        offset = 0
        for position, agent_id in enumerate(assigned_agents):
            count = int(counts[position].item())
            skewed_indices = remaining_indices[offset : offset + count]
            agent_indices[agent_id].extend(skewed_indices)
            offset += count

    for agent_id in range(n_agents):
        agent_indices[agent_id].sort()

    if return_agent_classes:
        return agent_indices, resolved_agent_classes
    return agent_indices