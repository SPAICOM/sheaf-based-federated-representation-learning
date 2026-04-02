"""Anchor-selection utilities for Sheaf-FRL."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


AnchorKeys = dict[int, list[tuple[int, int]]]
AnchorTensors = dict[int, torch.Tensor]

VALID_ANCHOR_STRATEGIES = {
    'prototype',
    'random',
    'balanced',
    'semantic_pilots',
    'clustered_pilots',
    'dynamic',
}


@dataclass(frozen=True)
class AnchorConfig:
    """Configuration for anchor construction and normalization."""

    strategy: str
    num_anchors: int
    parseval_normalization: bool
    l2_normalization: bool
    parseval_eps: float = 1e-4


def supported_anchor_strategy(anchor_strategy: str) -> str:
    """Validate and normalize the configured anchor strategy name."""
    strategy = str(anchor_strategy)
    if strategy not in VALID_ANCHOR_STRATEGIES:
        raise ValueError(
            f'Unknown anchor_strategy: {strategy}. Valid options: '
            f'{sorted(VALID_ANCHOR_STRATEGIES)}'
        )
    return strategy


def parseval_normalize(
    anchor_matrix: torch.Tensor,
    *,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Apply Parseval normalization to anchor features."""
    covariance = torch.matmul(anchor_matrix.T, anchor_matrix)
    covariance = covariance + eps * torch.eye(
        covariance.size(0),
        device=covariance.device,
        dtype=covariance.dtype,
    )

    original_dtype = covariance.dtype
    covariance_double = covariance.to(torch.float64)

    try:
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance_double)
    except torch._C._LinAlgError:
        covariance_double = covariance_double + (eps * 10) * torch.eye(
            covariance.size(0),
            device=covariance.device,
            dtype=torch.float64,
        )
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance_double)

    eigenvalues = eigenvalues.to(original_dtype)
    eigenvectors = eigenvectors.to(original_dtype)
    inv_sqrt_eigenvalues = torch.rsqrt(eigenvalues.clamp(min=eps))

    covariance_inv = torch.matmul(
        eigenvectors * inv_sqrt_eigenvalues.unsqueeze(0),
        eigenvectors.T,
    )
    return torch.matmul(anchor_matrix, covariance_inv)


def l2_normalize(anchor_matrix: torch.Tensor) -> torch.Tensor:
    """Apply row-wise L2 normalization to anchor features."""
    return F.normalize(anchor_matrix, p=2, dim=1)


def normalize_anchor_matrix(
    anchor_matrix: torch.Tensor,
    config: AnchorConfig,
) -> torch.Tensor:
    """Apply the configured anchor normalization to a full anchor matrix."""
    if config.parseval_normalization:
        return parseval_normalize(anchor_matrix, eps=float(config.parseval_eps))
    if config.l2_normalization:
        return l2_normalize(anchor_matrix)
    return anchor_matrix


def sorted_global_classes(
    labels_per_agent: dict[int, torch.Tensor],
) -> list[int]:
    """Return the sorted union of labels observed across agents."""
    if not labels_per_agent:
        return []
    all_labels = torch.cat(
        [labels.detach().cpu() for labels in labels_per_agent.values()]
    )
    return sorted(int(label) for label in torch.unique(all_labels).tolist())


def build_balanced_class_plan(
    global_classes: list[int],
    *,
    num_anchors: int,
) -> dict[int, int]:
    """Distribute the anchor budget as evenly as possible across classes."""
    if not global_classes:
        return {}

    budget = max(1, int(num_anchors))
    class_order = [
        global_classes[idx]
        for idx in torch.randperm(len(global_classes)).tolist()
    ]

    if budget <= len(class_order):
        return {class_label: 1 for class_label in class_order[:budget]}

    plan = {class_label: 1 for class_label in class_order}
    remaining = budget - len(class_order)
    cursor = 0
    while remaining > 0:
        class_label = class_order[cursor % len(class_order)]
        plan[class_label] += 1
        remaining -= 1
        cursor += 1
    return plan


def build_random_class_plan(
    global_classes: list[int],
    *,
    num_anchors: int,
) -> dict[int, int]:
    """Sample a random class allocation plan under the anchor budget."""
    if not global_classes:
        return {}

    budget = max(1, int(num_anchors))
    draws = torch.randint(len(global_classes), (budget,))
    plan: dict[int, int] = {}
    for draw in draws.tolist():
        class_label = global_classes[draw]
        plan[class_label] = plan.get(class_label, 0) + 1
    return plan


def build_class_anchor_plan(
    labels_per_agent: dict[int, torch.Tensor],
    config: AnchorConfig,
) -> dict[int, int]:
    """Build a shared class-to-slot plan for the current anchor strategy."""
    strategy = supported_anchor_strategy(config.strategy)
    global_classes = sorted_global_classes(labels_per_agent)

    match strategy:
        case 'prototype':
            return {class_label: 1 for class_label in global_classes}
        case 'balanced' | 'clustered_pilots' | 'dynamic':
            return build_balanced_class_plan(
                global_classes,
                num_anchors=config.num_anchors,
            )
        case 'random':
            return build_random_class_plan(
                global_classes,
                num_anchors=config.num_anchors,
            )
        case _:
            raise ValueError(f'Unsupported anchor strategy: {strategy}')


def class_distance_order(class_latents: torch.Tensor) -> torch.Tensor:
    """Order class samples from most central to most peripheral."""
    class_center = class_latents.mean(dim=0, keepdim=True)
    distances = torch.linalg.vector_norm(class_latents - class_center, dim=1)
    return torch.argsort(distances)


def chunk_mean_anchors(
    class_latents: torch.Tensor,
    n_slots: int,
    ordered_indices: torch.Tensor | None = None,
) -> list[torch.Tensor]:
    """Split class samples into chunks and average each chunk."""
    if n_slots < 1 or len(class_latents) == 0:
        return []

    valid_slots = min(int(n_slots), len(class_latents))
    if ordered_indices is None:
        ordered_indices = torch.randperm(
            len(class_latents),
            device=class_latents.device,
        )
    else:
        ordered_indices = ordered_indices[: len(class_latents)]

    splits = torch.tensor_split(ordered_indices, valid_slots)
    return [
        class_latents[split].mean(dim=0)
        for split in splits
        if split.numel() > 0
    ]


def random_sample_anchors(
    class_latents: torch.Tensor,
    n_slots: int,
) -> list[torch.Tensor]:
    """Draw raw class-conditioned sample anchors without replacement."""
    if n_slots < 1 or len(class_latents) == 0:
        return []

    valid_slots = min(int(n_slots), len(class_latents))
    perm = torch.randperm(
        len(class_latents),
        device=class_latents.device,
    )[:valid_slots]
    return [class_latents[idx] for idx in perm.tolist()]


def build_class_anchors(
    class_latents: torch.Tensor,
    n_slots: int,
    *,
    strategy: str,
) -> list[torch.Tensor]:
    """Build semantically keyed anchors for one class."""
    if len(class_latents) == 0 or n_slots < 1:
        return []

    match strategy:
        case 'prototype':
            return [class_latents.mean(dim=0)]
        case 'random':
            return random_sample_anchors(class_latents, n_slots)
        case 'balanced' | 'dynamic':
            return chunk_mean_anchors(class_latents, n_slots)
        case 'clustered_pilots':
            return chunk_mean_anchors(
                class_latents=class_latents,
                n_slots=n_slots,
                ordered_indices=class_distance_order(class_latents),
            )
        case _:
            raise ValueError(f'Unsupported anchor strategy: {strategy}')


def build_anchor_bundles(
    latents: AnchorTensors,
    labels_per_agent: dict[int, torch.Tensor],
    config: AnchorConfig,
) -> tuple[AnchorTensors, AnchorKeys]:
    """Build per-agent anchors together with explicit semantic keys."""
    strategy = supported_anchor_strategy(config.strategy)
    if strategy == 'semantic_pilots':
        raise ValueError(
            'semantic_pilots requires shared pilot batches and cannot be '
            'built from local class labels.'
        )
    class_plan = build_class_anchor_plan(labels_per_agent, config)

    anchor_tensors: AnchorTensors = {}
    anchor_keys: AnchorKeys = {}

    for idx, anchor_source in latents.items():
        labels = labels_per_agent[idx]
        rows: list[torch.Tensor] = []
        keys: list[tuple[int, int]] = []

        for class_label in sorted(class_plan):
            class_mask = labels == class_label
            if not class_mask.any():
                continue

            class_rows = build_class_anchors(
                class_latents=anchor_source[class_mask],
                n_slots=class_plan[class_label],
                strategy=strategy,
            )
            for slot_idx, anchor_row in enumerate(class_rows):
                rows.append(anchor_row)
                keys.append((int(class_label), slot_idx))

        if not rows:
            continue

        anchor_matrix = torch.stack(rows)
        anchor_tensors[idx] = normalize_anchor_matrix(anchor_matrix, config)
        anchor_keys[idx] = keys

    return anchor_tensors, anchor_keys


def build_semantic_pilot_bundles(
    latents: AnchorTensors,
    sample_ids_per_agent: dict[int, torch.Tensor],
    config: AnchorConfig,
) -> tuple[AnchorTensors, AnchorKeys]:
    """Build anchors keyed by shared pilot sample identifiers."""
    anchor_tensors: AnchorTensors = {}
    anchor_keys: AnchorKeys = {}

    for idx, anchor_source in latents.items():
        sample_ids = sample_ids_per_agent[idx]
        unique_ids = sorted(
            int(sample_id)
            for sample_id in torch.unique(sample_ids.detach().cpu()).tolist()
        )

        rows: list[torch.Tensor] = []
        keys: list[tuple[int, int]] = []
        for sample_id in unique_ids:
            mask = sample_ids == sample_id
            if mask.any():
                rows.append(anchor_source[mask].mean(dim=0))
                keys.append((sample_id, 0))

        if not rows:
            continue

        anchor_matrix = torch.stack(rows)
        anchor_tensors[idx] = normalize_anchor_matrix(anchor_matrix, config)
        anchor_keys[idx] = keys

    return anchor_tensors, anchor_keys


def shared_anchor_rows(
    A_i: torch.Tensor,
    keys_i: list[tuple[int, int]],
    A_j: torch.Tensor,
    keys_j: list[tuple[int, int]],
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Extract anchor rows whose semantic keys are shared across agents."""
    if not keys_i or not keys_j:
        return None

    key_to_idx_i = {key: pos for pos, key in enumerate(keys_i)}
    key_to_idx_j = {key: pos for pos, key in enumerate(keys_j)}
    shared_keys = sorted(set(key_to_idx_i) & set(key_to_idx_j))
    if not shared_keys:
        return None

    idx_i = torch.tensor(
        [key_to_idx_i[key] for key in shared_keys],
        device=A_i.device,
        dtype=torch.long,
    )
    idx_j = torch.tensor(
        [key_to_idx_j[key] for key in shared_keys],
        device=A_j.device,
        dtype=torch.long,
    )

    return A_i.index_select(0, idx_i), A_j.index_select(0, idx_j)


__all__ = [
    'AnchorConfig',
    'AnchorKeys',
    'AnchorTensors',
    'VALID_ANCHOR_STRATEGIES',
    'build_anchor_bundles',
    'build_class_anchor_plan',
    'build_semantic_pilot_bundles',
    'l2_normalize',
    'normalize_anchor_matrix',
    'parseval_normalize',
    'shared_anchor_rows',
    'sorted_global_classes',
    'supported_anchor_strategy',
]
