"""Anchor-selection utilities for Sheaf-FRL."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

AnchorKeys = dict[int, list[tuple[int, int]]]
AnchorTensors = dict[int, torch.Tensor]

'''
VALID_ANCHOR_STRATEGIES = {
    'prototype',
    'uniform',
    'geometric',
    'semantic_pilots',
    'clustering',
}
'''

@dataclass(frozen=True)
class AnchorConfig:
    """Configuration for anchor construction, communication and normalization."""
    # strategy: str
    # num_anchors: int
    parseval_normalization: bool
    l2_normalization: bool
    parseval_eps: float = 1e-4
    filter_unseen_classes: bool = False
    use_prototypes: bool = False
    sparse_communication: bool = False
    sparse_epsilon: float = 1e-5

'''
def supported_anchor_strategy(anchor_strategy: str) -> str:
    """Validate the configured anchor strategy name."""
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
'''

def parseval_normalize(
    anchor_matrix: torch.Tensor,
    *,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Apply Parseval prewhitening to pilot anchor rows.

    `anchor_matrix` is `[num_pilots, latent_dim]`
    The prewhitening formulation treats latent dimensions as variables and
    pilots as observations, so it operates on `anchor_matrix.T`. The final
    `sqrt(num_pilots - 1)` scaling converts unit covariance into the
    Parseval frame convention ``A.T @ A ~= I`` for the returned row-major
    anchor matrix.
    """
    if anchor_matrix.ndim != 2:
        raise ValueError('anchor_matrix must be a 2D tensor')

    num_pilots = anchor_matrix.size(0)
    if num_pilots <= 1 or anchor_matrix.numel() == 0:
        return anchor_matrix
    if eps < 0:
        raise ValueError('eps must be non-negative')

    original_dtype = anchor_matrix.dtype
    feature_by_pilot = anchor_matrix.T.to(torch.float64)
    mean = feature_by_pilot.mean(dim=1, keepdim=True)
    centered = feature_by_pilot - mean

    covariance_denominator = num_pilots - 1
    covariance = torch.matmul(centered, centered.T) / covariance_denominator
    jitter = float(eps)
    eye = torch.eye(
        covariance.size(0),
        device=covariance.device,
        dtype=covariance.dtype,
    )
    covariance = covariance + jitter * eye

    cholesky, info = torch.linalg.cholesky_ex(covariance)
    if bool(torch.any(info != 0)):
        fallback_jitter = max(jitter * 10, 1e-12)
        cholesky = torch.linalg.cholesky(covariance + fallback_jitter * eye)

    whitened = torch.linalg.solve(cholesky, centered)
    parseval_rows = whitened.T / torch.sqrt(
        torch.tensor(
            covariance_denominator,
            device=anchor_matrix.device,
            dtype=torch.float64,
        )
    )
    return parseval_rows.to(dtype=original_dtype)

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

'''
def sorted_global_classes(labels_per_agent: dict[int, torch.Tensor]) -> list[int]:
    """Return the sorted union of labels observed across agents."""
    if not labels_per_agent:
        return []
    all_labels = torch.cat([labels.detach().cpu() for labels in labels_per_agent.values()])
    return sorted(int(label) for label in torch.unique(all_labels).tolist())


def _global_class_counts(labels_per_agent: dict[int, torch.Tensor]) -> dict[int, int]:
    """Count how often each class appears across all agents."""
    counts: dict[int, int] = {}
    for labels in labels_per_agent.values():
        unique_labels, class_counts = torch.unique(labels.detach().cpu(), return_counts=True)
        for class_label, count in zip(unique_labels.tolist(), class_counts.tolist()):
            class_idx = int(class_label)
            counts[class_idx] = counts.get(class_idx, 0) + int(count)
    return counts
'''
def shared_anchor_rows(
    A_i: torch.Tensor,
    A_j: torch.Tensor,
    labels_i: torch.Tensor,
    labels_j: torch.Tensor,
    seen_i: set[int],
    seen_j: set[int],
    config: AnchorConfig,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Extract paired anchor rows, matching classes across both agents."""
    present_i = set(labels_i.tolist())
    present_j = set(labels_j.tolist())
    target_classes = present_i.intersection(present_j)
    if config.filter_unseen_classes:
        target_classes = target_classes.intersection(seen_i, seen_j)

    if not target_classes:
        return None

    if config.use_prototypes:
        proto_i, proto_j = [], []
        for c in sorted(target_classes):
            mask_i = labels_i == c
            mask_j = labels_j == c
            if mask_i.any() and mask_j.any():
                proto_i.append(A_i[mask_i].mean(dim=0))
                proto_j.append(A_j[mask_j].mean(dim=0))

        if not proto_i:
            return None

        return torch.stack(proto_i), torch.stack(proto_j)

    target_tensor_i = torch.tensor(
        sorted(target_classes),
        device=labels_i.device,
        dtype=labels_i.dtype,
    )
    target_tensor_j = torch.tensor(
        sorted(target_classes),
        device=labels_j.device,
        dtype=labels_j.dtype,
    )
    mask_i = torch.isin(labels_i, target_tensor_i)
    mask_j = torch.isin(labels_j, target_tensor_j)
    if not mask_i.any() or not mask_j.any():
        return None

    selected_labels_i = labels_i[mask_i]
    selected_labels_j = labels_j[mask_j]
    if (
        selected_labels_i.shape != selected_labels_j.shape
        or not torch.equal(selected_labels_i, selected_labels_j)
    ):
        return None

    return A_i[mask_i], A_j[mask_j]


def communication_anchor_payload(
    anchor_matrix: torch.Tensor,
    labels: torch.Tensor,
    config: AnchorConfig,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Return the communicated anchor payload for one exchange step.

    Communication cost ignores unseen-class filtering because agents do not
    know which classes their neighbors have observed before exchanging pilot
    features. When prototype mode is enabled, each agent sends one prototype
    per class present in its local pilot batch; otherwise it sends the full
    pilot embedding matrix. When sparse communication is enabled, the helper
    compares the dense payload against a masked sparse representation made of
    the surviving values plus their coordinate indices, and returns whichever
    is smaller.
    """
    if not config.use_prototypes:
        payload = anchor_matrix
    else:
        unique_classes = torch.unique(labels, sorted=True)
        if unique_classes.numel() == 0:
            return anchor_matrix[:0]
        else:
            prototypes = []
            for class_label in unique_classes.tolist():
                mask = labels == class_label
                if mask.any():
                    prototypes.append(anchor_matrix[mask].mean(dim=0))
            payload = torch.stack(prototypes, dim=0) if prototypes else anchor_matrix[:0]

    if config.sparse_communication and payload.numel() > 0:
        # mask elements that survive the threshold
        sparse_mask = payload.abs() > config.sparse_epsilon
        # 1D tensor of non-zero values after sparsification
        values = payload[sparse_mask]
        
        # coordinate rows of the surviving entries in the original payload
        indices = torch.nonzero(sparse_mask, as_tuple=False).to(torch.int32)
        
        # theoretical bytes of dense vs. sparse representation
        dense_bytes = payload.numel() * payload.element_size()
        sparse_bytes = (values.numel() * values.element_size()) + (indices.numel() * indices.element_size())
        
        if sparse_bytes < dense_bytes:
            return (values, indices)

    return payload


'''
def build_coverage_class_plan(global_classes: list[int], *, num_anchors: int) -> dict[int, int]:
    """Distribute the anchor budget as evenly as possible across classes."""
    if not global_classes:
        return {}

    budget = max(1, int(num_anchors))
    class_order = list(global_classes)

    if budget <= len(class_order):
        return {class_label: 1 for class_label in class_order[:budget]}

    plan = dict.fromkeys(class_order, 1)
    remaining = budget - len(class_order)
    cursor = 0
    while remaining > 0:
        class_label = class_order[cursor % len(class_order)]
        plan[class_label] += 1
        remaining -= 1
        cursor += 1
    return plan


def build_proportional_class_plan(labels_per_agent: dict[int, torch.Tensor], *, num_anchors: int) -> dict[int, int]:
    """Allocate anchor budget proportionally to empirical class mass."""
    class_counts = _global_class_counts(labels_per_agent)
    if not class_counts:
        return {}

    ordered_classes = sorted(class_counts)
    budget = max(1, int(num_anchors))
    if budget <= len(ordered_classes):
        selected = sorted(
            ordered_classes,
            key=lambda class_label: (-class_counts[class_label], class_label),
        )[:budget]
        return {class_label: 1 for class_label in sorted(selected)}

    total_count = sum(class_counts.values())
    fractional_targets = {
        class_label: budget * class_counts[class_label] / total_count
        for class_label in ordered_classes
    }
    plan = {class_label: max(1, int(fractional_targets[class_label])) for class_label in ordered_classes}
    allocated = sum(plan.values())

    if allocated > budget:
        removable = sorted(
            ordered_classes,
            key=lambda class_label: (fractional_targets[class_label] - plan[class_label], class_label),
        )
        for class_label in removable:
            while allocated > budget and plan[class_label] > 1:
                plan[class_label] -= 1
                allocated -= 1
            if allocated == budget:
                break
    elif allocated < budget:
        expandable = sorted(
            ordered_classes,
            key=lambda class_label: (plan[class_label] - fractional_targets[class_label], class_label),
        )
        cursor = 0
        while allocated < budget:
            class_label = expandable[cursor % len(expandable)]
            plan[class_label] += 1
            allocated += 1
            cursor += 1

    return {class_label: count for class_label, count in plan.items() if count > 0}


def build_class_anchor_plan(labels_per_agent: dict[int, torch.Tensor], config: AnchorConfig) -> dict[int, int]:
    """Build a shared class-to-slot plan for the current anchor strategy."""
    strategy = supported_anchor_strategy(config.strategy)
    global_classes = sorted_global_classes(labels_per_agent)

    match strategy:
        case 'prototype':
            return dict.fromkeys(global_classes, 1)
        case 'uniform':
            return build_proportional_class_plan(labels_per_agent, num_anchors=config.num_anchors)
        case 'geometric' | 'clustering':
            return build_coverage_class_plan(global_classes, num_anchors=config.num_anchors)
        case _:
            raise ValueError(f'Unsupported anchor strategy: {strategy}')


def farthest_point_indices(anchor_source: torch.Tensor, n_points: int) -> torch.Tensor:
    """Select a diverse subset with farthest-point sampling."""
    if n_points < 1 or len(anchor_source) == 0:
        return torch.empty(0, dtype=torch.long, device=anchor_source.device)

    valid_points = min(int(n_points), len(anchor_source))
    source = anchor_source.detach()
    center = source.mean(dim=0, keepdim=True)
    distances_to_center = torch.linalg.vector_norm(source - center, dim=1)
    first_idx = int(torch.argmax(distances_to_center).item())

    selected = [first_idx]
    min_distances = torch.cdist(source, source[first_idx : first_idx + 1]).squeeze(1)

    while len(selected) < valid_points:
        candidate_idx = int(torch.argmax(min_distances).item())
        if candidate_idx in selected:
            break
        selected.append(candidate_idx)
        candidate_distances = torch.cdist(source, source[candidate_idx : candidate_idx + 1]).squeeze(1)
        min_distances = torch.minimum(min_distances, candidate_distances)

    return torch.tensor(selected, dtype=torch.long, device=anchor_source.device)


def random_sample_anchors(class_latents: torch.Tensor, n_slots: int) -> list[torch.Tensor]:
    """Draw raw class-conditioned sample anchors without replacement."""
    if n_slots < 1 or len(class_latents) == 0:
        return []

    valid_slots = min(int(n_slots), len(class_latents))
    perm = torch.randperm(len(class_latents), device=class_latents.device)[:valid_slots]
    return [class_latents[idx] for idx in perm.tolist()]


def geometric_sample_anchors(class_latents: torch.Tensor, n_slots: int) -> list[torch.Tensor]:
    """Use farthest-point sampling to cover each class with diverse anchors."""
    selected = farthest_point_indices(class_latents, n_slots)
    return [class_latents[idx] for idx in selected.tolist()]


def cluster_centroid_anchors(class_latents: torch.Tensor, n_slots: int, *, max_iter: int = 10) -> list[torch.Tensor]:
    """Approximate K-means centroids for class-conditioned coverage."""
    if n_slots < 1 or len(class_latents) == 0:
        return []

    valid_slots = min(int(n_slots), len(class_latents))
    if valid_slots == len(class_latents):
        return [class_latents[idx] for idx in range(len(class_latents))]

    with torch.no_grad():
        init_idx = farthest_point_indices(class_latents, valid_slots)
        source = class_latents.detach()
        centers = source.index_select(0, init_idx)

        for _ in range(max_iter):
            distances = torch.cdist(source, centers)
            assignments = torch.argmin(distances, dim=1)

            new_centers = []
            for cluster_idx in range(valid_slots):
                mask = assignments == cluster_idx
                if mask.any():
                    new_centers.append(source[mask].mean(dim=0))
                else:
                    new_centers.append(centers[cluster_idx])
            new_centers = torch.stack(new_centers)

            if torch.allclose(new_centers, centers):
                centers = new_centers
                break
            centers = new_centers

        final_distances = torch.cdist(source, centers)
        assignments = torch.argmin(final_distances, dim=1)

    centroids: list[torch.Tensor] = []
    for cluster_idx in range(valid_slots):
        mask = assignments == cluster_idx
        if mask.any():
            centroids.append(class_latents[mask].mean(dim=0))

    return centroids


def build_class_anchors(class_latents: torch.Tensor, n_slots: int, *, strategy: str) -> list[torch.Tensor]:
    """Build semantically keyed anchors for one class according to the specified strategy."""
    if len(class_latents) == 0 or n_slots < 1:
        return []

    match strategy:
        case 'prototype':
            return [class_latents.mean(dim=0)]
        case 'uniform':
            return random_sample_anchors(class_latents, n_slots)
        case 'geometric':
            return geometric_sample_anchors(class_latents, n_slots)
        case 'clustering':
            return cluster_centroid_anchors(class_latents, n_slots)
        case _:
            raise ValueError(f'Unsupported anchor strategy: {strategy}')


def build_anchor_bundles(
    latents: AnchorTensors, 
    labels_per_agent: dict[int, torch.Tensor], 
    config: AnchorConfig
) -> tuple[AnchorTensors, AnchorKeys]:
    """Build per-agent anchors together with explicit semantic keys."""
    strategy = supported_anchor_strategy(config.strategy)
    if strategy == 'semantic_pilots':
        raise ValueError('semantic_pilots requires shared pilot batches and cannot be built from local class labels.')
    
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
    config: AnchorConfig
) -> tuple[AnchorTensors, AnchorKeys]:
    """Build anchors keyed by shared pilot sample identifiers."""
    anchor_tensors: AnchorTensors = {}
    anchor_keys: AnchorKeys = {}

    for idx, anchor_source in latents.items():
        sample_ids = sample_ids_per_agent[idx]
        unique_ids = sorted(int(sample_id) for sample_id in torch.unique(sample_ids.detach().cpu()).tolist())

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


def _group_anchor_positions(keys: list[tuple[int, int]]) -> dict[int, list[int]]:
    """Group anchor row positions by their primary semantic identifier."""
    grouped_positions: dict[int, list[int]] = {}
    for position, key in enumerate(keys):
        grouped_positions.setdefault(int(key[0]), []).append(position)
    return grouped_positions


def _greedy_match_positions(distance_matrix: torch.Tensor) -> list[tuple[int, int]]:
    """Greedily match row indices from the smallest available distances.
    Used to match anchors that are most similar intraclass and to avoid
    collaps/distortions trying to match an husky to a bulldog"""
    if distance_matrix.numel() == 0:
        return []

    n_rows_i, n_rows_j = distance_matrix.shape
    flat_order = torch.argsort(distance_matrix.reshape(-1))

    used_i: set[int] = set()
    used_j: set[int] = set()
    matches: list[tuple[int, int]] = []

    for flat_idx in flat_order.tolist():
        row_i = int(flat_idx // n_rows_j)
        row_j = int(flat_idx % n_rows_j)
        if row_i in used_i or row_j in used_j:
            continue

        used_i.add(row_i)
        used_j.add(row_j)
        matches.append((row_i, row_j))

        if len(matches) == min(n_rows_i, n_rows_j):
            break

    return sorted(matches)


def shared_anchor_rows(
    A_i: torch.Tensor,
    keys_i: list[tuple[int, int]],
    A_j: torch.Tensor,
    keys_j: list[tuple[int, int]],
    *,
    match_by: str = 'exact',
    projected_A_j: torch.Tensor | None = None,
    class_matching: bool = True,
    intraclass_matching: bool = True,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Extract anchor rows shared across agents under a chosen matching rule."""
    if not keys_i or not keys_j:
        return None

    if match_by == 'exact':
        # using exact keys as for semantic pilots strategy
        key_to_idx_i = {key: pos for pos, key in enumerate(keys_i)}
        key_to_idx_j = {key: pos for pos, key in enumerate(keys_j)}
        shared_keys = sorted(set(key_to_idx_i) & set(key_to_idx_j))
        if not shared_keys:
            return None

        idx_i = torch.tensor([key_to_idx_i[key] for key in shared_keys], device=A_i.device, dtype=torch.long)
        idx_j = torch.tensor([key_to_idx_j[key] for key in shared_keys], device=A_j.device, dtype=torch.long)
        return A_i.index_select(0, idx_i), A_j.index_select(0, idx_j)

    if match_by != 'class':
        raise ValueError(f'Unknown match_by strategy: {match_by}')

    matching_rows_j = projected_A_j if projected_A_j is not None else A_j

    if not class_matching:
        # aligning the unfiltered but same sized anchor set with no class logic
        min_rows = min(A_i.shape[0], A_j.shape[0])
        if min_rows == 0:
            return None
        return A_i[:min_rows], A_j[:min_rows]

    grouped_i = _group_anchor_positions(keys_i)
    grouped_j = _group_anchor_positions(keys_j)
    shared_classes = sorted(set(grouped_i) & set(grouped_j))
    if not shared_classes:
        return None

    matched_positions_i: list[int] = []
    matched_positions_j: list[int] = []

    for class_label in shared_classes:
        positions_i = grouped_i[class_label]
        positions_j = grouped_j[class_label]

        if intraclass_matching:
            # further match within the class to preserve finer-grained semantics and avoid collapses/distortions
            idx_i = torch.tensor(positions_i, device=A_i.device, dtype=torch.long)
            idx_j = torch.tensor(positions_j, device=matching_rows_j.device, dtype=torch.long)

            class_rows_i = A_i.index_select(0, idx_i)
            class_rows_j = matching_rows_j.index_select(0, idx_j)
            distances = torch.cdist(class_rows_i, class_rows_j)

            for local_i, local_j in _greedy_match_positions(distances):
                matched_positions_i.append(positions_i[local_i])
                matched_positions_j.append(positions_j[local_j])

        else:
            # Just pair them up blindly in the order they were generated
            min_class_rows = min(len(positions_i), len(positions_j))
            matched_positions_i.extend(positions_i[:min_class_rows])
            matched_positions_j.extend(positions_j[:min_class_rows])

    if not matched_positions_i:
        return None

    gathered_i = torch.tensor(matched_positions_i, device=A_i.device, dtype=torch.long)
    gathered_j = torch.tensor(matched_positions_j, device=A_j.device, dtype=torch.long)
    return A_i.index_select(0, gathered_i), A_j.index_select(0, gathered_j)

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
'''

__all__ = [
    'AnchorConfig',
    'l2_normalize',
    'normalize_anchor_matrix',
    'parseval_normalize',
    'shared_anchor_rows',
]
