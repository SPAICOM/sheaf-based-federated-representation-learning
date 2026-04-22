"""Utility functions shared across all datamodules.

This module provides helper functions for dataset splitting and manipulation
used by all datamodule classes.
"""

from math import ceil
from typing import TypeVar

import torch
from torch.utils.data import ConcatDataset, Dataset

T = TypeVar('T')


class PairwiseDataset(Dataset):
    """Dataset that pairs two aligned datasets to produce per-edge samples.

    Used for ``comm_data='pairwise_pilots'`` to build one pilot dataloader
    per agent pair ``(i, j)``.  Both datasets **must** be constructed from
    the same ordered ``sample_ids`` list so that position *k* in
    ``dataset_i`` and position *k* in ``dataset_j`` correspond to the same
    underlying physical sample.

    Parameters
    ----------
    dataset_i : Dataset
        Pilot dataset for agent *i*.  Each item must follow the format
        ``(x, label)`` or ``(x, label, sample_id)``.
    dataset_j : Dataset
        Pilot dataset for agent *j*.  Must have the same length as
        ``dataset_i``.

    Returns
    -------
    tuple
        ``(x_i, x_j, label)`` when the wrapped datasets do **not** carry
        sample identifiers, or ``(x_i, x_j, label, sample_id)`` otherwise.
        ``sample_id`` is taken from ``dataset_i``.

    Raises
    ------
    ValueError
        If ``dataset_i`` and ``dataset_j`` have different lengths.
    """

    def __init__(self, dataset_i: Dataset, dataset_j: Dataset) -> None:
        if len(dataset_i) != len(dataset_j):
            raise ValueError(
                'PairwiseDataset requires both datasets to have the same '
                f'length, got {len(dataset_i)} and {len(dataset_j)}.'
            )
        self.dataset_i = dataset_i
        self.dataset_j = dataset_j

    def __len__(self) -> int:
        return len(self.dataset_i)

    def __getitem__(self, idx: int):
        item_i = self.dataset_i[idx]
        item_j = self.dataset_j[idx]
        x_i, label = item_i[0], item_i[1]
        x_j = item_j[0]
        if len(item_i) == 3:
            return x_i, x_j, label, item_i[2]
        return x_i, x_j, label


def compute_split_indices(
    total_size: int,
    val_split: float,
    test_split: float,
    seed: int,
    pilot_split: float = 0.0,
    pilot_num_samples: int | None = None,
) -> dict[str, list[int]]:
    """Compute deterministic pilot/train/val/test indices from one dataset.

    Creates reproducible data splits for federated learning scenarios where
    each agent needs consistent train/validation/test splits. Optionally
    creates a pilot set for calibration or inspection purposes.

    Parameters
    ----------
    total_size : int
        Total number of samples in the dataset. Must be at least 1.
    val_split : float
        Fraction of data to allocate for validation (after pilot removal).
        Must be non-negative.
    test_split : float
        Fraction of data to allocate for testing (after pilot removal).
        Must be non-negative.
    seed : int
        Random seed for reproducible shuffling and splitting.
    pilot_split : float, optional
        Fraction of data to allocate for pilot set (default: 0.0).
        Must be in [0, 1). If pilot_num_samples is specified, this is ignored.
    pilot_num_samples : int or None, optional
        Absolute number of samples for pilot set (default: None).
        If specified, takes precedence over pilot_split. Must be non-negative.

    Returns
    -------
    dict[str, list[int]]
        Dictionary with keys 'pilot', 'train', 'val', 'test' mapping to
        lists of indices for each split.

    Raises
    ------
    ValueError
        If any parameter constraints are violated:
        - total_size < 1
        - val_split or test_split < 0
        - val_split + test_split >= 1
        - pilot_split not in [0, 1)
        - pilot_num_samples < 0 (when not None)
        - Pilot split leaves no samples for train/val/test

    Notes
    -----
    The splitting strategy:
    1. Extract pilot set first (if requested)
    2. Split remaining data into train/val/test with proportions:
       - train: remaining - val_count - test_count
       - val: remaining_size * val_split (rounded)
       - test: remaining_size * test_split (rounded)
    3. Adjust val/test counts if they exceed available samples
       by preferentially reducing the larger split
    """
    if total_size < 1:
        raise ValueError('total_size must be at least 1')
    if val_split < 0 or test_split < 0:
        raise ValueError('val_split and test_split must be non-negative')
    if val_split + test_split >= 1:
        raise ValueError('val_split + test_split must be smaller than 1')
    if not 0 <= pilot_split < 1:
        raise ValueError('pilot_split must be in [0, 1)')
    if pilot_num_samples is not None and pilot_num_samples < 0:
        raise ValueError('pilot_num_samples must be non-negative')

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(total_size, generator=generator).tolist()

    pilot_count = 0
    if pilot_num_samples is not None and pilot_num_samples > 0:
        pilot_count = min(int(pilot_num_samples), total_size - 1)
    elif pilot_split > 0:
        pilot_count = min(
            max(int(round(total_size * pilot_split)), 1),
            total_size - 1,
        )

    pilot_indices = permutation[:pilot_count]
    remaining_indices = permutation[pilot_count:]
    remaining_size = len(remaining_indices)
    if remaining_size < 1:
        raise ValueError('Pilot split leaves no samples for train/val/test.')

    val_count = int(round(remaining_size * val_split))
    test_count = int(round(remaining_size * test_split))

    max_held_out = max(remaining_size - 1, 0)
    while val_count + test_count > max_held_out:
        if test_count >= val_count and test_count > 0:
            test_count -= 1
        elif val_count > 0:
            val_count -= 1
        else:
            break

    train_count = remaining_size - val_count - test_count
    train_indices = remaining_indices[:train_count]
    val_indices = remaining_indices[train_count : train_count + val_count]
    test_indices = remaining_indices[train_count + val_count :]

    return {
        'pilot': pilot_indices,
        'train': train_indices,
        'val': val_indices,
        'test': test_indices,
    }


def assign_agents_to_random_groups(
    n_agents: int,
    group_values: list[T],
    seed: int,
) -> dict[int, T]:
    """Assign agents to seeded random groups as evenly as possible."""
    if n_agents < 1:
        raise ValueError('n_agents must be positive')
    if not group_values:
        raise ValueError('group_values must be non-empty')

    shuffled_agents = torch.randperm(
        n_agents,
        generator=torch.Generator().manual_seed(seed),
    ).tolist()
    base_group_size, remainder = divmod(n_agents, len(group_values))

    assignment: dict[int, T] = {}
    start = 0
    for group_idx, group_value in enumerate(group_values):
        current_group_size = base_group_size + (
            1 if group_idx < remainder else 0
        )
        for agent_idx in shuffled_agents[start : start + current_group_size]:
            assignment[agent_idx] = group_value
        start += current_group_size

    return {agent_idx: assignment[agent_idx] for agent_idx in range(n_agents)}


def repeat_dataset_to_num_samples(
    dataset: Dataset,
    target_num_samples: int,
) -> Dataset:
    """Repeat a dataset until it has at least the requested size.

    Creates a new dataset by concatenating multiple copies of the input dataset
    to reach or exceed the target number of samples. Useful for creating
    pilot datasets that need to be sampled multiple times during training.

    Parameters
    ----------
    dataset : Dataset
        PyTorch dataset to repeat. Must implement __len__ method.
    target_num_samples : int
        Target number of samples for the repeated dataset.
        If less than or equal to len(dataset), returns the original dataset.

    Returns
    -------
    Dataset
        Either the original dataset (if target_num_samples <= len(dataset))
        or a ConcatDataset containing multiple copies of the original dataset
        sufficient to reach at least target_num_samples.

    Notes
    -----
    The repeat factor is calculated as ceil(target_num_samples / len(dataset)).
    If the dataset is empty (len(dataset) == 0), returns the original dataset
    to avoid division by zero.
    """
    if target_num_samples <= len(dataset) or len(dataset) == 0:
        return dataset

    repeat_factor = ceil(target_num_samples / len(dataset))
    return ConcatDataset([dataset] * repeat_factor)
