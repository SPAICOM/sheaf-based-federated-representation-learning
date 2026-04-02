"""Shared datamodule utilities."""

from math import ceil

import torch
from torch.utils.data import ConcatDataset, Dataset


def compute_split_indices(
    total_size: int,
    val_split: float,
    test_split: float,
    seed: int,
    pilot_split: float = 0.0,
    pilot_num_samples: int | None = None,
) -> dict[str, list[int]]:
    """Compute deterministic pilot/train/val/test indices from one dataset."""
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
    val_indices = remaining_indices[train_count:train_count + val_count]
    test_indices = remaining_indices[train_count + val_count:]

    return {
        'pilot': pilot_indices,
        'train': train_indices,
        'val': val_indices,
        'test': test_indices,
    }


def repeat_dataset_to_num_samples(
    dataset: Dataset,
    target_num_samples: int,
) -> Dataset:
    """Repeat a dataset until it has at least the requested size."""
    if target_num_samples <= len(dataset) or len(dataset) == 0:
        return dataset

    repeat_factor = ceil(target_num_samples / len(dataset))
    return ConcatDataset([dataset] * repeat_factor)
