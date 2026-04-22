"""Client-first classification datamodule for heterogeneous non-IID splits.

This module mirrors :mod:`classification_datamodule` but changes the split
construction for non-IID settings:

1. Remove any shared pilot set from the full dataset.
2. Partition the remaining samples across clients exactly once.
3. Split each client's local pool into train/val/test with a stratified split.

That ordering preserves each client's local label and quantity skew across
its held-out splits, which the global-first ``ClassificationDataModule`` does
not guarantee.
"""

from __future__ import annotations

import torch
from datasets import concatenate_datasets, load_dataset
from PIL import Image
from torchvision import transforms

from src.datamodules.classification_datamodule import (
    ClassificationDataModule,
    ClassificationDataset,
)
from src.datamodules.utils import compute_split_indices
from src.utils.data_partitioner import (
    partition_non_iid,
    partition_non_iid_fair,
    partition_non_iid_with_margin,
)


class HeteroClassificationDataModule(ClassificationDataModule):
    """Classification datamodule with client-first non-IID splitting.

    The public interface intentionally matches ``ClassificationDataModule`` so
    existing experiment wiring can switch targets with no code changes.
    """

    _CLIENT_FIRST_SPLITS = {
        'non_iid',
        'non_iid_with_margin',
        'non_iid_fair',
    }

    def __init__(self, *args, safety_margin: int = 10, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.safety_margin = int(safety_margin)

    def _partition_client_indices(
        self,
        labels: list[int],
    ) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
        """Partition the full dataset once to assign local client pools."""
        if self.split_strategy == 'non_iid':
            return partition_non_iid(
                labels=labels,
                n_agents=self.n_agents,
                classes_per_agent=self.classes_per_agent,
                seed=self.seed,
                alpha=self.alpha,
                return_agent_classes=True,
            )

        if self.split_strategy == 'non_iid_with_margin':
            return partition_non_iid_with_margin(
                labels=labels,
                n_agents=self.n_agents,
                classes_per_agent=self.classes_per_agent,
                seed=self.seed,
                alpha=self.alpha,
                return_agent_classes=True,
                safety_margin=self.safety_margin,
            )

        if self.split_strategy == 'non_iid_fair':
            return partition_non_iid_fair(
                labels=labels,
                n_agents=self.n_agents,
                classes_per_agent=self.classes_per_agent,
                seed=self.seed,
                alpha=self.alpha,
                return_agent_classes=True,
                safety_margin=self.safety_margin,
            )

        raise ValueError(
            f'Unsupported hetero split_strategy: {self.split_strategy}'
        )

    def _local_stratified_split_indices(
        self,
        labels: list[int],
        *,
        seed: int,
        train_class_minimum: int = 0,
    ) -> dict[str, list[int]]:
        """Split one client's local pool while preserving class skew.

        The split is stratified by label so each held-out split follows the
        client's local class distribution as closely as integer counts allow.
        """
        if not labels:
            return {'train': [], 'val': [], 'test': []}

        labels_tensor = torch.tensor(labels, dtype=torch.long)
        generator = torch.Generator().manual_seed(seed)

        train_indices: list[int] = []
        val_indices: list[int] = []
        test_indices: list[int] = []

        for class_label in torch.unique(labels_tensor, sorted=True).tolist():
            class_indices = torch.where(labels_tensor == int(class_label))[0]
            shuffled_indices = class_indices[
                torch.randperm(len(class_indices), generator=generator)
            ].tolist()
            class_size = len(shuffled_indices)

            val_count = int(round(class_size * self.val_split))
            test_count = int(round(class_size * self.test_split))

            min_train = min(
                class_size,
                max(int(train_class_minimum), 1),
            )
            max_held_out = max(class_size - min_train, 0)

            while val_count + test_count > max_held_out:
                if test_count >= val_count and test_count > 0:
                    test_count -= 1
                elif val_count > 0:
                    val_count -= 1
                else:
                    break

            train_count = class_size - val_count - test_count
            if train_count <= 0 and class_size > 0:
                if test_count >= val_count and test_count > 0:
                    test_count -= 1
                elif val_count > 0:
                    val_count -= 1
                train_count = class_size - val_count - test_count

            train_indices.extend(shuffled_indices[:train_count])
            val_indices.extend(
                shuffled_indices[train_count : train_count + val_count]
            )
            test_indices.extend(shuffled_indices[train_count + val_count :])

        return {
            'train': sorted(train_indices),
            'val': sorted(val_indices),
            'test': sorted(test_indices),
        }

    def setup(self, stage: str | None = None) -> None:
        """Load the dataset and build client-first non-IID splits."""
        if self.split_strategy not in self._CLIENT_FIRST_SPLITS:
            super().setup(stage=stage)
            return

        ds = load_dataset(f'{self.repo}/{self.name}')
        splits = [ds[s] for s in ds]
        all_data = concatenate_datasets(splits)

        self.train_datasets: dict[int, ClassificationDataset] = {}
        self.val_datasets: dict[int, ClassificationDataset] = {}
        self.test_datasets: dict[int, ClassificationDataset] = {}
        self.pilot_datasets: dict[int, ClassificationDataset] = {}
        self.num_classes: dict[int, int] = {}

        if self.n_agents is None:
            max_key = 0
            if self.agent_rotations:
                max_key = max(max_key, max(self.agent_rotations.keys()))
            if self.agent_classes:
                max_key = max(max_key, max(self.agent_classes.keys()))
            self.n_agents = max_key + 1

        label_key = self.attributes[0]

        if self.pilot_split > 0 or self.pilot_num_samples:
            pilot_split_indices = compute_split_indices(
                total_size=len(all_data),
                val_split=0.0,
                test_split=0.0,
                seed=self.seed,
                pilot_split=self.pilot_split,
                pilot_num_samples=self.pilot_num_samples,
            )
            pilot_indices = pilot_split_indices['pilot']
            pilot_data = all_data.select(pilot_indices)
            remaining_data = all_data.select(pilot_split_indices['train'])

            if pilot_indices:
                self._build_pilot_datasets(
                    pilot_data=pilot_data,
                    pilot_indices=pilot_indices,
                    label_key=label_key,
                )
        else:
            remaining_data = all_data

        client_indices, sampled_agent_classes = self._partition_client_indices(
            labels=remaining_data[label_key]
        )
        self.agent_classes = sampled_agent_classes

        train_class_minimum = (
            self.safety_margin
            if (
                self.starve_clients
                and self.split_strategy in {'non_iid_with_margin', 'non_iid_fair'}
            )
            else 0
        )

        for client_idx in range(self.n_agents):
            local_indices = client_indices[client_idx]
            local_data = remaining_data.select(local_indices)
            local_split_indices = self._local_stratified_split_indices(
                labels=local_data[label_key],
                seed=self.seed + client_idx,
                train_class_minimum=train_class_minimum,
            )

            rotation = self.agent_rotations.get(client_idx, 0)
            self.train_datasets[client_idx] = ClassificationDataset(
                local_data.select(local_split_indices['train']),
                self.data_key,
                label_key,
                rotation_angle=rotation,
            )
            self.val_datasets[client_idx] = ClassificationDataset(
                local_data.select(local_split_indices['val']),
                self.data_key,
                label_key,
                rotation_angle=rotation,
            )
            self.test_datasets[client_idx] = ClassificationDataset(
                local_data.select(local_split_indices['test']),
                self.data_key,
                label_key,
                rotation_angle=rotation,
            )
            self.num_classes[client_idx] = len(
                set(self.train_datasets[client_idx].dataset[label_key])
            )

        if self.starve_clients:
            self._starve_training_datasets(label_key)

        sample, _ = self.train_datasets[0][0]
        if isinstance(sample, Image.Image):
            self.input_shape = transforms.ToTensor()(sample).shape
        else:
            self.input_shape = sample.shape

        self.num_classes['label'] = len(set(all_data[label_key]))
        self.models = list(range(self.n_agents))
        self.input_dims = {str(i): self.input_shape[0] for i in self.models}
