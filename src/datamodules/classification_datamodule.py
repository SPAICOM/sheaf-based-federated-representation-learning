"""
Classification datamodule for loading and distributing HuggingFace datasets.

This module provides data loading capabilities for image classification
tasks (e.g., CIFAR10, CIFAR100, MNIST) with support for federated learning
scenarios where data is split across multiple agents.

Features:
- Load datasets directly from HuggingFace
- Split data uniformly or by class across agents
- Apply rotation transformations per agent
- Create train/val/test partitions
"""

import lightning as l
import torch
from datasets import concatenate_datasets, load_dataset
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.utils.data_partitioner import partition_non_iid


def _collate_fn(batch):
    """Collate function for handling mixed image and tensor data in batches.

    This function handles the case where batch items may contain either
    PIL Images or pre-converted tensors, converting images to tensors
    while passing through existing tensors.

    Parameters
    ----------
    batch : list[tuple]
        List of (data, label) tuples from the dataset.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Stacked data tensor and stacked label tensor.
    """
    data_list, label_list = zip(*batch)
    data_out = []
    for data in data_list:
        if isinstance(data, Image.Image):
            data_out.append(transforms.ToTensor()(data))
        else:
            data_out.append(data)
    return torch.stack(data_out), torch.stack(label_list)


class ClassificationDataset(Dataset):
    """Dataset wrapper for classification data from HuggingFace.

    Wraps a HuggingFace dataset containing data and associated labels.

    Parameters
    ----------
    hf_dataset : datasets.Dataset
        HuggingFace dataset containing data and label columns.
    data_key : str
        Name of the column containing the data (e.g., 'img', 'image').
    label_key : str
        Name of the column containing label data.
    rotation_angle : float, optional
        Rotation angle in degrees (0, 90, 180, 270). Default: 0.
    """

    def __init__(
        self,
        hf_dataset,
        data_key: str,
        label_key: str = 'label',
        rotation_angle: float = 0,
    ) -> None:
        self.dataset = hf_dataset
        self.data_key = data_key
        self.label_key = label_key
        # Normalize rotation angle to [0, 360) range
        self.rotation_angle = rotation_angle % 360

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(
        self, idx: int
    ) -> tuple[Image.Image | torch.Tensor, torch.Tensor]:
        """Get a single sample from the dataset.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve.

        Returns
        -------
        tuple
            Tuple of (data, label).
        """
        item = self.dataset[idx]

        # Extract data from the specified column
        # Convert list data to tensor (e.g., for vector features)
        data = item[self.data_key]
        if isinstance(data, list):
            data = torch.tensor(data, dtype=torch.float32)

        # Extract label from specified column
        # Convert integer labels to long tensors for classification
        label = item[self.label_key]
        if isinstance(label, int):
            label = torch.tensor(label, dtype=torch.long)

        return data, label


class ClassificationDataModule(l.LightningDataModule):
    """DataModule for standard classification datasets from HuggingFace.

    Loads datasets like CIFAR10, CIFAR100, and MNIST directly from
    HuggingFace. Merges all available splits and re-splits into custom
    train/val/test partitions. Supports multiple agents with data splitting.

    Parameters
    ----------
    repo : str
        HuggingFace repository namespace (e.g., 'uoft-cs', 'ylecun').
    name : str
        Name of the dataset (e.g., 'cifar10', 'cifar100', 'mnist').
    data_key : str
        Name of the column containing the data (e.g., 'img', 'image').
    attributes : list[str]
        List of attribute columns to use as labels.
    n_agents : int, optional
        Number of agents to split data across. If not provided, inferred
        from the maximum key in agent_rotations or agent_classes.
    agent_rotations : dict[int, float], optional
        Dictionary mapping agent indices to rotation angles in degrees.
        Each agent sees images rotated by its specified angle.
        Default: None (no rotation for all agents).
    split_strategy : str, optional
        How to split data across agents. Options: 'uniform' (default),
        'class_partition' (each agent sees specific classes),
        'non_iid' (each agent is randomly assigned ``classes_per_agent``
        classes and samples are distributed among assigned agents).
    agent_classes : dict[int, list[int]], optional
        Dictionary mapping agent indices to list of class labels they see.
        Only used when split_strategy='class_partition'.
    classes_per_agent : int, optional
        Number of classes randomly assigned to each agent. Only used when
        split_strategy='non_iid'. Default: 2.
    alpha : float, optional
        Dirichlet concentration parameter controlling statistical skew
        when split_strategy='non_iid'. Lower values produce more skew
        (one agent gets most samples), higher values approach uniform.
        Default: 0.5.
    batch_size : int, optional
        Batch size for dataloaders (default: 64).
    num_workers : int, optional
        Number of worker processes for data loading (default: 4).
    mode : str, optional
        Mode for CombinedLoader (default: 'min_size').
    val_split : float, optional
        Fraction of data for validation (default: 0.1).
    test_split : float, optional
        Fraction of data for testing (default: 0.1).
    seed : int, optional
        Random seed for reproducibility (default: 42).

    Attributes
    ----------
    train_datasets : dict[int, ClassificationDataset]
        Dictionary mapping agent indices to training datasets.
    val_datasets : dict[int, ClassificationDataset]
        Dictionary mapping agent indices to validation datasets.
    test_datasets : dict[int, ClassificationDataset]
        Dictionary mapping agent indices to test datasets.
    num_classes : dict[int, int]
        Dictionary mapping agent indices to number of classes they see.
    input_shape : tuple
        Shape of input data.

    Raises
    ------
    ValueError
        If split_strategy='class_partition' and an agent has no classes
        assigned (agent_classes[i] is empty).
    """

    def __init__(
        self,
        repo: str,
        name: str,
        data_key: str,
        attributes: list[str],
        n_agents: int | None = None,
        agent_rotations: dict[int, float] | None = None,
        split_strategy: str = 'uniform',
        agent_classes: dict[int, list[int]] | None = None,
        classes_per_agent: int = 2,
        alpha: float = 0.5,
        batch_size: int = 64,
        num_workers: int = 4,
        mode: str = 'min_size',
        val_split: float = 0.1,
        test_split: float = 0.1,
        seed: int = 42,
    ) -> None:
        super().__init__()

        self.repo = repo
        self.name = name
        self.data_key = data_key
        self.attributes = attributes
        self.n_agents = n_agents
        self.agent_rotations = agent_rotations or {}
        self.split_strategy = split_strategy
        self.agent_classes = agent_classes or {}
        self.classes_per_agent = classes_per_agent
        self.alpha = alpha

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mode = mode

        self.val_split = val_split
        self.test_split = test_split
        self.seed = seed

    def _filter_by_classes(
        self, dataset, allowed_classes: set[int]
    ) -> list[int]:
        """Get indices of samples whose label is in allowed_classes.

        Parameters
        ----------
        dataset : datasets.Dataset
            Dataset to filter.
        allowed_classes : set[int]
            Set of allowed class labels.

        Returns
        -------
        list[int]
            List of indices of samples with allowed labels.
        """
        # Extract label key from attributes; use first attribute as label
        label_key = self.attributes[0]
        indices = []
        # Iterate through dataset to find samples matching allowed classes
        for idx, item in enumerate(dataset):
            if item[label_key] in allowed_classes:
                indices.append(idx)
        return indices

    def _split_data_uniform(self, train, val, test, label_key: str) -> None:
        """Split data evenly across agents.

        Parameters
        ----------
        train : datasets.Dataset
            Training data.
        val : datasets.Dataset
            Validation data.
        test : datasets.Dataset
            Test data.
        label_key : str
            Name of the label column.
        """
        if self.n_agents == 1:
            rotation = self.agent_rotations.get(0, 0)
            self.train_datasets[0] = ClassificationDataset(
                train, self.data_key, label_key, rotation_angle=rotation
            )
            self.val_datasets[0] = ClassificationDataset(
                val, self.data_key, label_key, rotation_angle=rotation
            )
            self.test_datasets[0] = ClassificationDataset(
                test, self.data_key, label_key, rotation_angle=rotation
            )
            self.num_classes[0] = len(set(train[label_key]))
        else:
            train_indices = torch.randperm(
                len(train), generator=torch.Generator().manual_seed(self.seed)
            )
            val_indices = torch.randperm(
                len(val), generator=torch.Generator().manual_seed(self.seed)
            )
            test_indices = torch.randperm(
                len(test), generator=torch.Generator().manual_seed(self.seed)
            )

            train_splits = train_indices.split(len(train) // self.n_agents)
            val_splits = val_indices.split(len(val) // self.n_agents)
            test_splits = test_indices.split(len(test) // self.n_agents)

            for i in range(self.n_agents):
                rotation = self.agent_rotations.get(i, 0)
                self.train_datasets[i] = ClassificationDataset(
                    train.select(train_splits[i].tolist()),
                    self.data_key,
                    label_key,
                    rotation_angle=rotation,
                )
                self.val_datasets[i] = ClassificationDataset(
                    val.select(val_splits[i].tolist()),
                    self.data_key,
                    label_key,
                    rotation_angle=rotation,
                )
                self.test_datasets[i] = ClassificationDataset(
                    test.select(test_splits[i].tolist()),
                    self.data_key,
                    label_key,
                    rotation_angle=rotation,
                )
                self.num_classes[i] = len(
                    set(train.select(train_splits[i].tolist())[label_key])
                )

    def _split_data_class_partition(
        self, train, val, test, label_key: str
    ) -> None:
        """Split data by class assignment per agent.

        Parameters
        ----------
        train : datasets.Dataset
            Training data.
        val : datasets.Dataset
            Validation data.
        test : datasets.Dataset
            Test data.
        label_key : str
            Name of the label column.

        Raises
        ------
        ValueError
            If an agent has no classes assigned.
        """
        for i in range(self.n_agents):
            allowed_classes = self.agent_classes.get(i, None)
            if allowed_classes is None:
                allowed = set(train[label_key])
            else:
                allowed = set(allowed_classes)
            train_idx = self._filter_by_classes(train, allowed)
            val_idx = self._filter_by_classes(val, allowed)
            test_idx = self._filter_by_classes(test, allowed)

            rotation = self.agent_rotations.get(i, 0)
            self.train_datasets[i] = ClassificationDataset(
                train.select(train_idx),
                self.data_key,
                label_key,
                rotation_angle=rotation,
            )
            self.val_datasets[i] = ClassificationDataset(
                val.select(val_idx),
                self.data_key,
                label_key,
                rotation_angle=rotation,
            )
            self.test_datasets[i] = ClassificationDataset(
                test.select(test_idx),
                self.data_key,
                label_key,
                rotation_angle=rotation,
            )
            self.num_classes[i] = len(allowed)

    def _split_data_non_iid(
        self, train, val, test, label_key: str
    ) -> None:
        """Split data using automatic Non-IID partitioning.

        Each agent is randomly assigned ``classes_per_agent`` classes.
        Samples for each class are distributed among the agents assigned
        to that class with random Dirichlet-drawn proportions, producing
        statistical skew (``alpha`` controls the degree of imbalance).
        No manual ``agent_classes`` needed.

        Parameters
        ----------
        train : datasets.Dataset
            Training data.
        val : datasets.Dataset
            Validation data.
        test : datasets.Dataset
            Test data.
        label_key : str
            Name of the label column.
        """
        for split_data, target_dict in [
            (train, 'train_datasets'),
            (val, 'val_datasets'),
            (test, 'test_datasets'),
        ]:
            labels = split_data[label_key]
            partition = partition_non_iid(
                labels=labels,
                n_agents=self.n_agents,
                classes_per_agent=self.classes_per_agent,
                seed=self.seed,
                alpha=self.alpha,
            )
            for i in range(self.n_agents):
                indices = partition[i]
                rotation = self.agent_rotations.get(i, 0)
                getattr(self, target_dict)[i] = ClassificationDataset(
                    split_data.select(indices),
                    self.data_key,
                    label_key,
                    rotation_angle=rotation,
                )

        # Set per-agent num_classes from the actual labels each agent received
        for i in range(self.n_agents):
            agent_labels = self.train_datasets[i].dataset[label_key]
            self.num_classes[i] = len(set(agent_labels))

    def prepare_data(self) -> None:
        """Download dataset from HuggingFace."""
        load_dataset(f'{self.repo}/{self.name}')

    def setup(self, stage: str | None = None) -> None:
        """Load and split the dataset across agents.

        Parameters
        ----------
        stage : str or None
            Current stage ('fit', 'validate', 'test', or None for all).
        """
        # Load dataset from HuggingFace using repo/name combination
        ds = load_dataset(f'{self.repo}/{self.name}')

        # Merge all splits to maximize data before re-splitting
        # according to custom ratios
        splits = [ds[s] for s in ds]
        all_data = concatenate_datasets(splits)

        # Two-stage split to avoid data leakage:
        # Stage 1: Split off (validation + test) from training data
        split_1 = all_data.train_test_split(
            test_size=self.val_split + self.test_split,
            seed=self.seed,
        )

        train = split_1['train']
        temp = split_1['test']

        # Stage 2: Split the held-out portion into validation and test sets
        # This ensures test set is never seen during training/validation
        split_2 = temp.train_test_split(
            test_size=self.test_split / (self.val_split + self.test_split),
            seed=self.seed,
        )

        # Initialize dictionaries to hold datasets for each agent
        self.train_datasets: dict[int, ClassificationDataset] = {}
        self.val_datasets: dict[int, ClassificationDataset] = {}
        self.test_datasets: dict[int, ClassificationDataset] = {}
        self.num_classes: dict[int, int] = {}

        # Infer number of agents if not explicitly provided
        # Use the maximum key from agent_rotations or agent_classes
        if self.n_agents is None:
            max_key = 0
            if self.agent_rotations:
                max_key = max(max_key, max(self.agent_rotations.keys()))
            if self.agent_classes:
                max_key = max(max_key, max(self.agent_classes.keys()))
            self.n_agents = max_key + 1

        # Use first attribute as the label key for classification
        label_key = self.attributes[0]

        # Choose splitting strategy based on configuration
        if self.split_strategy == 'uniform':
            # Distribute data evenly across all agents
            self._split_data_uniform(
                train, split_2['train'], split_2['test'], label_key
            )
        elif self.split_strategy == 'class_partition':
            # Assign specific classes to each agent
            self._split_data_class_partition(
                train, split_2['train'], split_2['test'], label_key
            )
        elif self.split_strategy == 'non_iid':
            # Automatic Non-IID: each agent gets classes_per_agent random
            # classes, samples distributed among assigned agents
            self._split_data_non_iid(
                train, split_2['train'], split_2['test'], label_key
            )
        else:
            raise ValueError(f'Unknown split_strategy: {self.split_strategy}')

        # Determine input shape from first sample
        # Handle both PIL Images (need conversion) and pre-converted tensors
        sample, _ = self.train_datasets[0][0]
        if isinstance(sample, Image.Image):
            self.input_shape = transforms.ToTensor()(sample).shape
        else:
            self.input_shape = sample.shape

        # Store number of classes for the 'label' key (for compatibility)
        self.num_classes['label'] = self.num_classes.get(0)

        # Create list of agent indices and determine input dimensions dynamically
        self.models = list(range(self.n_agents))
        self.input_dims = {str(i): self.input_shape[0] for i in self.models}

    def _make_loader(
        self, dataset: Dataset, shuffle: bool
    ) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=_collate_fn,
        )

    def train_dataloader(self) -> CombinedLoader:
        return CombinedLoader(
            {
                i: self._make_loader(self.train_datasets[i], True)
                for i in self.train_datasets
            },
            mode=self.mode,
        )

    def val_dataloader(self) -> CombinedLoader:
        return CombinedLoader(
            {
                i: self._make_loader(self.val_datasets[i], False)
                for i in self.val_datasets
            },
            mode=self.mode,
        )

    def test_dataloader(self) -> CombinedLoader:
        return CombinedLoader(
            {
                i: self._make_loader(self.test_datasets[i], False)
                for i in self.test_datasets
            },
            mode=self.mode,
        )
