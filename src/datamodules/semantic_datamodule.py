"""
Semantic datamodule for loading pre-computed embedding datasets.

This module provides data loading capabilities for semantic embedding tasks
where pre-computed embeddings from vision models are used with associated
attribute labels. Designed for federated learning with multiple agents.

Features:
- Load pre-computed embeddings from HuggingFace datasets
- Support for multiple model embeddings per dataset
- Custom train/val/test splits per agent
- Class filtering per agent
"""

import lightning as l
import torch
from datasets import concatenate_datasets, load_dataset
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from torch.utils.data import DataLoader, Dataset

from src.datamodules.utils import (
    PairwiseDataset,
    compute_split_indices,
    repeat_dataset_to_num_samples,
)


class SemanticDataset(Dataset):
    """Dataset wrapper for semantic embedding data with attributes.

    Wraps a HuggingFace dataset containing pre-computed embeddings and
    associated attributes for federated learning scenarios.

    Parameters
    ----------
    hf_dataset : datasets.Dataset
        HuggingFace dataset containing 'embedding' and attribute columns.
    attributes : list[str]
        List of attribute column names to extract as labels.

    Notes
    -----
    - The dataset must contain an 'embedding' column with float tensors.
    - Attributes are converted to torch tensors for compatibility with PyTorch.
    """

    def __init__(
        self,
        hf_dataset,
        attributes: list[str],
        sample_ids: list[int] | None = None,
    ) -> None:
        """Initialize the semantic dataset.

        Parameters
        ----------
        hf_dataset : datasets.Dataset
            HuggingFace dataset containing embeddings and attributes.
        attributes : list[str]
            List of attribute column names to extract as labels.
        """
        self.dataset = hf_dataset
        self.attributes = attributes
        self.sample_ids = sample_ids

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.dataset)

    def __getitem__(
        self, idx: int
    ) -> (
        tuple[torch.Tensor, torch.Tensor | dict[str, torch.Tensor]]
        | tuple[
            torch.Tensor, torch.Tensor | dict[str, torch.Tensor], torch.Tensor
        ]
    ):
        """Get a single sample from the dataset.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve.

        Returns
        -------
        tuple
            Tuple of (embedding, labels) where labels is either a single tensor
            (if one attribute) or a dictionary of tensors (if multiple).
        """
        item = self.dataset[idx]

        # Extract embedding column and convert to float tensor
        embedding = torch.tensor(item['embedding'], dtype=torch.float32)

        # Extract specified attribute columns as labels
        attrs = {k: item[k] for k in self.attributes}

        # Return single tensor for single attribute, dict for multiple
        if len(attrs) == 1:
            labels = torch.tensor(list(attrs.values())[0])
        else:
            labels = {k: torch.tensor(v) for k, v in attrs.items()}

        if self.sample_ids is not None:
            sample_id = torch.tensor(self.sample_ids[idx], dtype=torch.long)
            return embedding, labels, sample_id

        return embedding, labels


class SemanticDataModule(l.LightningDataModule):
    """Per-model resplitting DataModule for semantic datasets.

    For each model, this DataModule:
    1. Merges all available splits (train/val/test)
    2. Re-splits into custom train / validation / test partitions

    Parameters
    ----------
    repo : str
        HuggingFace repository path.
    name : str
        Name of the dataset configuration.
    agents : dict, optional
        Dictionary mapping agent indices to their model configurations.
        Each agent should have a 'model' key specifying which embedding to use.
        Example: {0: {model: aimv2_1b_patch14_224.apple_pt}}
    models : list[str], optional
        List of model names/configurations to load (backward compatibility).
        Ignored if 'agents' is provided.
    agent_classes : dict, optional
        Dictionary mapping agent indices to list of class labels they see.
        If not provided or agent not specified, all classes are used.
    attributes : list[str]
        List of attribute columns to use as labels.
    batch_size : int, optional
        Batch size for dataloaders (default: 32).
    num_workers : int, optional
        Number of worker processes for data loading (default: 4).
    mode : str, optional
        Mode for CombinedLoader ('min_size', 'max_size', etc.)
        (default: 'min_size').
    val_split : float, optional
        Fraction of data for validation (default: 0.1).
    test_split : float, optional
        Fraction of data for testing (default: 0.1).
    seed : int, optional
        Random seed for reproducibility (default: 42).

    Attributes
    ----------
    train_datasets : dict[str, SemanticDataset]
        Dictionary mapping agent indices to training datasets.
    val_datasets : dict[str, SemanticDataset]
        Dictionary mapping agent indices to validation datasets.
    test_datasets : dict[str, SemanticDataset]
        Dictionary mapping agent indices to test datasets.
    input_dims : dict[str, int]
        Dictionary mapping agent indices to input feature dimensions.
    num_classes : dict
        Dictionary containing the number of classes for categorical attributes.
    """

    def __init__(
        self,
        repo: str,
        name: str,
        attributes: list[str],
        agents: dict | None = None,
        models: list[str] | None = None,
        agent_classes: dict | None = None,
        batch_size: int = 32,
        num_workers: int = 4,
        mode: str = 'min_size',
        val_split: float = 0.1,
        test_split: float = 0.1,
        pilot_split: float = 0.0,
        pilot_num_samples: int | None = None,
        pilot_batch_size: int | None = None,
        comm_data: str = 'private_pilots',
        seed: int = 42,
    ) -> None:
        """Initialize the semantic data module.

        Parameters
        ----------
        repo : str
            HuggingFace repository path.
        name : str
            Name of the dataset configuration.
        attributes : list[str]
            List of attribute columns to use as labels.
        agents : dict, optional
            Dictionary mapping agent indices to their model configurations.
        models : list[str], optional
            List of model names (backward compatibility).
        agent_classes : dict, optional
            Dictionary mapping agent indices to allowed classes.
        batch_size : int, optional
            Batch size for dataloaders (default: 32).
        num_workers : int, optional
            Number of worker processes (default: 4).
        mode : str, optional
            Mode for CombinedLoader (default: 'min_size').
        val_split : float, optional
            Fraction of data for validation (default: 0.1).
        test_split : float, optional
            Fraction of data for testing (default: 0.1).
        comm_data : str, optional
            Communication dataset strategy (default: 'private_pilots'):
            - 'private_pilots': one pilot dataloader per agent
            - 'shared_global_pilots': single global pilot dataloader
            - 'pairwise_pilots': one pilot dataloader per agent pair
        seed : int, optional
            Random seed for reproducibility (default: 42).
        """
        super().__init__()

        self.repo = repo
        self.name = name
        self.attributes = attributes
        self.agents = agents
        self.models = models
        self.agent_classes = agent_classes or {}

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mode = mode

        self.val_split = val_split
        self.test_split = test_split
        self.pilot_split = pilot_split
        self.pilot_num_samples = pilot_num_samples
        self.pilot_batch_size = (
            batch_size if pilot_batch_size is None else pilot_batch_size
        )
        self.comm_data = comm_data
        self.seed = seed

    def _merge_all_splits(self, ds) -> torch.utils.data.Dataset:
        """Merge all available splits into one dataset.

        HuggingFace datasets often provide train/validation/test splits
        separately. This method concatenates all splits to maximize the
        available data before re-splitting according to our custom
        proportions.

        Parameters
        ----------
        ds : datasets.DatasetDict
            Dictionary of HuggingFace dataset splits.

        Returns
        -------
        datasets.Dataset
            Concatenated dataset containing all splits.
        """
        # Collect all available splits (train, validation, test, etc.)
        # and concatenate them into one dataset to maximize available data
        # before applying our custom train/val/test split proportions
        splits = [ds[s] for s in ds]
        return concatenate_datasets(splits)

    def _resplit(self, dataset):
        """Split dataset into train / val / test using stratified sampling.

        Uses a two-stage split to avoid data leakage and ensure
        reproducibility:
        1. First split: separate (train) from (val + test) based on
           combined ratio (val_split + test_split)
        2. Second split: separate val from test within the held-out portion

        This ensures the test set is never seen during training/validation,
        preventing data leakage and providing a unbiased estimate of
        generalization performance.

        Parameters
        ----------
        dataset : datasets.Dataset
            Dataset to split.

        Returns
        -------
        tuple[datasets.Dataset, datasets.Dataset, datasets.Dataset]
            Tuple of (train, val, test) datasets.
        """
        # First split: separate training data from validation+test data
        # We use (val_split + test_split) as the test size to hold out
        # the combined validation and test portions
        split_1 = dataset.train_test_split(
            test_size=self.val_split + self.test_split, seed=self.seed
        )

        train = split_1['train']
        temp = split_1['test']  # This contains both validation and test data

        # Second split: separate validation from test data
        # Want test = test_split of original
        # But temp = (val_split+test_split) of original
        # Need test_split/(val_split+test_split) of temp for test
        split_2 = temp.train_test_split(
            test_size=self.test_split / (self.val_split + self.test_split),
            seed=self.seed,
        )

        # split_2['train'] is validation data
        # split_2['test'] is test data
        return train, split_2['train'], split_2['test']

    def prepare_data(self) -> None:
        """Download datasets (called only once in distributed setting)."""
        if self.agents:
            for agent_cfg in self.agents.values():
                load_dataset(f'{self.repo}/{self.name}', agent_cfg['model'])
        elif self.models:
            for m in self.models:
                load_dataset(f'{self.repo}/{self.name}', m)

    def setup(self, stage: str | None = None) -> None:
        """Set up datasets for each agent.

        Loads datasets, merges splits, and re-splits into custom partitions.
        Infers input dimensions and number of classes from the data.

        Parameters
        ----------
        stage : str or None
            Current stage ('fit', 'validate', 'test', or None for all).

        Raises
        ------
        ValueError
            If 'label' attribute is not found in the dataset.
        """
        # Initialize dictionaries to hold datasets for each agent/model
        # Using string keys for compatibility with PyTorch Lightning
        self.train_datasets: dict[str, SemanticDataset] = {}
        self.val_datasets: dict[str, SemanticDataset] = {}
        self.test_datasets: dict[str, SemanticDataset] = {}
        self.pilot_datasets: dict[str, SemanticDataset] = {}

        # Determine which models/embeddings to load for each agent
        # Priority: agents dict > models list (for backward compatibility)
        if self.agents:
            # Extract model configuration from agents dict
            # Convert string keys to int for internal processing
            agent_models = {
                int(i): cfg['model'] for i, cfg in self.agents.items()
            }
        elif self.models:
            # Use models list directly with sequential agent indices
            # enumerate() produces (index, value) tuples
            agent_models = dict(enumerate(self.models))
        else:
            raise ValueError('Either "agents" or "models" must be provided')

        # Variables to track split indices and expected dataset length
        split_indices: dict[str, list[int]] | None = None
        expected_merged_length: int | None = None

        # Load datasets for each agent/model combination
        for i, m in agent_models.items():
            # Load dataset with specific model configuration
            # from HuggingFace Hub
            ds = load_dataset(f'{self.repo}/{self.name}', m)

            # Merge all splits (train/val/test) to maximize data
            # before applying custom train/val/test split proportions
            merged = self._merge_all_splits(ds)

            # First agent: set expected length & compute split indices
            # All agents need same length for calibration pilots
            if expected_merged_length is None:
                expected_merged_length = len(merged)
                split_indices = compute_split_indices(
                    total_size=expected_merged_length,
                    val_split=self.val_split,
                    test_split=self.test_split,
                    seed=self.seed,
                    pilot_split=self.pilot_split,
                    pilot_num_samples=self.pilot_num_samples,
                )
            elif len(merged) != expected_merged_length:
                raise ValueError(
                    'All semantic agent datasets must have the same length to '
                    'share calibration pilots.'
                )

            assert split_indices is not None
            # Select samples for each split using precomputed indices
            train = merged.select(split_indices['train'])
            val = merged.select(split_indices['val'])
            test = merged.select(split_indices['test'])
            pilot = merged.select(split_indices['pilot'])

            # Use string key for PyTorch Lightning compatibility
            key = str(i)

            # Create semantic datasets for each split
            # Each dataset wraps the HF dataset with attribute extraction logic
            self.train_datasets[key] = SemanticDataset(train, self.attributes)
            self.val_datasets[key] = SemanticDataset(val, self.attributes)
            self.test_datasets[key] = SemanticDataset(test, self.attributes)
            if split_indices['pilot']:
                self.pilot_datasets[key] = SemanticDataset(
                    pilot,
                    self.attributes,
                    sample_ids=split_indices['pilot'],
                )

        # Store agent/model indices as integers for internal use
        self.models = list(agent_models.keys())
        self.n_agents = len(self.models)

        # Determine input dimensions from embedding size
        # Get first sample from first dataset to infer feature dimension
        self.input_dims = {}
        for m, ds in self.train_datasets.items():
            # ds[0] returns (embedding, labels) or
            # (embedding, labels, sample_id)
            # We only need the embedding to determine input dimensions
            sample = ds[0]
            x = sample[0]  # First element is always the embedding tensor
            self.input_dims[m] = x.shape[0]

        # Determine number of classes if 'label' attribute exists
        self.num_classes = {}

        if 'label' in self.attributes:
            # Get first dataset to inspect label values
            first_ds = next(iter(self.train_datasets.values())).dataset

            # Verify 'label' column exists in dataset
            if 'label' not in first_ds.column_names:
                raise ValueError('Attribute "label" not found in dataset')

            # Extract label values for analysis
            values = list(first_ds['label'])

            # Count unique classes if labels are categorical (int or bool)
            # Set to None for continuous labels (float)
            if isinstance(values[0], (int, bool)):
                self.num_classes['label'] = len(set(values))
            else:
                self.num_classes['label'] = None

    def _make_loader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        """Create a DataLoader for the given dataset.

        Parameters
        ----------
        dataset : Dataset
            PyTorch dataset to load.
        shuffle : bool
            Whether to shuffle the data.

        Returns
        -------
        DataLoader
            Configured DataLoader instance.
        """
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
        )

    def _make_pilot_loader(
        self,
        dataset: Dataset,
        target_num_batches: int,
    ) -> DataLoader:
        target_num_samples = target_num_batches * self.pilot_batch_size
        repeated_dataset = repeat_dataset_to_num_samples(
            dataset,
            target_num_samples,
        )
        return DataLoader(
            repeated_dataset,
            batch_size=self.pilot_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def _add_pilot_loaders(
        self, loaders: dict, target_num_batches: int
    ) -> None:
        if self.comm_data == 'private_pilots':
            loaders.update(
                {
                    f'pilot_{m}': self._make_pilot_loader(
                        ds, target_num_batches
                    )
                    for m, ds in self.pilot_datasets.items()
                }
            )
        elif self.comm_data == 'shared_global_pilots':
            first_key = next(iter(self.pilot_datasets.keys()))
            loaders['global_pilot'] = self._make_pilot_loader(
                self.pilot_datasets[first_key], target_num_batches
            )
        elif self.comm_data == 'pairwise_pilots':
            agent_keys = list(self.pilot_datasets.keys())
            for i in range(len(agent_keys)):
                for j in range(i + 1, len(agent_keys)):
                    key_i, key_j = agent_keys[i], agent_keys[j]
                    pairwise_ds = PairwiseDataset(
                        self.pilot_datasets[key_i],
                        self.pilot_datasets[key_j],
                    )
                    loaders[f'pilot_{key_i}_{key_j}'] = (
                        self._make_pilot_loader(
                            pairwise_ds, target_num_batches
                        )
                    )

    def train_dataloader(self) -> CombinedLoader:
        """Create and return the training data loader.

        Returns
        -------
        CombinedLoader
            CombinedLoader containing all training datasets.
        """
        loaders = {
            m: self._make_loader(ds, True)
            for m, ds in self.train_datasets.items()
        }
        if self.pilot_datasets:
            target_num_batches = max(
                (len(dataset) + self.batch_size - 1) // self.batch_size
                for dataset in self.train_datasets.values()
            )
            self._add_pilot_loaders(loaders, target_num_batches)
        return CombinedLoader(loaders, mode=self.mode)  # type: ignore[arg-type]

    def val_dataloader(self) -> CombinedLoader:
        """Create and return the validation data loader.

        Returns
        -------
        CombinedLoader
            CombinedLoader containing all validation datasets.
        """
        loaders = {
            m: self._make_loader(ds, False)
            for m, ds in self.val_datasets.items()
        }
        if self.pilot_datasets:
            target_num_batches = max(
                (len(dataset) + self.batch_size - 1) // self.batch_size
                for dataset in self.val_datasets.values()
            )
            self._add_pilot_loaders(loaders, target_num_batches)
        return CombinedLoader(loaders, mode=self.mode)  # type: ignore[arg-type]

    def test_dataloader(self) -> CombinedLoader:
        """Create and return the test data loader.

        Returns
        -------
        CombinedLoader
            CombinedLoader containing all test datasets.
        """
        loaders = {
            m: self._make_loader(ds, False)
            for m, ds in self.test_datasets.items()
        }
        if self.pilot_datasets:
            target_num_batches = max(
                (len(dataset) + self.batch_size - 1) // self.batch_size
                for dataset in self.test_datasets.values()
            )
            self._add_pilot_loaders(loaders, target_num_batches)
        return CombinedLoader(loaders, mode=self.mode)  # type: ignore[arg-type]
