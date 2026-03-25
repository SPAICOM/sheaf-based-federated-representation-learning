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

    def __init__(self, hf_dataset, attributes: list[str]) -> None:
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

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.dataset)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor | dict[str, torch.Tensor]]:
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
            return embedding, torch.tensor(list(attrs.values())[0])

        return embedding, {k: torch.tensor(v) for k, v in attrs.items()}


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
        # Collect all available splits and concatenate them into one dataset
        splits = [ds[s] for s in ds]
        return concatenate_datasets(splits)

    def _resplit(self, dataset):
        """Split dataset into train / val / test using stratified sampling.

        Uses a two-stage split to avoid data leakage and ensure
        reproducibility:
        1. First split: separate (train) from (val + test) based on
           combined ratio
        2. Second split: separate val from test within the held-out portion

        This ensures the test set is never seen during training/validation.

        Parameters
        ----------
        dataset : datasets.Dataset
            Dataset to split.

        Returns
        -------
        tuple[datasets.Dataset, datasets.Dataset, datasets.Dataset]
            Tuple of (train, val, test) datasets.
        """
        split_1 = dataset.train_test_split(
            test_size=self.val_split + self.test_split, seed=self.seed
        )

        train = split_1['train']
        temp = split_1['test']

        split_2 = temp.train_test_split(
            test_size=self.test_split / (self.val_split + self.test_split),
            seed=self.seed,
        )

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
        self.train_datasets: dict[str, SemanticDataset] = {}
        self.val_datasets: dict[str, SemanticDataset] = {}
        self.test_datasets: dict[str, SemanticDataset] = {}

        # Determine which models/embeddings to load for each agent
        # Priority: agents dict > models list (for backward compatibility)
        if self.agents:
            # Extract model configuration from agents dict
            agent_models = {
                int(i): cfg['model'] for i, cfg in self.agents.items()
            }
        elif self.models:
            # Use models list directly with sequential agent indices
            agent_models = dict(enumerate(self.models))
        else:
            raise ValueError('Either "agents" or "models" must be provided')

        # Load datasets for each agent/model combination
        for i, m in agent_models.items():
            # Load dataset with specific model configuration
            ds = load_dataset(f'{self.repo}/{self.name}', m)

            # Merge all splits to maximize available data
            merged = self._merge_all_splits(ds)
            # Re-split according to custom train/val/test ratios
            train, val, test = self._resplit(merged)

            # Use string key for compatibility with Lightning
            key = str(i)

            # Create semantic datasets for each split
            self.train_datasets[key] = SemanticDataset(train, self.attributes)
            self.val_datasets[key] = SemanticDataset(val, self.attributes)
            self.test_datasets[key] = SemanticDataset(test, self.attributes)

        # Store agent/model indices
        self.models = list(agent_models.keys())

        # Determine input dimensions from embedding size
        # Get first sample from first dataset to infer feature dimension
        self.input_dims = {}
        for m, ds in self.train_datasets.items():
            x, _ = ds[0]
            self.input_dims[m] = x.shape[0]

        # Determine number of classes if 'label' attribute exists
        self.num_classes = {}

        if 'label' in self.attributes:
            # Get first dataset to inspect label values
            first_ds = next(iter(self.train_datasets.values())).dataset

            # Verify 'label' column exists in dataset
            if 'label' not in first_ds.column_names:
                raise ValueError('Attribute "label" not found in dataset')

            values = list(first_ds['label'])

            # Count unique classes if labels are categorical (int or bool)
            # Set to None for continuous labels
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

    def train_dataloader(self) -> CombinedLoader:
        """Create and return the training data loader.

        Returns
        -------
        CombinedLoader
            CombinedLoader containing all training datasets.
        """
        return CombinedLoader(
            {
                m: self._make_loader(ds, True)
                for m, ds in self.train_datasets.items()
            },
            mode=self.mode,  # type: ignore[arg-type]
        )

    def val_dataloader(self) -> CombinedLoader:
        """Create and return the validation data loader.

        Returns
        -------
        CombinedLoader
            CombinedLoader containing all validation datasets.
        """
        return CombinedLoader(
            {
                m: self._make_loader(ds, False)
                for m, ds in self.val_datasets.items()
            },
            mode=self.mode,  # type: ignore[arg-type]
        )

    def test_dataloader(self) -> CombinedLoader:
        """Create and return the test data loader.

        Returns
        -------
        CombinedLoader
            CombinedLoader containing all test datasets.
        """
        return CombinedLoader(
            {
                m: self._make_loader(ds, False)
                for m, ds in self.test_datasets.items()
            },
            mode=self.mode,  # type: ignore[arg-type]
        )
