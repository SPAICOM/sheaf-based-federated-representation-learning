import lightning as l
import torch
from datasets import concatenate_datasets, load_dataset
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from torch.utils.data import DataLoader, Dataset


class SemanticDataset(Dataset):
    def __init__(self, hf_dataset, attributes: list[str]) -> None:
        self.dataset = hf_dataset
        self.attributes = attributes

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor | dict[str, torch.Tensor]]:
        item = self.dataset[idx]

        embedding = torch.tensor(item['embedding'], dtype=torch.float32)

        attrs = {k: item[k] for k in self.attributes}

        if len(attrs) == 1:
            return embedding, torch.tensor(list(attrs.values())[0])

        return embedding, {k: torch.tensor(v) for k, v in attrs.items()}


class SemanticDataModule(l.LightningDataModule):
    """
    Per-model resplitting DataModule.

    For each model:
        1. Merge all available splits
        2. Re-split into train / validation / test
    """

    def __init__(
        self,
        repo: str,
        name: str,
        models: list[str],
        attributes: list[str],
        batch_size: int = 32,
        num_workers: int = 4,
        mode: str = 'min_size',
        val_split: float = 0.1,
        test_split: float = 0.1,
        seed: int = 42,
    ) -> None:
        super().__init__()

        self.repo = repo
        self.name = name
        self.models = models
        self.attributes = attributes

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mode = mode

        self.val_split = val_split
        self.test_split = test_split
        self.seed = seed

    def _merge_all_splits(self, ds) -> torch.utils.data.Dataset:
        """
        Merge all available splits into one dataset.
        """
        splits = [ds[s] for s in ds]
        return concatenate_datasets(splits)

    def _resplit(self, dataset):
        """
        Split dataset into train / val / test.
        """
        # train vs (val+test)
        split_1 = dataset.train_test_split(
            test_size=self.val_split + self.test_split, seed=self.seed
        )

        train = split_1['train']
        temp = split_1['test']

        # val vs test
        split_2 = temp.train_test_split(
            test_size=self.test_split / (self.val_split + self.test_split),
            seed=self.seed,
        )

        return train, split_2['train'], split_2['test']

    def prepare_data(self) -> None:
        """
        Download datasets (called only once in distributed setting).
        """
        for m in self.models:
            load_dataset(f'{self.repo}/{self.name}', m)

    def setup(self, stage: str | None = None) -> None:
        self.train_datasets: dict[str, SemanticDataset] = {}
        self.val_datasets: dict[str, SemanticDataset] = {}
        self.test_datasets: dict[str, SemanticDataset] = {}

        for i, m in enumerate(self.models):
            ds = load_dataset(f'{self.repo}/{self.name}', m)

            merged = self._merge_all_splits(ds)
            train, val, test = self._resplit(merged)

            key = str(i)

            self.train_datasets[key] = SemanticDataset(train, self.attributes)
            self.val_datasets[key] = SemanticDataset(val, self.attributes)
            self.test_datasets[key] = SemanticDataset(test, self.attributes)

        # Infer input dimensions
        self.input_dims = {}
        for m, ds in self.train_datasets.items():
            x, _ = ds[0]
            self.input_dims[m] = x.shape[0]

        # Infer attribute structure
        self.num_classes = {}

        if 'label' in self.attributes:
            first_ds = next(iter(self.train_datasets.values())).dataset

            if 'label' not in first_ds.column_names:
                raise ValueError('Attribute "label" not found in dataset')

            values = list(first_ds['label'])

            if isinstance(values[0], (int, bool)):
                self.num_classes['label'] = len(set(values))
            else:
                self.num_classes['label'] = None

    def _make_loader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
        )

    def train_dataloader(self) -> CombinedLoader:
        return CombinedLoader(
            {
                m: self._make_loader(ds, True)
                for m, ds in self.train_datasets.items()
            },
            mode=self.mode,
        )

    def val_dataloader(self) -> CombinedLoader:
        return CombinedLoader(
            {
                m: self._make_loader(ds, False)
                for m, ds in self.val_datasets.items()
            },
            mode=self.mode,
        )

    def test_dataloader(self) -> CombinedLoader:
        return CombinedLoader(
            {
                m: self._make_loader(ds, False)
                for m, ds in self.test_datasets.items()
            },
            mode=self.mode,
        )
