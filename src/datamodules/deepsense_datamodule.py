"""
DeepSense datamodule for multimodal sensor fusion.

This module provides data loading for the DeepSense Scenario 30 dataset,
which contains synchronized RGB images, mmWave radar, and LiDAR data for
LOS/NLOS classification in indoor environments.

Features:
- Load from downloaded zip archive (Google Drive)
- Three modalities: RGB (3x64x64), mmWave (1x16x64), LiDAR (2x32x32)
- Flexible agent-modality mapping (default: 1:1)
- Split strategies: 'full', 'uniform', 'class_partition', 'non_iid'
- Shared pilot datasets for federated calibration
- CombinedLoader with per-modality loaders

Modality mapping (fixed):
- 0: RGB image
- 1: mmWave radar
- 2: LiDAR

Labels:
- 0: LOS (Line of Sight)
- 1: NLOS (Non-Line of Sight)
"""

import zipfile
from pathlib import Path

import gdown
import lightning as l
import numpy as np
import pandas as pd
import scipy.io
import torch
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.datamodules.utils import (
    PairwiseDataset,
    compute_split_indices,
    repeat_dataset_to_num_samples,
)
from src.utils.data_partitioner import partition_non_iid


class DeepSenseDataset(Dataset):
    """Base dataset for DeepSense Scenario 30.

    Loads multimodal samples from the scenario CSV containing paths to
    RGB images, mmWave power vectors, and LiDAR matrices.
    """

    MODALITY_SHAPES = {
        0: (3, 64, 64),  # RGB
        1: (1, 16, 64),  # mmWave
        2: (2, 32, 32),  # LiDAR
    }

    def __init__(
        self,
        root: str,
        scenario: str,
    ) -> None:
        self.root = Path(root)
        self.csv_path = self.root / f'scenario{scenario}.csv'
        if not self.csv_path.exists():
            raise FileNotFoundError(
                f'DeepSense CSV not found at {self.csv_path}'
            )

        self.df = pd.read_csv(self.csv_path)
        self.df.columns = self.df.columns.str.strip()

        self.transform = transforms.Compose(
            [
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_path(self, rel_path: str) -> Path:
        clean = str(rel_path).replace('./', '').replace('\\', '/')
        return self.root / clean

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        x_rgb = self._load_rgb(row['unit1_rgb'])
        x_mmwave = self._load_mmwave(row['unit1_pwr_60ghz'])
        x_lidar = self._load_lidar(row['unit1_lidar'])
        label = self._load_label(row['unit1_blockage'])

        return {0: x_rgb, 1: x_mmwave, 2: x_lidar, 'label': label}

    def _load_rgb(self, path: str) -> torch.Tensor:
        try:
            img_path = self._resolve_path(path)
            image_pil = Image.open(img_path).convert('RGB')
            return self.transform(image_pil)
        except Exception:
            return torch.randn(3, 64, 64) * 1e-5

    def _load_mmwave(self, path: str) -> torch.Tensor:
        try:
            pwr_path = self._resolve_path(path)
            with open(pwr_path) as f:
                content = (
                    f.read()
                    .strip()
                    .replace('[', '')
                    .replace(']', '')
                    .replace(',', ' ')
                )
                vals = np.fromstring(content, sep=' ')

            x_vec = torch.tensor(vals, dtype=torch.float32)
            x_vec = torch.nan_to_num(x_vec, nan=0.0)

            if x_vec.shape[0] < 64:
                x_vec = torch.cat([x_vec, torch.zeros(64 - x_vec.shape[0])])
            elif x_vec.shape[0] > 64:
                x_vec = x_vec[:64]

            x_mmwave = x_vec.unsqueeze(0).repeat(16, 1)
            return x_mmwave.unsqueeze(0)
        except Exception:
            return torch.randn(1, 16, 64) * 1e-5

    def _load_lidar(self, path: str) -> torch.Tensor:
        try:
            lidar_path = self._resolve_path(path)
            mat_data = scipy.io.loadmat(lidar_path)

            if 'lidar_data' in mat_data:
                raw_lidar = mat_data['lidar_data']
            else:
                raw_lidar = mat_data[
                    [k for k in mat_data if not k.startswith('_')][0]
                ]

            x_lidar = torch.tensor(raw_lidar).abs().float()
            x_lidar = torch.log1p(x_lidar)
            if x_lidar.max() > 0:
                x_lidar /= x_lidar.max()

            if x_lidar.dim() == 2:
                x_lidar = x_lidar.unsqueeze(0)

            x_lidar = torch.nn.functional.interpolate(
                x_lidar.unsqueeze(0), size=(32, 32), mode='bilinear'
            ).squeeze(0)

            if x_lidar.shape[0] == 1:
                x_lidar = x_lidar.repeat(2, 1, 1)
            elif x_lidar.shape[0] > 2:
                x_lidar = x_lidar[:2]

            return x_lidar
        except Exception:
            return torch.randn(2, 32, 32) * 1e-5

    def _load_label(self, path: str) -> torch.Tensor:
        label_path = self._resolve_path(path)
        with open(label_path) as f:
            val = float(f.read().strip())
        label = 0 if val == 0.0 else 1
        return torch.tensor(label, dtype=torch.long)


class DeepSenseModalityDataset(Dataset):
    """Dataset that returns a single modality per sample.

    Wraps the base DeepSenseDataset and extracts only one modality
    for assignment to specific agents in federated learning.
    """

    def __init__(
        self,
        base_dataset: DeepSenseDataset,
        modality_idx: int,
        sample_ids: list[int] | None = None,
        return_sample_ids: bool = False,
    ) -> None:
        self.base_dataset = base_dataset
        self.modality_idx = modality_idx
        self.sample_ids = sample_ids
        self.return_sample_ids = return_sample_ids

    def __len__(self) -> int:
        if self.sample_ids is not None:
            return len(self.sample_ids)
        return len(self.base_dataset)

    def __getitem__(
        self, idx: int
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        if self.sample_ids is not None:
            actual_idx = self.sample_ids[idx]
        else:
            actual_idx = idx

        sample = self.base_dataset[actual_idx]
        x = sample[self.modality_idx]
        label = sample['label']

        if self.return_sample_ids and self.sample_ids is not None:
            sample_id = torch.tensor(self.sample_ids[idx], dtype=torch.long)
            return x, label, sample_id

        return x, label


class DeepSenseDataModule(l.LightningDataModule):
    """Lightning datamodule for DeepSense multimodal dataset.

    Loads DeepSense Scenario 30 data and distributes modalities across
    agents in federated learning scenarios. Each agent can receive one
    or more modalities based on the agent_modalities mapping.

    Parameters
    ----------
    data_path : str
        Path to the DeepSense data directory.
    gdrive_folder_id : str, optional
        Google Drive folder ID for downloading the dataset.
        If None, uses DEEPSENSE_GDRIVE_FOLDER_ID from .env.
    split_strategy : str
        How to split data across agents. Options:
        - 'full': Each agent gets ALL samples of assigned modalities
        - 'uniform': Random split of samples within each modality
        - 'class_partition': Each agent gets specific classes
        - 'non_iid': Dirichlet-skewed class distribution
    agent_modalities : dict[int, list[int]], optional
        Mapping of agent indices to modality indices.
        Default: {0: [0], 1: [1], 2: [2]} (1:1 mapping)
    n_agents : int
        Total number of agents (default: 3)
    batch_size : int
        Batch size for dataloaders (default: 32)
    num_workers : int
        Number of worker processes (default: 4)
    mode : str
        CombinedLoader mode (default: 'min_size')
    val_split : float
        Fraction of data for validation (default: 0.1)
    test_split : float
        Fraction of data for testing (default: 0.1)
    pilot_split : float
        Fraction of data for pilot datasets (default: 0.0)
    pilot_num_samples : int, optional
        Fixed number of pilot samples per agent
    comm_data : str
        Communication dataset strategy exposed through the CombinedLoader.
        Applies only when pilots are enabled (``pilot_split > 0`` or
        ``pilot_num_samples`` is set).  Valid values:

        ``'private_pilots'`` *(default)*
            One modality-specific dataloader per agent, keyed
            ``pilot_0``, ``pilot_1``, ….  Each agent owns its own
            pilot view of the shared sample subset.
        ``'shared_global_pilots'``
            Same shared random sample subset for all agents, but each
            agent still receives its own modality-specific dataloader.
            Keys become ``global_pilot_0``, ``global_pilot_1``, … so
            orchestrators can distinguish this mode from private pilots.
        ``'pairwise_pilots'``
            One dataloader per agent pair ``(i, j)`` with ``i < j``,
            keyed ``pilot_i_j``.  Each batch yields
            ``(x_i, x_j, label, sample_id)`` where ``x_i`` and ``x_j``
            are the two agents' modality views of the **same** underlying
            sample (sample intersection of their pilot datasets, which
            equals the full pilot set under ``split_strategy='full'``).
            Created for **all** possible pairs; the orchestrator filters
            to actual graph edges.
    seed : int
        Random seed for reproducibility (default: 42)

    Attributes
    ----------
    train_datasets : dict[int, DeepSenseModalityDataset]
        Dictionary mapping agent indices to training datasets.
    val_datasets : dict[int, DeepSenseModalityDataset]
        Dictionary mapping agent indices to validation datasets.
    test_datasets : dict[int, DeepSenseModalityDataset]
        Dictionary mapping agent indices to test datasets.
    pilot_datasets : dict[int, DeepSenseModalityDataset]
        Dictionary mapping agent indices to pilot datasets.
    num_classes : dict
        Number of classes per agent and global label count.
    input_shape : dict[int, tuple]
        Input shape per agent (determined by assigned modality).
    input_dims : dict[str, int]
        Input dimension per agent.
    """

    MODALITY_NAMES = {0: 'rgb', 1: 'mmwave', 2: 'lidar'}
    MODALITY_SHAPES = {
        0: (3, 64, 64),  # RGB
        1: (1, 16, 64),  # mmWave
        2: (2, 32, 32),  # LiDAR
    }

    def __init__(
        self,
        data_path: str = './data/DeepSense',
        gdrive_file_id: str | None = None,
        scenario: str | None = None,
        split_strategy: str = 'full',
        agent_modalities: dict[int, list[int]] | None = None,
        n_agents: int = 3,
        batch_size: int = 32,
        num_workers: int = 4,
        mode: str = 'min_size',
        val_split: float = 0.1,
        test_split: float = 0.1,
        pilot_split: float = 0.0,
        pilot_num_samples: int | None = None,
        comm_data: str = 'private_pilots',
        seed: int = 42,
    ) -> None:
        super().__init__()

        self.data_path = Path(data_path)
        self.gdrive_file_id = gdrive_file_id
        self.scenario = scenario
        self.split_strategy = split_strategy
        self.n_agents = n_agents
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mode = mode
        self.val_split = val_split
        self.test_split = test_split
        self.pilot_split = pilot_split
        self.pilot_num_samples = pilot_num_samples
        self.comm_data = comm_data
        self.seed = seed

        if agent_modalities is None:
            self.agent_modalities = {i: [i] for i in range(n_agents)}
        else:
            self.agent_modalities = agent_modalities

    def prepare_data(self) -> None:
        """Download and extract DeepSense dataset if not present.

        Downloads scenario30.zip directly from Google Drive by file ID.
        """
        csv_path = self.data_path / f'scenario{self.scenario}.csv'
        data_dir = self.data_path / 'unit1'
        if csv_path.exists() and data_dir.exists():
            return

        if not self.gdrive_file_id:
            raise RuntimeError(
                'DeepSense dataset not found. '
                'Provide gdrive_file_id in your config yaml.'
            )

        self.data_path.mkdir(parents=True, exist_ok=True)
        zip_path = self.data_path / f'scenario{self.scenario}.zip'

        if not zip_path.exists():
            print(f'[prepare_data] Downloading {self.scenario}.zip …')
            gdown.download(
                id=self.gdrive_file_id, output=str(zip_path), quiet=False
            )

        if not zip_path.exists():
            raise RuntimeError(f'Download failed, zip not found at {zip_path}')

        print(f'[prepare_data] Extracting {zip_path} …')
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(self.data_path)

    def _build_modality_datasets(
        self,
        indices: dict[int, list[int]],
        split_name: str,
    ) -> dict[int, list[DeepSenseModalityDataset]]:
        result = {}
        for agent_id, mod_list in self.agent_modalities.items():
            result[agent_id] = []
            for mod_idx in mod_list:
                dataset = DeepSenseModalityDataset(
                    self._base_dataset,
                    mod_idx,
                    sample_ids=indices.get(agent_id),
                )
                result[agent_id].append(dataset)
        return result

    def _split_data_full(
        self,
        split_indices: dict[str, list[int]],
    ) -> None:
        for agent_id in range(self.n_agents):
            train_mods = []
            val_mods = []
            test_mods = []

            for mod_idx in self.agent_modalities[agent_id]:
                train_ds = DeepSenseModalityDataset(
                    self._base_dataset,
                    mod_idx,
                    sample_ids=split_indices['train'],
                )
                val_ds = DeepSenseModalityDataset(
                    self._base_dataset,
                    mod_idx,
                    sample_ids=split_indices['val'],
                )
                test_ds = DeepSenseModalityDataset(
                    self._base_dataset,
                    mod_idx,
                    sample_ids=split_indices['test'],
                )
                train_mods.append(train_ds)
                val_mods.append(val_ds)
                test_mods.append(test_ds)

            self.train_datasets[agent_id] = train_mods
            self.val_datasets[agent_id] = val_mods
            self.test_datasets[agent_id] = test_mods

    def _split_data_uniform(
        self,
        split_indices: dict[str, list[int]],
    ) -> None:
        for agent_id in range(self.n_agents):
            train_mods = []
            val_mods = []
            test_mods = []

            for mod_idx in self.agent_modalities[agent_id]:
                n_samples = len(split_indices['train'])
                n_per_agent = n_samples // self.n_agents

                perm = torch.randperm(
                    n_samples,
                    generator=torch.Generator().manual_seed(self.seed),
                )
                start = agent_id * n_per_agent
                end = (
                    start + n_per_agent
                    if agent_id < self.n_agents - 1
                    else n_samples
                )

                train_ids = [
                    split_indices['train'][perm[i].item()]
                    for i in range(start, end)
                ]

                val_n = len(split_indices['val'])
                val_per = val_n // self.n_agents
                val_perm = torch.randperm(
                    val_n,
                    generator=torch.Generator().manual_seed(self.seed + 1),
                )
                val_start = agent_id * val_per
                val_end = (
                    val_start + val_per
                    if agent_id < self.n_agents - 1
                    else val_n
                )
                val_ids = [
                    split_indices['val'][val_perm[i].item()]
                    for i in range(val_start, val_end)
                ]

                test_n = len(split_indices['test'])
                test_per = test_n // self.n_agents
                test_perm = torch.randperm(
                    test_n,
                    generator=torch.Generator().manual_seed(self.seed + 2),
                )
                test_start = agent_id * test_per
                test_end = (
                    test_start + test_per
                    if agent_id < self.n_agents - 1
                    else test_n
                )
                test_ids = [
                    split_indices['test'][test_perm[i].item()]
                    for i in range(test_start, test_end)
                ]

                train_mods.append(
                    DeepSenseModalityDataset(
                        self._base_dataset, mod_idx, train_ids
                    )
                )
                val_mods.append(
                    DeepSenseModalityDataset(
                        self._base_dataset, mod_idx, val_ids
                    )
                )
                test_mods.append(
                    DeepSenseModalityDataset(
                        self._base_dataset, mod_idx, test_ids
                    )
                )

            self.train_datasets[agent_id] = train_mods
            self.val_datasets[agent_id] = val_mods
            self.test_datasets[agent_id] = test_mods

    def _split_data_class_partition(
        self,
        split_indices: dict[str, list[int]],
        all_labels: np.ndarray,
    ) -> None:
        if not self.agent_classes:
            raise ValueError(
                'class_partition requires agent_classes to be set'
            )

        for agent_id in range(self.n_agents):
            train_mods = []
            val_mods = []
            test_mods = []

            for mod_idx in self.agent_modalities[agent_id]:
                allowed_classes = set(self.agent_classes.get(agent_id, []))

                train_idx = [
                    i
                    for i in split_indices['train']
                    if all_labels[i] in allowed_classes
                ]
                val_idx = [
                    i
                    for i in split_indices['val']
                    if all_labels[i] in allowed_classes
                ]
                test_idx = [
                    i
                    for i in split_indices['test']
                    if all_labels[i] in allowed_classes
                ]

                train_mods.append(
                    DeepSenseModalityDataset(
                        self._base_dataset, mod_idx, train_idx
                    )
                )
                val_mods.append(
                    DeepSenseModalityDataset(
                        self._base_dataset, mod_idx, val_idx
                    )
                )
                test_mods.append(
                    DeepSenseModalityDataset(
                        self._base_dataset, mod_idx, test_idx
                    )
                )

            self.train_datasets[agent_id] = train_mods
            self.val_datasets[agent_id] = val_mods
            self.test_datasets[agent_id] = test_mods

    def _split_data_non_iid(
        self,
        split_indices: dict[str, list[int]],
        all_labels: np.ndarray,
    ) -> None:
        train_partition, sampled_classes = partition_non_iid(
            labels=all_labels[split_indices['train']],
            n_agents=self.n_agents,
            classes_per_agent=2,
            seed=self.seed,
            alpha=0.5,
            return_agent_classes=True,
        )
        self.agent_classes = sampled_classes

        for agent_id in range(self.n_agents):
            train_mods = []
            val_mods = []
            test_mods = []

            for mod_idx in self.agent_modalities[agent_id]:
                train_idx = [
                    split_indices['train'][i]
                    for i in train_partition[agent_id]
                ]
                val_idx = [
                    split_indices['val'][i]
                    for i in range(len(split_indices['val']))
                ]
                test_idx = [
                    split_indices['test'][i]
                    for i in range(len(split_indices['test']))
                ]

                train_mods.append(
                    DeepSenseModalityDataset(
                        self._base_dataset, mod_idx, train_idx
                    )
                )
                val_mods.append(
                    DeepSenseModalityDataset(
                        self._base_dataset, mod_idx, val_idx
                    )
                )
                test_mods.append(
                    DeepSenseModalityDataset(
                        self._base_dataset, mod_idx, test_idx
                    )
                )

            self.train_datasets[agent_id] = train_mods
            self.val_datasets[agent_id] = val_mods
            self.test_datasets[agent_id] = test_mods

    def setup(self, stage: str | None = None) -> None:
        if stage == 'test' or stage is None:
            self._base_dataset = DeepSenseDataset(
                str(self.data_path), self.scenario
            )

            split_indices = compute_split_indices(
                total_size=len(self._base_dataset),
                val_split=self.val_split,
                test_split=self.test_split,
                seed=self.seed,
                pilot_split=self.pilot_split,
                pilot_num_samples=self.pilot_num_samples,
            )

            self.train_datasets: dict[int, list[DeepSenseModalityDataset]] = {}
            self.val_datasets: dict[int, list[DeepSenseModalityDataset]] = {}
            self.test_datasets: dict[int, list[DeepSenseModalityDataset]] = {}
            self.pilot_datasets: dict[int, list[DeepSenseModalityDataset]] = {}
            self.num_classes: dict = {}

            all_labels = np.array(
                [
                    self._base_dataset[i]['label'].item()
                    for i in range(len(self._base_dataset))
                ]
            )
            self.num_classes['label'] = 2  # DeepSense is binary: LOS (0) / NLOS (1)

            if self.split_strategy == 'full':
                self._split_data_full(split_indices)
            elif self.split_strategy == 'uniform':
                self._split_data_uniform(split_indices)
            elif self.split_strategy == 'class_partition':
                self._split_data_class_partition(split_indices, all_labels)
            elif self.split_strategy == 'non_iid':
                self._split_data_non_iid(split_indices, all_labels)
            else:
                raise ValueError(
                    f'Unknown split_strategy: {self.split_strategy}'
                )

            if split_indices.get('pilot'):
                for agent_id in range(self.n_agents):
                    self.pilot_datasets[agent_id] = [
                        DeepSenseModalityDataset(
                            self._base_dataset,
                            mod_idx,
                            sample_ids=split_indices['pilot'],
                            return_sample_ids=True,
                        )
                        for mod_idx in self.agent_modalities[agent_id]
                    ]

            self.input_shape = {}
            self.input_dims = {}
            for agent_id, mod_list in self.agent_modalities.items():
                shape = self.MODALITY_SHAPES[mod_list[0]]
                self.input_shape[agent_id] = shape
                self.input_dims[str(agent_id)] = shape[0]

    def _make_loader(
        self,
        datasets: list[DeepSenseModalityDataset],
        shuffle: bool,
    ) -> DataLoader:
        if len(datasets) == 1:
            dataset = datasets[0]
        else:
            dataset = torch.utils.data.ConcatDataset(datasets)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
        )

    def _make_pilot_loader(
        self,
        dataset: Dataset | list[DeepSenseModalityDataset],
        target_num_batches: int,
    ) -> DataLoader:
        if isinstance(dataset, list):
            ds: Dataset = (
                torch.utils.data.ConcatDataset(dataset)
                if len(dataset) > 1
                else dataset[0]
            )
        else:
            ds = dataset
        target_num_samples = target_num_batches * self.batch_size
        repeated = repeat_dataset_to_num_samples(ds, target_num_samples)
        return DataLoader(
            repeated,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def _create_loaders(self, split: str) -> CombinedLoader:
        split_map: dict[str, tuple[dict, bool]] = {
            'train': (self.train_datasets, True),
            'val': (self.val_datasets, False),
            'test': (self.test_datasets, False),
        }
        if split not in split_map:
            raise ValueError(
                f'Unknown split {split!r}. Valid options: {list(split_map)}'
            )
        datasets, shuffle = split_map[split]
        loaders: dict = {
            i: self._make_loader(ds, shuffle) for i, ds in datasets.items()
        }

        if self.pilot_datasets:
            target_batches = max(
                (len(ds) + self.batch_size - 1) // self.batch_size
                for ds in self.pilot_datasets.values()
            )
            if self.comm_data == 'private_pilots':
                loaders.update({
                    f'pilot_{i}': self._make_pilot_loader(ds, target_batches)
                    for i, ds in self.pilot_datasets.items()
                })
            elif self.comm_data == 'shared_global_pilots':
                loaders.update({
                    f'global_pilot_{i}': self._make_pilot_loader(ds, target_batches)
                    for i, ds in self.pilot_datasets.items()
                })
            elif self.comm_data == 'pairwise_pilots':
                loaders.update({
                    f'pilot_{i}_{j}': self._make_pilot_loader(
                        PairwiseDataset(self.pilot_datasets[i], self.pilot_datasets[j]),
                        target_batches,
                    )
                    for i in range(self.n_agents)
                    for j in range(i + 1, self.n_agents)
                })

        return CombinedLoader(loaders, mode=self.mode)

    def train_dataloader(self) -> CombinedLoader:
        return self._create_loaders('train')

    def val_dataloader(self) -> CombinedLoader:
        return self._create_loaders('val')

    def test_dataloader(self) -> CombinedLoader:
        return self._create_loaders('test')
