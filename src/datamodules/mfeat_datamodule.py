"""
Multiple Features (MFeat) datamodule for multi-view digit recognition.

This module provides data loading for the UCI MFeat dataset, which contains
2000 handwritten digit patterns (0-9, 200 per class) described by six
independent feature sets extracted from the same underlying images:

- ``fac``: 216 profile correlations
- ``fou``:  76 Fourier coefficients of the character shapes
- ``kar``:  64 Karhunen-Loève coefficients
- ``mor``:   6 morphological features
- ``pix``: 240 pixel averages in 2x3 windows
- ``zon``:  47 Zernike moments

Labels are encoded implicitly by row order: rows 0-199 → digit 0,
rows 200-399 → digit 1, …, rows 1800-1999 → digit 9.

Features:
- Download from Google Drive via gdown
- One PyTorch Dataset per (agent, split) backed by a single loaded array
- Each agent receives a designated modality; different agents see different
  feature views of the same underlying samples
- Split strategies: ``'full'``, ``'uniform'``, ``'class_partition'``, ``'non_iid'``
- Shared pilot dataset for federated calibration
- ``CombinedLoader`` structure matching ``MHealthDataModule``
"""

import zipfile
from pathlib import Path

import gdown
import lightning as l
import numpy as np
import torch
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from torch.utils.data import DataLoader, Dataset

from src.datamodules.utils import (
    PairwiseDataset,
    compute_split_indices,
    repeat_dataset_to_num_samples,
    sample_random_subset,
)
from src.utils.data_partitioner import (
    build_shared_class_partition,
    partition_by_agent_classes,
    partition_non_iid,
)

# ── Modality registry ─────────────────────────────────────────────────────────

MFEAT_MODALITIES: dict[str, int] = {
    'fac': 216,
    'fou': 76,
    'kar': 64,
    'mor': 6,
    'pix': 240,
    'zer': 47,
}

_NUM_CLASSES = 10
_SAMPLES_PER_CLASS = 200
_TOTAL_SAMPLES = _NUM_CLASSES * _SAMPLES_PER_CLASS  # 2000


# ── I/O ───────────────────────────────────────────────────────────────────────


def _load_mfeat_modality(data_path: Path, modality: str) -> np.ndarray:
    """Load one MFeat feature file as a float32 array of shape (2000, n_feat).

    The files are plain-text, space-separated, with no header.  They are
    found either as ``mfeat-<mod>`` (no extension) or ``mfeat-<mod>.txt``.
    """
    candidates = [
        data_path / f'mfeat-{modality}',
        data_path / f'mfeat-{modality}.txt',
        data_path / f'mfeat_{modality}',
        data_path / f'mfeat_{modality}.txt',
    ]
    for p in candidates:
        if p.exists():
            return np.loadtxt(p, dtype='float32')
    raise FileNotFoundError(
        f'MFeat modality file for "{modality}" not found in {data_path}. '
        f'Tried: {[str(c) for c in candidates]}'
    )


def _build_labels(n_total: int = _TOTAL_SAMPLES) -> np.ndarray:
    """Build integer labels from implicit row ordering (200 samples/class)."""
    return np.repeat(np.arange(_NUM_CLASSES), _SAMPLES_PER_CLASS).astype('int64')[
        :n_total
    ]


# ── Dataset ───────────────────────────────────────────────────────────────────


class MFeatDataset(Dataset):
    """PyTorch dataset wrapping a single MFeat modality array.

    Parameters
    ----------
    features : np.ndarray
        Feature matrix of shape ``(N, n_feat)``.
    labels : np.ndarray
        Integer label array of shape ``(N,)``.
    indices : list[int] or None
        Subset of row indices to expose.  ``None`` → all rows.
    sample_ids : list[int] or None
        Global sample identifiers returned as a third element when set.
    normalize : bool or tuple[np.ndarray, np.ndarray]
        Per-feature normalisation.  Pass ``(mean, std)`` computed on the
        *training* split to avoid target-leakage in val/test sets.
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        indices: list[int] | None = None,
        sample_ids: list[int] | None = None,
        normalize: bool | tuple[np.ndarray, np.ndarray] = True,
    ) -> None:
        if indices is not None:
            self._x = features[indices].copy()
            self._y = labels[indices].copy()
        else:
            self._x = features.copy()
            self._y = labels.copy()

        self.sample_ids = sample_ids

        if isinstance(normalize, tuple):
            mean, std = normalize
            self._x = (self._x - mean) / std
        elif normalize:
            mean = self._x.mean(axis=0, keepdims=True)
            std = self._x.std(axis=0, keepdims=True)
            std = np.where(std < 1e-8, 1.0, std)
            self._x = (self._x - mean) / std

    def __len__(self) -> int:
        return len(self._x)

    def __getitem__(
        self, idx: int
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        x = torch.from_numpy(self._x[idx])
        y = torch.tensor(int(self._y[idx]), dtype=torch.long)
        if self.sample_ids is not None:
            sid = torch.tensor(self.sample_ids[idx], dtype=torch.long)
            return x, y, sid
        return x, y


# ── DataModule ────────────────────────────────────────────────────────────────


class MFeatDataModule(l.LightningDataModule):
    """DataModule for the UCI Multiple Features (MFeat) dataset.

    Each agent is assigned one feature modality so that all agents observe
    the same underlying digit samples but from different representational
    views.  This naturally models a vertical / multi-view federated setting.

    Parameters
    ----------
    data_path : str or Path
        Directory containing the six ``mfeat-*`` feature files.
    gdrive_file_id : str or None
        Google Drive file ID of the dataset archive.  Required when
        ``data_path`` does not yet exist.
    n_agents : int
        Number of federated agents.  Must not exceed the number of entries
        in ``agent_modalities``.
    agent_modalities : dict[int, str] or None
        Mapping ``{agent_id: modality_name}``.  Valid modality names are
        the keys of ``MFEAT_MODALITIES`` (``'fac'``, ``'fou'``, ``'kar'``,
        ``'mor'``, ``'pix'``, ``'zon'``).
        Defaults to the five most expressive modalities (``fou``, ``fac``,
        ``kar``, ``pix``, ``zon``) assigned to agents 0–4.
    split_strategy : str
        How samples are distributed across agents:

        ``'full'`` *(default)*
            All agents share the same train/val/test sample indices; each
            simply views them through its own modality.  Suitable for
            vertical FL experiments.
        ``'uniform'``
            Samples are shuffled and evenly divided among agents.
        ``'class_partition'``
            Each agent receives a specific subset of digit classes.
            Requires ``agent_classes`` or ``shared_classes``.
        ``'non_iid'``
            Dirichlet-skewed class assignment (Acar et al., 2021).
        ``'random_subset'``
            Each agent draws an independent random subset of the training
            pool (with possible overlap across agents).  Val and test sets
            are shared.  Combine with ``class_imbalance=True`` to also
            apply a different random class distribution per agent.

    agent_classes : dict[int, list[int]] or None
        Per-agent class list for ``'class_partition'``.
    shared_classes : int or None
        Number of globally shared classes when ``'class_partition'`` is
        used without explicit ``agent_classes``.
    classes_per_agent : int
        Target classes per agent for ``'non_iid'`` (default: 3).
    alpha : float
        Dirichlet concentration for ``'non_iid'`` (default: 0.5).
    subset_fraction : float
        Fraction of training samples each agent draws in
        ``'random_subset'`` (default: 0.8).  Agents sample independently,
        so subsets can overlap.
    class_imbalance : bool
        When ``True`` and ``split_strategy='random_subset'``, each agent's
        subset is drawn with a random class distribution sampled from
        ``Dirichlet(imbalance_alpha)``.  Default: ``False``.
    imbalance_alpha : float
        Dirichlet concentration for per-agent class proportions when
        ``class_imbalance=True``.  Smaller values produce more skewed
        distributions (default: 0.5).
    exclude_null_activity : bool
        Ignored for MFeat (all labels are valid).  Present for interface
        symmetry with ``MHealthDataModule``.
    window_size : int or None
        Ignored for MFeat (no temporal structure).  Present for interface
        symmetry with ``MHealthDataModule``.
    stride : int
        Ignored for MFeat.  Present for interface symmetry.
    batch_size : int
        DataLoader batch size (default: 64).
    num_workers : int
        DataLoader worker count (default: 0).
    mode : str
        ``CombinedLoader`` mode (default: ``'min_size'``).
    val_split : float
        Fraction of data for validation (default: 0.1).
    test_split : float
        Fraction of data for testing (default: 0.1).
    pilot_split : float
        Fraction of data for the shared pilot set (default: 0.0).
    pilot_num_samples : int or None
        Absolute pilot size; overrides ``pilot_split``.
    comm_data : str
        Pilot loader strategy: ``'private_pilots'``,
        ``'shared_global_pilots'``, or ``'pairwise_pilots'``.
    seed : int
        Random seed (default: 42).

    Attributes
    ----------
    train_datasets, val_datasets, test_datasets, pilot_datasets :
        ``dict[int, MFeatDataset]`` populated after ``setup()``.
    input_shape : tuple[int]
        ``(n_features,)`` for agent 0's modality.
    num_classes : dict[str, int]
        ``{'label': 10}`` after ``setup()``.
    models : list[int]
        Agent indices.
    input_dims : dict[str, int]
        Per-agent feature dimension keyed by string agent index.
    """

    _DEFAULT_MODALITIES: dict[int, str] = {
        0: 'fou',
        1: 'fac',
        2: 'kar',
        3: 'pix',
        4: 'zer',
    }

    def __init__(
        self,
        data_path: str | Path,
        gdrive_file_id: str | None = None,
        n_agents: int = 5,
        agent_modalities: dict[int, str] | None = None,
        split_strategy: str = 'full',
        agent_classes: dict[int, list[int]] | None = None,
        shared_classes: int | None = None,
        classes_per_agent: int = 3,
        alpha: float = 0.5,
        subset_fraction: float = 0.8,
        class_imbalance: bool = False,
        imbalance_alpha: float = 0.5,
        exclude_null_activity: bool = True,
        window_size: int | None = None,
        stride: int = 1,
        batch_size: int = 64,
        num_workers: int = 0,
        mode: str = 'min_size',
        val_split: float = 0.1,
        test_split: float = 0.1,
        pilot_split: float = 0.0,
        pilot_num_samples: int | None = None,
        pilot_batch_size: int | None = None,
        comm_data: str = 'private_pilots',
        seed: int = 42,
    ) -> None:
        super().__init__()

        self.data_path = Path(data_path)
        self.gdrive_file_id = gdrive_file_id
        self.n_agents = n_agents

        if agent_modalities is None:
            self.agent_modalities: dict[int, str] = {
                k: v
                for k, v in self._DEFAULT_MODALITIES.items()
                if k < n_agents
            }
        else:
            self.agent_modalities = {int(k): v for k, v in agent_modalities.items()}

        self.split_strategy = split_strategy
        self.agent_classes = agent_classes or {}
        self.shared_classes = shared_classes
        self.classes_per_agent = classes_per_agent
        self.alpha = alpha
        self.subset_fraction = subset_fraction
        self.class_imbalance = class_imbalance
        self.imbalance_alpha = imbalance_alpha

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

    # ── Validation ────────────────────────────────────────────────────────────

    def _validate_modalities(self) -> None:
        for agent_id, mod in self.agent_modalities.items():
            if mod not in MFEAT_MODALITIES:
                raise ValueError(
                    f'Unknown modality "{mod}" for agent {agent_id}. '
                    f'Valid options: {list(MFEAT_MODALITIES)}'
                )

    # ── Download / extraction ─────────────────────────────────────────────────

    def prepare_data(self) -> None:
        """Download and extract MFeat dataset from Google Drive if not present."""
        dest = self.data_path

        existing = list(dest.glob('mfeat-*')) + list(dest.glob('mfeat_*'))
        if dest.exists() and existing:
            return

        if not self.gdrive_file_id:
            raise RuntimeError(
                'MFeat dataset not found. '
                'Provide gdrive_file_id in your config yaml.'
            )

        dest.mkdir(parents=True, exist_ok=True)
        zip_path = dest / 'mfeat.zip'

        print('[prepare_data] Downloading mfeat.zip …')
        gdown.download(id=self.gdrive_file_id, output=str(zip_path), quiet=False)

        if not zip_path.exists():
            raise RuntimeError(f'Download failed, zip not found at {zip_path}')

        print(f'[prepare_data] Extracting {zip_path} …')
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(dest)

        # Lift files up to dest root if they landed in a nested subdirectory
        data_files = list(dest.rglob('mfeat-*')) + list(dest.rglob('mfeat_*'))
        if not data_files:
            raise RuntimeError(
                f'No mfeat-* files found after extraction in {dest}'
            )

        top_level = {f for f in data_files if f.parent == dest}
        if not top_level:
            import shutil

            src_dir = data_files[0].parent
            for f in data_files:
                shutil.copy2(f, dest / f.name)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _norm_stats(self, arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = arr.mean(axis=0, keepdims=True)
        std = arr.std(axis=0, keepdims=True)
        std = np.where(std < 1e-8, 1.0, std)
        return mean, std

    def _make_dataset(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        indices: list[int],
        sample_ids: list[int] | None = None,
        normalize: bool | tuple[np.ndarray, np.ndarray] = True,
    ) -> MFeatDataset:
        return MFeatDataset(
            features=features,
            labels=labels,
            indices=indices,
            sample_ids=sample_ids,
            normalize=normalize,
        )

    def _build_pilot_datasets(
        self,
        pilot_indices: list[int],
    ) -> None:
        self.pilot_datasets = {}
        for i in range(self.n_agents):
            mod = self.agent_modalities[i]
            arr = self._modality_arrays[mod]
            labels = self._labels
            self.pilot_datasets[i] = self._make_dataset(
                arr,
                labels,
                pilot_indices,
                sample_ids=pilot_indices,
                normalize=self._norm_stats(arr[pilot_indices]),
            )

    # ── Split-strategy implementations ───────────────────────────────────────

    def _split_full(self, split_indices: dict[str, list[int]]) -> None:
        """All agents share the same sample indices, each via its own modality."""
        for i in range(self.n_agents):
            mod = self.agent_modalities[i]
            arr = self._modality_arrays[mod]
            labels = self._labels
            train_idx = split_indices['train']
            norm = self._norm_stats(arr[train_idx])

            self.train_datasets[i] = self._make_dataset(
                arr, labels, train_idx, normalize=norm
            )
            self.val_datasets[i] = self._make_dataset(
                arr, labels, split_indices['val'], normalize=norm
            )
            self.test_datasets[i] = self._make_dataset(
                arr, labels, split_indices['test'], normalize=norm
            )

    def _split_uniform(self, split_indices: dict[str, list[int]]) -> None:
        """Shuffle and evenly divide train samples; val/test shared."""
        gen = torch.Generator().manual_seed(self.seed)
        train_idx = split_indices['train']
        perm = torch.randperm(len(train_idx), generator=gen).tolist()
        chunks = [
            [train_idx[j] for j in perm[k::self.n_agents]]
            for k in range(self.n_agents)
        ]

        for i in range(self.n_agents):
            mod = self.agent_modalities[i]
            arr = self._modality_arrays[mod]
            labels = self._labels
            norm = self._norm_stats(arr[chunks[i]])

            self.train_datasets[i] = self._make_dataset(
                arr, labels, chunks[i], normalize=norm
            )
            self.val_datasets[i] = self._make_dataset(
                arr, labels, split_indices['val'], normalize=norm
            )
            self.test_datasets[i] = self._make_dataset(
                arr, labels, split_indices['test'], normalize=norm
            )

    def _split_class_partition(self, split_indices: dict[str, list[int]]) -> None:
        if not self.agent_classes:
            raise ValueError(
                "split_strategy='class_partition' requires agent_classes or "
                'shared_classes to be provided.'
            )
        for split_name, target in [
            ('train', self.train_datasets),
            ('val', self.val_datasets),
            ('test', self.test_datasets),
        ]:
            idx = split_indices[split_name]
            split_labels = self._labels[idx].tolist()
            partition = partition_by_agent_classes(
                labels=split_labels,
                agent_classes=self.agent_classes,
                n_agents=self.n_agents,
                seed=self.seed,
            )
            for i in range(self.n_agents):
                mod = self.agent_modalities[i]
                arr = self._modality_arrays[mod]
                agent_idx = [idx[j] for j in partition[i]]
                norm = (
                    self._norm_stats(arr[agent_idx])
                    if split_name == 'train'
                    else self._train_norm_cache[i]
                )
                if split_name == 'train':
                    self._train_norm_cache[i] = norm
                target[i] = self._make_dataset(
                    arr, self._labels, agent_idx, normalize=norm
                )

    def _split_non_iid(self, split_indices: dict[str, list[int]]) -> None:
        train_idx = split_indices['train']
        train_labels = self._labels[train_idx].tolist()

        train_partition, sampled_classes = partition_non_iid(
            labels=train_labels,
            n_agents=self.n_agents,
            classes_per_agent=self.classes_per_agent,
            seed=self.seed,
            alpha=self.alpha,
            return_agent_classes=True,
        )
        self.agent_classes = sampled_classes

        norm_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for i in range(self.n_agents):
            mod = self.agent_modalities[i]
            arr = self._modality_arrays[mod]
            agent_train_idx = [train_idx[j] for j in train_partition[i]]
            norm_cache[i] = self._norm_stats(arr[agent_train_idx])
            self.train_datasets[i] = self._make_dataset(
                arr, self._labels, agent_train_idx, normalize=norm_cache[i]
            )

        for split_name, target, seed_offset in [
            ('val', self.val_datasets, 1),
            ('test', self.test_datasets, 2),
        ]:
            idx = split_indices[split_name]
            split_labels = self._labels[idx].tolist()
            partition = partition_non_iid(
                labels=split_labels,
                n_agents=self.n_agents,
                classes_per_agent=self.classes_per_agent,
                seed=self.seed + seed_offset,
                alpha=self.alpha,
                agent_classes=self.agent_classes,
            )
            for i in range(self.n_agents):
                mod = self.agent_modalities[i]
                arr = self._modality_arrays[mod]
                agent_idx = [idx[j] for j in partition[i]]
                target[i] = self._make_dataset(
                    arr, self._labels, agent_idx, normalize=norm_cache[i]
                )

    def _split_random_subset(self, split_indices: dict[str, list[int]]) -> None:
        """Each agent draws an independent random subset of training samples.

        Subsets can overlap across agents.  Val and test indices are shared
        so that evaluation remains comparable across agents.
        When ``self.class_imbalance`` is ``True`` each agent's subset is
        drawn with a different Dirichlet-sampled class distribution.
        """
        train_idx = split_indices['train']
        subset_size = max(1, int(round(len(train_idx) * self.subset_fraction)))

        for i in range(self.n_agents):
            mod = self.agent_modalities[i]
            arr = self._modality_arrays[mod]

            agent_train_idx = sample_random_subset(
                indices=train_idx,
                subset_size=subset_size,
                seed=self.seed + i * 100,
                labels=self._labels if self.class_imbalance else None,
                alpha=self.imbalance_alpha if self.class_imbalance else None,
            )

            norm = self._norm_stats(arr[agent_train_idx])
            self.train_datasets[i] = self._make_dataset(
                arr, self._labels, agent_train_idx, normalize=norm
            )
            self.val_datasets[i] = self._make_dataset(
                arr, self._labels, split_indices['val'], normalize=norm
            )
            self.test_datasets[i] = self._make_dataset(
                arr, self._labels, split_indices['test'], normalize=norm
            )

    # ── LightningDataModule interface ─────────────────────────────────────────

    def setup(self, stage: str | None = None) -> None:
        """Load modality arrays and build per-agent train/val/test/pilot datasets."""
        self._validate_modalities()

        # Load all modalities needed by at least one agent
        needed = set(self.agent_modalities.values())
        self._modality_arrays: dict[str, np.ndarray] = {
            mod: _load_mfeat_modality(self.data_path, mod) for mod in needed
        }

        # Verify all arrays have the same number of rows
        n_rows = {mod: arr.shape[0] for mod, arr in self._modality_arrays.items()}
        if len(set(n_rows.values())) > 1:
            raise RuntimeError(
                f'Inconsistent sample counts across modalities: {n_rows}'
            )
        total = next(iter(n_rows.values()))
        self._labels: np.ndarray = _build_labels(total)

        self.train_datasets: dict[int, MFeatDataset] = {}
        self.val_datasets: dict[int, MFeatDataset] = {}
        self.test_datasets: dict[int, MFeatDataset] = {}
        self.pilot_datasets: dict[int, MFeatDataset] = {}
        self._train_norm_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

        split_indices = compute_split_indices(
            total_size=total,
            val_split=self.val_split,
            test_split=self.test_split,
            seed=self.seed,
            pilot_split=self.pilot_split,
            pilot_num_samples=self.pilot_num_samples,
        )

        if split_indices['pilot']:
            self._build_pilot_datasets(split_indices['pilot'])

        if (
            self.split_strategy == 'class_partition'
            and not self.agent_classes
            and self.shared_classes is not None
        ):
            self.agent_classes = build_shared_class_partition(
                unique_classes=list(range(_NUM_CLASSES)),
                n_agents=self.n_agents,
                shared_classes=int(self.shared_classes),
                seed=self.seed,
            )

        if self.split_strategy == 'full':
            self._split_full(split_indices)
        elif self.split_strategy == 'uniform':
            self._split_uniform(split_indices)
        elif self.split_strategy == 'class_partition':
            self._split_class_partition(split_indices)
        elif self.split_strategy == 'non_iid':
            self._split_non_iid(split_indices)
        elif self.split_strategy == 'random_subset':
            self._split_random_subset(split_indices)
        else:
            raise ValueError(
                f'Unknown split_strategy: {self.split_strategy!r}. '
                "Valid options: 'full', 'uniform', 'class_partition', "
                "'non_iid', 'random_subset'"
            )

        self.num_classes: dict[str, int] = {'label': _NUM_CLASSES}
        self.models: list[int] = list(range(self.n_agents))

        sample_x, *_ = self.train_datasets[0][0]
        self.input_shape: tuple[int, ...] = tuple(sample_x.shape)

        self.input_dims: dict[str, int] = {
            str(i): MFEAT_MODALITIES[self.agent_modalities[i]]
            for i in self.models
        }

    # ── DataLoader factories ──────────────────────────────────────────────────

    def _make_loader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
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
        repeated = repeat_dataset_to_num_samples(dataset, target_num_samples)
        return DataLoader(
            repeated,
            batch_size=self.pilot_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def _create_loaders(self, split: str) -> CombinedLoader:
        split_map: dict[str, tuple[dict[int, MFeatDataset], bool]] = {
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
            target_num_batches = max(
                (len(ds) + self.batch_size - 1) // self.batch_size
                for ds in datasets.values()
            )
            if self.comm_data == 'private_pilots':
                loaders.update(
                    {
                        f'pilot_{i}': self._make_pilot_loader(
                            ds, target_num_batches
                        )
                        for i, ds in self.pilot_datasets.items()
                    }
                )
            elif self.comm_data == 'shared_global_pilots':
                loaders.update(
                    {
                        f'global_pilot_{i}': self._make_pilot_loader(
                            ds, target_num_batches
                        )
                        for i, ds in self.pilot_datasets.items()
                    }
                )
            elif self.comm_data == 'pairwise_pilots':
                loaders.update(
                    {
                        f'pilot_{i}_{j}': self._make_pilot_loader(
                            PairwiseDataset(
                                self.pilot_datasets[i], self.pilot_datasets[j]
                            ),
                            target_num_batches,
                        )
                        for i in range(self.n_agents)
                        for j in range(i + 1, self.n_agents)
                    }
                )

        return CombinedLoader(loaders, mode=self.mode)

    def train_dataloader(self) -> CombinedLoader:
        return self._create_loaders('train')

    def val_dataloader(self) -> CombinedLoader:
        return self._create_loaders('val')

    def test_dataloader(self) -> CombinedLoader:
        return self._create_loaders('test')
