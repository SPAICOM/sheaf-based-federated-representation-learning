"""
Mobile Health (MHEALTH) datamodule for sensor-based activity recognition.

This module provides data loading for the MHEALTH dataset, which contains
wearable sensor data (accelerometer, gyroscope, magnetometer, ECG) recorded
at 50 Hz from three body positions across 10 subjects performing 12 activities.

Features:
- Download from Google Drive via gdown
- Load from a merged CSV file or a directory of per-subject ``.log`` files
- Named feature groups: ``'chest'``, ``'left_ankle'``, ``'right_wrist'``
- Optional sliding-window segmentation for temporal modelling
- Split strategies: ``'uniform'``, ``'class_partition'``, ``'non_iid'``,
  ``'subject'`` (natural federated split — one agent per subject)
- Shared pilot dataset for federated calibration
- ``CombinedLoader`` with optional ``pilot_{i}`` loaders — symmetric with
  ``ClassificationDataModule`` and ``SemanticDataModule``
"""

import shutil
import zipfile
from pathlib import Path

import gdown
import lightning as l
import numpy as np
import pandas as pd
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

MHEALTH_KAGGLE_COLUMNS: dict[str, str] = {
    'alx': 'acc_ankle_x',
    'aly': 'acc_ankle_y',
    'alz': 'acc_ankle_z',
    'glx': 'gyro_ankle_x',
    'gly': 'gyro_ankle_y',
    'glz': 'gyro_ankle_z',
    'arx': 'acc_chest_x',
    'ary': 'acc_chest_y',
    'arz': 'acc_chest_z',
    'grx': 'gyro_ankle_x',
    'gry': 'gyro_ankle_y',
    'grz': 'gyro_ankle_z',
    'Activity': 'activity',
    'subject': 'subject',
}

# ── Column layout of the raw MHEALTH .log files ───────────────────────────────
MHEALTH_COLUMNS: list[str] = [
    'acc_chest_x',
    'acc_chest_y',
    'acc_chest_z',
    'ecg_lead1',
    'ecg_lead2',
    'acc_ankle_x',
    'acc_ankle_y',
    'acc_ankle_z',
    'gyro_ankle_x',
    'gyro_ankle_y',
    'gyro_ankle_z',
    'mag_ankle_x',
    'mag_ankle_y',
    'mag_ankle_z',
    'acc_wrist_x',
    'acc_wrist_y',
    'acc_wrist_z',
    'gyro_wrist_x',
    'gyro_wrist_y',
    'gyro_wrist_z',
    'mag_wrist_x',
    'mag_wrist_y',
    'mag_wrist_z',
    'activity',
]

# ── Named sensor groups for convenient feature selection ──────────────────────
MHEALTH_FEATURE_GROUPS: dict[str, list[str]] = {
    'chest': [
        'acc_chest_x',
        'acc_chest_y',
        'acc_chest_z',
        'ecg_lead1',
        'ecg_lead2',
    ],
    'left_ankle': [
        'acc_ankle_x',
        'acc_ankle_y',
        'acc_ankle_z',
        'gyro_ankle_x',
        'gyro_ankle_y',
        'gyro_ankle_z',
        'mag_ankle_x',
        'mag_ankle_y',
        'mag_ankle_z',
    ],
    'right_wrist': [
        'acc_wrist_x',
        'acc_wrist_y',
        'acc_wrist_z',
        'gyro_wrist_x',
        'gyro_wrist_y',
        'gyro_wrist_z',
        'mag_wrist_x',
        'mag_wrist_y',
        'mag_wrist_z',
    ],
}

# ── Per-sensor modality groupings (x,y,z axes treated as one 3-D modality) ───
MHEALTH_SENSOR_MODALITIES: dict[str, list[str]] = {
    'acc_chest': ['acc_chest_x', 'acc_chest_y', 'acc_chest_z'],
    'ecg': ['ecg_lead1', 'ecg_lead2'],
    'acc_ankle': ['acc_ankle_x', 'acc_ankle_y', 'acc_ankle_z'],
    'gyro_ankle': ['gyro_ankle_x', 'gyro_ankle_y', 'gyro_ankle_z'],
    'mag_ankle': ['mag_ankle_x', 'mag_ankle_y', 'mag_ankle_z'],
    'acc_wrist': ['acc_wrist_x', 'acc_wrist_y', 'acc_wrist_z'],
    'gyro_wrist': ['gyro_wrist_x', 'gyro_wrist_y', 'gyro_wrist_z'],
    'mag_wrist': ['mag_wrist_x', 'mag_wrist_y', 'mag_wrist_z'],
}

_LABEL_COL = 'activity'


# ── I/O helper ────────────────────────────────────────────────────────────────


def _load_mhealth_df(
    data_path: Path,
    label_col: str = _LABEL_COL,
    exclude_null_activity: bool = True,
) -> pd.DataFrame:
    """Load MHEALTH data from a CSV file or a directory of per-subject logs.

    Parameters
    ----------
    data_path : Path
        A single CSV/log file *or* a directory of space-separated ``.log``
        files (one per subject).  When loading a directory a ``'subject'``
        column (1-indexed) is appended automatically.
    label_col : str
        Name of the activity label column.
    exclude_null_activity : bool
        Remove rows where ``label_col == 0`` (null / transition activity).

    Returns
    -------
    pd.DataFrame
        Merged, labelled, and optionally filtered DataFrame.
    """

    def _load_single_file(
        fpath: Path, subject_id: int | None = None
    ) -> pd.DataFrame:
        df = pd.read_csv(fpath, sep='\t')

        if set(df.columns) == set(MHEALTH_KAGGLE_COLUMNS.keys()):
            df = df.rename(columns=MHEALTH_KAGGLE_COLUMNS)
        elif df.shape[1] == len(MHEALTH_COLUMNS):
            df.columns = MHEALTH_COLUMNS
        elif df.shape[1] == len(MHEALTH_COLUMNS) + 1:
            df.columns = MHEALTH_COLUMNS + ['subject']
        else:
            raise ValueError(
                f'Unexpected column count in {fpath}: '
                f'expected {len(MHEALTH_COLUMNS)}, got {df.shape[1]}'
            )

        if subject_id is not None:
            df['subject'] = subject_id
        return df

    if data_path.is_dir():
        log_files = sorted(data_path.glob('*.log'))
        if not log_files:
            log_files = sorted(data_path.glob('*.csv'))
        if not log_files:
            raise FileNotFoundError(
                f'No .log or .csv files found in directory: {data_path}'
            )

        frames = []
        for subject_id, fpath in enumerate(log_files, start=1):
            frames.append(_load_single_file(fpath, subject_id))
        merged = pd.concat(frames, ignore_index=True)
    else:
        merged = _load_single_file(data_path)

    if exclude_null_activity and label_col in merged.columns:
        merged = merged[merged[label_col] != 0].reset_index(drop=True)

    return merged


def _resolve_feature_cols(
    df: pd.DataFrame,
    feature_cols: list[str] | None,
    feature_groups: list[str] | None,
) -> list[str]:
    """Determine feature columns with priority: explicit > groups > all."""
    if feature_cols is not None:
        return feature_cols
    if feature_groups is not None:
        cols: list[str] = []
        for group in feature_groups:
            if group not in MHEALTH_FEATURE_GROUPS:
                raise ValueError(
                    f'Unknown feature_group "{group}". '
                    f'Valid options: {list(MHEALTH_FEATURE_GROUPS)}'
                )
            cols.extend(MHEALTH_FEATURE_GROUPS[group])
        return cols
    return [c for c in MHEALTH_COLUMNS if c != _LABEL_COL]


# ── Dataset ───────────────────────────────────────────────────────────────────


class MHealthDataset(Dataset):
    """PyTorch dataset wrapping MHEALTH DataFrame rows.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    feature_cols : list[str]
        Feature column names.
    label_col : str
        Label column name.
    window_size : int or None
        Sliding-window length.  ``None`` → point-wise (row-level) sampling.
    stride : int
        Sliding-window step size.
    sample_ids : list[int] or None
        Global row indices carried along for pilot identification.  When set,
        ``__getitem__`` returns ``(features, label, sample_id)``.
    normalize : bool or tuple of arrays
        If True, apply standard normalization using precomputed stats.
        If a tuple (mean, std), use those for normalization.
        If False, no normalization applied.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        label_col: str = _LABEL_COL,
        window_size: int | None = None,
        stride: int = 1,
        sample_ids: list[int] | None = None,
        normalize: bool | tuple[np.ndarray, np.ndarray] = True,
    ) -> None:
        self.feature_cols = feature_cols
        self.label_col = label_col
        self.window_size = window_size
        self.stride = stride
        self.sample_ids = sample_ids

        self.features: np.ndarray = (
            df[feature_cols].to_numpy(dtype='float32').copy()
        )
        self.labels: np.ndarray = df[label_col].to_numpy(dtype='int64').copy()

        if isinstance(normalize, tuple):
            self.mean = normalize[0].copy()
            self.std = normalize[1].copy()
            self._normalize = True
        elif normalize:
            self.mean = np.mean(self.features, axis=0, keepdims=True).copy()
            self.std = np.std(self.features, axis=0, keepdims=True).copy()
            self.std = np.where(self.std < 1e-8, 1.0, self.std)
            self._normalize = True
        else:
            self.mean = np.zeros((1, len(feature_cols)), dtype='float32')
            self.std = np.ones((1, len(feature_cols)), dtype='float32')
            self._normalize = False

        if self._normalize:
            self.features = (self.features - self.mean) / self.std

        if window_size is not None:
            n = len(self.features)
            self._windows = list(range(0, max(n - window_size + 1, 0), stride))
        else:
            self._windows = None

    def __len__(self) -> int:
        return (
            len(self._windows)
            if self._windows is not None
            else len(self.features)
        )

    def __getitem__(
        self, idx: int
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        if self._windows is not None:
            start = self._windows[idx]
            x = torch.from_numpy(
                self.features[start : start + self.window_size].copy()
            )  # (window_size, n_features)
            window_labels = self.labels[start : start + self.window_size]
            label_val = int(
                torch.mode(
                    torch.from_numpy(window_labels.copy())
                ).values.item()
            )
            label = torch.tensor(label_val, dtype=torch.long)
        else:
            x = torch.from_numpy(self.features[idx].copy())
            label = torch.tensor(int(self.labels[idx]), dtype=torch.long)

        if self.sample_ids is not None:
            sample_id = torch.tensor(self.sample_ids[idx], dtype=torch.long)
            return x, label, sample_id

        return x, label


# ── DataModule ────────────────────────────────────────────────────────────────


class MHealthDataModule(l.LightningDataModule):
    """DataModule for the Kaggle MHEALTH dataset.

    Loads wearable-sensor data for human activity recognition, distributes it
    across multiple agents, and exposes ``CombinedLoader`` instances that
    mirror the structure of ``ClassificationDataModule`` and
    ``SemanticDataModule``.

    Parameters
    ----------
    data_path : str or Path
        Path to a merged CSV file *or* a directory of per-subject ``.log``
        files.
    n_agents : int or None
        Number of federated agents.  Inferred automatically from the number of
        subjects when ``split_strategy='subject'`` and left ``None``.
    feature_cols : list[str] or None
        Explicit list of sensor columns to use as features.  Takes priority
        over ``feature_groups``.
    feature_groups : list[str] or None
        Named sensor groups to include (``'chest'``, ``'left_ankle'``,
        ``'right_wrist'``).  Ignored when ``feature_cols`` is set.  Defaults
        to all 23 sensor columns.
    label_col : str
        Activity label column name (default: ``'activity'``).
    exclude_null_activity : bool
        Drop rows with label ``0`` (null/transition activity, default: True).
    window_size : int or None
        Sliding-window length in samples.  ``None`` → point-wise (row-level)
        sampling.
    stride : int
        Sliding-window step size (default: 1).
    split_strategy : str
        Data distribution strategy:

        - ``'subject'`` *(default)*: one agent per subject; splits computed
          within each subject.
        - ``'uniform'``: random even split across agents.
        - ``'class_partition'``: each agent sees a specific subset of activity
          classes.
        - ``'non_iid'``: Dirichlet-skewed automatic class assignment.

    agent_classes : dict[int, list[int]] or None
        Per-agent class lists for ``'class_partition'``.
    shared_classes : int or None
        Globally shared class count when ``'class_partition'`` is used without
        explicit ``agent_classes``.
    classes_per_agent : int
        Target classes per agent for ``'non_iid'`` (default: 3).
    alpha : float
        Dirichlet concentration for ``'non_iid'`` (default: 0.5).
    batch_size : int
        DataLoader batch size (default: 64).
    num_workers : int
        DataLoader worker processes (default: 4).
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
    pilot_batch_size : int or None
        Batch size for pilot loaders; defaults to ``batch_size``.
    seed : int
        Random seed for reproducibility (default: 42).

    Attributes
    ----------
    train_datasets, val_datasets, test_datasets, pilot_datasets :
        ``dict[int, MHealthDataset]`` populated after ``setup()``.
    feature_cols : list[str]
        Resolved feature column names.
    input_shape : tuple[int, ...]
        Shape of one input sample: ``(n_features,)`` or
        ``(window_size, n_features)``.
    num_classes : dict[str, int]
        ``{'activity': <global_n_classes>}`` after ``setup()``.
    models : list[int]
        Agent indices (kept for compatibility with other datamodules).
    input_dims : dict[str, int]
        Last feature dimension per agent key.
    """

    def __init__(
        self,
        data_path: str | Path,
        gdrive_file_id: str | None = None,
        n_agents: int | None = None,
        feature_cols: list[str] | None = None,
        feature_groups: list[str] | None = None,
        agent_modalities: dict[int, str] | None = None,
        label_col: str = _LABEL_COL,
        exclude_null_activity: bool = True,
        window_size: int | None = None,
        stride: int = 1,
        split_strategy: str = 'subject',
        agent_classes: dict[int, list[int]] | None = None,
        shared_classes: int | None = None,
        classes_per_agent: int = 3,
        alpha: float = 0.5,
        subset_fraction: float = 0.8,
        class_imbalance: bool = False,
        imbalance_alpha: float = 0.5,
        batch_size: int = 64,
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
        super().__init__()

        self.data_path = Path(data_path)
        self.gdrive_file_id = gdrive_file_id
        self.n_agents = n_agents
        self._feature_cols_param = feature_cols
        self.feature_groups = feature_groups
        self.agent_modalities: dict[int, str] = agent_modalities or {}
        self.label_col = label_col
        self.exclude_null_activity = exclude_null_activity
        self.window_size = window_size
        self.stride = stride
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

    def prepare_data(self) -> None:
        """Download and extract MHEALTH dataset from Google Drive if not present.

        The folder may contain either a zip archive (which is extracted) or
        the raw .log/.csv files directly; both layouts are handled automatically.
        """
        dest = self.data_path

        if dest.exists() and (
            list(dest.glob('*.log')) or list(dest.glob('*.csv'))
        ):
            return

        if not self.gdrive_file_id:
            raise RuntimeError(
                'MHEALTH dataset not found. '
                'Provide gdrive_file_id in your config yaml.'
            )

        dest.mkdir(parents=True, exist_ok=True)
        zip_path = dest / 'mhealth_dataset.zip'

        print('[prepare_data] Downloading mhealth_dataset.zip …')
        gdown.download(
            id=self.gdrive_file_id, output=str(zip_path), quiet=False
        )

        if not zip_path.exists():
            raise RuntimeError(f'Download failed, zip not found at {zip_path}')

        print(f'[prepare_data] Extracting {zip_path} …')
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(dest)

        # Lift data files up to dest root if they landed in a subdirectory
        data_files = list(dest.rglob('*.log')) + list(dest.rglob('*.csv'))
        if not data_files:
            raise RuntimeError(
                f'No .log or .csv files found after extraction in {dest}'
            )

        data_dir = data_files[0].parent
        if data_dir != dest:
            for f in data_dir.glob('*'):
                if f.suffix in ('.log', '.csv'):
                    shutil.copy2(f, dest / f.name)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _compute_norm_stats(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute per-channel mean and std for normalization."""
        features = df[feature_cols].to_numpy(dtype='float32')
        mean = np.mean(features, axis=0, keepdims=True).copy()
        std = np.std(features, axis=0, keepdims=True).copy()
        std = np.where(std < 1e-8, 1.0, std)
        return mean, std

    def _make_dataset(
        self,
        df: pd.DataFrame,
        sample_ids: list[int] | None = None,
        feature_cols: list[str] | None = None,
        normalize: bool | tuple[np.ndarray, np.ndarray] = True,
    ) -> MHealthDataset:
        cols = feature_cols if feature_cols is not None else self.feature_cols
        available_cols = [c for c in cols if c in df.columns]
        if len(available_cols) < len(cols):
            missing = set(cols) - set(available_cols)
            print(
                f'[setup] Note: {len(missing)} columns not available '
                f'in dataset: {missing}'
            )
        return MHealthDataset(
            df=df,
            feature_cols=available_cols,
            label_col=self.label_col,
            window_size=self.window_size,
            stride=self.stride,
            sample_ids=sample_ids,
            normalize=normalize,
        )

    def _select(self, df: pd.DataFrame, indices: list[int]) -> pd.DataFrame:
        return df.iloc[indices].reset_index(drop=True)

    def _build_pilot_datasets(
        self,
        pilot_df: pd.DataFrame,
        pilot_indices: list[int],
    ) -> None:
        self.pilot_datasets = {}
        for i in range(self.n_agents):
            self.pilot_datasets[i] = self._make_dataset(
                pilot_df,
                sample_ids=pilot_indices,
                feature_cols=self._agent_feature_cols.get(i),
            )

    # ── Split-strategy implementations ───────────────────────────────────────

    def _split_data_uniform(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> None:
        norm_stats_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for i in range(self.n_agents):
            fcols = self._agent_feature_cols.get(i)
            norm_stats_cache[i] = self._compute_norm_stats(train_df, fcols)

        if self.n_agents == 1:
            self.train_datasets[0] = self._make_dataset(
                train_df, normalize=norm_stats_cache[0]
            )
            self.val_datasets[0] = self._make_dataset(
                val_df, normalize=norm_stats_cache[0]
            )
            self.test_datasets[0] = self._make_dataset(
                test_df, normalize=norm_stats_cache[0]
            )
            return

        gen = torch.Generator().manual_seed(self.seed)
        for split_df, target in [
            (train_df, self.train_datasets),
            (val_df, self.val_datasets),
            (test_df, self.test_datasets),
        ]:
            idx = torch.randperm(len(split_df), generator=gen)
            chunks = idx.split(len(split_df) // self.n_agents)
            for i in range(self.n_agents):
                target[i] = self._make_dataset(
                    self._select(split_df, chunks[i].tolist()),
                    feature_cols=self._agent_feature_cols.get(i),
                    normalize=norm_stats_cache[i],
                )

    def _split_data_class_partition(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> None:
        if not self.agent_classes:
            raise ValueError(
                "split_strategy='class_partition' requires agent_classes or "
                'shared_classes to be provided'
            )
        norm_stats_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for i in range(self.n_agents):
            fcols = self._agent_feature_cols.get(i)
            norm_stats_cache[i] = self._compute_norm_stats(train_df, fcols)

        for split_df, target in [
            (train_df, self.train_datasets),
            (val_df, self.val_datasets),
            (test_df, self.test_datasets),
        ]:
            partition = partition_by_agent_classes(
                labels=split_df[self.label_col].tolist(),
                agent_classes=self.agent_classes,
                n_agents=self.n_agents,
                seed=self.seed,
            )
            for i in range(self.n_agents):
                if not self.agent_classes.get(i):
                    raise ValueError(
                        f'agent {i} has no assigned classes in agent_classes'
                    )
                target[i] = self._make_dataset(
                    self._select(split_df, partition[i]),
                    feature_cols=self._agent_feature_cols.get(i),
                    normalize=norm_stats_cache[i],
                )

    def _split_data_non_iid(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> None:
        train_partition, sampled_classes = partition_non_iid(
            labels=train_df[self.label_col].tolist(),
            n_agents=self.n_agents,
            classes_per_agent=self.classes_per_agent,
            seed=self.seed,
            alpha=self.alpha,
            return_agent_classes=True,
        )
        self.agent_classes = sampled_classes

        norm_stats_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for i in range(self.n_agents):
            fcols = self._agent_feature_cols.get(i)
            norm_stats_cache[i] = self._compute_norm_stats(train_df, fcols)

        for split_df, target, seed_offset in [
            (train_df, self.train_datasets, 0),
            (val_df, self.val_datasets, 1),
            (test_df, self.test_datasets, 2),
        ]:
            if seed_offset == 0:
                partition = train_partition
            else:
                partition = partition_non_iid(
                    labels=split_df[self.label_col].tolist(),
                    n_agents=self.n_agents,
                    classes_per_agent=self.classes_per_agent,
                    seed=self.seed + seed_offset,
                    alpha=self.alpha,
                    agent_classes=self.agent_classes,
                )
            for i in range(self.n_agents):
                target[i] = self._make_dataset(
                    self._select(split_df, partition[i]),
                    feature_cols=self._agent_feature_cols.get(i),
                    normalize=norm_stats_cache[i],
                )

    def _split_data_subject(self, df: pd.DataFrame) -> None:
        """One agent per subject; train/val/test split within each subject."""
        if 'subject' not in df.columns:
            raise ValueError(
                "split_strategy='subject' requires a 'subject' column. "
                'Load from a directory of per-subject .log files or include a '
                "'subject' column in the CSV."
            )
        subjects = sorted(df['subject'].unique())
        if self.n_agents is None:
            self.n_agents = len(subjects)
        subjects = subjects[: self.n_agents]

        for i, subj in enumerate(subjects):
            subj_df = df[df['subject'] == subj].reset_index(drop=True)
            idxs = compute_split_indices(
                total_size=len(subj_df),
                val_split=self.val_split,
                test_split=self.test_split,
                seed=self.seed,
                pilot_split=0.0,
                pilot_num_samples=None,
            )
            fcols = self._agent_feature_cols.get(i)
            train_df = self._select(subj_df, idxs['train'])
            norm_stats = self._compute_norm_stats(train_df, fcols)
            self.train_datasets[i] = self._make_dataset(
                train_df, feature_cols=fcols, normalize=norm_stats
            )
            self.val_datasets[i] = self._make_dataset(
                self._select(subj_df, idxs['val']),
                feature_cols=fcols,
                normalize=norm_stats,
            )
            self.test_datasets[i] = self._make_dataset(
                self._select(subj_df, idxs['test']),
                feature_cols=fcols,
                normalize=norm_stats,
            )

    def _split_data_random_subset(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> None:
        """Each agent draws an independent random subset of training rows.

        Subsets can overlap across agents.  Val and test DataFrames are
        shared.  When ``self.class_imbalance`` is ``True`` each agent's
        subset is drawn with a Dirichlet-sampled class distribution.
        """
        train_idx = list(range(len(train_df)))
        subset_size = max(1, int(round(len(train_idx) * self.subset_fraction)))
        labels_arr = train_df[self.label_col].to_numpy()

        for i in range(self.n_agents):
            fcols = self._agent_feature_cols.get(i)

            agent_train_idx = sample_random_subset(
                indices=train_idx,
                subset_size=subset_size,
                seed=self.seed + i * 100,
                labels=labels_arr if self.class_imbalance else None,
                alpha=self.imbalance_alpha if self.class_imbalance else None,
            )

            agent_train_df = self._select(train_df, agent_train_idx)
            norm_stats = self._compute_norm_stats(
                agent_train_df, fcols or self.feature_cols
            )
            self.train_datasets[i] = self._make_dataset(
                agent_train_df, feature_cols=fcols, normalize=norm_stats
            )
            self.val_datasets[i] = self._make_dataset(
                val_df, feature_cols=fcols, normalize=norm_stats
            )
            self.test_datasets[i] = self._make_dataset(
                test_df, feature_cols=fcols, normalize=norm_stats
            )

    # ── LightningDataModule interface ─────────────────────────────────────────

    def setup(self, stage: str | None = None) -> None:
        """Load data and populate per-agent train/val/test/pilot datasets.

        Parameters
        ----------
        stage : str or None
            Ignored; all splits are always prepared.
        """
        df = _load_mhealth_df(
            self.data_path,
            label_col=self.label_col,
            exclude_null_activity=self.exclude_null_activity,
        )

        self.feature_cols: list[str] = _resolve_feature_cols(
            df, self._feature_cols_param, self.feature_groups
        )

        # Remap labels to 0-indexed.  The raw MHEALTH labels are 1–12 (label 0
        # is the null/transition activity that is excluded above).
        # CrossEntropyLoss requires contiguous indices starting from 0, so a
        # label of 12 with num_classes=12 would trigger an out-of-bounds CUDA
        # assert without this remapping.
        if self.label_col in df.columns:
            min_label = df[self.label_col].min()
            if min_label != 0:
                df[self.label_col] = df[self.label_col] - min_label

        # Build per-agent feature column mapping.
        # When agent_modalities is set, each agent receives only the 3 (or 2
        # for ECG) columns belonging to its assigned sensor modality, so that
        # modality-specific encoders (MHealthAccelerometerEncoder, etc.) see
        # the correct 3-D (or 2-D) input vectors instead of a flat concat of
        # all sensor channels.
        if self.agent_modalities:
            for mod in self.agent_modalities.values():
                if mod not in MHEALTH_SENSOR_MODALITIES:
                    raise ValueError(
                        f'Unknown modality "{mod}". '
                        f'Valid options: {list(MHEALTH_SENSOR_MODALITIES)}'
                    )
            self._agent_feature_cols: dict[int, list[str] | None] = {
                i: MHEALTH_SENSOR_MODALITIES[mod]
                for i, mod in self.agent_modalities.items()
            }
        else:
            self._agent_feature_cols = {}

        self.train_datasets: dict[int, MHealthDataset] = {}
        self.val_datasets: dict[int, MHealthDataset] = {}
        self.test_datasets: dict[int, MHealthDataset] = {}
        self.pilot_datasets: dict[int, MHealthDataset] = {}
        self.num_classes: dict[str, int] = {}

        if self.split_strategy == 'subject':
            # Extract a global pilot before subject partitioning so that every
            # agent sees the same anchor points.
            if self.pilot_split > 0.0 or self.pilot_num_samples:
                pilot_idxs = compute_split_indices(
                    total_size=len(df),
                    val_split=0.0,
                    test_split=0.0,
                    seed=self.seed,
                    pilot_split=self.pilot_split,
                    pilot_num_samples=self.pilot_num_samples,
                )['pilot']
                mask = torch.ones(len(df), dtype=torch.bool)
                mask[pilot_idxs] = False
                pilot_df = self._select(df, pilot_idxs)
                df = df[mask.numpy()].reset_index(drop=True)
                # n_agents not yet known; set temporarily from subjects
                if self.n_agents is None:
                    self.n_agents = len(df['subject'].unique())
                self._build_pilot_datasets(pilot_df, pilot_idxs)
            self._split_data_subject(df)
        else:
            # Infer n_agents for non-subject strategies
            if self.n_agents is None:
                max_key = (
                    max(self.agent_classes.keys()) if self.agent_classes else 0
                )
                self.n_agents = max_key + 1

            split_indices = compute_split_indices(
                total_size=len(df),
                val_split=self.val_split,
                test_split=self.test_split,
                seed=self.seed,
                pilot_split=self.pilot_split,
                pilot_num_samples=self.pilot_num_samples,
            )

            pilot_df = self._select(df, split_indices['pilot'])
            train_df = self._select(df, split_indices['train'])
            val_df = self._select(df, split_indices['val'])
            test_df = self._select(df, split_indices['test'])

            if split_indices['pilot']:
                self._build_pilot_datasets(pilot_df, split_indices['pilot'])

            if (
                self.split_strategy == 'class_partition'
                and not self.agent_classes
                and self.shared_classes is not None
            ):
                self.agent_classes = build_shared_class_partition(
                    unique_classes=sorted(
                        df[self.label_col].unique().tolist()
                    ),
                    n_agents=self.n_agents,
                    shared_classes=int(self.shared_classes),
                    seed=self.seed,
                )

            if self.split_strategy == 'uniform':
                self._split_data_uniform(train_df, val_df, test_df)
            elif self.split_strategy == 'class_partition':
                self._split_data_class_partition(train_df, val_df, test_df)
            elif self.split_strategy == 'non_iid':
                self._split_data_non_iid(train_df, val_df, test_df)
            elif self.split_strategy == 'random_subset':
                self._split_data_random_subset(train_df, val_df, test_df)
            else:
                raise ValueError(
                    f'Unknown split_strategy: {self.split_strategy!r}. '
                    "Valid options: 'subject', 'uniform', "
                    "'class_partition', 'non_iid', 'random_subset'"
                )

        # Global label space (used by classifier heads)
        self.num_classes['label'] = len(df[self.label_col].unique())

        # Infer input shape from first sample of agent 0
        sample_x, *_ = self.train_datasets[0][0]
        self.input_shape: tuple[int, ...] = tuple(sample_x.shape)

        self.models: list[int] = list(range(self.n_agents))

        # input_dims: per-agent last feature dimension.
        # When agent_modalities is set, each agent's dimension equals the
        # number of channels in its sensor modality (3 for acc/gyro/mag,
        # 2 for ECG) so that modality-specific encoders receive the correct
        # input size.
        if self.agent_modalities:
            self.input_dims: dict[str, int] = {
                str(i): len(
                    MHEALTH_SENSOR_MODALITIES[self.agent_modalities[i]]
                )
                for i in self.models
                if i in self.agent_modalities
            }
        else:
            n_feat = self.input_shape[-1]
            self.input_dims = {str(i): n_feat for i in self.models}

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
        split_map: dict[str, tuple[dict[int, MHealthDataset], bool]] = {
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
