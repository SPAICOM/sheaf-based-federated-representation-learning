"""Tests for src.datamodules.mhealth_datamodule."""

import shutil
import tempfile
from pathlib import Path

import pytest
import torch

from src.datamodules.mhealth_datamodule import MHealthDataModule


@pytest.fixture(scope='module')
def data_dir():
    """Create a temporary directory and download MHEALTH dataset."""
    tmp_dir = Path(tempfile.mkdtemp())
    dm = MHealthDataModule(
        data_path=tmp_dir,
        batch_size=256,
        num_workers=0,
        val_split=0.1,
        test_split=0.1,
        seed=42,
    )
    dm.prepare_data()
    yield dm.data_path
    shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.fixture(scope='module')
def dm_base(data_dir):
    """Base datamodule with all features, no windowing."""
    dm = MHealthDataModule(
        data_path=data_dir,
        batch_size=256,
        num_workers=0,
        val_split=0.1,
        test_split=0.1,
        seed=42,
    )
    dm.setup()
    return dm


def get_batch_data(dataloader):
    """Extract batch dict from CombinedLoader iterator."""
    batch, *_ = next(iter(dataloader))
    return batch


class TestMHealthDataModule:
    """Tests for MHealthDataModule."""

    def test_subject_split_single_agent(self, dm_base):
        """Test subject split with single agent (one subject)."""
        assert dm_base.n_agents == 1

        batch = get_batch_data(dm_base.train_dataloader())
        x, y = batch[0]
        assert x.dim() == 2
        assert x.shape[0] == 256
        assert y.shape[0] == 256
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)

    def test_input_shape_is_tuple(self, dm_base):
        """Test that input_shape is a tuple with feature dimension."""
        assert isinstance(dm_base.input_shape, tuple)
        assert len(dm_base.input_shape) == 1

    def test_subject_split_windowed(self, data_dir):
        """Test subject split with sliding window."""
        dm = MHealthDataModule(
            data_path=data_dir,
            batch_size=256,
            num_workers=0,
            val_split=0.1,
            test_split=0.1,
            seed=42,
            window_size=100,
            stride=50,
        )
        dm.setup()

        batch = get_batch_data(dm.train_dataloader())
        x, y = batch[0]
        assert x.dim() == 3
        assert x.shape[1] == 100

    def test_uniform_split_multiple_agents(self, data_dir):
        """Test uniform split across multiple agents."""
        dm = MHealthDataModule(
            data_path=data_dir,
            batch_size=256,
            num_workers=0,
            val_split=0.1,
            test_split=0.1,
            seed=42,
            split_strategy='uniform',
            n_agents=3,
            feature_groups=['chest', 'left_ankle'],
        )
        dm.setup()

        assert dm.n_agents == 3

        batch = get_batch_data(dm.train_dataloader())
        assert len(batch) == 3
        for i in range(3):
            x, y = batch[i]
            assert x.shape[0] == 256

    def test_non_iid_split(self, data_dir):
        """Test non-IID Dirichlet split."""
        dm = MHealthDataModule(
            data_path=data_dir,
            batch_size=256,
            num_workers=0,
            val_split=0.1,
            test_split=0.1,
            seed=42,
            split_strategy='non_iid',
            n_agents=4,
            classes_per_agent=3,
            alpha=0.3,
        )
        dm.setup()

        assert dm.n_agents == 4

        batch = get_batch_data(dm.train_dataloader())
        assert len(batch) == 4

    def test_pilot_dataset_creation(self, data_dir):
        """Test pilot dataset creation."""
        dm = MHealthDataModule(
            data_path=data_dir,
            batch_size=256,
            num_workers=0,
            val_split=0.1,
            test_split=0.1,
            seed=42,
            split_strategy='uniform',
            n_agents=2,
            pilot_num_samples=200,
        )
        dm.setup()

        assert dm.pilot_datasets is not None
        assert len(dm.pilot_datasets) == 2

        batch = get_batch_data(dm.train_dataloader())
        assert 'pilot_0' in batch
        assert 'pilot_1' in batch
        x_pilot, y_pilot, sample_ids = batch['pilot_0']
        assert x_pilot.shape[0] == y_pilot.shape[0]

    def test_feature_groups_selection(self, data_dir):
        """Test named feature groups."""
        dm = MHealthDataModule(
            data_path=data_dir,
            batch_size=256,
            num_workers=0,
            val_split=0.1,
            test_split=0.1,
            seed=42,
            feature_groups=['chest'],
        )
        dm.setup()

        assert len(dm.feature_cols) > 0
        assert 'acc_chest_x' in dm.feature_cols

        batch = get_batch_data(dm.train_dataloader())
        x, y = batch[0]
        assert x.shape[1] <= len(dm.feature_cols)

    def test_val_and_test_loaders(self, dm_base):
        """Test validation and test dataloaders."""
        val_batch = get_batch_data(dm_base.val_dataloader())
        test_batch = get_batch_data(dm_base.test_dataloader())

        val_x, val_y = val_batch[0]
        test_x, test_y = test_batch[0]

        assert val_x.shape[0] <= 256
        assert test_x.shape[0] <= 256

    def test_num_classes(self, dm_base):
        """Test num_classes attribute."""
        assert 'label' in dm_base.num_classes
        assert dm_base.num_classes['label'] > 0

    def test_input_dims(self, dm_base):
        """Test input_dims attribute."""
        assert len(dm_base.input_dims) > 0
        assert '0' in dm_base.input_dims
