"""Tests for src.datamodules.deepsense_datamodule."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from src.datamodules.deepsense_datamodule import (
    DeepSenseDataModule,
    DeepSenseDataset,
    DeepSenseModalityDataset,
)


@pytest.fixture
def mock_deepsense_dir(tmp_path):
    csv_content = """unit1_rgb,unit1_pwr_60ghz,unit1_lidar,unit1_label
data/img1.png,data/pwr1.txt,data/lidar1.mat,data/label1.txt
data/img2.png,data/pwr2.txt,data/lidar2.mat,data/label2.txt
data/img3.png,data/pwr3.txt,data/lidar3.mat,data/label3.txt
data/img4.png,data/pwr4.txt,data/lidar4.mat,data/label4.txt
data/img5.png,data/pwr5.txt,data/lidar5.mat,data/label5.txt
"""
    csv_path = tmp_path / 'scenario30.csv'
    csv_path.write_text(csv_content)
    return tmp_path


class TestDeepSenseModalityDataset:
    """Tests for DeepSenseModalityDataset."""

    def test_single_modality(self, mock_deepsense_dir):
        base_ds = DeepSenseDataset(str(mock_deepsense_dir))
        mod_ds = DeepSenseModalityDataset(base_ds, modality_idx=0)

        assert len(mod_ds) == 5

        x, label = mod_ds[0]
        assert x.shape == (3, 64, 64)
        assert label.dtype == torch.long

    def test_modality_shapes(self, mock_deepsense_dir):
        base_ds = DeepSenseDataset(str(mock_deepsense_dir))

        for mod_idx, expected_shape in [
            (0, (3, 64, 64)),
            (1, (1, 16, 64)),
            (2, (2, 32, 32)),
        ]:
            mod_ds = DeepSenseModalityDataset(base_ds, modality_idx=mod_idx)
            x, _ = mod_ds[0]
            assert x.shape == expected_shape

    def test_sample_ids_filtering(self, mock_deepsense_dir):
        base_ds = DeepSenseDataset(str(mock_deepsense_dir))
        mod_ds = DeepSenseModalityDataset(
            base_ds, modality_idx=1, sample_ids=[0, 2]
        )

        assert len(mod_ds) == 2


class TestDeepSenseDataModule:
    """Tests for DeepSenseDataModule."""

    def test_default_agent_modalities(self):
        dm = DeepSenseDataModule(
            data_path='./data/DeepSense',
            gdrive_folder_id='test_folder',
            split_strategy='full',
            n_agents=3,
        )

        assert dm.agent_modalities == {0: [0], 1: [1], 2: [2]}

    def test_custom_agent_modalities(self):
        dm = DeepSenseDataModule(
            data_path='./data/DeepSense',
            gdrive_folder_id='test_folder',
            split_strategy='full',
            agent_modalities={0: [0, 1], 1: [2], 2: [1]},
            n_agents=3,
        )

        assert dm.agent_modalities == {0: [0, 1], 1: [2], 2: [1]}

    def test_gdrive_folder_id_required_for_download(self, tmp_path):
        dm = DeepSenseDataModule(
            data_path=str(tmp_path),
            gdrive_folder_id=None,
        )

        with pytest.raises(RuntimeError, match='gdrive_folder_id'):
            dm.prepare_data()

    def test_prepare_data_skips_existing(
        self, mock_deepsense_dir, monkeypatch
    ):
        dm = DeepSenseDataModule(
            data_path=str(mock_deepsense_dir),
            gdrive_folder_id='test_folder',
        )
        dm.prepare_data()

    def test_split_full_strategy(self, mock_deepsense_dir):
        dm = DeepSenseDataModule(
            data_path=str(mock_deepsense_dir),
            gdrive_folder_id=None,
            split_strategy='full',
            n_agents=2,
            batch_size=2,
            val_split=0.2,
            test_split=0.2,
            seed=42,
        )
        dm.setup()

        assert dm.n_agents == 2
        assert dm.input_shape is not None
        assert 'label' in dm.num_classes

    def test_split_uniform_strategy(self, mock_deepsense_dir):
        dm = DeepSenseDataModule(
            data_path=str(mock_deepsense_dir),
            gdrive_folder_id=None,
            split_strategy='uniform',
            n_agents=2,
            batch_size=2,
            val_split=0.2,
            test_split=0.2,
            seed=42,
        )
        dm.setup()

        assert dm.n_agents == 2

    def test_unknown_split_strategy_raises(self, mock_deepsense_dir):
        dm = DeepSenseDataModule(
            data_path=str(mock_deepsense_dir),
            gdrive_folder_id=None,
            split_strategy='unknown',
            n_agents=2,
            batch_size=2,
        )

        with pytest.raises(ValueError, match='Unknown split_strategy'):
            dm.setup()

    def test_input_dims(self, mock_deepsense_dir):
        dm = DeepSenseDataModule(
            data_path=str(mock_deepsense_dir),
            gdrive_folder_id=None,
            split_strategy='full',
            n_agents=3,
            batch_size=2,
        )
        dm.setup()

        assert dm.input_dims is not None
        for i in range(3):
            assert str(i) in dm.input_dims

    def test_pilot_datasets_created(self, mock_deepsense_dir):
        dm = DeepSenseDataModule(
            data_path=str(mock_deepsense_dir),
            gdrive_folder_id=None,
            split_strategy='full',
            n_agents=2,
            batch_size=2,
            pilot_split=0.1,
            seed=42,
        )
        dm.setup()

        assert dm.pilot_datasets is not None
        assert len(dm.pilot_datasets) == 2
