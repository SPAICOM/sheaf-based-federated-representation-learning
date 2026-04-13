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


SCENARIO = '26'


@pytest.fixture
def mock_deepsense_dir(tmp_path):
    csv_content = """unit1_rgb,unit1_pwr_60ghz,unit1_lidar,unit1_blockage
data/img1.png,data/pwr1.txt,data/lidar1.mat,data/label1.txt
data/img2.png,data/pwr2.txt,data/lidar2.mat,data/label2.txt
data/img3.png,data/pwr3.txt,data/lidar3.mat,data/label3.txt
data/img4.png,data/pwr4.txt,data/lidar4.mat,data/label4.txt
data/img5.png,data/pwr5.txt,data/lidar5.mat,data/label5.txt
"""
    csv_path = tmp_path / f'scenario{SCENARIO}.csv'
    csv_path.write_text(csv_content)
    return tmp_path


class TestDeepSenseModalityDataset:
    """Tests for DeepSenseModalityDataset."""

    def test_single_modality(self, mock_deepsense_dir):
        base_ds = DeepSenseDataset(str(mock_deepsense_dir), scenario=SCENARIO)
        mod_ds = DeepSenseModalityDataset(base_ds, modality_idx=0)

        assert len(mod_ds) == 5

        x, label = mod_ds[0]
        assert x.shape == (3, 64, 64)
        assert label.dtype == torch.long

    def test_modality_shapes(self, mock_deepsense_dir):
        base_ds = DeepSenseDataset(str(mock_deepsense_dir), scenario=SCENARIO)

        for mod_idx, expected_shape in [
            (0, (3, 64, 64)),
            (1, (1, 16, 64)),
            (2, (2, 32, 32)),
        ]:
            mod_ds = DeepSenseModalityDataset(base_ds, modality_idx=mod_idx)
            x, _ = mod_ds[0]
            assert x.shape == expected_shape

    def test_sample_ids_filtering(self, mock_deepsense_dir):
        base_ds = DeepSenseDataset(str(mock_deepsense_dir), scenario=SCENARIO)
        mod_ds = DeepSenseModalityDataset(
            base_ds, modality_idx=1, sample_ids=[0, 2]
        )

        assert len(mod_ds) == 2


class TestDeepSenseDataModule:
    """Tests for DeepSenseDataModule."""

    def test_default_agent_modalities(self):
        dm = DeepSenseDataModule(
            data_path='./data/DeepSense',
            gdrive_file_id='test_folder',
            split_strategy='full',
            n_agents=3,
        )

        assert dm.agent_modalities == {0: [0], 1: [1], 2: [2]}

    def test_custom_agent_modalities(self):
        dm = DeepSenseDataModule(
            data_path='./data/DeepSense',
            gdrive_file_id='test_folder',
            split_strategy='full',
            agent_modalities={0: [0, 1], 1: [2], 2: [1]},
            n_agents=3,
        )

        assert dm.agent_modalities == {0: [0, 1], 1: [2], 2: [1]}

    def test_gdrive_file_id_required_for_download(self, tmp_path):
        dm = DeepSenseDataModule(
            data_path=str(tmp_path),
            gdrive_file_id=None,
        )

        with pytest.raises(RuntimeError, match='gdrive_file_id'):
            dm.prepare_data()

    def test_prepare_data_skips_existing(self, mock_deepsense_dir):
        dm = DeepSenseDataModule(
            data_path=str(mock_deepsense_dir),
            scenario=SCENARIO,
            gdrive_file_id='some_file_id',
        )
        dm.prepare_data()

    def test_prepare_data_downloads_and_extracts(self, tmp_path):
        """gdown.download is called with the file ID and the zip is extracted."""
        import zipfile as zf_mod

        file_id = 'abc123'
        dm = DeepSenseDataModule(
            data_path=str(tmp_path),
            scenario=SCENARIO,
            gdrive_file_id=file_id,
        )
        zip_path = tmp_path / f'scenario{SCENARIO}.zip'

        def fake_download(id, output, **kwargs):
            with zf_mod.ZipFile(output, 'w') as zf:
                zf.writestr(f'scenario{SCENARIO}.csv', 'unit1_rgb,unit1_blockage\n')

        with patch(
            'src.datamodules.deepsense_datamodule.gdown.download',
            side_effect=fake_download,
        ) as mock_dl:
            dm.prepare_data()

        mock_dl.assert_called_once_with(
            id=file_id, output=str(zip_path), quiet=False
        )
        assert (tmp_path / f'scenario{SCENARIO}.csv').exists()

    def test_prepare_data_skips_download_if_zip_exists(self, tmp_path):
        """gdown is not called when the zip is already on disk."""
        import zipfile as zf_mod

        zip_path = tmp_path / f'scenario{SCENARIO}.zip'
        with zf_mod.ZipFile(zip_path, 'w') as zf:
            zf.writestr(f'scenario{SCENARIO}.csv', 'unit1_rgb,unit1_blockage\n')

        dm = DeepSenseDataModule(
            data_path=str(tmp_path),
            scenario=SCENARIO,
            gdrive_file_id='some_file_id',
        )

        with patch('src.datamodules.deepsense_datamodule.gdown.download') as mock_dl:
            dm.prepare_data()

        mock_dl.assert_not_called()
        assert (tmp_path / f'scenario{SCENARIO}.csv').exists()

    def test_prepare_data_raises_if_download_produces_no_zip(self, tmp_path):
        """RuntimeError raised when gdown completes but zip is still absent."""
        dm = DeepSenseDataModule(
            data_path=str(tmp_path),
            scenario=SCENARIO,
            gdrive_file_id='bad_id',
        )

        with patch('src.datamodules.deepsense_datamodule.gdown.download'):
            with pytest.raises(RuntimeError, match='Download failed'):
                dm.prepare_data()

    def test_prepare_data_creates_directory_if_missing(self, tmp_path):
        """data_path directory is created automatically when absent."""
        new_dir = tmp_path / 'nested' / 'deepsense'
        assert not new_dir.exists()

        dm = DeepSenseDataModule(
            data_path=str(new_dir),
            scenario=SCENARIO,
            gdrive_file_id='some_file_id',
        )

        with patch('src.datamodules.deepsense_datamodule.gdown.download'):
            with pytest.raises(RuntimeError):
                dm.prepare_data()

        assert new_dir.exists()

    def test_split_full_strategy(self, mock_deepsense_dir):
        dm = DeepSenseDataModule(
            data_path=str(mock_deepsense_dir),
            scenario=SCENARIO,
            gdrive_file_id=None,
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
            scenario=SCENARIO,
            gdrive_file_id=None,
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
            scenario=SCENARIO,
            gdrive_file_id=None,
            split_strategy='unknown',
            n_agents=2,
            batch_size=2,
        )

        with pytest.raises(ValueError, match='Unknown split_strategy'):
            dm.setup()

    def test_input_dims(self, mock_deepsense_dir):
        dm = DeepSenseDataModule(
            data_path=str(mock_deepsense_dir),
            scenario=SCENARIO,
            gdrive_file_id=None,
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
            scenario=SCENARIO,
            gdrive_file_id=None,
            split_strategy='full',
            n_agents=2,
            batch_size=2,
            pilot_split=0.1,
            seed=42,
        )
        dm.setup()

        assert dm.pilot_datasets is not None
        assert len(dm.pilot_datasets) == 2
