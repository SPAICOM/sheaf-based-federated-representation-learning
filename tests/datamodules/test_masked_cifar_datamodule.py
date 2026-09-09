"""Tests for masked CIFAR-10 inpainting datamodule."""

from unittest.mock import patch

import pytest
import torch
from datasets import Dataset, DatasetDict
from PIL import Image

from src.datamodules.mask_generators import SpatialRandomMaskGenerator
from src.datamodules.masked_cifar_datamodule import MaskedCIFARDataModule


def _mock_cifar(num_samples: int = 80) -> DatasetDict:
    images = []
    labels = []
    for idx in range(num_samples):
        image = Image.new(
            'RGB',
            (32, 32),
            color=(idx % 255, (idx * 2) % 255, (idx * 3) % 255),
        )
        images.append(image)
        labels.append(idx % 10)
    return DatasetDict({'train': Dataset.from_dict({'img': images, 'label': labels})})


class TestMaskedCIFARDataModule:
    def test_fixed_overlap_pilots_keep_shared_targets_with_different_views(self):
        with patch(
            'src.datamodules.masked_cifar_datamodule.load_dataset',
            return_value=_mock_cifar(),
        ):
            dm = MaskedCIFARDataModule(
                n_agents=2,
                mask_mode='fixed_overlap',
                fixed_overlap_px=2.0,
                val_split=0.1,
                test_split=0.1,
                pilot_split=0.1,
                batch_size=4,
                num_workers=0,
                seed=7,
            )
            dm.setup()

        x0, target0, sid0 = dm.pilot_datasets[0][0]
        x1, target1, sid1 = dm.pilot_datasets[1][0]

        assert x0.shape == (3, 32, 32)
        assert sid0.item() == sid1.item()
        assert torch.equal(target0, target1)
        assert not torch.equal(x0, x1)
        assert dm.input_dims == {'0': 3, '1': 3}
        assert dm.num_classes['label'] == 10

    def test_private_splits_use_different_sample_ids_by_default(self):
        with patch(
            'src.datamodules.masked_cifar_datamodule.load_dataset',
            return_value=_mock_cifar(),
        ):
            dm = MaskedCIFARDataModule(
                n_agents=2,
                mask_mode='fixed_overlap',
                val_split=0.1,
                test_split=0.1,
                batch_size=4,
                num_workers=0,
                seed=11,
            )
            dm.setup()

        _x0, target0 = dm.train_datasets[0][0]
        _x1, target1 = dm.train_datasets[1][0]

        assert not torch.equal(target0, target1)

    def test_agent_random_spatial_masks_use_different_focus_regions(self):
        with patch(
            'src.datamodules.masked_cifar_datamodule.load_dataset',
            return_value=_mock_cifar(),
        ):
            dm = MaskedCIFARDataModule(
                n_agents=2,
                mask_mode='agent_random_spatial',
                random_focus_regions=['top_right', 'bottom_left'],
                random_private_visible_probability=1.0,
                random_shared_visible_probability=0.0,
                random_block_size=4,
                val_split=0.1,
                test_split=0.1,
                batch_size=4,
                num_workers=0,
                seed=11,
            )
            dm.setup()

        mask0 = dm.train_datasets[0]._mask(0)
        mask1 = dm.train_datasets[1]._mask(0)

        assert mask0[:, :16, 16:].mean() > 0.9
        assert mask0[:, 16:, :16].mean() < 0.1
        assert mask1[:, 16:, :16].mean() > 0.9
        assert mask1[:, :16, 16:].mean() < 0.1

    def test_agent_random_spatial_defaults_to_four_focus_regions(self):
        generator = SpatialRandomMaskGenerator(
            n_agents=4,
            private_visible_probability=1.0,
            shared_visible_probability=0.0,
            block_size=4,
        )

        masks = [generator.get_mask(agent_idx, sample_id=0) for agent_idx in range(4)]

        assert masks[0][:, :16, 16:].mean() > 0.9
        assert masks[1][:, 16:, :16].mean() > 0.9
        assert masks[2][:, :16, :16].mean() > 0.9
        assert masks[3][:, 16:, 16:].mean() > 0.9
        with pytest.raises(ValueError, match='got 4 for 5 agents'):
            SpatialRandomMaskGenerator(n_agents=5)

    def test_agent_random_spatial_shared_probability_controls_overlap(self):
        with patch(
            'src.datamodules.masked_cifar_datamodule.load_dataset',
            return_value=_mock_cifar(),
        ):
            dm = MaskedCIFARDataModule(
                n_agents=2,
                mask_mode='agent_random_spatial',
                random_focus_regions=['top_right', 'bottom_left'],
                random_private_visible_probability=0.0,
                random_shared_visible_probability=1.0,
                random_block_size=4,
                val_split=0.1,
                test_split=0.1,
                batch_size=4,
                num_workers=0,
                seed=17,
            )
            dm.setup()

        mask0 = dm.train_datasets[0]._mask(0)
        mask1 = dm.train_datasets[1]._mask(0)

        assert torch.equal(mask0, mask1)
        assert torch.all(mask0 == 1.0)

    def test_train_loader_returns_masked_inputs_and_full_targets(self):
        with patch(
            'src.datamodules.masked_cifar_datamodule.load_dataset',
            return_value=_mock_cifar(),
        ):
            dm = MaskedCIFARDataModule(
                n_agents=2,
                mask_mode='fixed_overlap',
                batch_size=4,
                num_workers=0,
                seed=13,
            )
            dm.setup()

        batch, _batch_idx, _loader_idx = next(iter(dm.train_dataloader()))
        x, target = batch[0]

        assert x.shape == (4, 3, 32, 32)
        assert target.shape == (4, 3, 32, 32)
