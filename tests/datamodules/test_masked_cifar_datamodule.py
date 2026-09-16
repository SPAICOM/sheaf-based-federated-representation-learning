"""Tests for masked CIFAR-10 inpainting datamodule."""

from unittest.mock import patch

import pytest
import torch
from datasets import Dataset, DatasetDict
from PIL import Image

from src.datamodules.mask_generators import (
    ConstantVisibleSharedMaskGenerator,
    SpatialRandomMaskGenerator,
)
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
    return DatasetDict(
        {'train': Dataset.from_dict({'img': images, 'label': labels})}
    )


class TestMaskedCIFARDataModule:
    def test_fixed_overlap_pilots_keep_shared_targets_with_different_views(
        self,
    ):
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

        masks = [
            generator.get_mask(agent_idx, sample_id=0)
            for agent_idx in range(4)
        ]

        assert masks[0][:, :16, 16:].mean() > 0.9
        assert masks[1][:, 16:, :16].mean() > 0.9
        assert masks[2][:, :16, :16].mean() > 0.9
        assert masks[3][:, 16:, 16:].mean() > 0.9
        with pytest.raises(ValueError, match='got 4 for 5 agents'):
            SpatialRandomMaskGenerator(n_agents=5)

    def test_agent_random_spatial_off_focus_probability_adds_private_context(
        self,
    ):
        generator = SpatialRandomMaskGenerator(
            n_agents=2,
            focus_regions=['left', 'right'],
            private_visible_probability=1.0,
            off_focus_visible_probability=1.0,
            shared_visible_probability=0.0,
            block_size=4,
            seed=19,
        )

        mask0 = generator.get_mask(agent_idx=0, sample_id=0)
        mask1 = generator.get_mask(agent_idx=1, sample_id=0)

        assert torch.all(mask0 == 1.0)
        assert torch.all(mask1 == 1.0)

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

    def test_agent_random_spatial_datamodule_wires_off_focus_probability(self):
        with patch(
            'src.datamodules.masked_cifar_datamodule.load_dataset',
            return_value=_mock_cifar(),
        ):
            dm = MaskedCIFARDataModule(
                n_agents=2,
                mask_mode='agent_random_spatial',
                random_focus_regions=['left', 'right'],
                random_private_visible_probability=0.0,
                random_off_focus_visible_probability=1.0,
                random_shared_visible_probability=0.0,
                random_block_size=4,
                val_split=0.1,
                test_split=0.1,
                batch_size=4,
                num_workers=0,
                seed=23,
            )
            dm.setup()

        mask0 = dm.train_datasets[0]._mask(0)
        mask1 = dm.train_datasets[1]._mask(0)

        assert mask0[:, :, :16].mean() < 0.1
        assert mask0[:, :, 16:].mean() > 0.9
        assert mask1[:, :, 16:].mean() < 0.1
        assert mask1[:, :, :16].mean() > 0.9

    def test_constant_visible_shared_keeps_private_disjoint_from_shared(self):
        generator = ConstantVisibleSharedMaskGenerator(
            n_agents=2,
            focus_regions=['left', 'right'],
            visible_fraction=0.5,
            shared_visible_probability=0.25,
            block_size=4,
            seed=29,
        )

        shared0, private0 = generator.get_components(agent_idx=0, sample_id=3)
        shared1, private1 = generator.get_components(agent_idx=1, sample_id=3)
        mask0 = generator.get_mask(agent_idx=0, sample_id=3)
        mask1 = generator.get_mask(agent_idx=1, sample_id=3)

        assert torch.equal(shared0, shared1)
        assert torch.logical_and(shared0.bool(), private0.bool()).sum() == 0
        assert torch.logical_and(shared1.bool(), private1.bool()).sum() == 0
        assert torch.logical_and(private0.bool(), private1.bool()).sum() == 0
        assert torch.equal(
            mask0.bool(), torch.logical_or(shared0.bool(), private0.bool())
        )
        assert torch.equal(
            mask1.bool(), torch.logical_or(shared1.bool(), private1.bool())
        )
        assert torch.isclose(mask0.mean(), torch.tensor(0.5))
        assert torch.isclose(mask1.mean(), torch.tensor(0.5))
        # 25% of the 50% visible budget is shared: 8/64 blocks.
        assert torch.isclose(shared0.mean(), torch.tensor(0.125))

    def test_constant_visible_shared_allows_shared_above_visible_fraction(self):
        generator = ConstantVisibleSharedMaskGenerator(
            n_agents=2,
            visible_fraction=0.5,
            shared_visible_probability=0.75,
            block_size=4,
        )

        shared, private = generator.get_components(agent_idx=0, sample_id=0)
        assert torch.isclose(shared.mean(), torch.tensor(0.375))
        assert torch.isclose(private.mean(), torch.tensor(0.125))

    def test_constant_visible_shared_rejects_impossible_private_disjointness(
        self,
    ):
        with pytest.raises(ValueError, match='agent-private blocks disjoint'):
            ConstantVisibleSharedMaskGenerator(
                n_agents=2,
                visible_fraction=0.75,
                shared_visible_probability=0.2,
                block_size=4,
            )

    def test_constant_visible_shared_datamodule_wires_mode(self):
        with patch(
            'src.datamodules.masked_cifar_datamodule.load_dataset',
            return_value=_mock_cifar(),
        ):
            dm = MaskedCIFARDataModule(
                n_agents=2,
                mask_mode='constant_visible_shared',
                random_focus_regions=['left', 'right'],
                constant_visible_fraction=0.5,
                constant_shared_visible_probability=0.25,
                random_block_size=4,
                val_split=0.1,
                test_split=0.1,
                batch_size=4,
                num_workers=0,
                seed=31,
                return_mask=True,
            )
            dm.setup()

        x0, target0, mask0 = dm.train_datasets[0][0]
        x1, target1, mask1 = dm.train_datasets[1][0]

        assert mask0.shape == (1, 32, 32)
        assert mask1.shape == (1, 32, 32)
        assert torch.isclose(mask0.mean(), torch.tensor(0.5))
        assert torch.isclose(mask1.mean(), torch.tensor(0.5))
        assert torch.equal(x0, target0 * mask0)
        assert torch.equal(x1, target1 * mask1)

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

    def test_return_mask_adds_visible_mask_to_batches(self):
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
                return_mask=True,
            )
            dm.setup()

        batch, _batch_idx, _loader_idx = next(iter(dm.train_dataloader()))
        x, target, mask = batch[0]

        assert x.shape == (4, 3, 32, 32)
        assert target.shape == (4, 3, 32, 32)
        assert mask.shape == (4, 1, 32, 32)
        assert torch.equal(x, target * mask)

    def test_include_mask_in_input_concatenates_visible_mask_channel(self):
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
                return_mask=True,
                include_mask_in_input=True,
            )
            dm.setup()

        batch, _batch_idx, _loader_idx = next(iter(dm.train_dataloader()))
        x, target, mask = batch[0]

        assert x.shape == (4, 4, 32, 32)
        assert target.shape == (4, 3, 32, 32)
        assert mask.shape == (4, 1, 32, 32)
        assert torch.equal(x[:, :3], target * mask)
        assert torch.equal(x[:, 3:], mask)
        assert dm.input_dims == {'0': 4, '1': 4}
        assert dm.target_dims == {'0': 3, '1': 3}

    def test_region_overlap_mask_mode_returns_masks(self):
        with patch(
            'src.datamodules.masked_cifar_datamodule.load_dataset',
            return_value=_mock_cifar(),
        ):
            dm = MaskedCIFARDataModule(
                n_agents=2,
                mask_mode='region_overlap',
                region_visible_fraction=0.55,
                region_overlap_fraction=0.1,
                batch_size=4,
                num_workers=0,
                seed=19,
                return_mask=True,
            )
            dm.setup()

        x0, target0, mask0 = dm.train_datasets[0][0]
        x1, target1, mask1 = dm.train_datasets[1][0]

        assert mask0.shape == (1, 32, 32)
        assert mask1.shape == (1, 32, 32)
        assert torch.isclose(mask0.mean(), torch.tensor(0.55), atol=0.05)
        assert torch.isclose(mask1.mean(), torch.tensor(0.55), atol=0.05)
        assert torch.logical_or(mask0.bool(), mask1.bool()).all()
        assert torch.equal(x0, target0 * mask0)
        assert torch.equal(x1, target1 * mask1)

    def test_region_overlap_visible_fraction_controls_overlap(self):
        with patch(
            'src.datamodules.masked_cifar_datamodule.load_dataset',
            return_value=_mock_cifar(),
        ):
            dm = MaskedCIFARDataModule(
                n_agents=2,
                mask_mode='region_overlap',
                region_visible_fraction=0.6,
                region_overlap_fraction=0.2,
                batch_size=4,
                num_workers=0,
                seed=23,
                return_mask=True,
            )
            dm.setup()

        mask0 = dm.train_datasets[0]._mask(0).bool()
        mask1 = dm.train_datasets[1]._mask(0).bool()
        overlap = torch.logical_and(mask0, mask1).float().mean()

        assert torch.logical_or(mask0, mask1).all()
        assert overlap > 0.0

    def test_region_overlap_fraction_controls_shared_area(self):
        with patch(
            'src.datamodules.masked_cifar_datamodule.load_dataset',
            return_value=_mock_cifar(),
        ):
            dm = MaskedCIFARDataModule(
                n_agents=2,
                mask_mode='region_overlap',
                region_visible_fraction=0.625,
                region_overlap_fraction=0.25,
                batch_size=4,
                num_workers=0,
                seed=29,
                return_mask=True,
            )
            dm.setup()

        mask0 = dm.train_datasets[0]._mask(0).bool()
        mask1 = dm.train_datasets[1]._mask(0).bool()
        overlap = torch.logical_and(mask0, mask1).float().mean()

        assert torch.logical_or(mask0, mask1).all()
        assert torch.isclose(overlap, torch.tensor(0.25), atol=0.02)

    def test_region_overlap_boundary_jitter_moves_masks_per_sample(self):
        with patch(
            'src.datamodules.masked_cifar_datamodule.load_dataset',
            return_value=_mock_cifar(),
        ):
            dm = MaskedCIFARDataModule(
                n_agents=2,
                mask_mode='region_overlap',
                region_visible_fraction=0.55,
                region_overlap_fraction=0.1,
                region_boundary_jitter_px=2,
                batch_size=4,
                num_workers=0,
                seed=31,
                return_mask=True,
            )
            dm.setup()

        mask0_sample0 = dm.train_datasets[0]._mask(0).bool()
        mask0_sample1 = dm.train_datasets[0]._mask(1).bool()
        mask1_sample0 = dm.train_datasets[1]._mask(0).bool()
        expected_mask0 = dm.train_datasets[0].mask_generator.__class__(
            n_agents=2,
            visible_fraction=0.55,
            overlap_fraction=0.1,
            image_size=32,
            boundary_jitter_px=0,
            seed=31,
        ).get_mask(0).bool()
        expected_mask1 = dm.train_datasets[1].mask_generator.__class__(
            n_agents=2,
            visible_fraction=0.55,
            overlap_fraction=0.1,
            image_size=32,
            boundary_jitter_px=0,
            seed=31,
        ).get_mask(1).bool()

        overlap = torch.logical_and(mask0_sample0, mask1_sample0).float().mean()
        expected_overlap = (
            torch.logical_and(expected_mask0, expected_mask1).float().mean()
        )

        assert not torch.equal(mask0_sample0, mask0_sample1)
        assert torch.logical_or(mask0_sample0, mask1_sample0).all()
        assert torch.isclose(overlap, expected_overlap)
