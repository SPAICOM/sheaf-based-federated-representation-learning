"""Masked CIFAR-10 datamodule for federated inpainting."""

from collections.abc import Sequence

import lightning as l
import torch
from datasets import concatenate_datasets, load_dataset
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.datamodules.mask_generators import (
    ConstantVisibleSharedMaskGenerator,
    OverlapMaskGenerator,
    RegionOverlapMaskGenerator,
    SpatialRandomMaskGenerator,
)
from src.datamodules.utils import (
    PairwiseDataset,
    compute_split_indices,
    repeat_dataset_to_num_samples,
)


def _collate_masked_batch(batch):
    xs, targets = [], []
    masks = [] if len(batch[0]) >= 3 and batch[0][2].ndim >= 2 else None
    ids = [] if len(batch[0]) in {3, 4} and batch[0][-1].ndim == 0 else None
    for item in batch:
        xs.append(item[0])
        targets.append(item[1])
        if masks is not None:
            masks.append(item[2])
        if ids is not None:
            ids.append(item[-1])
    out = (torch.stack(xs), torch.stack(targets))
    if masks is not None:
        out = out + (torch.stack(masks),)
    if ids is not None:
        out = out + (torch.stack(ids),)
    return out


class MaskedCIFARDataset(Dataset):
    """Return masked/full images, optionally with mask and sample id."""

    def __init__(
        self,
        hf_dataset,
        data_key: str,
        agent_idx: int,
        mask_mode: str,
        image_size: int = 32,
        fixed_overlap_px: float = 2.0,
        min_fixed_overlap_px: float = 1.0,
        region_visible_fraction: float = 0.5,
        region_overlap_fraction: float = 0.3,
        region_boundary_jitter_px: int = 0,
        random_focus_regions: Sequence[str] | None = None,
        random_private_visible_probability: float = 0.7,
        random_off_focus_visible_probability: float = 0.0,
        random_shared_visible_probability: float = 0.1,
        constant_visible_fraction: float = 0.7,
        constant_shared_visible_probability: float = 0.1,
        random_block_size: int = 4,
        n_agents: int = 1,
        seed: int = 42,
        sample_ids: list[int] | None = None,
        return_mask: bool = False,
        include_mask_in_input: bool = False,
    ) -> None:
        self.dataset = hf_dataset
        self.data_key = data_key
        self.agent_idx = int(agent_idx)
        self.mask_mode = mask_mode
        self.image_size = int(image_size)
        self.seed = int(seed)
        self.sample_ids = sample_ids
        self.return_mask = bool(return_mask)
        self.include_mask_in_input = bool(include_mask_in_input)
        self.mask_epoch = torch.zeros((), dtype=torch.long).share_memory_()
        self.to_tensor = transforms.ToTensor()

        valid_modes = {
            'fixed_overlap',
            'region_overlap',
            'agent_random_spatial',
            'constant_visible_shared',
        }
        if self.mask_mode not in valid_modes:
            raise ValueError(
                f"mask_mode must be one of {valid_modes}, got '{mask_mode}'"
            )
        if self.mask_mode == 'fixed_overlap':
            self.mask_generator = OverlapMaskGenerator(
                n_agents=n_agents,
                overlap_px=fixed_overlap_px,
                image_size=self.image_size,
                min_overlap_px=min_fixed_overlap_px,
            )
        elif self.mask_mode == 'region_overlap':
            self.mask_generator = RegionOverlapMaskGenerator(
                n_agents=n_agents,
                visible_fraction=region_visible_fraction,
                overlap_fraction=region_overlap_fraction,
                image_size=self.image_size,
                boundary_jitter_px=region_boundary_jitter_px,
                seed=self.seed,
            )
        elif self.mask_mode == 'agent_random_spatial':
            self.mask_generator = SpatialRandomMaskGenerator(
                n_agents=n_agents,
                focus_regions=random_focus_regions,
                private_visible_probability=random_private_visible_probability,
                off_focus_visible_probability=random_off_focus_visible_probability,
                shared_visible_probability=random_shared_visible_probability,
                block_size=random_block_size,
                image_size=self.image_size,
                seed=self.seed,
            )
        else:
            self.mask_generator = ConstantVisibleSharedMaskGenerator(
                n_agents=n_agents,
                focus_regions=random_focus_regions,
                visible_fraction=constant_visible_fraction,
                shared_visible_probability=constant_shared_visible_probability,
                block_size=random_block_size,
                image_size=self.image_size,
                seed=self.seed,
            )

    def __len__(self) -> int:
        return len(self.dataset)

    def _full_image(self, idx: int) -> torch.Tensor:
        image = self.dataset[idx][self.data_key]
        if isinstance(image, Image.Image):
            return self.to_tensor(image)
        if isinstance(image, list):
            return torch.tensor(image, dtype=torch.float32)
        return image.float()

    def _mask(self, idx: int) -> torch.Tensor:
        sample_id = (
            self.sample_ids[idx] if self.sample_ids is not None else idx
        )
        sample_id = int(sample_id) + 10_000_019 * int(self.mask_epoch.item())
        if self.mask_mode == 'fixed_overlap':
            return self.mask_generator.get_mask(self.agent_idx)
        if self.mask_mode == 'region_overlap':
            return self.mask_generator.get_mask(self.agent_idx, int(sample_id))
        return self.mask_generator.get_mask(self.agent_idx, int(sample_id))

    def __getitem__(self, idx: int):
        full = self._full_image(idx)
        mask = self._mask(idx).to(dtype=full.dtype)
        masked = full * mask
        x = torch.cat([masked, mask], dim=0) if self.include_mask_in_input else masked
        out = (x, full, mask) if self.return_mask else (x, full)
        if self.sample_ids is not None:
            out = out + (torch.tensor(self.sample_ids[idx], dtype=torch.long),)
        return out


class MaskedCIFARDataModule(l.LightningDataModule):
    """CIFAR-10 multi-agent inpainting datamodule.

    Private train/val/test samples are split across agents by default; global
    pilots are shared by sample id and each agent observes them through its own
    mask distribution. The supervised target is always the full image.
    """

    def __init__(
        self,
        repo: str = 'uoft-cs',
        name: str = 'cifar10',
        data_key: str = 'img',
        n_agents: int = 2,
        mask_mode: str = 'fixed_overlap',
        fixed_overlap_px: float = 2.0,
        min_fixed_overlap_px: float = 1.0,
        region_visible_fraction: float = 0.5,
        region_overlap_fraction: float = 0.3,
        region_boundary_jitter_px: int = 0,
        random_focus_regions: Sequence[str] | None = None,
        random_private_visible_probability: float = 0.7,
        random_off_focus_visible_probability: float = 0.0,
        random_shared_visible_probability: float = 0.1,
        random_focus_visible_probability: float | None = None,
        random_overlap_visible_probability: float | None = None,
        constant_visible_fraction: float = 0.7,
        constant_shared_visible_probability: float = 0.1,
        random_block_size: int = 4,
        share_private_data: bool = False,
        private_train_fraction: float = 1.0,
        batch_size: int = 64,
        num_workers: int = 4,
        mode: str = 'min_size',
        val_split: float = 0.1,
        test_split: float = 0.1,
        monitor_test_during_fit: bool = False,
        pilot_split: float = 0.0,
        pilot_num_samples: int | None = None,
        pilot_batch_size: int | None = None,
        comm_data: str = 'shared_global_pilots',
        seed: int = 42,
        return_mask: bool = False,
        include_mask_in_input: bool = False,
    ) -> None:
        super().__init__()
        self.repo = repo
        self.name = name
        self.data_key = data_key
        self.n_agents = int(n_agents)
        self.mask_mode = mask_mode
        self.fixed_overlap_px = float(fixed_overlap_px)
        self.min_fixed_overlap_px = float(min_fixed_overlap_px)
        self.region_visible_fraction = float(region_visible_fraction)
        self.region_overlap_fraction = float(region_overlap_fraction)
        self.region_boundary_jitter_px = int(region_boundary_jitter_px)
        self.random_focus_regions = list(random_focus_regions or [])
        if random_focus_visible_probability is not None:
            random_private_visible_probability = (
                random_focus_visible_probability
            )
        if random_overlap_visible_probability is not None:
            random_shared_visible_probability = (
                random_overlap_visible_probability
            )
        self.random_private_visible_probability = float(
            random_private_visible_probability
        )
        self.random_off_focus_visible_probability = float(
            random_off_focus_visible_probability
        )
        self.random_shared_visible_probability = float(
            random_shared_visible_probability
        )
        self.constant_visible_fraction = float(constant_visible_fraction)
        self.constant_shared_visible_probability = float(
            constant_shared_visible_probability
        )
        self.random_block_size = int(random_block_size)
        self.share_private_data = bool(share_private_data)
        self.private_train_fraction = float(private_train_fraction)
        if not 0 < self.private_train_fraction <= 1:
            raise ValueError("private_train_fraction must be in (0, 1].")
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.mode = mode
        self.val_split = float(val_split)
        self.test_split = float(test_split)
        self.monitor_test_during_fit = monitor_test_during_fit
        self.pilot_split = float(pilot_split)
        self.pilot_num_samples = pilot_num_samples
        self.pilot_batch_size = (
            self.batch_size if pilot_batch_size is None else pilot_batch_size
        )
        _valid_comm = {
            'private_pilots',
            'shared_global_pilots',
            'pairwise_pilots',
        }
        if comm_data not in _valid_comm:
            raise ValueError(
                f"Unknown comm_data '{comm_data}'. Valid options: {_valid_comm}"
            )
        self.comm_data = comm_data
        self.seed = int(seed)
        self.return_mask = bool(return_mask)
        self.include_mask_in_input = bool(include_mask_in_input)
        self.image_size = 32

    def prepare_data(self) -> None:
        load_dataset(f'{self.repo}/{self.name}')

    def _agent_dataset(
        self, data, agent_idx: int, sample_ids: list[int] | None
    ):
        return MaskedCIFARDataset(
            data,
            data_key=self.data_key,
            agent_idx=agent_idx,
            mask_mode=self.mask_mode,
            image_size=self.image_size,
            fixed_overlap_px=self.fixed_overlap_px,
            min_fixed_overlap_px=self.min_fixed_overlap_px,
            region_visible_fraction=self.region_visible_fraction,
            region_overlap_fraction=self.region_overlap_fraction,
            region_boundary_jitter_px=self.region_boundary_jitter_px,
            random_focus_regions=self.random_focus_regions,
            random_private_visible_probability=(
                self.random_private_visible_probability
            ),
            random_off_focus_visible_probability=(
                self.random_off_focus_visible_probability
            ),
            random_shared_visible_probability=(
                self.random_shared_visible_probability
            ),
            constant_visible_fraction=self.constant_visible_fraction,
            constant_shared_visible_probability=(
                self.constant_shared_visible_probability
            ),
            random_block_size=self.random_block_size,
            n_agents=self.n_agents,
            seed=self.seed,
            sample_ids=sample_ids,
            return_mask=self.return_mask,
            include_mask_in_input=self.include_mask_in_input,
        )

    def _build_agent_datasets(
        self,
        data,
        sample_ids: list[int] | None,
    ) -> dict[int, MaskedCIFARDataset]:
        return {
            i: self._agent_dataset(data, i, sample_ids)
            for i in range(self.n_agents)
        }

    def _build_private_agent_datasets(
        self,
        data,
        split_seed: int,
    ) -> dict[int, MaskedCIFARDataset]:
        if self.share_private_data:
            return self._build_agent_datasets(data, None)

        perm = torch.randperm(
            len(data), generator=torch.Generator().manual_seed(split_seed)
        ).tolist()
        chunks = [perm[i :: self.n_agents] for i in range(self.n_agents)]
        return {
            i: self._agent_dataset(data.select(chunks[i]), i, None)
            for i in range(self.n_agents)
        }

    def setup(self, stage: str | None = None) -> None:
        ds = load_dataset(f'{self.repo}/{self.name}')
        all_data = concatenate_datasets([ds[s] for s in ds])
        split_indices = compute_split_indices(
            total_size=len(all_data),
            val_split=self.val_split,
            test_split=self.test_split,
            seed=self.seed,
            pilot_split=self.pilot_split,
            pilot_num_samples=self.pilot_num_samples,
        )

        pilot = all_data.select(split_indices['pilot'])
        train_ids = split_indices['train']
        budget = max(self.n_agents, int(len(train_ids) * self.private_train_fraction))
        train = all_data.select(train_ids[:budget])
        val = all_data.select(split_indices['val'])
        test = all_data.select(split_indices['test'])

        self.train_datasets = self._build_private_agent_datasets(
            train, self.seed
        )
        self.val_datasets = self._build_private_agent_datasets(
            val, self.seed + 1
        )
        self.test_datasets = self._build_private_agent_datasets(
            test, self.seed + 2
        )
        self.pilot_datasets = (
            self._build_agent_datasets(pilot, split_indices['pilot'])
            if split_indices['pilot']
            else {}
        )

        input_channels = 4 if self.include_mask_in_input else 3
        self.input_shape = (input_channels, self.image_size, self.image_size)
        self.input_dims = {
            str(i): input_channels for i in range(self.n_agents)
        }
        self.target_shape = (3, self.image_size, self.image_size)
        self.target_dims = {str(i): 3 for i in range(self.n_agents)}
        self.models = list(range(self.n_agents))
        self.num_classes = {'label': 10}

    def _make_loader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        generator = None
        if shuffle:
            generator = torch.Generator().manual_seed(self.seed)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=_collate_masked_batch,
            drop_last=shuffle,
            generator=generator,
        )

    def _make_pilot_loader(
        self,
        dataset: Dataset,
        target_num_batches: int,
    ) -> DataLoader:
        target_num_samples = target_num_batches * self.pilot_batch_size
        repeated_dataset = repeat_dataset_to_num_samples(
            dataset, target_num_samples
        )
        return DataLoader(
            repeated_dataset,
            batch_size=self.pilot_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=_collate_masked_batch,
        )

    def _create_loaders(self, split: str) -> CombinedLoader:
        split_map = {
            'train': (self.train_datasets, True),
            'val': (self.val_datasets, False),
            'test': (self.test_datasets, False),
        }
        if split not in split_map:
            raise ValueError(f'Unknown split {split!r}')
        datasets, shuffle = split_map[split]
        loaders = {
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

    def val_dataloader(self) -> CombinedLoader | list[CombinedLoader]:
        val_loader = self._create_loaders('val')
        if not self.monitor_test_during_fit:
            return val_loader
        return [val_loader, self.test_dataloader()]

    def test_dataloader(self) -> CombinedLoader:
        return self._create_loaders('test')
