"""Mask generators for masked-image federated learning."""

from collections.abc import Sequence

import torch
import torch.nn.functional as F


class RegionOverlapMaskGenerator:
    """Fixed cyclic stripe masks with guaranteed global coverage and an
    independently controllable overlap.

    Agent centres are spaced uniformly at ``image_size / n_agents``. The
    visible window is derived from ``overlap_fraction`` as
    ``spacing * (1 + overlap_fraction)``, so the union covers the whole image
    and the duplicated adjacent coverage is controlled by a single scalar.

    ``visible_fraction`` is accepted as a lower-bound sanity check for
    backwards-compatible configs. If it asks for more visibility than the
    requested overlap implies, the two constraints are inconsistent.
    """

    def __init__(
        self,
        n_agents,
        visible_fraction=0.5,
        overlap_fraction=0.3,
        image_size=32,
        fill_value=0.0,
        enforce_full_coverage=True,
        boundary_jitter_px=0,
        seed=42,
    ):
        self.n_agents = int(n_agents)
        self.W = int(image_size)
        self.fill_value = fill_value
        self.boundary_jitter_px = max(0, int(boundary_jitter_px))
        self.seed = int(seed)
        self.spacing = self.W / self.n_agents

        overlap_fraction = float(overlap_fraction)
        if overlap_fraction < 0.0:
            raise ValueError('overlap_fraction must be >= 0.')

        min_visible_for_coverage = 1.0 / self.n_agents
        implied_visible_fraction = (1.0 + overlap_fraction) / self.n_agents
        if (
            enforce_full_coverage
            and visible_fraction < min_visible_for_coverage
        ):
            raise ValueError(
                f'visible_fraction={visible_fraction} lascia una zona non '
                f'vista da nessun agente con n_agents={self.n_agents} (serve >= '
                f'{min_visible_for_coverage:.3f}). Alza visible_fraction o '
                'passa enforce_full_coverage=False per accettarlo.'
            )
        if visible_fraction > implied_visible_fraction + 1e-9:
            raise ValueError(
                f"visible_fraction={visible_fraction} richiede piu' overlap "
                f'di region_overlap_fraction={overlap_fraction}. Per '
                f'n_agents={self.n_agents}, usa visible_fraction <= '
                f'{implied_visible_fraction:.3f} oppure aumenta '
                'region_overlap_fraction.'
            )

        self.requested_visible_fraction = float(visible_fraction)
        self.overlap_fraction = overlap_fraction
        self.visible_px = implied_visible_fraction * self.W
        self.window = self.visible_px
        self.actual_overlap_px = max(0.0, self.window - self.spacing)
        self.orphan_px = max(0.0, self.spacing - self.window)
        self.centers = [
            (a * self.spacing) % self.W for a in range(self.n_agents)
        ]

    def _sample_shift(self, sample_id: int | None) -> float:
        if self.boundary_jitter_px <= 0 or sample_id is None:
            return 0.0
        generator = torch.Generator().manual_seed(
            self.seed + 1_000_003 * int(sample_id)
        )
        return float(
            torch.randint(
                low=-self.boundary_jitter_px,
                high=self.boundary_jitter_px + 1,
                size=(1,),
                generator=generator,
            ).item()
        )

    def get_mask(self, agent_idx, sample_id: int | None = None):
        mask = torch.zeros(1, self.W, self.W)
        c = (self.centers[agent_idx] + self._sample_shift(sample_id)) % self.W
        half = self.window / 2.0
        lo, hi = (c - half) % self.W, (c + half) % self.W
        cols = torch.arange(self.W)
        visible = (
            (cols >= lo) & (cols < hi)
            if lo < hi
            else (cols >= lo) | (cols < hi)
        )
        mask[:, :, visible] = 1.0
        return mask

    def __call__(self, img, agent_idx, sample_id: int | None = None):
        mask = self.get_mask(agent_idx, sample_id).to(
            dtype=img.dtype, device=img.device
        )
        fill = (
            img.mean(dim=(1, 2), keepdim=True)
            if self.fill_value == 'mean'
            else float(self.fill_value)
        )
        return img * mask + fill * (1 - mask), mask


class OverlapMaskGenerator:
    """Fixed cyclic stripe masks with continuous pairwise overlap control."""

    def __init__(
        self,
        n_agents: int,
        overlap_px: float = 2.0,
        image_size: int = 32,
        min_overlap_px: float = 1.0,
        fill_value: float | str = 0.0,
    ):
        self.n_agents = int(n_agents)
        self.overlap_px = max(float(overlap_px), float(min_overlap_px))
        self.image_size = int(image_size)
        self.fill_value = fill_value
        self.spacing = self.image_size / self.n_agents
        self.window = self.spacing + self.overlap_px
        self.centers = [
            agent_idx * self.spacing for agent_idx in range(self.n_agents)
        ]

    def get_mask(self, agent_idx: int) -> torch.Tensor:
        mask = torch.zeros(1, self.image_size, self.image_size)
        center = self.centers[int(agent_idx)]
        half_window = self.window / 2.0
        lo = (center - half_window) % self.image_size
        hi = (center + half_window) % self.image_size
        cols = torch.arange(self.image_size)
        if lo < hi:
            visible = (cols >= lo) & (cols < hi)
        else:
            visible = (cols >= lo) | (cols < hi)
        mask[:, :, visible] = 1.0
        return mask

    def __call__(
        self, img: torch.Tensor, agent_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask = self.get_mask(agent_idx).to(dtype=img.dtype, device=img.device)
        fill = (
            img.mean(dim=(1, 2), keepdim=True)
            if self.fill_value == 'mean'
            else float(self.fill_value)
        )
        return img * mask + fill * (1 - mask), mask


class SpatialRandomMaskGenerator:
    """Random block masks with explicit shared and private visibility.

    The shared mask is identical for every agent on the same sample. Each
    agent then adds a private spatially biased mask: high probability inside
    its focus region, low probability elsewhere.
    """

    _DEFAULT_REGIONS = ('top_right', 'bottom_left', 'top_left', 'bottom_right')
    _REGIONS = {
        'top_left': (0.0, 0.5, 0.0, 0.5),
        'top_right': (0.0, 0.5, 0.5, 1.0),
        'bottom_left': (0.5, 1.0, 0.0, 0.5),
        'bottom_right': (0.5, 1.0, 0.5, 1.0),
        'top': (0.0, 0.5, 0.0, 1.0),
        'bottom': (0.5, 1.0, 0.0, 1.0),
        'left': (0.0, 1.0, 0.0, 0.5),
        'right': (0.0, 1.0, 0.5, 1.0),
        'center': (0.25, 0.75, 0.25, 0.75),
    }

    def __init__(
        self,
        n_agents: int,
        focus_regions: Sequence[str] | None = None,
        private_visible_probability: float = 0.7,
        off_focus_visible_probability: float = 0.0,
        shared_visible_probability: float = 0.1,
        block_size: int = 4,
        image_size: int = 32,
        seed: int = 42,
    ):
        self.n_agents = int(n_agents)
        self.image_size = int(image_size)
        self.block_size = max(1, int(block_size))
        self.seed = int(seed)
        self.private_visible_probability = float(private_visible_probability)
        self.off_focus_visible_probability = float(
            off_focus_visible_probability
        )
        self.shared_visible_probability = float(shared_visible_probability)
        self.focus_regions = self._resolve_focus_regions(focus_regions)

    def _resolve_focus_regions(
        self, focus_regions: Sequence[str] | None
    ) -> list[tuple[float, float, float, float]]:
        regions = list(focus_regions or self._DEFAULT_REGIONS[: self.n_agents])

        if len(regions) != self.n_agents:
            raise ValueError(
                'focus_regions must have one entry per agent when provided, '
                f'got {len(regions)} for {self.n_agents} agents'
            )
        unknown = sorted(set(regions) - set(self._REGIONS))
        if unknown:
            raise ValueError(f'Unknown focus regions: {unknown}')
        return [self._REGIONS[name] for name in regions]

    def _private_probability_grid(self, agent_idx: int) -> torch.Tensor:
        grid_size = (self.image_size + self.block_size - 1) // self.block_size
        probs = torch.full(
            (1, grid_size, grid_size),
            self.off_focus_visible_probability,
        )
        r0, r1, c0, c1 = self.focus_regions[int(agent_idx)]
        row_start = int(round(r0 * grid_size))
        row_end = max(row_start + 1, int(round(r1 * grid_size)))
        col_start = int(round(c0 * grid_size))
        col_end = max(col_start + 1, int(round(c1 * grid_size)))
        probs[:, row_start:row_end, col_start:col_end] = (
            self.private_visible_probability
        )
        return probs.clamp(0.0, 1.0)

    def _upsample(self, lowres: torch.Tensor) -> torch.Tensor:
        mask = F.interpolate(
            lowres.unsqueeze(0).float(),
            size=(self.image_size, self.image_size),
            mode='nearest',
        )
        return mask.squeeze(0)

    def _shared_mask(self, sample_id: int) -> torch.Tensor:
        grid_size = (self.image_size + self.block_size - 1) // self.block_size
        generator = torch.Generator().manual_seed(
            self.seed + 1_000_003 * int(sample_id)
        )
        lowres = (
            torch.rand(1, grid_size, grid_size, generator=generator)
            < self.shared_visible_probability
        )
        return self._upsample(lowres)

    def _private_mask(self, agent_idx: int, sample_id: int) -> torch.Tensor:
        generator = torch.Generator().manual_seed(
            self.seed + 100_003 * int(agent_idx) + int(sample_id)
        )
        probs = self._private_probability_grid(agent_idx)
        lowres = torch.rand(probs.shape, generator=generator) < probs
        return self._upsample(lowres)

    def get_mask(self, agent_idx: int, sample_id: int = 0) -> torch.Tensor:
        shared = self._shared_mask(sample_id)
        private = self._private_mask(agent_idx, sample_id)
        return torch.logical_or(shared.bool(), private.bool()).float()

    def __call__(
        self, img: torch.Tensor, agent_idx: int, sample_id: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask = self.get_mask(agent_idx, sample_id).to(
            dtype=img.dtype, device=img.device
        )
        return img * mask, mask


class ConstantVisibleSharedMaskGenerator:
    """Block masks with fixed visibility and variable shared mass.

    ``shared_visible_probability`` is the fraction of each agent's visible
    blocks that is shared. For every sample, the generator first selects an
    identical shared block set for all agents. Each agent then adds private
    blocks sampled without
    replacement from blocks not in the shared set and not assigned to another
    agent, preferring its focus region. Therefore every agent sees the same
    number of blocks and private blocks are disjoint from both shared blocks and
    other agents' private blocks.
    """

    _DEFAULT_REGIONS = SpatialRandomMaskGenerator._DEFAULT_REGIONS
    _REGIONS = SpatialRandomMaskGenerator._REGIONS

    def __init__(
        self,
        n_agents: int,
        focus_regions: Sequence[str] | None = None,
        visible_fraction: float = 0.7,
        shared_visible_probability: float = 0.1,
        block_size: int = 4,
        image_size: int = 32,
        seed: int = 42,
    ):
        self.n_agents = int(n_agents)
        self.image_size = int(image_size)
        self.block_size = max(1, int(block_size))
        self.seed = int(seed)
        self.visible_fraction = float(visible_fraction)
        self.shared_visible_probability = float(shared_visible_probability)
        self.focus_regions = self._resolve_focus_regions(focus_regions)

        if not 0.0 <= self.visible_fraction <= 1.0:
            raise ValueError('visible_fraction must be in [0, 1].')
        if not 0.0 <= self.shared_visible_probability <= 1.0:
            raise ValueError(
                'shared_visible_probability must be in [0, 1].'
            )

        self.grid_size = (
            self.image_size + self.block_size - 1
        ) // self.block_size
        self.num_blocks = self.grid_size * self.grid_size
        self.visible_blocks = int(round(self.visible_fraction * self.num_blocks))
        # Shared visibility is a fraction of the fixed visible budget, not of
        # the full image. This keeps total visibility constant throughout the
        # sweep while moving mass from private to shared blocks.
        self.shared_blocks = int(
            round(self.shared_visible_probability * self.visible_blocks)
        )
        self.private_blocks_per_agent = self.visible_blocks - self.shared_blocks
        private_capacity = self.num_blocks - self.shared_blocks
        required_private = self.n_agents * self.private_blocks_per_agent
        if required_private > private_capacity:
            max_visible = (
                self.shared_blocks + private_capacity // self.n_agents
            ) / self.num_blocks
            raise ValueError(
                'Cannot make agent-private blocks disjoint with '
                f'visible_fraction={self.visible_fraction} and '
                f'shared_visible_probability={self.shared_visible_probability}. '
                f'Use visible_fraction <= {max_visible:.3f} for '
                f'n_agents={self.n_agents}.'
            )

    def _resolve_focus_regions(
        self, focus_regions: Sequence[str] | None
    ) -> list[tuple[float, float, float, float]]:
        regions = list(focus_regions or self._DEFAULT_REGIONS[: self.n_agents])
        if len(regions) != self.n_agents:
            raise ValueError(
                'focus_regions must have one entry per agent when provided, '
                f'got {len(regions)} for {self.n_agents} agents'
            )
        unknown = sorted(set(regions) - set(self._REGIONS))
        if unknown:
            raise ValueError(f'Unknown focus regions: {unknown}')
        return [self._REGIONS[name] for name in regions]

    def _upsample(self, lowres: torch.Tensor) -> torch.Tensor:
        mask = F.interpolate(
            lowres.unsqueeze(0).float(),
            size=(self.image_size, self.image_size),
            mode='nearest',
        )
        return mask.squeeze(0)

    def _shared_indices(self, sample_id: int) -> torch.Tensor:
        generator = torch.Generator().manual_seed(
            self.seed + 1_000_003 * int(sample_id)
        )
        perm = torch.randperm(self.num_blocks, generator=generator)
        return perm[: self.shared_blocks]

    def _focus_indices(self, agent_idx: int) -> torch.Tensor:
        r0, r1, c0, c1 = self.focus_regions[int(agent_idx)]
        row_start = int(round(r0 * self.grid_size))
        row_end = max(row_start + 1, int(round(r1 * self.grid_size)))
        col_start = int(round(c0 * self.grid_size))
        col_end = max(col_start + 1, int(round(c1 * self.grid_size)))
        rows = torch.arange(row_start, row_end)
        cols = torch.arange(col_start, col_end)
        rr, cc = torch.meshgrid(rows, cols, indexing='ij')
        return (rr * self.grid_size + cc).flatten()

    def _all_private_indices(
        self, sample_id: int, shared: torch.Tensor
    ) -> list[torch.Tensor]:
        if self.private_blocks_per_agent <= 0:
            return [
                torch.empty(0, dtype=torch.long)
                for _agent_idx in range(self.n_agents)
            ]
        shared_mask = torch.zeros(self.num_blocks, dtype=torch.bool)
        shared_mask[shared.long()] = True
        available = torch.arange(self.num_blocks)[~shared_mask]

        assigned = torch.zeros(self.num_blocks, dtype=torch.bool)
        private_by_agent: list[torch.Tensor] = []
        for agent_idx in range(self.n_agents):
            generator = torch.Generator().manual_seed(
                self.seed + 100_003 * int(agent_idx) + int(sample_id)
            )
            focus = self._focus_indices(agent_idx)
            focus_available = focus[~shared_mask[focus] & ~assigned[focus]]
            focus_available = focus_available[
                torch.randperm(focus_available.numel(), generator=generator)
            ]
            selected = focus_available[: self.private_blocks_per_agent]

            if selected.numel() < self.private_blocks_per_agent:
                selected_mask = torch.zeros(self.num_blocks, dtype=torch.bool)
                selected_mask[selected.long()] = True
                fallback = available[
                    ~assigned[available] & ~selected_mask[available]
                ]
                fallback = fallback[
                    torch.randperm(fallback.numel(), generator=generator)
                ]
                selected = torch.cat(
                    [
                        selected,
                        fallback[
                            : self.private_blocks_per_agent - selected.numel()
                        ],
                    ]
                )

            assigned[selected.long()] = True
            private_by_agent.append(selected)
        return private_by_agent

    def _private_indices(
        self,
        agent_idx: int,
        sample_id: int,
        shared: torch.Tensor,
    ) -> torch.Tensor:
        return self._all_private_indices(sample_id, shared)[int(agent_idx)]

    def get_mask(self, agent_idx: int, sample_id: int = 0) -> torch.Tensor:
        shared = self._shared_indices(sample_id)
        private = self._private_indices(agent_idx, sample_id, shared)
        lowres = torch.zeros(1, self.grid_size, self.grid_size)
        indices = torch.cat([shared, private]).long()
        lowres.view(-1)[indices] = 1.0
        return self._upsample(lowres)

    def get_components(
        self, agent_idx: int, sample_id: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shared = self._shared_indices(sample_id)
        private = self._private_indices(agent_idx, sample_id, shared)
        shared_lowres = torch.zeros(1, self.grid_size, self.grid_size)
        private_lowres = torch.zeros(1, self.grid_size, self.grid_size)
        shared_lowres.view(-1)[shared.long()] = 1.0
        private_lowres.view(-1)[private.long()] = 1.0
        return self._upsample(shared_lowres), self._upsample(private_lowres)

    def __call__(
        self, img: torch.Tensor, agent_idx: int, sample_id: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask = self.get_mask(agent_idx, sample_id).to(
            dtype=img.dtype, device=img.device
        )
        return img * mask, mask
