"""Tests for SheafFRL's full-pool anchor selection and step-batch penalty.

Covers the redesign where the epoch-end map refit sources its candidate pool
from the FULL pilot split (``dm.pilot_datasets``) instead of the last loader
batch, and where ``anchor_selection='all'`` computes the per-step sheaf
penalty on the current rotating pilot batch instead of a fixed cache.
"""

from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.orchestrators.sheaf_frl import SheafFRL

POOL_SIZE = 40
PILOT_BATCH_SIZE = 8
LATENT_DIM = 3
IN_FEATURES = 4
NUM_CLASSES = 4


class _ToyAgent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(IN_FEATURES, LATENT_DIM, bias=False)
        self.decoder = nn.Linear(LATENT_DIM, NUM_CLASSES, bias=False)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x.float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encode(x))

    def compute_loss(self, y_hat, y) -> torch.Tensor:
        return F.cross_entropy(y_hat, y)

    def task_performance(self, y_hat, y) -> torch.Tensor:
        return (y_hat.argmax(dim=1) == y).float().mean()


class _ToyPilotDataset:
    """Minimal pilot dataset: (x, y, sample_id) rows with ``sample_ids``."""

    def __init__(self, n: int = POOL_SIZE, seed: int = 0) -> None:
        g = torch.Generator().manual_seed(seed)
        self.x = torch.randn(n, IN_FEATURES, generator=g)
        self.y = torch.randint(0, NUM_CLASSES, (n,), generator=g)
        self.sample_ids = list(range(n))

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, i: int):
        return self.x[i], self.y[i], self.sample_ids[i]


def _build(anchor_selection: str, num_anchors: int = 4) -> SheafFRL:
    orchestrator = SheafFRL(
        agents={0: _ToyAgent(), 1: _ToyAgent()},
        neighbors={0: {1}, 1: {0}},
        optimizer={'_target_': 'torch.optim.SGD', 'lr': 0.1},
        max_lmb=0.1,
        latent_dims={0: LATENT_DIM, 1: LATENT_DIM},
        anchor_selection=anchor_selection,
        num_anchors=num_anchors,
    )
    pool = _ToyPilotDataset()
    dm = SimpleNamespace(
        pilot_datasets={0: pool, 1: pool},
        pilot_batch_size=PILOT_BATCH_SIZE,
    )
    orchestrator._trainer = SimpleNamespace(
        datamodule=dm,
        current_epoch=0,
        num_training_batches=10,
    )
    # Standalone _shared_eval calls: no trainer loop to route logs through.
    orchestrator.log = lambda *a, **k: None
    orchestrator.log_dict = lambda *a, **k: None
    orchestrator._task_latent_buffer = {}
    orchestrator._agent_target_classes = None
    return orchestrator


def _edge_key(orchestrator: SheafFRL) -> tuple[int, int]:
    assert len(orchestrator.stiefel_matrices) == 1
    key = next(iter(orchestrator.stiefel_matrices))
    node_i, node_j = map(int, key.split('_'))
    return node_i, node_j


def test_refit_pool_exceeds_pilot_batch_size() -> None:
    """'all' anchors span the whole pilot split, not one loader batch."""
    orchestrator = _build('all')
    orchestrator._rebuild_edge_anchor_caches()

    pair = _edge_key(orchestrator)
    cache = orchestrator._edge_anchor_cache[pair]
    assert cache['sample_ids'].shape[0] == POOL_SIZE
    assert POOL_SIZE > PILOT_BATCH_SIZE

    Z_i_frozen, _, Z_j_frozen, _ = orchestrator._frozen_edge_anchors[pair]
    assert Z_i_frozen.shape[0] == POOL_SIZE
    assert Z_j_frozen.shape[0] == POOL_SIZE


def test_random_selection_respects_num_anchors() -> None:
    """num_anchors < pool size actually reduces the anchor set."""
    orchestrator = _build('random', num_anchors=4)
    orchestrator._rebuild_edge_anchor_caches()

    pair = _edge_key(orchestrator)
    cache = orchestrator._edge_anchor_cache[pair]
    assert cache['sample_ids'].shape[0] == 4

    Z_i_frozen, _, _, _ = orchestrator._frozen_edge_anchors[pair]
    assert Z_i_frozen.shape[0] == 4


def test_random_live_reencode_bounded_by_pilot_batch_size() -> None:
    """A large num_anchors doesn't blow up the per-step live re-encode.

    num_anchors=20 > PILOT_BATCH_SIZE=8: the fixed cache legitimately holds
    20 rows (the full communication/map-fit budget), but a single step's
    live re-encode (_edge_live_anchors, real forward pass, every edge, every
    step) must stay bounded to pilot_batch_size, cycling through the cache
    across steps rather than re-encoding all 20 at once.
    """
    orchestrator = _build('random', num_anchors=20)
    orchestrator._rebuild_edge_anchor_caches()
    pair = _edge_key(orchestrator)
    cache = orchestrator._edge_anchor_cache[pair]
    assert cache['sample_ids'].shape[0] == 20

    node_i, node_j = pair
    matched = orchestrator._edge_step_anchors(node_i, node_j)
    assert matched is not None
    Z_i, y_i, Z_j, y_j = matched
    assert Z_i.shape[0] == PILOT_BATCH_SIZE

    # Rotating window: the next step's call advances the offset, so it
    # touches different rows of the cache.
    matched2 = orchestrator._edge_step_anchors(node_i, node_j)
    Z_i2, y_i2, Z_j2, y_j2 = matched2
    assert Z_i2.shape[0] == PILOT_BATCH_SIZE
    assert not torch.equal(Z_i, Z_i2)


def test_random_small_num_anchors_unbounded_by_pilot_batch_size() -> None:
    """num_anchors <= pilot_batch_size: unchanged (every anchor, every step)."""
    orchestrator = _build('random', num_anchors=4)
    orchestrator._rebuild_edge_anchor_caches()
    pair = _edge_key(orchestrator)
    node_i, node_j = pair

    matched = orchestrator._edge_step_anchors(node_i, node_j)
    Z_i, y_i, Z_j, y_j = matched
    assert Z_i.shape[0] == 4


def test_proto_kmeans_assignment_fits_from_full_pool() -> None:
    """K-means fits when K <= full pool size (was impossible with K > batch)."""
    orchestrator = _build('proto_kmeans', num_anchors=4)
    orchestrator._rebuild_edge_anchor_caches()

    pair = _edge_key(orchestrator)
    assert pair in orchestrator._edge_kmeans_assign
    assert len(orchestrator._edge_kmeans_assign[pair]) == POOL_SIZE

    Z_i_frozen, _, _, _ = orchestrator._frozen_edge_anchors[pair]
    assert Z_i_frozen.shape[0] <= 4


def test_proto_kmeans_assignment_refits_every_refresh() -> None:
    """The clustering is refit at each refresh, tracking the current encoder."""
    orchestrator = _build('proto_kmeans', num_anchors=4)
    orchestrator._rebuild_edge_anchor_caches()
    pair = _edge_key(orchestrator)
    first = dict(orchestrator._edge_kmeans_assign[pair])

    # Change the encoder: the whitened pool moves, so a refit must be able to
    # produce a different assignment (same sample-id key set either way).
    for agent in orchestrator.agents.values():
        with torch.no_grad():
            agent.encoder.weight.add_(
                torch.randn_like(agent.encoder.weight)
            )
    orchestrator._rebuild_edge_anchor_caches()
    second = orchestrator._edge_kmeans_assign[pair]

    assert set(second) == set(first)
    assert second is not first
    assert second != first


def test_step_penalty_uses_rotating_batch_without_cache() -> None:
    """'all' computes a nonzero step penalty from the batch alone (no cache)."""
    orchestrator = _build('all')
    assert not orchestrator._edge_anchor_cache

    def _batch(offset: int) -> dict:
        g = torch.Generator().manual_seed(100 + offset)
        x = torch.randn(PILOT_BATCH_SIZE, IN_FEATURES, generator=g)
        y = torch.randint(0, NUM_CLASSES, (PILOT_BATCH_SIZE,), generator=g)
        sids = torch.arange(
            offset * PILOT_BATCH_SIZE, (offset + 1) * PILOT_BATCH_SIZE
        )
        return {
            0: [x, y],
            1: [x, y],
            'global_pilot': [x, y, sids],
        }

    _, loss_a = orchestrator._shared_eval(_batch(0), 0, 'train')
    penalty_a = orchestrator._edge_step_anchors(*_edge_key(orchestrator))
    # _step_pilot_latents is cleared after every loss computation.
    assert penalty_a is None
    assert not orchestrator._step_pilot_latents
    assert loss_a.ndim == 0

    # Different pilot batches must produce different penalties: the penalty
    # consumes the rotating batch, not a fixed set.
    torch.manual_seed(0)
    _, loss_b = orchestrator._shared_eval(_batch(1), 1, 'train')
    assert not torch.allclose(loss_a, loss_b)


def test_vectorised_matcher_pairs_rows_by_key() -> None:
    """The unique-key fast path pairs rows of the same sample id, key-sorted."""
    orchestrator = _build('all')
    keys_i = torch.tensor([5, 1, 9, 3])
    keys_j = torch.tensor([3, 7, 5, 2, 9])
    A_i = keys_i.float().unsqueeze(1) * 10.0
    A_j = keys_j.float().unsqueeze(1) * 100.0
    y_i = keys_i.clone()
    y_j = keys_j.clone()

    matched = orchestrator._match_pilots_with_labels(
        A_i, keys_i, y_i, A_j, keys_j, y_j
    )
    assert matched is not None
    A_i_m, y_i_m, A_j_m, y_j_m, keys_m = matched
    assert keys_m.tolist() == [3, 5, 9]
    assert A_i_m.squeeze(1).tolist() == [30.0, 50.0, 90.0]
    assert A_j_m.squeeze(1).tolist() == [300.0, 500.0, 900.0]
    assert y_i_m.tolist() == y_j_m.tolist() == [3, 5, 9]


def test_step_exchange_recorded_for_all_strategy() -> None:
    """Training steps under 'all' add per-edge KB to the train accounting."""
    orchestrator = _build('all')
    before = orchestrator._communication_by_split['train']['kilobytes']

    g = torch.Generator().manual_seed(7)
    x = torch.randn(PILOT_BATCH_SIZE, IN_FEATURES, generator=g)
    y = torch.randint(0, NUM_CLASSES, (PILOT_BATCH_SIZE,), generator=g)
    batch = {
        0: [x, y],
        1: [x, y],
        'global_pilot': [x, y, torch.arange(PILOT_BATCH_SIZE)],
    }
    orchestrator._shared_eval(batch, 0, 'train')

    after = orchestrator._communication_by_split['train']['kilobytes']
    assert after > before
