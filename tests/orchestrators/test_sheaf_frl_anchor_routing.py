"""Focused tests for SheafFRL anchor routing."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.orchestrators.sheaf_frl import SheafFRL


class _CountingAgent(nn.Module):
    def __init__(
        self,
        in_features: int = 4,
        latent_dim: int = 3,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.encoder = nn.Linear(in_features, latent_dim, bias=False)
        self.decoder = nn.Linear(latent_dim, num_classes, bias=False)
        self.encode_calls = 0

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        self.encode_calls += 1
        return self.encoder(x.float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encode(x))

    def compute_loss(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        return F.cross_entropy(y_hat, y)

    def task_performance(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        return (y_hat.argmax(dim=1) == y).float().mean()


def _build_orchestrator(
    anchor_strategy: str,
    *,
    use_prototypes: bool = False,
    num_anchors: int = 32,
) -> tuple[SheafFRL, _CountingAgent]:
    agent = _CountingAgent()
    orchestrator = SheafFRL(
        agents={0: agent},
        neighbors={0: set()},
        optimizer={'_target_': 'torch.optim.SGD', 'lr': 0.1},
        max_lmb=0.1,
        latent_dims={0: 3},
        anchor_strategy=anchor_strategy,
        num_anchors=num_anchors,
        parseval_normalization=False,
        l2_normalization=False,
        use_prototypes=use_prototypes,
    )
    return orchestrator, agent


def test_invalid_anchor_strategy_raises() -> None:
    with pytest.raises(ValueError, match='Unknown anchor_strategy'):
        _build_orchestrator('dynamic')


def test_batch_anchors_reuses_cached_task_latents_without_extra_encoding() -> None:
    orchestrator, agent = _build_orchestrator(
        'batch_anchors',
        use_prototypes=False,
    )
    batch = {
        0: [
            torch.randn(4, 4),
            torch.tensor([0, 1, 0, 1], dtype=torch.long),
        ]
    }

    orchestrator.on_train_start()
    outputs, total_loss = orchestrator._shared_eval(batch, 0, 'train')

    assert orchestrator.hparams.anchor_strategy == 'batch_anchors'
    assert agent.encode_calls == 1
    assert len(orchestrator.epoch_latents_cache[0]) == 1
    assert orchestrator.epoch_latents_cache[0][0].shape[0] == 4
    assert torch.equal(
        orchestrator.epoch_labels_cache[0][0],
        torch.tensor([0, 1, 0, 1], dtype=torch.long),
    )
    assert '0' in outputs
    assert total_loss.ndim == 0


def test_pilots_strategy_uses_pilot_batch_encoding() -> None:
    orchestrator, agent = _build_orchestrator('pilots')
    batch = {
        0: [
            torch.randn(4, 4),
            torch.tensor([0, 1, 0, 1], dtype=torch.long),
        ],
        'pilot_0': [
            torch.randn(2, 4),
            torch.tensor([0, 1], dtype=torch.long),
            torch.arange(2, dtype=torch.long),
        ],
    }

    outputs, total_loss = orchestrator._shared_eval(batch, 0, 'test')

    assert orchestrator.hparams.anchor_strategy == 'pilots'
    assert agent.encode_calls == 2
    assert '0' in outputs
    assert total_loss.ndim == 0


def test_extract_pilot_batch_prefers_agent_specific_global_pilot() -> None:
    orchestrator, _agent = _build_orchestrator('pilots')
    batch = {
        'global_pilot': [
            torch.randn(2, 4),
            torch.tensor([9, 9], dtype=torch.long),
            torch.tensor([90, 91], dtype=torch.long),
        ],
        'global_pilot_0': [
            torch.randn(3, 4),
            torch.tensor([1, 2, 3], dtype=torch.long),
            torch.tensor([10, 11, 12], dtype=torch.long),
        ],
    }

    _x_pilot, y_pilot, sample_ids = orchestrator._extract_pilot_batch(batch, 0)

    assert torch.equal(y_pilot, torch.tensor([1, 2, 3], dtype=torch.long))
    assert torch.equal(sample_ids, torch.tensor([10, 11, 12], dtype=torch.long))


def test_pilots_uniform_caches_raw_latents_and_sample_ids() -> None:
    orchestrator, agent = _build_orchestrator(
        'pilots',
        use_prototypes=False,
        num_anchors=1,
    )
    batch = {
        0: [
            torch.randn(4, 4),
            torch.tensor([0, 1, 0, 1], dtype=torch.long),
        ],
        'pilot_0': [
            torch.randn(3, 4),
            torch.tensor([5, 6, 7], dtype=torch.long),
            torch.tensor([10, 11, 12], dtype=torch.long),
        ],
    }

    orchestrator.on_train_start()
    orchestrator._shared_eval(batch, 0, 'train')

    assert agent.encode_calls == 2
    assert orchestrator.epoch_latents_cache[0][0].shape[0] == 3
    assert torch.equal(
        orchestrator.epoch_labels_cache[0][0],
        torch.tensor([10, 11, 12], dtype=torch.long),
    )


def test_batch_anchors_aligns_two_agents_via_class_prototypes() -> None:
    agent_0 = _CountingAgent(num_classes=3)
    agent_1 = _CountingAgent(num_classes=3)
    orchestrator = SheafFRL(
        agents={0: agent_0, 1: agent_1},
        neighbors={0: {1}, 1: {0}},
        optimizer={'_target_': 'torch.optim.SGD', 'lr': 0.1},
        max_lmb=0.1,
        latent_dims={0: 3, 1: 3},
        anchor_strategy='batch_anchors',
        parseval_normalization=False,
        l2_normalization=False,
        use_prototypes=False,
    )
    batch = {
        0: [
            torch.randn(4, 4),
            torch.tensor([0, 0, 1, 1], dtype=torch.long),
        ],
        1: [
            torch.randn(4, 4),
            torch.tensor([1, 1, 2, 2], dtype=torch.long),
        ],
    }

    orchestrator.on_train_start()
    outputs, total_loss = orchestrator._shared_eval(batch, 0, 'train')

    assert agent_0.encode_calls == 1
    assert agent_1.encode_calls == 1
    assert orchestrator.epoch_latents_cache[0][0].shape[0] == 4
    assert orchestrator.epoch_latents_cache[1][0].shape[0] == 4
    assert torch.equal(
        orchestrator.epoch_labels_cache[0][0],
        torch.tensor([0, 0, 1, 1], dtype=torch.long),
    )
    assert torch.equal(
        orchestrator.epoch_labels_cache[1][0],
        torch.tensor([1, 1, 2, 2], dtype=torch.long),
    )
    assert set(outputs) == {'0', '1'}
    assert total_loss.ndim == 0


def test_train_step_uses_sheaf_penalty_every_iteration() -> None:
    agent_0 = _CountingAgent(in_features=4, latent_dim=2, num_classes=2)
    agent_1 = _CountingAgent(in_features=4, latent_dim=2, num_classes=2)
    with torch.no_grad():
        identity = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        classifier = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        agent_0.encoder.weight.copy_(identity)
        agent_1.encoder.weight.copy_(identity)
        agent_0.decoder.weight.copy_(classifier)
        agent_1.decoder.weight.copy_(classifier)

    orchestrator = SheafFRL(
        agents={0: agent_0, 1: agent_1},
        neighbors={0: {1}, 1: {0}},
        optimizer={'_target_': 'torch.optim.SGD', 'lr': 0.1},
        lambda_sheaf=0.5,
        latent_dims={0: 2, 1: 2},
        local_steps=3,
        anchor_strategy='batch_anchors',
        parseval_normalization=False,
        l2_normalization=False,
        use_prototypes=False,
    )
    batch = {
        0: [
            torch.tensor(
                [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            ),
            torch.tensor([0, 1], dtype=torch.long),
        ],
        1: [
            torch.tensor(
                [[2.0, 0.0, 0.0, 0.0], [0.0, 3.0, 0.0, 0.0]],
            ),
            torch.tensor([0, 1], dtype=torch.long),
        ],
    }

    task_loss = torch.tensor(0.0)
    for idx, agent in orchestrator.agents.items():
        x, y = batch[int(idx)]
        latent = agent.encode(x)
        task_loss = task_loss + agent.compute_loss(agent.decoder(latent), y)
    agent_0.encode_calls = 0
    agent_1.encode_calls = 0

    orchestrator.on_train_start()
    _, total_loss = orchestrator._shared_eval(batch, 0, 'train')

    assert agent_0.encode_calls == 1
    assert agent_1.encode_calls == 1
    assert total_loss > task_loss
    train_metrics = orchestrator._communication_metrics('train')
    assert train_metrics['train/communication_rounds'] == 1.0


def test_uniform_strategy_respects_num_anchors_budget() -> None:
    orchestrator = SheafFRL(
        agents={0: _CountingAgent(), 1: _CountingAgent()},
        neighbors={0: {1}, 1: {0}},
        optimizer={'_target_': 'torch.optim.SGD', 'lr': 0.1},
        lambda_sheaf=0.1,
        latent_dims={0: 3, 1: 3},
        anchor_strategy='batch_anchors',
        num_anchors=2,
        parseval_normalization=False,
        l2_normalization=False,
        use_prototypes=False,
    )
    raw_latents = {
        0: torch.randn(6, 3),
        1: torch.randn(6, 3),
    }
    raw_labels = {
        0: torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long),
        1: torch.tensor([0, 1, 1, 2, 2, 2], dtype=torch.long),
    }

    anchor_tensors, anchor_keys = orchestrator._build_anchor_bundles(
        raw_latents,
        raw_labels,
    )

    assert all(anchor.shape[0] <= 2 for anchor in anchor_tensors.values())
    assert all(len(keys) <= 2 for keys in anchor_keys.values())


def test_epoch_end_updates_stiefel_from_cached_anchor_cross_covariance() -> None:
    orchestrator = SheafFRL(
        agents={0: _CountingAgent(), 1: _CountingAgent()},
        neighbors={0: {1}, 1: {0}},
        optimizer={'_target_': 'torch.optim.SGD', 'lr': 0.1},
        lambda_sheaf=0.1,
        latent_dims={0: 3, 1: 3},
        anchor_strategy='batch_anchors',
        parseval_normalization=False,
        l2_normalization=False,
        filter_unseen_classes=False,
        use_prototypes=False,
    )

    A_0 = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
    )
    A_1 = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, -1.0, 0.0]],
    )
    labels = torch.tensor([0, 1, 1], dtype=torch.long)
    orchestrator.epoch_latents_cache[0].append(A_0)
    orchestrator.epoch_latents_cache[1].append(A_1)
    orchestrator.epoch_labels_cache[0].append(labels)
    orchestrator.epoch_labels_cache[1].append(labels.clone())

    epoch_anchor_tensors, epoch_anchor_keys = orchestrator._build_anchor_bundles(
        {0: A_0, 1: A_1},
        {0: labels, 1: labels.clone()},
    )
    shared_rows = orchestrator._match_bundle_rows(
        A_i=epoch_anchor_tensors[1],
        keys_i=epoch_anchor_keys[1],
        A_j=epoch_anchor_tensors[0],
        keys_j=epoch_anchor_keys[0],
        seen_i=set(),
        seen_j=set(),
    )
    assert shared_rows is not None
    centered_1 = shared_rows[0] - shared_rows[0].mean(dim=0, keepdim=True)
    centered_0 = shared_rows[1] - shared_rows[1].mean(dim=0, keepdim=True)
    covariance = centered_1.T @ centered_0
    U, _, Vh = torch.linalg.svd(covariance, full_matrices=False)
    expected = U @ Vh

    orchestrator.on_train_epoch_end()

    assert torch.allclose(
        orchestrator.stiefel_matrices['1_0'],
        expected,
    )
