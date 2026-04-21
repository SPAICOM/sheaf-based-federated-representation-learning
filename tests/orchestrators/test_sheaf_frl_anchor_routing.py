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
) -> tuple[SheafFRL, _CountingAgent]:
    agent = _CountingAgent()
    orchestrator = SheafFRL(
        agents={0: agent},
        neighbors={0: set()},
        optimizer={'_target_': 'torch.optim.SGD', 'lr': 0.1},
        lambda_sheaf=0.1,
        latent_dims={0: 3},
        anchor_strategy=anchor_strategy,
        parseval_normalization=False,
        l2_normalization=False,
        use_prototypes=use_prototypes,
    )
    return orchestrator, agent


def test_invalid_anchor_strategy_raises() -> None:
    with pytest.raises(ValueError, match='Unknown anchor_strategy'):
        _build_orchestrator('dynamic')


def test_batch_anchors_reuses_cached_task_latents_and_caches_prototypes() -> None:
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
    assert orchestrator.epoch_latents_cache[0][0].shape[0] == 2
    assert torch.equal(
        orchestrator.epoch_labels_cache[0][0],
        torch.tensor([0, 1], dtype=torch.long),
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


def test_batch_anchors_aligns_two_agents_via_class_prototypes() -> None:
    agent_0 = _CountingAgent(num_classes=3)
    agent_1 = _CountingAgent(num_classes=3)
    orchestrator = SheafFRL(
        agents={0: agent_0, 1: agent_1},
        neighbors={0: {1}, 1: {0}},
        optimizer={'_target_': 'torch.optim.SGD', 'lr': 0.1},
        lambda_sheaf=0.1,
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
    assert orchestrator.epoch_latents_cache[0][0].shape[0] == 2
    assert orchestrator.epoch_latents_cache[1][0].shape[0] == 2
    assert torch.equal(
        orchestrator.epoch_labels_cache[0][0],
        torch.tensor([0, 1], dtype=torch.long),
    )
    assert torch.equal(
        orchestrator.epoch_labels_cache[1][0],
        torch.tensor([1, 2], dtype=torch.long),
    )
    assert set(outputs) == {'0', '1'}
    assert total_loss.ndim == 0
