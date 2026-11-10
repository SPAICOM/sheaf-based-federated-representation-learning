"""Tests for src.orchestrators.fedre."""

from types import SimpleNamespace

import pytest
import torch

from src.agents.latent_classifier import LatentClassifier
from src.orchestrators.fedre import FedRE


class MockOptimizer:
    _target_ = 'torch.optim.SGD'
    lr = 0.1


def _make_agent(decoder_hidden_dims=None, seed=0) -> LatentClassifier:
    torch.manual_seed(seed)
    return LatentClassifier(
        in_features=8,
        num_classes=4,
        latent_dim=4,
        encoder_hidden_dims=[6],
        decoder_hidden_dims=decoder_hidden_dims,
    )


def _dataset(classes: list[int], n: int = 12):
    return [(torch.randn(8), classes[i % len(classes)]) for i in range(n)]


def _make_orchestrator(**kwargs) -> FedRE:
    orchestrator = FedRE(
        agents={0: _make_agent(seed=0), 1: _make_agent(seed=1)},
        neighbors={0: {1}, 1: {0}},
        optimizer=MockOptimizer(),
        **kwargs,
    )
    orchestrator.on_train_start()
    return orchestrator


_DM = SimpleNamespace(train_datasets={0: _dataset([0, 1]), 1: _dataset([1, 2])})


def _heads_equal(a, b) -> bool:
    return all(
        torch.equal(x, y)
        for x, y in zip(a.state_dict().values(), b.state_dict().values())
    )


class TestFedRE:
    def test_rejects_different_decoders(self):
        with pytest.raises(ValueError, match='identical decoder'):
            FedRE(
                agents={
                    0: _make_agent(decoder_hidden_dims=[8]),
                    1: _make_agent(decoder_hidden_dims=[6]),
                },
                neighbors={0: {1}, 1: {0}},
                optimizer=MockOptimizer(),
            )

    def test_entangle_is_a_convex_combination_of_prototypes(self):
        Z = torch.tensor([[1.0, 0.0], [3.0, 0.0], [0.0, 4.0], [0.0, 4.0]])
        y = torch.tensor([1, 1, 3, 3])

        rep, soft_label = FedRE._entangle(Z, y, num_classes=5)

        assert soft_label.sum().item() == pytest.approx(1.0)
        assert soft_label[[0, 2, 4]].abs().sum() == 0
        # Prototypes are (2, 0) and (0, 4), mixed with the label's weights.
        expected = soft_label[1] * Z[:2].mean(0) + soft_label[3] * Z[2:].mean(0)
        assert torch.allclose(rep, expected)

    def test_entangle_resamples_weights(self):
        Z, y = torch.randn(8, 3), torch.tensor([0, 1] * 4)
        _, first = FedRE._entangle(Z, y, num_classes=2)
        _, second = FedRE._entangle(Z, y, num_classes=2)
        assert not torch.equal(first, second)

    def test_server_head_is_not_a_local_parameter(self):
        orchestrator = _make_orchestrator()
        head = set(map(id, orchestrator._server_head.parameters()))
        assert not head & set(map(id, orchestrator.parameters()))

    def test_train_start_shares_agent_zero_head(self):
        orchestrator = _make_orchestrator()
        assert _heads_equal(
            orchestrator.agents['0'].decoder, orchestrator.agents['1'].decoder
        )
        assert _heads_equal(
            orchestrator.agents['0'].decoder, orchestrator._server_head
        )

    def test_server_step_minimizes_soft_cross_entropy(self):
        orchestrator = _make_orchestrator(server_lr=0.5)
        reps = torch.randn(2, 4)
        soft = torch.tensor([[0.3, 0.7, 0.0, 0.0], [0.0, 0.2, 0.8, 0.0]])
        losses = [
            orchestrator._train_server_head(reps, soft).item()
            for _ in range(30)
        ]
        assert losses[-1] < losses[0]

    def test_round_updates_head_broadcasts_and_keeps_encoders(self):
        orchestrator = _make_orchestrator()
        before_head = {
            k: v.clone()
            for k, v in orchestrator._server_head.state_dict().items()
        }
        before_encoder = {
            k: v.clone()
            for k, v in orchestrator.agents['1'].encoder.state_dict().items()
        }
        kb_before = orchestrator._communication_by_split['train']['kilobytes']

        orchestrator._server_round(_DM)

        changed = orchestrator._server_head.state_dict()
        assert any(not torch.equal(before_head[k], changed[k]) for k in changed)
        for agent in orchestrator.agents.values():
            assert _heads_equal(agent.decoder, orchestrator._server_head)
        after_encoder = orchestrator.agents['1'].encoder.state_dict()
        assert all(torch.equal(before_encoder[k], after_encoder[k]) for k in after_encoder)

        split = orchestrator._communication_by_split['train']
        assert split['kilobytes'] > kb_before
        assert split['rounds'] == 1

    def test_local_backward_then_round_then_backward(self):
        orchestrator = _make_orchestrator()
        optimizer = torch.optim.SGD(orchestrator.parameters(), lr=0.1)
        batch = {
            0: [torch.randn(4, 8), torch.tensor([0, 1, 0, 1])],
            1: [torch.randn(4, 8), torch.tensor([1, 2, 1, 2])],
        }
        for _ in range(2):
            outputs = orchestrator(batch)
            loss = sum(
                orchestrator.agents[idx].compute_loss(y_hat, y)
                for idx, (y_hat, y) in outputs.items()
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            orchestrator._server_round(_DM)

    def test_missing_agent_data_raises(self):
        orchestrator = _make_orchestrator()
        dm = SimpleNamespace(train_datasets={0: _dataset([0, 1])})
        with pytest.raises(ValueError, match='No training data'):
            orchestrator._server_round(dm)

    def test_native_message_is_identity(self):
        orchestrator = _make_orchestrator()
        Z = torch.randn(3, 4)
        assert orchestrator.hparams.alignment_method is None
        assert torch.equal(orchestrator.send_message(0, 1, Z), Z)
