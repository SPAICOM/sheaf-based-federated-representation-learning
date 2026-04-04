"""Tests for src.orchestrators.fedper."""

from types import SimpleNamespace

import torch

from src.agents.latent_classifier import LatentClassifier
from src.orchestrators.fedper import FedPer


class MockOptimizer:
    """Mock optimizer config for testing."""

    _target_ = 'torch.optim.Adam'
    lr = 0.001


def _make_agent(
    *,
    decoder_hidden_dims: list[int] | None = None,
) -> LatentClassifier:
    return LatentClassifier(
        in_features=8,
        num_classes=3,
        latent_dim=4,
        encoder_hidden_dims=[6],
        decoder_hidden_dims=decoder_hidden_dims,
    )


def _fill_module(module, value: float) -> None:
    for parameter in module.parameters():
        parameter.data.fill_(value)
    for buffer in module.buffers():
        if torch.is_floating_point(buffer):
            buffer.data.fill_(value)


class DummyDataset:
    def __init__(self, size: int):
        self.size = size

    def __len__(self) -> int:
        return self.size


class DummyDataModule:
    def __init__(self, counts: dict[int, int]):
        self.train_datasets = {
            agent_idx: DummyDataset(size) for agent_idx, size in counts.items()
        }


class TestFedPer:
    def test_initialization_allows_personalized_heads(self):
        orchestrator = FedPer(
            agents={
                0: _make_agent(decoder_hidden_dims=[8]),
                1: _make_agent(decoder_hidden_dims=[12, 8]),
            },
            neighbors={0: {1}, 1: {0}},
            optimizer=MockOptimizer(),
        )
        assert orchestrator is not None

    def test_synchronize_base_layers_only_updates_encoder(self):
        agent0 = _make_agent(decoder_hidden_dims=[8])
        agent1 = _make_agent(decoder_hidden_dims=[12, 8])

        _fill_module(agent0.encoder, 0.0)
        _fill_module(agent1.encoder, 2.0)
        _fill_module(agent0.decoder, 10.0)
        _fill_module(agent1.decoder, 20.0)

        orchestrator = FedPer(
            agents={0: agent0, 1: agent1},
            neighbors={0: {1}, 1: {0}},
            optimizer=MockOptimizer(),
            sample_counts={0: 1, 1: 3},
            sync_on_train_start=False,
        )

        orchestrator._synchronize_base_layers()

        for parameter in orchestrator.agents['0'].encoder.parameters():
            assert torch.allclose(parameter, torch.full_like(parameter, 1.5))
        for parameter in orchestrator.agents['1'].encoder.parameters():
            assert torch.allclose(parameter, torch.full_like(parameter, 1.5))

        for parameter in orchestrator.agents['0'].decoder.parameters():
            assert torch.allclose(parameter, torch.full_like(parameter, 10.0))
        for parameter in orchestrator.agents['1'].decoder.parameters():
            assert torch.allclose(parameter, torch.full_like(parameter, 20.0))

    def test_on_train_start_infers_sample_counts_and_syncs(self):
        agent0 = _make_agent()
        agent1 = _make_agent()
        _fill_module(agent0.encoder, 1.0)
        _fill_module(agent1.encoder, 3.0)

        orchestrator = FedPer(
            agents={0: agent0, 1: agent1},
            neighbors={0: {1}, 1: {0}},
            optimizer=MockOptimizer(),
            sync_on_train_start=True,
        )
        orchestrator._trainer = SimpleNamespace(
            datamodule=DummyDataModule({0: 1, 1: 3}),
            should_stop=False,
        )

        orchestrator.on_train_start()

        for parameter in orchestrator.agents['0'].encoder.parameters():
            assert torch.allclose(parameter, torch.full_like(parameter, 2.5))
        for parameter in orchestrator.agents['1'].encoder.parameters():
            assert torch.allclose(parameter, torch.full_like(parameter, 2.5))

    def test_on_train_batch_end_aggregates_on_schedule_and_stops(self):
        orchestrator = FedPer(
            agents={0: _make_agent(), 1: _make_agent()},
            neighbors={0: {1}, 1: {0}},
            optimizer=MockOptimizer(),
            local_steps=2,
            global_steps=1,
            sync_on_train_start=False,
        )

        calls = []

        def _mark_sync():
            calls.append(orchestrator.global_step)

        orchestrator._synchronize_base_layers = _mark_sync
        orchestrator._trainer = SimpleNamespace(
            global_step=1,
            should_stop=False,
            datamodule=None,
        )

        orchestrator.on_train_batch_end(None, {}, 0)
        assert calls == []
        assert orchestrator._trainer.should_stop is False

        orchestrator._trainer.global_step = 2
        orchestrator.on_train_batch_end(None, {}, 1)

        assert calls == [2]
        assert orchestrator._global_aggregation_count == 1
        assert orchestrator._trainer.should_stop is True
