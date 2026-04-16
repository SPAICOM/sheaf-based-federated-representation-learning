"""Tests for src.orchestrators.heterofl."""

from types import SimpleNamespace

import pytest
import torch
from lightning.pytorch import LightningDataModule, Trainer
from torch.utils.data import DataLoader, TensorDataset

from src.agents.hetero_cnn_classifier import HeteroCNNClassifier
from src.orchestrators.heterofl import HeteroFL

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockOptimizer:
    """Minimal mock optimizer config accepted by hydra instantiate."""

    _target_ = 'torch.optim.Adam'
    lr = 0.001


def _make_agent(
    *,
    in_features: int = 3,
    num_classes: int = 5,
    hidden_dims: list[int] | None = None,
    rate: float = 1.0,
) -> HeteroCNNClassifier:
    return HeteroCNNClassifier(
        in_features=in_features,
        num_classes=num_classes,
        encoder_hidden_dims=hidden_dims or [4],
        rate=rate,
    )


def _fill_module(module: torch.nn.Module, value: float) -> None:
    for param in module.parameters():
        param.data.fill_(value)
    for buf in module.buffers():
        if torch.is_floating_point(buf):
            buf.data.fill_(value)


class _ToyDataModule(LightningDataModule):
    """Minimal datamodule yielding small image batches for two agents."""

    def setup(self, stage=None):
        self.train_datasets = {
            0: TensorDataset(
                torch.randn(16, 3, 16, 16),
                torch.randint(0, 5, (16,)),
            ),
            1: TensorDataset(
                torch.randn(16, 3, 16, 16),
                torch.randint(0, 5, (16,)),
            ),
        }

    def train_dataloader(self):
        return {
            idx: DataLoader(ds, batch_size=4)
            for idx, ds in self.train_datasets.items()
        }


class DummyDataset:
    def __init__(self, size: int):
        self.size = size

    def __len__(self) -> int:
        return self.size


class DummyDataModule:
    def __init__(self, counts: dict[int, int]):
        self.train_datasets = {k: DummyDataset(v) for k, v in counts.items()}


# ---------------------------------------------------------------------------
# Initialization and validation
# ---------------------------------------------------------------------------


class TestHeteroFLInit:
    """Initialization and agent-validation tests."""

    def test_basic_initialization(self):
        orc = HeteroFL(
            agents={0: _make_agent(rate=1.0), 1: _make_agent(rate=0.5)},
            neighbors={0: {1}, 1: {0}},
            optimizer=MockOptimizer(),
            rates={0: 1.0, 1: 0.5},
        )
        assert orc is not None

    def test_default_schedule_is_epoch_interval_one_no_cap(self):
        orc = HeteroFL(
            agents={0: _make_agent()},
            neighbors={0: set()},
            optimizer=MockOptimizer(),
            rates={0: 1.0},
        )
        assert orc.hparams.aggregation_interval == 1
        assert orc.hparams.aggregation_unit == 'epoch'
        assert orc.hparams.max_global_rounds is None

    def test_missing_encoder_attribute_raises(self):
        import torch.nn as nn

        class NoEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.decoder = nn.Linear(1, 1)

        with pytest.raises(TypeError, match='enc-dec'):
            HeteroFL(
                agents={0: NoEncoder()},
                neighbors={0: set()},
                optimizer=MockOptimizer(),
                rates={0: 1.0},
            )

    def test_missing_decoder_attribute_raises(self):
        import torch.nn as nn

        class NoDecoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Linear(1, 1)

        with pytest.raises(TypeError, match='enc-dec'):
            HeteroFL(
                agents={0: NoDecoder()},
                neighbors={0: set()},
                optimizer=MockOptimizer(),
                rates={0: 1.0},
            )

    def test_agent_missing_from_rates_raises(self):
        with pytest.raises(ValueError, match='rates'):
            HeteroFL(
                agents={0: _make_agent(), 1: _make_agent()},
                neighbors={0: {1}, 1: {0}},
                optimizer=MockOptimizer(),
                rates={0: 1.0},  # agent 1 absent
            )

    def test_rate_zero_raises(self):
        with pytest.raises(ValueError, match='Rate'):
            HeteroFL(
                agents={0: _make_agent()},
                neighbors={0: set()},
                optimizer=MockOptimizer(),
                rates={0: 0.0},
            )

    def test_rate_above_one_raises(self):
        with pytest.raises(ValueError, match='Rate'):
            HeteroFL(
                agents={0: _make_agent()},
                neighbors={0: set()},
                optimizer=MockOptimizer(),
                rates={0: 1.5},
            )

    def test_rates_stored_as_float(self):
        orc = HeteroFL(
            agents={0: _make_agent(), 1: _make_agent(rate=0.5)},
            neighbors={0: {1}, 1: {0}},
            optimizer=MockOptimizer(),
            rates={0: 1, 1: 0.5},  # int key / int value for agent 0
        )
        assert orc._rates[0] == 1.0
        assert isinstance(orc._rates[0], float)


# ---------------------------------------------------------------------------
# Schedule resolution
# ---------------------------------------------------------------------------


class TestHeteroFLScheduleResolution:
    """Tests for aggregation-schedule configuration and alias handling."""

    def test_legacy_local_steps_maps_to_step_unit(self):
        orc = HeteroFL(
            agents={0: _make_agent(), 1: _make_agent()},
            neighbors={0: {1}, 1: {0}},
            optimizer=MockOptimizer(),
            rates={0: 1.0, 1: 1.0},
            local_steps=5,
            global_steps=10,
        )
        assert orc.hparams.aggregation_interval == 5
        assert orc.hparams.aggregation_unit == 'step'
        assert orc.hparams.max_global_rounds == 10

    def test_invalid_aggregation_unit_raises(self):
        with pytest.raises(ValueError, match='aggregation_unit'):
            HeteroFL(
                agents={0: _make_agent()},
                neighbors={0: set()},
                optimizer=MockOptimizer(),
                rates={0: 1.0},
                aggregation_unit='batch',
            )

    def test_conflicting_local_steps_and_interval_raises(self):
        with pytest.raises(ValueError):
            HeteroFL(
                agents={0: _make_agent()},
                neighbors={0: set()},
                optimizer=MockOptimizer(),
                rates={0: 1.0},
                aggregation_interval=2,
                aggregation_unit='step',
                local_steps=3,
            )

    def test_conflicting_global_steps_and_max_rounds_raises(self):
        with pytest.raises(ValueError):
            HeteroFL(
                agents={0: _make_agent()},
                neighbors={0: set()},
                optimizer=MockOptimizer(),
                rates={0: 1.0},
                max_global_rounds=5,
                global_steps=5,
            )

    def test_explicit_step_schedule_stored_correctly(self):
        orc = HeteroFL(
            agents={0: _make_agent()},
            neighbors={0: set()},
            optimizer=MockOptimizer(),
            rates={0: 1.0},
            aggregation_interval=3,
            aggregation_unit='step',
            max_global_rounds=20,
        )
        assert orc.hparams.aggregation_interval == 3
        assert orc.hparams.aggregation_unit == 'step'
        assert orc.hparams.max_global_rounds == 20


# ---------------------------------------------------------------------------
# Neighbor aggregation correctness
# ---------------------------------------------------------------------------


class TestHeteroFLAggregation:
    """Unit tests for _neighbor_aggregate with controlled encoder values."""

    def test_decoder_not_modified_after_aggregation(self):
        agent0 = _make_agent(rate=1.0)
        agent1 = _make_agent(rate=0.5)
        _fill_module(agent0.encoder, 0.0)
        _fill_module(agent1.encoder, 2.0)
        decoder0_snapshot = {
            k: v.clone() for k, v in agent0.decoder.state_dict().items()
        }

        orc = HeteroFL(
            agents={0: agent0, 1: agent1},
            neighbors={0: {1}, 1: {0}},
            optimizer=MockOptimizer(),
            rates={0: 1.0, 1: 0.5},
            sample_counts={0: 1, 1: 1},
            sync_on_train_start=False,
        )
        orc._neighbor_aggregate()

        for k, v in agent0.decoder.state_dict().items():
            assert torch.equal(v, decoder0_snapshot[k]), (
                f'Decoder param {k} was modified by aggregation'
            )

    def test_uniform_rates_weighted_average_all_positions(self):
        """With identical rates, every position
        receives the weighted average."""

        agent0 = _make_agent(rate=1.0, hidden_dims=[4])
        agent1 = _make_agent(rate=1.0, hidden_dims=[4])
        _fill_module(agent0.encoder, 0.0)
        _fill_module(agent1.encoder, 4.0)

        orc = HeteroFL(
            agents={0: agent0, 1: agent1},
            neighbors={0: {1}, 1: {0}},
            optimizer=MockOptimizer(),
            rates={0: 1.0, 1: 1.0},
            sample_counts={0: 1, 1: 3},
            sync_on_train_start=False,
        )
        orc._neighbor_aggregate()

        # expected: (0.0*1 + 4.0*3) / (1+3) = 3.0 everywhere
        for param in orc.agents['0'].encoder.parameters():
            assert torch.allclose(param, torch.full_like(param, 3.0))
        for param in orc.agents['1'].encoder.parameters():
            assert torch.allclose(param, torch.full_like(param, 3.0))

    def test_narrow_node_covered_positions_are_averaged(self):
        """Narrow node's positions are all in the intersection — averaged
        correctly."""
        agent0 = _make_agent(rate=1.0, hidden_dims=[4])  # 4 out-channels
        agent1 = _make_agent(rate=0.5, hidden_dims=[4])  # 2 out-channels
        _fill_module(agent0.encoder, 0.0)
        _fill_module(agent1.encoder, 4.0)

        orc = HeteroFL(
            agents={0: agent0, 1: agent1},
            neighbors={0: {1}, 1: {0}},
            optimizer=MockOptimizer(),
            rates={0: 1.0, 1: 0.5},
            sample_counts={0: 1, 1: 1},
            sync_on_train_start=False,
        )
        orc._neighbor_aggregate()

        # Agent1 has 2 out-channels. Both nodes fully overlap at [0:2].
        # expected: (0.0*1 + 4.0*1) / (1+1) = 2.0 for all agent1 positions
        for param in orc.agents['1'].encoder.parameters():
            assert torch.allclose(param, torch.full_like(param, 2.0))

    def test_wide_node_partitioned_positions(self):
        """Wide node positions within intersection are averaged;
        positions outside intersection retain only the wide node's value."""
        agent0 = _make_agent(rate=1.0, hidden_dims=[4])  # 4 out-channels
        agent1 = _make_agent(rate=0.5, hidden_dims=[4])  # 2 out-channels
        _fill_module(agent0.encoder, 2.0)
        _fill_module(agent1.encoder, 8.0)

        orc = HeteroFL(
            agents={0: agent0, 1: agent1},
            neighbors={0: {1}, 1: {0}},
            optimizer=MockOptimizer(),
            rates={0: 1.0, 1: 0.5},
            sample_counts={0: 1, 1: 1},
            sync_on_train_start=False,
        )
        orc._neighbor_aggregate()

        # Inspect every parameter of the wide node.
        # dim-0 positions 0:2 overlap with agent1, averaged:(2*1+8*1)/(1+1)=5.0
        # dim-0 positions 2:4 are exclusive to agent0, unchanged: 2.0
        for param in orc.agents['0'].encoder.parameters():
            narrow_channels = 2  # int(4 * 0.5)
            shared = param[:narrow_channels]
            exclusive = param[narrow_channels:]
            assert torch.allclose(shared, torch.full_like(shared, 5.0)), (
                'Shared (intersection) positions should equal weighted average'
            )
            assert torch.allclose(
                exclusive, torch.full_like(exclusive, 2.0)
            ), 'Exclusive positions should retain the wide node value'

    def test_no_neighbors_self_only_keeps_values_unchanged(self):
        """A node with no neighbors averages only itself
        Parameters stay unchanged."""
        agent0 = _make_agent(rate=1.0, hidden_dims=[4])
        _fill_module(agent0.encoder, 5.0)
        original = {
            k: v.clone() for k, v in agent0.encoder.state_dict().items()
        }

        orc = HeteroFL(
            agents={0: agent0},
            neighbors={0: set()},
            optimizer=MockOptimizer(),
            rates={0: 1.0},
            sample_counts={0: 1},
            sync_on_train_start=False,
        )
        orc._neighbor_aggregate()

        for name, val in agent0.encoder.state_dict().items():
            assert torch.allclose(val, original[name])

    def test_aggregation_is_order_independent(self):
        """Results must not depend on the iteration order over agents."""

        def _run(agent_order: list[int]) -> dict[int, dict[str, torch.Tensor]]:
            agents = {
                i: _make_agent(rate=[1.0, 0.5][i], hidden_dims=[4])
                for i in agent_order
            }
            for i, a in agents.items():
                _fill_module(a.encoder, float(i))
            orc = HeteroFL(
                agents=agents,
                neighbors={0: {1}, 1: {0}},
                optimizer=MockOptimizer(),
                rates={0: 1.0, 1: 0.5},
                sample_counts={0: 1, 1: 1},
                sync_on_train_start=False,
            )
            orc._neighbor_aggregate()
            return {
                k: {
                    n: t.clone()
                    for n, t in orc.agents[str(k)].encoder.state_dict().items()
                }
                for k in [0, 1]
            }

        result_a = _run([0, 1])
        result_b = _run([1, 0])

        for k in [0, 1]:
            for name in result_a[k]:
                assert torch.allclose(result_a[k][name], result_b[k][name]), (
                    f'Order-dependent result for agent {k}, param {name}'
                )

    def test_zero_weight_neighbor_contributes_nothing(self):
        """A neighbor with sample_count=0 must not affect the aggregate."""
        agent0 = _make_agent(rate=1.0, hidden_dims=[4])
        agent1 = _make_agent(rate=1.0, hidden_dims=[4])
        _fill_module(agent0.encoder, 3.0)
        _fill_module(agent1.encoder, 99.0)  # should be suppressed

        orc = HeteroFL(
            agents={0: agent0, 1: agent1},
            neighbors={0: {1}, 1: {0}},
            optimizer=MockOptimizer(),
            rates={0: 1.0, 1: 1.0},
            sample_counts={0: 1, 1: 0},
            sync_on_train_start=False,
        )
        orc._neighbor_aggregate()

        for param in orc.agents['0'].encoder.parameters():
            assert torch.allclose(param, torch.full_like(param, 3.0))


# ---------------------------------------------------------------------------
# on_train_start behaviour
# ---------------------------------------------------------------------------


class TestHeteroFLTrainStart:
    """Tests for the on_train_start hook."""

    def test_sync_on_start_averages_encoder_values(self):
        agent0 = _make_agent(rate=1.0, hidden_dims=[4])
        agent1 = _make_agent(rate=1.0, hidden_dims=[4])
        _fill_module(agent0.encoder, 0.0)
        _fill_module(agent1.encoder, 4.0)

        orc = HeteroFL(
            agents={0: agent0, 1: agent1},
            neighbors={0: {1}, 1: {0}},
            optimizer=MockOptimizer(),
            rates={0: 1.0, 1: 1.0},
            sample_counts={0: 1, 1: 1},
            sync_on_train_start=True,
        )
        orc.on_train_start()

        for param in orc.agents['0'].encoder.parameters():
            assert torch.allclose(param, torch.full_like(param, 2.0))
        for param in orc.agents['1'].encoder.parameters():
            assert torch.allclose(param, torch.full_like(param, 2.0))

    def test_no_sync_on_start_leaves_encoders_unchanged(self):
        agent0 = _make_agent(rate=1.0, hidden_dims=[4])
        agent1 = _make_agent(rate=1.0, hidden_dims=[4])
        _fill_module(agent0.encoder, 0.0)
        _fill_module(agent1.encoder, 4.0)
        original = {
            k: v.clone() for k, v in agent0.encoder.state_dict().items()
        }

        orc = HeteroFL(
            agents={0: agent0, 1: agent1},
            neighbors={0: {1}, 1: {0}},
            optimizer=MockOptimizer(),
            rates={0: 1.0, 1: 1.0},
            sample_counts={0: 1, 1: 1},
            sync_on_train_start=False,
        )
        orc.on_train_start()

        for k, v in agent0.encoder.state_dict().items():
            assert torch.equal(v, original[k])

    def test_on_train_start_infers_sample_counts_from_datamodule(self):
        orc = HeteroFL(
            agents={0: _make_agent(), 1: _make_agent()},
            neighbors={0: {1}, 1: {0}},
            optimizer=MockOptimizer(),
            rates={0: 1.0, 1: 1.0},
            sync_on_train_start=False,
        )
        orc._trainer = SimpleNamespace(
            datamodule=DummyDataModule({0: 40, 1: 160}),
            should_stop=False,
        )
        orc.on_train_start()

        counts = orc._infer_client_sample_counts()
        assert counts[0] == 40
        assert counts[1] == 160


# ---------------------------------------------------------------------------
# Epoch-based aggregation hook
# ---------------------------------------------------------------------------


class TestHeteroFLEpochHook:
    """Tests for on_train_epoch_end scheduling behaviour."""

    def test_does_not_aggregate_before_interval_is_reached(self):
        orc = HeteroFL(
            agents={0: _make_agent(), 1: _make_agent()},
            neighbors={0: {1}, 1: {0}},
            optimizer=MockOptimizer(),
            rates={0: 1.0, 1: 1.0},
            aggregation_interval=2,
            aggregation_unit='epoch',
            sync_on_train_start=False,
        )
        calls = []
        orc._synchronize = lambda: calls.append(True)
        orc.log_dict = lambda *a, **kw: None
        orc._trainer = SimpleNamespace(current_epoch=0, should_stop=False)

        orc.on_train_epoch_end()  # epoch 1 — not a multiple of 2
        assert calls == []

    def test_aggregates_when_interval_is_reached(self):
        orc = HeteroFL(
            agents={0: _make_agent(), 1: _make_agent()},
            neighbors={0: {1}, 1: {0}},
            optimizer=MockOptimizer(),
            rates={0: 1.0, 1: 1.0},
            aggregation_interval=2,
            aggregation_unit='epoch',
            sync_on_train_start=False,
        )
        calls = []
        orc._synchronize = lambda: calls.append(True)
        orc.log_dict = lambda *a, **kw: None
        orc._trainer = SimpleNamespace(current_epoch=0, should_stop=False)

        orc.on_train_epoch_end()  # epoch 1
        assert calls == []

        orc._trainer.current_epoch = 1
        orc.on_train_epoch_end()  # epoch 2 — fire
        assert len(calls) == 1
        assert orc._global_aggregation_count == 1

    def test_does_not_aggregate_twice_for_same_epoch(self):
        orc = HeteroFL(
            agents={0: _make_agent(), 1: _make_agent()},
            neighbors={0: {1}, 1: {0}},
            optimizer=MockOptimizer(),
            rates={0: 1.0, 1: 1.0},
            aggregation_interval=1,
            aggregation_unit='epoch',
            sync_on_train_start=False,
        )
        calls = []
        orc._synchronize = lambda: calls.append(True)
        orc.log_dict = lambda *a, **kw: None
        orc._trainer = SimpleNamespace(current_epoch=0, should_stop=False)

        orc.on_train_epoch_end()
        orc.on_train_epoch_end()  # same epoch again — must not re-fire
        assert len(calls) == 1

    def test_stops_trainer_after_global_round_cap(self):
        orc = HeteroFL(
            agents={0: _make_agent(), 1: _make_agent()},
            neighbors={0: {1}, 1: {0}},
            optimizer=MockOptimizer(),
            rates={0: 1.0, 1: 1.0},
            aggregation_interval=1,
            aggregation_unit='epoch',
            max_global_rounds=1,
            sync_on_train_start=False,
        )
        orc._synchronize = lambda: None
        orc.log_dict = lambda *a, **kw: None
        orc._trainer = SimpleNamespace(current_epoch=0, should_stop=False)

        orc.on_train_epoch_end()

        assert orc._global_aggregation_count == 1
        assert orc._trainer.should_stop is True

    def test_epoch_hook_ignores_step_unit(self):
        """Epoch hook must be silent when unit is 'step'."""
        orc = HeteroFL(
            agents={0: _make_agent(), 1: _make_agent()},
            neighbors={0: {1}, 1: {0}},
            optimizer=MockOptimizer(),
            rates={0: 1.0, 1: 1.0},
            aggregation_interval=1,
            aggregation_unit='step',
            sync_on_train_start=False,
        )
        calls = []
        orc._synchronize = lambda: calls.append(True)
        orc.log_dict = lambda *a, **kw: None
        orc._trainer = SimpleNamespace(current_epoch=0, should_stop=False)

        orc.on_train_epoch_end()
        assert calls == []


# ---------------------------------------------------------------------------
# Step-based aggregation hook
# ---------------------------------------------------------------------------


class TestHeteroFLStepHook:
    """Tests for on_train_batch_end scheduling behaviour."""

    def test_does_not_aggregate_before_step_interval(self):
        orc = HeteroFL(
            agents={0: _make_agent(), 1: _make_agent()},
            neighbors={0: {1}, 1: {0}},
            optimizer=MockOptimizer(),
            rates={0: 1.0, 1: 1.0},
            aggregation_interval=2,
            aggregation_unit='step',
            sync_on_train_start=False,
        )
        calls = []
        orc._synchronize = lambda: calls.append(True)
        orc._trainer = SimpleNamespace(global_step=1, should_stop=False)

        orc.on_train_batch_end(None, {}, 0)
        assert calls == []

    def test_aggregates_at_step_interval(self):
        orc = HeteroFL(
            agents={0: _make_agent(), 1: _make_agent()},
            neighbors={0: {1}, 1: {0}},
            optimizer=MockOptimizer(),
            rates={0: 1.0, 1: 1.0},
            aggregation_interval=2,
            aggregation_unit='step',
            sync_on_train_start=False,
        )
        calls = []
        orc._synchronize = lambda: calls.append(True)
        orc._trainer = SimpleNamespace(global_step=1, should_stop=False)

        orc.on_train_batch_end(None, {}, 0)
        assert calls == []

        orc._trainer.global_step = 2
        orc.on_train_batch_end(None, {}, 1)
        assert len(calls) == 1
        assert orc._global_aggregation_count == 1

    def test_does_not_aggregate_twice_for_same_step(self):
        orc = HeteroFL(
            agents={0: _make_agent(), 1: _make_agent()},
            neighbors={0: {1}, 1: {0}},
            optimizer=MockOptimizer(),
            rates={0: 1.0, 1: 1.0},
            aggregation_interval=1,
            aggregation_unit='step',
            sync_on_train_start=False,
        )
        calls = []
        orc._synchronize = lambda: calls.append(True)
        orc._trainer = SimpleNamespace(global_step=1, should_stop=False)

        orc.on_train_batch_end(None, {}, 0)
        orc.on_train_batch_end(None, {}, 0)  # same step again
        assert len(calls) == 1

    def test_stops_trainer_after_global_round_cap(self):
        orc = HeteroFL(
            agents={0: _make_agent(), 1: _make_agent()},
            neighbors={0: {1}, 1: {0}},
            optimizer=MockOptimizer(),
            rates={0: 1.0, 1: 1.0},
            aggregation_interval=1,
            aggregation_unit='step',
            max_global_rounds=1,
            sync_on_train_start=False,
        )
        orc._synchronize = lambda: None
        orc._trainer = SimpleNamespace(global_step=1, should_stop=False)

        orc.on_train_batch_end(None, {}, 0)

        assert orc._global_aggregation_count == 1
        assert orc._trainer.should_stop is True

    def test_step_hook_ignores_epoch_unit(self):
        """Step hook must be silent when unit is 'epoch'."""
        orc = HeteroFL(
            agents={0: _make_agent(), 1: _make_agent()},
            neighbors={0: {1}, 1: {0}},
            optimizer=MockOptimizer(),
            rates={0: 1.0, 1: 1.0},
            aggregation_interval=1,
            aggregation_unit='epoch',
            sync_on_train_start=False,
        )
        calls = []
        orc._synchronize = lambda: calls.append(True)
        orc._trainer = SimpleNamespace(global_step=1, should_stop=False)

        orc.on_train_batch_end(None, {}, 0)
        assert calls == []


# ---------------------------------------------------------------------------
# Sample-count inference
# ---------------------------------------------------------------------------


class TestHeteroFLSampleCounts:
    """Tests for sample-count inference and normalisation."""

    def test_explicit_counts_bypass_inference(self):
        orc = HeteroFL(
            agents={0: _make_agent(), 1: _make_agent()},
            neighbors={0: {1}, 1: {0}},
            optimizer=MockOptimizer(),
            rates={0: 1.0, 1: 1.0},
            sample_counts={0: 100, 1: 200},
            sync_on_train_start=False,
        )
        counts = orc._infer_client_sample_counts()
        assert counts == {0: 100, 1: 200}

    def test_infer_from_datamodule(self):
        orc = HeteroFL(
            agents={0: _make_agent(), 1: _make_agent()},
            neighbors={0: {1}, 1: {0}},
            optimizer=MockOptimizer(),
            rates={0: 1.0, 1: 1.0},
            sync_on_train_start=False,
        )
        orc._trainer = SimpleNamespace(
            datamodule=DummyDataModule({0: 50, 1: 150}),
            should_stop=False,
        )
        counts = orc._infer_client_sample_counts()
        assert counts[0] == 50
        assert counts[1] == 150

    def test_fallback_to_uniform_when_no_datamodule(self):
        orc = HeteroFL(
            agents={0: _make_agent(), 1: _make_agent()},
            neighbors={0: {1}, 1: {0}},
            optimizer=MockOptimizer(),
            rates={0: 1.0, 1: 1.0},
            sync_on_train_start=False,
        )
        orc._trainer = SimpleNamespace(datamodule=None, should_stop=False)
        counts = orc._infer_client_sample_counts()
        assert counts[0] == counts[1] == 1


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


class TestHeteroFLSharedEval:
    """Tests for _shared_eval."""

    def test_returns_scalar_loss_tensor(self):
        agent = _make_agent(rate=1.0)
        orc = HeteroFL(
            agents={0: agent},
            neighbors={0: set()},
            optimizer=MockOptimizer(),
            rates={0: 1.0},
        )
        x = torch.randn(4, 3, 16, 16)
        y = torch.randint(0, 5, (4,))
        _, loss = orc._shared_eval({0: [x, y]}, 0, 'train')

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0

    def test_returns_nonnegative_loss(self):
        agent = _make_agent(rate=1.0)
        orc = HeteroFL(
            agents={0: agent},
            neighbors={0: set()},
            optimizer=MockOptimizer(),
            rates={0: 1.0},
        )
        x = torch.randn(4, 3, 16, 16)
        y = torch.randint(0, 5, (4,))
        _, loss = orc._shared_eval({0: [x, y]}, 0, 'train')

        assert loss.item() >= 0.0


# ---------------------------------------------------------------------------
# Trainer integration
# ---------------------------------------------------------------------------


class TestHeteroFLTrainerIntegration:
    """End-to-end test running a full Lightning training loop."""

    def test_trainer_logs_nonzero_train_communication_metrics(self):
        """A one-epoch fit should produce positive kilobytes and one round."""
        orc = HeteroFL(
            agents={
                0: _make_agent(rate=1.0, hidden_dims=[4]),
                1: _make_agent(rate=0.5, hidden_dims=[4]),
            },
            neighbors={0: {1}, 1: {0}},
            optimizer={'_target_': 'torch.optim.SGD', 'lr': 0.01},
            rates={0: 1.0, 1: 0.5},
        )
        trainer = Trainer(
            max_epochs=1,
            accelerator='cpu',
            logger=False,
            enable_checkpointing=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
            limit_val_batches=0,
        )
        trainer.fit(orc, datamodule=_ToyDataModule())

        assert (
            float(trainer.callback_metrics['train/communication_kilobytes'])
            > 0.0
        )
        assert (
            float(trainer.callback_metrics['train/communication_rounds'])
            == 1.0
        )

    @pytest.mark.slow
    def test_train_one_epoch_with_two_hetero_agents(self):
        """Full training epoch with heterogeneous agents
        completes without error."""
        orc = HeteroFL(
            agents={
                0: _make_agent(rate=1.0, hidden_dims=[8, 16]),
                1: _make_agent(rate=0.5, hidden_dims=[8, 16]),
            },
            neighbors={0: {1}, 1: {0}},
            optimizer={'_target_': 'torch.optim.Adam', 'lr': 0.001},
            rates={0: 1.0, 1: 0.5},
            aggregation_interval=1,
            aggregation_unit='epoch',
        )
        trainer = Trainer(
            max_epochs=2,
            accelerator='cpu',
            logger=False,
            enable_checkpointing=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
            limit_val_batches=0,
        )
        trainer.fit(orc, datamodule=_ToyDataModule())
        assert (
            float(trainer.callback_metrics['train/communication_rounds'])
            == 2.0
        )
