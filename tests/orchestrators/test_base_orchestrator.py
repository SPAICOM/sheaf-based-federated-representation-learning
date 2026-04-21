"""Tests for src.orchestrators.base_orchestrator."""

import pytest
import torch
import torch.nn as nn

from src.orchestrators.base_orchestrator import BaseOrchestrator


class DummyAgent(nn.Module):
    """Dummy agent for testing."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        return self.fc(x)

    def encode(self, x):
        return x

    def compute_loss(self, y_hat, y):
        return nn.functional.cross_entropy(y_hat, y)

    def task_performance(self, y_hat, y):
        return (torch.argmax(y_hat, dim=1) == y).float().mean()


class ConcreteOrchestrator(BaseOrchestrator):
    """Concrete implementation for testing."""

    def __init__(self, agents, neighbors, optimizer):
        super().__init__(
            agents=agents, neighbors=neighbors, optimizer=optimizer
        )

    def on_train_epoch_end(self):
        pass

    def _shared_eval(self, batch, batch_idx, prefix):
        outputs = self(batch)
        total_loss = 0
        for idx in outputs:
            y_hat, y = outputs[idx]
            loss = self.agents[idx].compute_loss(y_hat, y)
            total_loss += loss
        return outputs, total_loss


class MockOptimizer:
    """Mock optimizer config for testing."""

    _target_ = 'torch.optim.Adam'
    lr = 0.001


class TestBaseOrchestrator:
    """Tests for BaseOrchestrator class."""

    def test_initialization(self):
        """Test basic initialization."""
        agent = DummyAgent()
        agents = {0: agent}
        neighbors = {0: set()}

        orchestrator = ConcreteOrchestrator(
            agents=agents,
            neighbors=neighbors,
            optimizer=MockOptimizer(),
        )
        assert orchestrator is not None

    def test_empty_agents_raises(self):
        """Test that empty agents raises assertion."""
        with pytest.raises(AssertionError):
            ConcreteOrchestrator(
                agents={},
                neighbors={},
                optimizer=MockOptimizer(),
            )

    def test_forward(self):
        """Test forward pass."""
        agent = DummyAgent()
        orchestrator = ConcreteOrchestrator(
            agents={0: agent},
            neighbors={0: set()},
            optimizer=MockOptimizer(),
        )

        x = torch.randn(8, 128)
        y = torch.randint(0, 10, (8,))
        batch = {'0': (x, y)}
        outputs = orchestrator(batch)
        assert '0' in outputs

    def test_training_step(self):
        """Test training step returns loss."""
        agent = DummyAgent()
        orchestrator = ConcreteOrchestrator(
            agents={0: agent},
            neighbors={0: set()},
            optimizer=MockOptimizer(),
        )

        x = torch.randn(8, 128)
        y = torch.randint(0, 10, (8,))
        batch = {'0': (x, y)}
        loss = orchestrator.training_step(batch, batch_idx=0)
        assert isinstance(loss, torch.Tensor)

    def test_communication_accounting_is_split_by_stage(self):
        """Train/test communication counters should be tracked separately."""
        agent = DummyAgent()
        orchestrator = ConcreteOrchestrator(
            agents={0: agent},
            neighbors={0: set()},
            optimizer=MockOptimizer(),
        )

        orchestrator.on_train_start()
        orchestrator._record_communication_round(prefix='train')
        orchestrator._record_communication(
            torch.ones(4),
            n_transmissions=2,
            prefix='train',
        )

        orchestrator.on_test_start()
        orchestrator._record_communication_round(prefix='test')
        orchestrator._record_communication(
            torch.ones(2),
            n_transmissions=1,
            prefix='test',
        )

        train_metrics = orchestrator._communication_metrics('train')
        test_metrics = orchestrator._communication_metrics('test')

        assert train_metrics['train/communication_rounds'] == 1.0
        assert test_metrics['test/communication_rounds'] == 1.0
        assert (
            train_metrics['train/communication_kilobytes']
            > test_metrics['test/communication_kilobytes']
        )

    def test_eval_logs_include_cumulative_train_communication(self):
        """Test-monitor/test logs should include cumulative train budget."""
        agent = DummyAgent()
        orchestrator = ConcreteOrchestrator(
            agents={0: agent},
            neighbors={0: set()},
            optimizer=MockOptimizer(),
        )

        logged_metrics = []

        def capture_log_dict(metrics, **kwargs):
            logged_metrics.append(dict(metrics))

        orchestrator.log_dict = capture_log_dict

        orchestrator.on_train_start()
        orchestrator._record_communication_round(prefix='train')
        orchestrator._record_communication(
            torch.ones(4),
            n_transmissions=2,
            prefix='train',
        )

        train_metrics = orchestrator._communication_metrics('train')

        orchestrator.on_validation_start()
        orchestrator._validation_prefixes_seen.add('test_monitor')
        orchestrator.on_validation_epoch_end()

        orchestrator.on_test_start()
        orchestrator.on_test_epoch_end()

        assert {
            'test_monitor/train_communication_kilobytes_cumulative': (
                train_metrics['train/communication_kilobytes']
            ),
            'test_monitor/train_communication_rounds_cumulative': 1.0,
        } in logged_metrics
        assert {
            'test/train_communication_kilobytes_cumulative': (
                train_metrics['train/communication_kilobytes']
            ),
            'test/train_communication_rounds_cumulative': 1.0,
        } in logged_metrics
