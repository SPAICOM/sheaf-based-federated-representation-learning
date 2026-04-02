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
