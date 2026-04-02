"""Tests for src.agents.latent_classifier."""

import torch

from src.agents.latent_classifier import LatentClassifier


class TestLatentClassifier:
    """Tests for LatentClassifier class."""

    def test_initialization(self):
        """Test basic initialization."""
        agent = LatentClassifier(
            in_features=128, num_classes=10, latent_dim=64
        )
        assert agent is not None

    def test_forward(self):
        """Test forward returns logits."""
        agent = LatentClassifier(
            in_features=128, num_classes=10, latent_dim=64
        )
        x = torch.randn(32, 128)
        logits = agent(x)
        assert logits.shape == (32, 10)

    def test_encode(self):
        """Test encode returns latent features."""
        agent = LatentClassifier(
            in_features=128, num_classes=10, latent_dim=64
        )
        x = torch.randn(32, 128)
        latent = agent.encode(x)
        assert latent.shape == (32, 64)

    def test_compute_loss(self):
        """Test loss computation."""
        agent = LatentClassifier(
            in_features=128, num_classes=10, latent_dim=64
        )
        x = torch.randn(32, 128)
        y = torch.randint(0, 10, (32,))
        logits = agent(x)
        loss = agent.compute_loss(logits, y)
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0

    def test_task_performance(self):
        """Test task performance returns accuracy."""
        agent = LatentClassifier(
            in_features=128, num_classes=10, latent_dim=64
        )
        x = torch.randn(32, 128)
        y = torch.randint(0, 10, (32,))
        logits = agent(x)
        accuracy = agent.task_performance(logits, y)
        assert isinstance(accuracy, torch.Tensor)
        assert 0 <= accuracy.item() <= 1
