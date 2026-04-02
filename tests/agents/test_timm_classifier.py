"""Tests for src.agents.timm_classifier."""

import torch

from src.agents.timm_classifier import TimmClassifier


class TestTimmClassifier:
    """Tests for TimmClassifier class."""

    def test_initialization(self):
        """Test basic initialization with small model."""
        agent = TimmClassifier(
            model_name='mobilenetv3_small_100',
            num_classes=10,
            pretrained=False,
        )
        assert agent is not None

    def test_forward(self):
        """Test forward returns logits."""
        agent = TimmClassifier(
            model_name='mobilenetv3_small_100',
            num_classes=10,
            pretrained=False,
        )
        x = torch.randn(4, 3, 224, 224)
        logits = agent(x)
        assert logits.shape == (4, 10)

    def test_encode(self):
        """Test encode returns latent features."""
        agent = TimmClassifier(
            model_name='mobilenetv3_small_100',
            num_classes=10,
            pretrained=False,
        )
        x = torch.randn(4, 3, 224, 224)
        features = agent.encode(x)
        # MobileNetV3 small outputs 576 features
        assert features.shape[0] == 4
        assert features.ndim == 2

    def test_encode_resizes_non_224(self):
        """Test encode handles non-224x224 inputs."""
        agent = TimmClassifier(
            model_name='mobilenetv3_small_100',
            num_classes=10,
            pretrained=False,
        )
        # 64x64 input should be resized to 224x224
        x = torch.randn(4, 3, 64, 64)
        features = agent.encode(x)
        assert features.shape[0] == 4

    def test_compute_loss(self):
        """Test loss computation."""
        agent = TimmClassifier(
            model_name='mobilenetv3_small_100',
            num_classes=10,
            pretrained=False,
        )
        x = torch.randn(4, 3, 224, 224)
        y = torch.randint(0, 10, (4,))
        logits = agent(x)
        loss = agent.compute_loss(logits, y)
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0

    def test_task_performance(self):
        """Test task performance returns accuracy."""
        agent = TimmClassifier(
            model_name='mobilenetv3_small_100',
            num_classes=10,
            pretrained=False,
        )
        x = torch.randn(8, 3, 224, 224)
        y = torch.randint(0, 10, (8,))
        logits = agent(x)
        accuracy = agent.task_performance(logits, y)
        assert isinstance(accuracy, torch.Tensor)
        assert 0 <= accuracy.item() <= 1
