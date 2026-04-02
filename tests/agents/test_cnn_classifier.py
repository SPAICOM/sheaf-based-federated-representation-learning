"""Tests for src.agents.cnn_classifier."""

import torch

from src.agents.cnn_classifier import CNNClassifier
from src.agents.personalized_classifier import PersonalizedClassifier
from src.agents.utils import CNN


class TestCNNClassifier:
    def _make(self, **kwargs) -> CNNClassifier:
        return CNNClassifier(**{'in_features': 3, 'num_classes': 10, **kwargs})

    def _x(self, batch: int = 4) -> torch.Tensor:
        return torch.randn(batch, 3, 16, 16)

    # --- inheritance ---

    def test_is_personalized_classifier(self):
        assert isinstance(self._make(), PersonalizedClassifier)

    def test_encoder_is_cnn(self):
        assert isinstance(self._make().encoder, CNN)

    # --- construction ---

    def test_default_encoder_hidden_dims(self):
        """Default encoder uses [32, 64, 128]; out_features is 128."""
        agent = self._make()
        assert agent.encoder.out_features == 128

    def test_custom_encoder_hidden_dims(self):
        agent = self._make(encoder_hidden_dims=[16, 32])
        assert agent.encoder.out_features == 32

    # --- forward / encode ---

    def test_forward_shape(self):
        assert self._make()(self._x()).shape == (4, 10)

    def test_encode_returns_flat(self):
        latent = self._make().encode(self._x())
        assert latent.ndim == 2
        assert latent.shape == (4, 128)

    def test_encode_custom_dims(self):
        agent = self._make(encoder_hidden_dims=[16, 32])
        assert agent.encode(self._x()).shape == (4, 32)

    # --- loss / performance ---

    def test_compute_loss_scalar(self):
        agent = self._make()
        loss = agent.compute_loss(agent(self._x()), torch.randint(0, 10, (4,)))
        assert loss.ndim == 0 and loss.item() >= 0

    def test_task_performance_range(self):
        agent = self._make()
        acc = agent.task_performance(
            agent(self._x()), torch.randint(0, 10, (4,))
        )
        assert 0.0 <= float(acc) <= 1.0

    # --- gradients ---

    def test_gradient_flow(self):
        agent = self._make()
        x = torch.randn(4, 3, 16, 16, requires_grad=True)
        agent(x).sum().backward()
        assert x.grad is not None
