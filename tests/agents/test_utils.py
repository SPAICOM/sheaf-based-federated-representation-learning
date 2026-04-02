"""Tests for src.agents.utils."""

import torch

from src.agents.utils import CNN, MLP


class TestCNN:
    """Tests for CNN encoder."""

    def test_out_features(self):
        """out_features matches the last hidden dim."""
        enc = CNN(in_features=3, hidden_dims=[16, 32])
        assert enc.out_features == 32

    def test_default_hidden_dims(self):
        """Default hidden dims are [32, 64, 128]."""
        enc = CNN(in_features=3)
        assert enc.out_features == 128

    def test_forward_shape(self):
        """Forward returns 4-D spatial feature maps."""
        enc = CNN(in_features=3, hidden_dims=[16, 32])
        x = torch.randn(4, 3, 16, 16)
        out = enc(x)
        assert out.ndim == 4
        assert out.shape[1] == 32

    def test_spatial_reduction(self):
        """Each MaxPool2d halves spatial size."""
        enc = CNN(in_features=3, hidden_dims=[16, 32])
        x = torch.randn(2, 3, 16, 16)
        out = enc(x)
        # 2 MaxPool2d layers: 16 -> 8 -> 4
        assert out.shape[2] == 4 and out.shape[3] == 4

    def test_gradient_flow(self):
        """Gradients flow back through the encoder."""
        enc = CNN(in_features=3, hidden_dims=[16])
        x = torch.randn(2, 3, 16, 16, requires_grad=True)
        enc(x).sum().backward()
        assert x.grad is not None


class TestMLP:
    """Tests for MLP class."""

    def test_basic_forward(self):
        """Test basic MLP forward pass."""
        mlp = MLP(input_dim=128, output_dim=10, hidden_dims=[64])
        x = torch.randn(32, 128)
        y = mlp(x)
        assert y.shape == (32, 10)

    def test_single_layer(self):
        """Test MLP with no hidden layers."""
        mlp = MLP(input_dim=128, output_dim=10)
        x = torch.randn(16, 128)
        y = mlp(x)
        assert y.shape == (16, 10)

    def test_multiple_hidden_layers(self):
        """Test MLP with multiple hidden layers."""
        mlp = MLP(input_dim=128, output_dim=10, hidden_dims=[64, 32])
        x = torch.randn(8, 128)
        y = mlp(x)
        assert y.shape == (8, 10)

    def test_gradient_flow(self):
        """Test that gradients flow through the network."""
        mlp = MLP(input_dim=128, output_dim=10, hidden_dims=[64])
        x = torch.randn(8, 128, requires_grad=True)
        y = mlp(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
