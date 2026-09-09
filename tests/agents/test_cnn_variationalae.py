"""Tests for src.agents.cnn_variationalae."""

import torch
import torch.nn as nn

import src.agents.cnn_variationalae as vae_module
from src.agents.base_agent import BaseAgent
from src.agents.cnn_variationalae import (
    CNNVAEDecoder,
    CNNVAEEncoder,
    CNNVariationalAE,
)


class TestCNNVariationalAE:
    def _make(self, **kwargs) -> CNNVariationalAE:
        return CNNVariationalAE(
            **{'in_features': 3, 'img_size': 16, 'latent_dim': 8, **kwargs}
        )

    def _x(self, batch: int = 4) -> torch.Tensor:
        return torch.rand(batch, 3, 16, 16)

    def test_is_base_agent(self):
        assert isinstance(self._make(), BaseAgent)

    def test_encoder_and_decoder_types(self):
        agent = self._make()
        assert isinstance(agent.encoder, CNNVAEEncoder)
        assert isinstance(agent.decoder, CNNVAEDecoder)

    def test_encode_returns_posterior_mean(self):
        latent = self._make().encode(self._x())
        assert latent.shape == (4, 8)

    def test_forward_shape(self):
        assert self._make()(self._x()).shape == (4, 3, 16, 16)

    def test_non_power_of_two_image_size(self):
        agent = CNNVariationalAE(
            in_features=1,
            img_size=28,
            latent_dim=8,
            encoder_hidden_dims=[16, 32],
        )
        assert agent(torch.rand(2, 1, 28, 28)).shape == (2, 1, 28, 28)

    def test_compute_loss_scalar(self):
        agent = self._make(beta=1e-3, reconstruction_l1_weight=0.1)
        x = self._x()
        loss = agent.compute_loss(agent(x), torch.randint(0, 10, (4,)))
        assert loss.ndim == 0 and torch.isfinite(loss)

    def test_compute_loss_exposes_logging_components(self):
        agent = self._make(
            beta=0.1,
            reconstruction_l1_weight=1.0,
            ssim_weight=0.5,
        )
        x = self._x()
        loss = agent.compute_loss(agent(x), x)

        assert torch.isclose(agent.last_loss_components['total'], loss.detach())
        assert 'mse' in agent.last_loss_components
        assert 'kl_weighted' in agent.last_loss_components
        assert 'ssim_loss' in agent.last_loss_components

    def test_compute_loss_uses_image_target_for_inpainting(self):
        agent = self._make(beta=0.0, weight_decay=0.0)
        full = self._x()
        masked = full * 0.0
        agent.encode(masked)
        loss = agent.compute_loss(full, full)
        assert torch.isclose(loss, full.new_zeros(()))

    def test_compute_loss_includes_perceptual_loss(self, monkeypatch):
        class FakePerceptualLoss(nn.Module):
            def forward(
                self, y_hat: torch.Tensor, target: torch.Tensor
            ) -> torch.Tensor:
                return y_hat.new_tensor(2.0)

        monkeypatch.setattr(
            vae_module, 'VGGPerceptualLoss', FakePerceptualLoss
        )
        perceptual_agent = self._make(
            beta=0.0, weight_decay=0.0, perceptual_weight=0.5
        )
        x = self._x()
        perceptual_agent.encode(x)

        perceptual_loss = perceptual_agent.compute_loss(
            x, torch.randint(0, 10, (4,))
        )

        assert torch.isclose(perceptual_loss, x.new_tensor(1.0))

    def test_compute_loss_includes_ssim_loss(self):
        agent = self._make(beta=0.0, weight_decay=0.0, ssim_weight=0.5)
        x = self._x()
        agent.encode(x)
        loss = agent.compute_loss(x, torch.randint(0, 10, (4,)))
        assert torch.isclose(loss, x.new_zeros(()), atol=1e-5)

    def test_reconstruction_warmup_uses_training_loss_steps(self, monkeypatch):
        class FakePerceptualLoss(nn.Module):
            def forward(
                self, y_hat: torch.Tensor, target: torch.Tensor
            ) -> torch.Tensor:
                return y_hat.new_tensor(2.0)

        monkeypatch.setattr(
            vae_module, 'VGGPerceptualLoss', FakePerceptualLoss
        )
        agent = self._make(
            beta=0.0,
            weight_decay=0.0,
            perceptual_weight=1.0,
            perceptual_warmup_steps=2,
        )
        x = self._x()

        agent.encode(x)
        first_loss = agent.compute_loss(x, torch.randint(0, 10, (4,)))
        agent.encode(x)
        second_loss = agent.compute_loss(x, torch.randint(0, 10, (4,)))
        agent.eval()
        agent.encode(x)
        eval_loss = agent.compute_loss(x, torch.randint(0, 10, (4,)))

        assert torch.isclose(first_loss, x.new_tensor(0.0))
        assert torch.isclose(second_loss, x.new_tensor(1.0))
        assert torch.isclose(eval_loss, x.new_tensor(2.0))

    def test_latent_sparsity_penalty_removed(self):
        assert not hasattr(self._make(), 'latent_sparsity_penalty')

    def test_task_performance_is_psnr(self):
        agent = self._make()
        x = self._x()
        psnr = agent.task_performance(agent(x), torch.randint(0, 10, (4,)))
        assert psnr.ndim == 0 and torch.isfinite(psnr)

    def test_gradient_flow(self):
        agent = self._make()
        x = self._x().requires_grad_()
        loss = agent.compute_loss(agent(x), torch.randint(0, 10, (4,)))
        loss.backward()
        assert x.grad is not None
