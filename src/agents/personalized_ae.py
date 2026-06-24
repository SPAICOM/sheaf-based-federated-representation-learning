"""Personalised autoencoder agent for federated learning.

Mirrors :class:`PersonalizedClassifier` but the private task is image
reconstruction. The bottleneck latent returned by ``encode`` is the
representation the sheaf orchestrator aligns across agents; the decoder
reconstructs the input image, and ``compute_loss`` is MSE reconstruction
plus the same optional sparsity / weight-decay penalties.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_agent import BaseAgent

_SPARSITY_TYPES = {'l1', 'sae-l1', 'l21'}


class PersonalizedAE(BaseAgent):
    """Autoencoder agent with injected encoder and decoder backbones.

    The encoder maps the input to a flat bottleneck vector; the decoder
    reconstructs the input from that vector. The bottleneck is what the
    sheaf orchestrator uses to build cross-agent maps and on which sparsity
    is optionally applied.

    Parameters
    ----------
    encoder : nn.Module
        Module mapping a 4-D image batch ``(B, C, H, W)`` to a flat latent
        tensor of shape ``(B, latent_dim)`` (or a tensor that flattens to
        that shape).
    decoder : nn.Module
        Module mapping ``(B, latent_dim)`` back to a reconstruction tensor
        of the same shape as the encoder input.
    latent_dim : int
        Dimensionality of the bottleneck latent.
    weight_decay : float, optional
        L2 weight-decay strength applied as an explicit term in the loss
        (default: 0.0).
    l1_reg : float, optional
        Strength of the bottleneck sparsity penalty (default: 0.0).
    sparsity_type : str, optional
        Sparsity penalty type, identical to ``PersonalizedClassifier``:
        ``'l1'`` (mean per-sample L1), ``'sae-l1'`` (L1 on a learned
        ``ReLU(Wz + b)`` projection), or ``'l21'`` (per-feature L2 summed
        over features). Default: ``'l1'``.
    pixel_max : float, optional
        Maximum pixel value used for the PSNR ``task_performance`` metric
        (default: 1.0 — consistent with ``torchvision.transforms.ToTensor``
        outputs in [0, 1]).
    num_classes : int | None, optional
        Accepted for compatibility with the classifier config wiring in the
        experiment scripts and silently ignored. AE agents do not use the
        global class count.
    """

    task_type = 'reconstruction'

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        latent_dim: int,
        weight_decay: float = 0.0,
        l1_reg: float = 0.0,
        sparsity_type: str = 'l1',
        pixel_max: float = 1.0,
        num_classes: int | None = None,
    ):
        super().__init__()

        if sparsity_type not in _SPARSITY_TYPES:
            raise ValueError(
                f"sparsity_type must be one of {_SPARSITY_TYPES}, got '{sparsity_type}'"
            )

        self._encoder = encoder
        self._decoder = decoder
        self._last_latent: torch.Tensor | None = None
        self._last_input: torch.Tensor | None = None

        self._sae_proj = (
            nn.Linear(latent_dim, latent_dim)
            if sparsity_type == 'sae-l1'
            else None
        )

        self.latent_dim = int(latent_dim)
        self.weight_decay = float(weight_decay)
        self.l1_reg = float(l1_reg)
        self.sparsity_type = sparsity_type
        self.pixel_max = float(pixel_max)

    @property
    def encoder(self) -> nn.Module:
        return self._encoder

    @property
    def decoder(self) -> nn.Module:
        return self._decoder

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode ``x`` to the bottleneck and cache ``x`` for reconstruction loss."""
        if isinstance(x, torch.Tensor) and x.ndim == 3:
            x = x.unsqueeze(0)

        self._last_input = x
        latent = self._encoder(x)
        if latent.ndim > 2:
            latent = latent.flatten(1)
        self._last_latent = latent
        return latent

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode then decode; returns reconstruction tensor of the input shape."""
        return self._decoder(self.encode(x))

    def weight_decay_penalty(self) -> torch.Tensor:
        if self.weight_decay <= 0.0:
            param = next(self.parameters(), None)
            return torch.tensor(0.0) if param is None else param.new_zeros(())
        return self.weight_decay * sum(
            p.pow(2).sum() for p in self.parameters() if p.requires_grad
        )

    def latent_sparsity_penalty(
        self, latents: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Sparsity penalty on the bottleneck latent (same modes as classifier)."""
        reference = latents if latents is not None else self._last_latent
        if latents is None:
            self._last_latent = None

        if reference is None:
            if self.l1_reg <= 0.0:
                param = next(self.parameters(), None)
                return (
                    torch.tensor(0.0) if param is None else param.new_zeros(())
                )
            raise RuntimeError(
                'Latent sparsity penalty requires encoder latents. '
                'Call encode/forward first or pass latents explicitly.'
            )
        if self.l1_reg <= 0.0 or reference.numel() == 0:
            return reference.new_zeros(())

        if self.sparsity_type == 'l1':
            return self.l1_reg * reference.abs().sum(dim=1).mean()

        if self.sparsity_type == 'sae-l1':
            z_prime = F.relu(self._sae_proj(reference))
            return self.l1_reg * z_prime.abs().sum(dim=1).mean()

        return self.l1_reg * reference.norm(p=2, dim=0).sum()

    def compute_loss(
        self, y_hat: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """MSE reconstruction loss against the cached input.

        ``y`` is the classification label provided by the datamodule and is
        ignored — the reconstruction target is the input cached during the
        most recent :meth:`encode` call.
        """
        target = self._last_input
        if target is None:
            raise RuntimeError(
                'PersonalizedAE.compute_loss requires a cached input from the '
                'most recent encode/forward call.'
            )
        recon_loss = F.mse_loss(y_hat, target)
        return (
            recon_loss
            + self.latent_sparsity_penalty()
            + self.weight_decay_penalty()
        )

    def task_performance(
        self, y_hat: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Peak signal-to-noise ratio (dB) between reconstruction and cached input."""
        target = self._last_input
        if target is None:
            return torch.tensor(float('nan'), device=y_hat.device)
        mse = F.mse_loss(y_hat, target).detach().clamp_min(1e-10)
        return 10.0 * torch.log10((self.pixel_max**2) / mse)
