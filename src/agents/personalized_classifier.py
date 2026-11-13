"""Personalized classifier agent for federated learning.

Accepts any nn.Module as the encoder backbone, allowing each agent
to carry a different architecture while sharing the same training
interface as CNNClassifier.
"""

import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import MulticlassAccuracy

from .base_agent import BaseAgent
from .utils import MLP

_SPARSITY_TYPES = {'l1', 'sae-l1', 'l21'}


class CenterNorm(nn.Module):
    """Bottleneck normalisation: z -> (z - mu) / ||z - mu||.

    Like BatchNorm1d without the affine part and with a per-sample scale:
    mu is the batch mean in training and a running mean (same momentum
    convention as BatchNorm) in eval; each centred vector is then divided by
    its own L2 norm, so every latent lies on the unit sphere. Only the agent's
    own task batches may move the running mean: see :func:`frozen_center_norm`.
    """

    def __init__(self, dim: int, momentum: float = 0.1, eps: float = 1e-6):
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.register_buffer('running_mean', torch.zeros(dim))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if self.training and z.shape[0] > 1:
            mu = z.mean(0)
            if self.momentum > 0:  # 0 while frozen: no write at all
                with torch.no_grad():
                    self.running_mean.lerp_(mu.detach(), self.momentum)
        else:
            mu = self.running_mean
        z = z - mu
        return z / z.norm(dim=1, keepdim=True).clamp_min(self.eps)


@contextlib.contextmanager
def frozen_center_norm(module: nn.Module):
    """Forwards inside this block cannot change any CenterNorm ``running_mean``.

    For pilot batches: train-mode layers still centre on the batch mean, with
    gradient, but the statistics used at evaluation come from task data only.
    """
    layers = [m for m in module.modules() if isinstance(m, CenterNorm)]
    saved = [m.momentum for m in layers]
    for m in layers:
        m.momentum = 0.0
    try:
        yield
    finally:
        for m, momentum in zip(layers, saved, strict=True):
            m.momentum = momentum


class PersonalizedClassifier(BaseAgent):
    """Classifier agent with an injected encoder backbone.

    The encoder is provided externally, enabling heterogeneous
    architectures across agents. The decoder is an MLP built from
    the encoder's output dimension to the number of classes.

    Parameters
    ----------
    encoder : nn.Module
        Any module that maps input tensors to a flat or spatial feature
        tensor. If the encoder returns a 4-D tensor (B, C, H, W), an
        AdaptiveAvgPool2d(1, 1) followed by a flatten is applied before
        the decoder.
    latent_dim : int
        Output dimensionality of the encoder (after pooling/flattening
        if spatial).
    num_classes : int
        Number of output classes.
    decoder_hidden_dims : list[int], optional
        Hidden layer dimensions for the decoder MLP (default: [256]).
    dropout : float, optional
        Dropout probability for the decoder (default: 0.0).
    activation : type[nn.Module], optional
        Activation function class (default: nn.ReLU).
    use_batchnorm : bool, optional
        Whether to use BatchNorm1d in the decoder (default: False).
    weight_decay : float, optional
        L2 regularization strength applied to all model parameters in the
        loss (default: 0.0).
    l1_reg : float, optional
        Regularization strength (default: 0.0).
    sparsity_type : str, optional
        Which sparsity penalty to apply on the encoder latents. One of:
        - ``"l1"``    : mean over the batch of per-sample L1 norms.
        - ``"sae-l1"``: sparse auto-encoder variant — applies a learned
          linear projection followed by ReLU on the latents to obtain
          Z' = ReLU(W Z + b), then computes the L1 penalty on Z'.
        - ``"l21"``   : batch L21 norm — for each feature dimension d the
          L2 norm over the batch is computed, then summed across dimensions:
          sum_d ||Z[:, d]||_2.
        Defaults to ``"l1"``.
    latent_norm : bool, optional
        Append a :class:`CenterNorm` layer at the bottleneck: subtract the
        batch (eval: running) mean and divide each latent by its L2 norm
        (default: False).
    """

    def __init__(
        self,
        encoder: nn.Module,
        latent_dim: int,
        num_classes: int,
        decoder_hidden_dims: list[int] | None = None,
        dropout: float = 0.0,
        activation: type[nn.Module] = nn.ReLU,
        use_batchnorm: bool = False,
        weight_decay: float = 0.0,
        l1_reg: float = 0.0,
        sparsity_type: str = 'l1',
        pilot_loss_weight: float = 0.0,
        latent_norm: bool = False,
    ):
        super().__init__()
        # >0 makes the shared pilots part of this agent's SUPERVISED task, not
        # just an alignment substrate: they are encoded+decoded and their loss
        # is added to the private task loss (see
        # BaseOrchestrator._local_pilot_loss).
        self.pilot_loss_weight = float(pilot_loss_weight)

        if sparsity_type not in _SPARSITY_TYPES:
            raise ValueError(
                f"sparsity_type must be one of {_SPARSITY_TYPES}, got '{sparsity_type}'"
            )

        if decoder_hidden_dims is None:
            decoder_hidden_dims = [256]

        self._encoder = encoder
        self._pool = nn.AdaptiveAvgPool2d((1, 1))
        # Optional bottleneck normalisation (CenterNorm): the latent the
        # decoder and every alignment / communication path see.
        self._latent_norm = CenterNorm(latent_dim) if latent_norm else None
        self._last_latent = None
        self._decoder = MLP(
            input_dim=latent_dim,
            output_dim=num_classes,
            hidden_dims=decoder_hidden_dims,
            activation=activation,
            dropout=dropout,
            use_batchnorm=use_batchnorm,
        )

        # SAE projection: Z' = ReLU(W Z + b), same dimension as latent space
        self._sae_proj = (
            nn.Linear(latent_dim, latent_dim)
            if sparsity_type == 'sae-l1'
            else None
        )

        self.accuracy = MulticlassAccuracy(num_classes=num_classes)
        self.weight_decay = weight_decay
        self.l1_reg = l1_reg
        self.sparsity_type = sparsity_type

    @property
    def encoder(self) -> nn.Module:
        return self._encoder

    @property
    def decoder(self) -> nn.Module:
        return self._decoder

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Pass through the encoder and return a flat latent vector.

        Note: a 3-D input (C, H, W) is silently unsqueezed to (1, C, H, W)
        to support single-sample inference. This means flat encoders (e.g.
        LatentClassifier's MLP) will receive a 4-D tensor and fail with a
        shape error if called with a single unqueued feature vector — callers
        are expected to always pass a proper 2-D batch (B, F) to flat
        encoders. 4-D encoder outputs (spatial feature maps) are reduced via
        AdaptiveAvgPool2d(1,1) before flattening.
        """
        if isinstance(x, torch.Tensor) and x.ndim == 3:
            x = x.unsqueeze(0)

        features = self._encoder(x)

        if features.ndim == 4:
            features = self._pool(features)

        latent = features.flatten(1)
        if self._latent_norm is not None:
            latent = self._latent_norm(latent)
        self._last_latent = latent
        return latent

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: encode then decode to class logits."""
        return self._decoder(self.encode(x))

    def weight_decay_penalty(self) -> torch.Tensor:
        """Return the L2 weight decay penalty over all model parameters."""
        if self.weight_decay <= 0.0:
            param = next(self.parameters(), None)
            return torch.tensor(0.0) if param is None else param.new_zeros(())
        return self.weight_decay * sum(
            p.pow(2).sum() for p in self.parameters() if p.requires_grad
        )

    def compute_loss(
        self, y_hat: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Compute cross-entropy loss plus optional sparsity and weight decay penalties."""
        loss = F.cross_entropy(y_hat, y.long())
        return loss + self.latent_sparsity_penalty() + self.weight_decay_penalty()

    def latent_sparsity_penalty(
        self, latents: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return the sparsity penalty on encoder latents.

        The penalty type is controlled by ``self.sparsity_type``:
        - ``"l1"``    : mean over batch of per-sample L1 norms.
        - ``"sae-l1"``: L1 penalty on Z' = ReLU(W Z + b).
        - ``"l21"``   : sum_d ||Z[:, d]||_2  (batch L21 norm).
        """
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

        # l21: for each feature dimension d, L2 norm over the batch, summed over d
        return self.l1_reg * reference.norm(p=2, dim=0).sum()

    def task_performance(
        self, y_hat: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Compute classification accuracy."""
        preds = torch.argmax(y_hat, dim=1)
        return self.accuracy(preds, y)
