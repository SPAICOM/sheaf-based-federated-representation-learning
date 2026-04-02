"""Personalized classifier agent for federated learning.

Accepts any nn.Module as the encoder backbone, allowing each agent
to carry a different architecture while sharing the same training
interface as CNNClassifier.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import MulticlassAccuracy

from .base_agent import BaseAgent
from .utils import MLP


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
    ):
        super().__init__()

        if decoder_hidden_dims is None:
            decoder_hidden_dims = [256]

        self._encoder = encoder
        self._pool = nn.AdaptiveAvgPool2d((1, 1))
        self._decoder = MLP(
            input_dim=latent_dim,
            output_dim=num_classes,
            hidden_dims=decoder_hidden_dims,
            activation=activation,
            dropout=dropout,
            use_batchnorm=use_batchnorm,
        )

        self.accuracy = MulticlassAccuracy(num_classes=num_classes)

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

        return features.flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: encode then decode to class logits."""
        return self._decoder(self.encode(x))

    def compute_loss(
        self, y_hat: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Compute cross-entropy loss."""
        return F.cross_entropy(y_hat, y.long())

    def task_performance(
        self, y_hat: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Compute classification accuracy."""
        preds = torch.argmax(y_hat, dim=1)
        return self.accuracy(preds, y)
