"""Simple CNN-based classifier agent for federated learning.

Uses a basic Convolutional Neural Network as the encoder backbone.
A custom MLP decoder maps the encoder features to the target classes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchmetrics.classification import MulticlassAccuracy
from torchvision import transforms

from .base_agent import BaseAgent
from .utils import MLP


class CNNClassifier(BaseAgent):
    """Simple CNN-based classifier with custom MLP decoder.

    Uses a basic Convolutional Neural Network as the encoder backbone.
    The encoder outputs feature maps which are pooled via AdaptiveAvgPool2d
    and flattened before being passed to the decoder.
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        encoder_hidden_dims: list[int] | None = None,
        decoder_hidden_dims: list[int] | None = None,
        dropout: float = 0.0,
        activation: type[nn.Module] = nn.ReLU,
        use_batchnorm: bool = False,
    ):
        """Initialize the CNN-based classifier.

        Parameters
        ----------
        in_features : int
            Number of input channels, injected automatically by experiment.py from the datamodule.
        num_classes : int
            Number of output classes.
        encoder_hidden_dims : list[int], optional
            Number of output channels for each Conv2d layer (default: [32, 64, 128]).
        decoder_hidden_dims : list[int], optional
            Hidden layer dimensions for the decoder MLP (default: [256]).
        dropout : float, optional
            Dropout probability for both encoder and decoder (default: 0.0).
        activation : type[nn.Module], optional
            Activation function class (default: nn.ReLU).
        use_batchnorm : bool, optional
            Whether to use BatchNorm in the encoder/decoder (default: False).
        """
        super().__init__()

        if encoder_hidden_dims is None:
            encoder_hidden_dims = [32, 64, 128]
        if decoder_hidden_dims is None:
            decoder_hidden_dims = [256]

        layers = []
        in_ch = in_features
        for out_ch in encoder_hidden_dims:
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(out_ch))
            
            layers.append(
                activation() if isinstance(activation, type) else activation
            )
            
            layers.append(nn.MaxPool2d(2))
            
            if dropout > 0:
                layers.append(nn.Dropout2d(p=dropout))
                
            in_ch = out_ch

        self._encoder = nn.Sequential(*layers)
        
        self._pool = nn.AdaptiveAvgPool2d((1, 1))

        encoder_dim = encoder_hidden_dims[-1]

        self._decoder = MLP(
            input_dim=encoder_dim,
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
        """Pass through encoder and pool to get latent features."""
        # Convert PIL to Tensor and add batch dimension
        if isinstance(x, Image.Image):
            x = transforms.ToTensor()(x).unsqueeze(0)
        # Convert single image (C, H, W) to batch (1, C, H, W)
        elif x.ndim == 3:
            x = x.unsqueeze(0)

        # Forward through Conv layers
        features = self._encoder(x)
        
        # Reduce spatial dimensions to 1x1
        if features.ndim == 4:
            features = self._pool(features)

        # Flatten spatial dimensions
        return features.flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the classifier."""
        return self._decoder(self.encode(x))

    def compute_loss(
        self, y_hat: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Compute cross-entropy loss."""
        return F.cross_entropy(y_hat, y.long())

    def task_performance(self, y_hat: torch.Tensor, y: torch.Tensor) -> float:
        """Compute classification accuracy."""
        preds = torch.argmax(y_hat, dim=1)
        return self.accuracy(preds, y)
