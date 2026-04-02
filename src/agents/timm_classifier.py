"""Timm-based classifier agent for federated learning.

Uses a pretrained model from the timm library as the encoder (backbone),
with the original classification head removed. A custom MLP decoder maps
the encoder features to the target number of classes.

Parameters
----------
model_name : str
    Name of the timm model (e.g., 'resnet50', 'vit_base_patch16_224').
num_classes : int
    Number of output classes.
pretrained : bool, optional
    Whether to load pretrained encoder weights (default: True).
decoder_hidden_dims : list[int], optional
    Hidden layer dimensions for the decoder MLP (default: [256]).
dropout : float, optional
    Dropout probability for the decoder (default: 0.0).
activation : type[nn.Module], optional
    Activation function class (default: nn.ReLU).
use_batchnorm : bool, optional
    Whether to use BatchNorm in the decoder (default: False).

Example
-------
    >>> agent = TimmClassifier(
    ...     model_name='resnet50',
    ...     num_classes=10,
    ...     pretrained=True,
    ... )
    >>> x = torch.randn(4, 3, 224, 224)
    >>> logits = agent(x)
    >>> print(logits.shape)  # torch.Size([4, 10])
    >>> features = agent.encode(x)
    >>> print(features.shape)  # torch.Size([4, 2048])
"""

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchmetrics.classification import MulticlassAccuracy
from torchvision import transforms

from .base_agent import BaseAgent
from .utils import MLP

class TimmClassifier(BaseAgent):
    """Timm-based classifier with custom MLP decoder.

    Uses a pretrained timm model as the encoder backbone. The original
    classification head is removed (num_classes=0) and replaced with a
    custom MLP decoder that maps encoder features to the target classes.

    The encoder outputs feature maps which are pooled and flattened before
    being passed to the decoder.
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        pretrained: bool = True,
        freeze_encoder: bool = False,
        decoder_hidden_dims: list[int] | None = None,
        dropout: float = 0.0,
        activation: type[nn.Module] = nn.ReLU,
        use_batchnorm: bool = False,
    ):
        """Initialize the timm-based classifier.

        Parameters
        ----------
        model_name : str
            Name of the timm model (e.g., 'resnet50', 'vit_base_patch16_224').
        num_classes : int
            Number of output classes.
        pretrained : bool, optional
            Whether to load pretrained encoder weights (default: True).
        freeze_encoder : bool, optional
            If True, freeze all encoder (backbone) parameters so only the
            decoder MLP is trained. Enables linear probing mode which avoids
            fine-tuning instability when per-agent data is small (default: False).
        decoder_hidden_dims : list[int], optional
            Hidden layer dimensions for the decoder MLP (default: [256]).
        dropout : float, optional
            Dropout probability for the decoder (default: 0.0).
        activation : type[nn.Module], optional
            Activation function class (default: nn.ReLU).
        use_batchnorm : bool, optional
            Whether to use BatchNorm in the decoder (default: False).
        """
        super().__init__()

        # Default decoder hidden dimensions if not specified
        if decoder_hidden_dims is None:
            decoder_hidden_dims = [256]

        # Create timm encoder (backbone) with pretrained weights
        # num_classes=0 removes the original classification head
        self._encoder = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
        )

        # Freeze encoder (linear probing mode): only the decoder MLP trains.
        # Critical when per-agent data is small relative to backbone capacity.
        if freeze_encoder:
            for param in self._encoder.parameters():
                param.requires_grad = False
            self._encoder.eval()  # also disables dropout/batchnorm in encoder

        # Global average pooling to reduce spatial dimensions to 1x1
        self._pool = nn.AdaptiveAvgPool2d((1, 1))

        # Determine encoder output dimension by running a forward pass
        # This handles variable-sized feature maps from different architectures
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            encoder_out = self._encoder(dummy)
            # Pool feature maps (H x W x C) to (1 x 1 x C)
            if encoder_out.ndim == 4:
                encoder_out = self._pool(encoder_out)
            # Flatten to get feature dimension
            encoder_dim = encoder_out.flatten(1).shape[1]

        # Build MLP decoder mapping encoder features to class predictions
        self._decoder = MLP(
            input_dim=encoder_dim,
            output_dim=num_classes,
            hidden_dims=decoder_hidden_dims,
            activation=activation,
            dropout=dropout,
            use_batchnorm=use_batchnorm,
        )

        # Initialize accuracy metric for task performance evaluation
        self.accuracy = MulticlassAccuracy(num_classes=num_classes)

    @property
    def encoder(self) -> nn.Module:
        """Timm encoder model (backbone)."""
        return self._encoder

    @property
    def decoder(self) -> nn.Module:
        """MLP decoder mapping encoder features to class predictions."""
        return self._decoder

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Pass through encoder and pool to get latent features.

        Parameters
        ----------
        x : torch.Tensor or PIL.Image
            Input tensor of shape (batch_size, 3, height, width) or
            PIL Image. Will be resized to 224x224 and converted to tensor.

        Returns
        -------
        torch.Tensor
            Latent features of shape (batch_size, encoder_dim).
        """
        # Handle different input formats and ensure proper shape/dimensions

        # Case 1: PIL Image - convert to tensor and resize to 224x224
        if isinstance(x, Image.Image):
            x = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                ]
            )(x)
            # Add batch dimension: (C, H, W) -> (1, C, H, W)
            x = x.unsqueeze(0)

        # Case 2: Single image as tensor (C, H, W) - add batch dimension
        elif x.ndim == 3:
            x = x.unsqueeze(0)
            # Resize to expected 224x224 if needed
            x = transforms.Resize((224, 224))(x)

        # Case 3: Batch of images (B, C, H, W) - resize if not 224x224
        elif x.ndim == 4:
            if x.shape[-1] != 224 or x.shape[-2] != 224:
                x = transforms.Resize((224, 224))(x)

        # Expand grayscale inputs to pseudo-RGB so MNIST-like datasets can be
        # used with RGB backbones without changing the pretrained stem.
        if x.ndim == 4 and x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        # Pass through encoder backbone to get feature maps
        features = self._encoder(x)

        # Pool spatial dimensions (H, W) -> (1, 1) for global average pooling
        if features.ndim == 4:
            features = self._pool(features)

        # Flatten spatial dimensions: (B, C, 1, 1) -> (B, C)
        return features.flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the classifier.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, 3, height, width).
            Expected to be 224x224 images.

        Returns
        -------
        torch.Tensor
            Output logits of shape (batch_size, num_classes).
        """
        return self._decoder(self.encode(x))

    def compute_loss(
        self, y_hat: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Compute cross-entropy loss.

        Parameters
        ----------
        y_hat : torch.Tensor
            Model outputs (logits) of shape (batch_size, num_classes).
        y : torch.Tensor
            Ground truth labels of shape (batch_size,).

        Returns
        -------
        torch.Tensor
            Scalar loss value (mean over batch).
        """
        return F.cross_entropy(y_hat, y.long())

    def task_performance(self, y_hat: torch.Tensor, y: torch.Tensor) -> float:
        """Compute classification accuracy.

        Parameters
        ----------
        y_hat : torch.Tensor
            Model outputs (logits) of shape (batch_size, num_classes).
        y : torch.Tensor
            Ground truth labels of shape (batch_size,).

        Returns
        -------
        float
            Accuracy (between 0 and 1).
        """
        preds = torch.argmax(y_hat, dim=1)
        return self.accuracy(preds, y)


if __name__ == '__main__':
    pass
