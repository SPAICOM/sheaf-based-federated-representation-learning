"""Latent-based classifier agent for federated learning."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import MulticlassAccuracy

from .base_agent import BaseAgent
from .utils import MLP


class LatentClassifier(BaseAgent):
    """A flexible MLP-based classifier with explicit encoder-decoder structure.

    The model consists of:
    - encoder: Maps input features to latent representation
    - decoder: Maps latent representation to class predictions

    Parameters
    ----------
    in_features : int
        Dimensionality of the input feature vector.
    num_classes : int
        Number of output classes.
    latent_dim : int
        Dimensionality of the latent representation.
    encoder_hidden_dims : list[int] | None, optional
        Hidden layer dimensions for the encoder. If None or empty,
        encoder is a single linear layer.
    decoder_hidden_dims : list[int], optional
        Hidden layer dimensions for the decoder (default: [256]).
    dropout : float, optional
        Dropout probability applied after each hidden layer (default: 0.0).
    activation : type[nn.Module], optional
        Activation function class (e.g., nn.ReLU, nn.GELU).
        Default: nn.ReLU.
    use_batchnorm : bool, optional
        Whether to include BatchNorm1d after each linear layer
        (default: False).

    Attributes
    ----------
    encoder : MLP
        Encoder network mapping input to latent space.
    decoder : MLP
        Decoder network mapping latent space to class predictions.
    accuracy : MulticlassAccuracy
        TorchMetrics accuracy calculator.

    Notes
    -----
    - The model outputs raw logits (no softmax applied).
    - Input must have shape (batch_size, in_features).
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        latent_dim: int,
        encoder_hidden_dims: list[int] | None = None,
        decoder_hidden_dims: list[int] | None = None,
        dropout: float = 0.0,
        activation: type[nn.Module] = nn.ReLU,
        use_batchnorm: bool = False,
    ):
        """Initialize the latent classifier with encoder-decoder architecture.

        Parameters
        ----------
        in_features : int
            Dimensionality of the input feature vector.
        num_classes : int
            Number of output classes.
        latent_dim : int
            Dimensionality of the latent representation.
        encoder_hidden_dims : list[int] | None, optional
            Hidden layer dimensions for the encoder. If None or empty,
            encoder is a single linear layer.
        decoder_hidden_dims : list[int], optional
            Hidden layer dimensions for the decoder (default: [256]).
        dropout : float, optional
            Dropout probability applied after each hidden layer (default: 0.0).
        activation : type[nn.Module], optional
            Activation function class (e.g., nn.ReLU, nn.GELU).
            Default: nn.ReLU.
        use_batchnorm : bool, optional
            Whether to include BatchNorm1d after each linear layer
            (default: False).
        """
        super().__init__()

        # Default decoder hidden dimensions if not specified
        if decoder_hidden_dims is None:
            decoder_hidden_dims = [256]

        # Encoder: maps input features -> latent representation
        # Compresses high-dimensional input into lower-dimensional space
        self.encoder = MLP(
            input_dim=in_features,
            output_dim=latent_dim,
            hidden_dims=encoder_hidden_dims,
            activation=activation,
            dropout=dropout,
            use_batchnorm=use_batchnorm,
        )

        # Decoder: maps latent representation -> class predictions (logits)
        # Expands from latent_dim to num_classes for classification
        self.decoder = MLP(
            input_dim=latent_dim,
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
        """Encoder network mapping input to latent space."""
        return self._encoder

    @encoder.setter
    def encoder(self, value: nn.Module):
        self._encoder = value

    @property
    def decoder(self) -> nn.Module:
        """Decoder network mapping latent space to predictions."""
        return self._decoder

    @decoder.setter
    def decoder(self, value: nn.Module):
        self._decoder = value

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Pass through encoder, returning latent features.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_features).

        Returns
        -------
        torch.Tensor
            Latent representation of shape (batch_size, latent_dim).
        """
        # Validate input shape: must be 2D (batch_size, features)
        if x.ndim != 2:
            raise ValueError(
                'Expected input of shape (batch_size, in_features),'
                f'got {x.shape}'
            )
        # Pass through encoder to get latent representation
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the classifier.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_features).

        Returns
        -------
        torch.Tensor
            Output logits of shape (batch_size, num_classes).
        """
        # Validate input shape: must be 2D (batch_size, features)
        if x.ndim != 2:
            raise ValueError(
                'Expected input of shape (batch_size, in_features),'
                f'got {x.shape}'
            )
        # Full forward pass: encode -> decode to get logits
        return self.decoder(self.encode(x))

    def compute_loss(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the loss for classification.

        Uses cross-entropy loss which combines log-softmax and negative
        log-likelihood. This is more numerically stable than computing
        softmax followed by log explicitly.

        Parameters
        ----------
        y_hat : torch.Tensor
            Model outputs (logits) of shape (batch_size, num_classes).
            Raw unnormalized scores from the final linear layer.
        y : torch.Tensor
            Ground truth labels of shape (batch_size,).
            Must be integer class indices in range [0, num_classes).

        Returns
        -------
        torch.Tensor
            Scalar loss value (mean over batch).

        Raises
        ------
        ValueError
            If shapes are inconsistent.
        """
        if y_hat.ndim != 2:
            raise ValueError(
                'Expected y_hat of shape (batch_size, num_classes),'
                f'got {y_hat.shape}'
            )

        if y.ndim != 1:
            raise ValueError(
                f'Expected y of shape (batch_size,), got {y.shape}'
            )

        return F.cross_entropy(y_hat, y.long())

    @torch.no_grad()
    def task_performance(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor,
    ) -> float:
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

        Raises
        ------
        ValueError
            If shapes are inconsistent.
        """
        if y_hat.ndim != 2:
            raise ValueError(
                'Expected y_hat of shape (batch_size, num_classes), '
                f'got {y_hat.shape}'
            )

        if y.ndim != 1:
            raise ValueError(
                f'Expected y of shape (batch_size,), got {y.shape}'
            )

        preds = torch.argmax(y_hat, dim=1)
        return self.accuracy(preds, y)


if __name__ == '__main__':
    pass
