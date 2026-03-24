"""Latent-based classifier agent for federated learning."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import MulticlassAccuracy

from .base_agent import BaseAgent


class LatentClassifier(BaseAgent):
    """A flexible multi-layer perceptron (MLP) classifier for vector inputs.

    This module is designed to operate on inputs that are already flattened
    feature vectors (e.g., outputs from a pretrained model or embedding layer).
    It supports configurable hidden layers, activation functions, batch
    normalization, and dropout.

    Parameters
    ----------
    in_features : int
        Dimensionality of the input feature vector.
    num_classes : int
        Number of output classes.
    hidden_dims : list[int] | None, optional
        Sizes of hidden layers. If None or empty, the model reduces to a
        single linear classifier.
    dropout : float, default=0.0
        Dropout probability applied after each hidden layer (if > 0).
    activation : callable, default=nn.ReLU
        Activation function class (e.g., nn.ReLU, nn.GELU).
    use_batchnorm : bool, default=False
        Whether to include BatchNorm1d after each linear layer.

    Attributes
    ----------
    classifier : nn.Sequential
        The sequential stack of linear, normalization, activation,
        and dropout layers.
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
        hidden_dims=None,
        dropout: float = 0.0,
        activation=nn.ReLU,
        use_batchnorm: bool = False,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = []

        layers = []
        prev_dim = in_features

        self.accuracy = MulticlassAccuracy(
            num_classes=num_classes,
        )

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))

            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h_dim))

            layers.append(
                activation
                if isinstance(activation, nn.Module)
                else activation()
            )

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, num_classes))

        self.classifier = nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of the classifier.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_features).

        Returns
        -------
        torch.Tensor
            Output logits of shape (batch_size, num_classes).

        Raises
        ------
        ValueError
            If the input tensor is not 2-dimensional.
        """
        if x.ndim != 2:
            raise ValueError(
                'Expected input of shape (batch_size, in_features),'
                f'got {x.shape}'
            )

        return self.classifier(x)

    def forward_with_features(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning logits and latent features.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_features).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Tuple of (logits, features) where features are the output of the
            last hidden layer.

        Raises
        ------
        ValueError
            If the input tensor is not 2-dimensional.
        """
        if x.ndim != 2:
            raise ValueError(
                'Expected input of shape (batch_size, in_features),'
                f'got {x.shape}'
            )

        features = self.classifier[:-1](x)
        logits = self.classifier[-1](features)

        return logits, features

    def compute_loss(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the loss for classification.

        Parameters
        ----------
        y_hat : torch.Tensor
            Model outputs (logits) of shape (batch_size, num_classes).
        y : torch.Tensor
            Ground truth labels of shape (batch_size,).

        Returns
        -------
        torch.Tensor
            Scalar loss value.

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
