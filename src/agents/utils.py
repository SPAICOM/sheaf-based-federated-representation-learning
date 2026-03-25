"""
Utility modules for building neural network components.

This module provides reusable neural network building blocks including
multi-layer perceptrons (MLP) with configurable activation, normalization,
and dropout.
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Fully connected neural network (multi-layer perceptron).

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input features.
    output_dim : int
        Dimensionality of the output features.
    hidden_dims : list[int] | None, optional
        List of hidden layer dimensions. If None or empty, the network
        reduces to a single linear layer.
    activation : type[nn.Module], optional
        Activation function class (e.g., nn.ReLU, nn.GELU).
        Default: nn.ReLU.
    dropout : float, optional
        Dropout probability applied after each hidden layer (default: 0.0).
    use_batchnorm : bool, optional
        Whether to include BatchNorm1d after each linear layer
        (default: False).

    Example
    -------
    >>> mlp = MLP(input_dim=128, output_dim=10, hidden_dims=[64, 32])
    >>> x = torch.randn(32, 128)
    >>> y = mlp(x)  # torch.Size([32, 10])
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list[int] | None = None,
        activation: type[nn.Module] = nn.ReLU,
        dropout: float = 0.0,
        use_batchnorm: bool = False,
    ):
        """Build the MLP network from input to output dimensions.

        Parameters
        ----------
        input_dim : int
            Dimensionality of the input features.
        output_dim : int
            Dimensionality of the output features.
        hidden_dims : list[int] | None, optional
            List of hidden layer dimensions. If None or empty, the network
            reduces to a single linear layer.
        activation : type[nn.Module], optional
            Activation function class (e.g., nn.ReLU, nn.GELU).
            Default: nn.ReLU.
        dropout : float, optional
            Dropout probability applied after each hidden layer (default: 0.0).
        use_batchnorm : bool, optional
            Whether to include BatchNorm1d after each linear layer
            (default: False).
        """
        super().__init__()

        # Default to empty list if no hidden layers specified
        if hidden_dims is None:
            hidden_dims = []

        layers = []
        prev_dim = input_dim

        # Build hidden layers sequentially
        # Each layer: Linear -> [BatchNorm] -> Activation -> [Dropout]
        for h_dim in hidden_dims:
            # Linear transformation from previous dimension to hidden dimension
            layers.append(nn.Linear(prev_dim, h_dim))

            # Optional batch normalization after linear transformation
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h_dim))

            # Activation function (handle both class and instance)
            layers.append(
                activation
                if isinstance(activation, nn.Module)
                else activation()
            )

            # Optional dropout for regularization
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            # Output of this layer becomes input to next
            prev_dim = h_dim

        # Final output layer (no activation, dropout, or batchnorm)
        layers.append(nn.Linear(prev_dim, output_dim))

        # Compose all layers into sequential network
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the MLP.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, output_dim).
        """
        return self.network(x)
