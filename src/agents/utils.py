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
        super().__init__()

        if hidden_dims is None:
            hidden_dims = []

        layers = []
        prev_dim = input_dim

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

        layers.append(nn.Linear(prev_dim, output_dim))

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
