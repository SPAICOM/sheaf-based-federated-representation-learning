"""Latent-based classifier agent for federated learning."""

import torch.nn as nn

from .personalized_classifier import PersonalizedClassifier
from .utils import MLP


class LatentClassifier(PersonalizedClassifier):
    """MLP-based classifier built on top of PersonalizedClassifier.

    Constructs an MLP encoder (in_features → latent_dim) and delegates
    everything else to PersonalizedClassifier.

    Parameters
    ----------
    in_features : int
        Dimensionality of the input feature vector.
    num_classes : int
        Number of output classes.
    latent_dim : int
        Dimensionality of the latent representation.
    encoder_hidden_dims : list[int] | None, optional
        Hidden layer dimensions for the encoder MLP. If None, the
        encoder is a single linear layer.
    decoder_hidden_dims : list[int], optional
        Hidden layer dimensions for the decoder MLP (default: [256]).
    dropout : float, optional
        Dropout probability (default: 0.0).
    activation : type[nn.Module], optional
        Activation function class (default: nn.ReLU).
    use_batchnorm : bool, optional
        Whether to use BatchNorm1d (default: False).
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
        l1_reg: float = 0.0,
    ):
        encoder = MLP(
            input_dim=in_features,
            output_dim=latent_dim,
            hidden_dims=encoder_hidden_dims,
            activation=activation,
            dropout=dropout,
            use_batchnorm=use_batchnorm,
        )

        super().__init__(
            encoder=encoder,
            latent_dim=latent_dim,
            num_classes=num_classes,
            decoder_hidden_dims=decoder_hidden_dims,
            dropout=dropout,
            activation=activation,
            use_batchnorm=use_batchnorm,
            l1_reg=l1_reg,
        )
