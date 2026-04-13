"""CNN-based classifier agent for federated learning.

Composes a CNN encoder (from utils.CNN) with an MLP decoder via
PersonalizedClassifier.
"""

import torch.nn as nn

from .personalized_classifier import PersonalizedClassifier
from .utils import HeteroCNN


class HeteroCNNClassifier(PersonalizedClassifier):
    """CNN-based classifier built on top of PersonalizedClassifier.

    Constructs a CNN encoder from the given architecture parameters and
    delegates everything else to PersonalizedClassifier.

    Parameters
    ----------
    in_features : int
        Number of input channels, injected automatically by
        experiment.py from the datamodule.
    num_classes : int
        Number of output classes.
    encoder_hidden_dims : list[int], optional
        Output channels for each Conv2d block (default: [32, 64, 128]).
    decoder_hidden_dims : list[int], optional
        Hidden layer dimensions for the decoder MLP (default: [256]).
    dropout : float, optional
        Dropout probability for encoder and decoder (default: 0.0).
    activation : type[nn.Module], optional
        Activation function class (default: nn.ReLU).
    use_batchnorm : bool, optional
        Whether to use batch normalisation (default: False).
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        encoder_hidden_dims: list[int] | None = None,
        decoder_hidden_dims: list[int] | None = None,
        rate: float = 1.0,
        dropout: float = 0.0,
        activation: type[nn.Module] = nn.ReLU,
        use_batchnorm: bool = False,
    ):
        if encoder_hidden_dims is None:
            encoder_hidden_dims = [32, 64, 128]

        encoder = CNN(
            in_features=in_features,
            hidden_dims=encoder_hidden_dims,
            dropout=dropout,
            activation=activation,
            use_batchnorm=use_batchnorm,
        )

        super().__init__(
            encoder=encoder,
            latent_dim=encoder.out_features,
            num_classes=num_classes,
            decoder_hidden_dims=decoder_hidden_dims,
            dropout=dropout,
            activation=activation,
            use_batchnorm=use_batchnorm,
        )
