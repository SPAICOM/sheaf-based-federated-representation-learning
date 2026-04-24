"""CNN-based classifier agent for federated learning.

Composes a CNN encoder (from utils.CNN) with an MLP decoder via
PersonalizedClassifier.
"""

import torch.nn as nn

from .personalized_classifier import PersonalizedClassifier
from .utils import HeteroCNN, HeteroMLP


class HeteroCNNClassifier(PersonalizedClassifier):
    """HeteroCNN encoder + HeteroMLP decoder classifier for heterogeneous FL.

    Both encoder and decoder channel widths are scaled by ``rate``, following
    the HeteroFL width-scaling scheme (Diao et al., ICLR 2021).

    Parameters
    ----------
    in_features : int
        Number of input channels, injected automatically by the experiment
        script from the datamodule.
    num_classes : int
        Number of output classes.
    encoder_hidden_dims : list[int], optional
        Output channels for each Conv2d block before rate scaling
        (default: [32, 64, 128]).
    decoder_hidden_dims : list[int], optional
        Hidden layer dimensions for the decoder before rate scaling
        (default: [256]).
    rate : float, optional
        Width scaling factor in (0, 1].  1.0 = full width (default).
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
        if decoder_hidden_dims is None:
            decoder_hidden_dims = [256]

        encoder = HeteroCNN(
            in_features=in_features,
            hidden_dims=encoder_hidden_dims,
            rate=rate,
            dropout=dropout,
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

        # Replace the MLP decoder built by PersonalizedClassifier with a
        # HeteroMLP so that classifier-head widths also scale with rate.
        self._decoder = HeteroMLP(
            input_dim=encoder.out_features,
            output_dim=num_classes,
            hidden_dims=decoder_hidden_dims,
            rate=rate,
            activation=activation,
            dropout=dropout,
            use_batchnorm=use_batchnorm,
        )
