"""CNN autoencoder agent for federated learning.

Composes :class:`CNNAEEncoder` (Conv+MaxPool stack → flatten → linear
bottleneck) with :class:`CNNAEDecoder` (linear → reshape → ConvTranspose
stack → sigmoid) via :class:`PersonalizedAE`. Architectural defaults
mirror :class:`CNNClassifier` so the two agent types can be compared on
the same dataset.
"""

import torch.nn as nn

from .personalized_ae import PersonalizedAE
from .utils import CNNAEDecoder, CNNAEEncoder


class CNNAutoencoder(PersonalizedAE):
    """CNN-based autoencoder built on top of :class:`PersonalizedAE`.

    Parameters
    ----------
    in_features : int
        Number of input channels, injected by the experiment script from
        the datamodule (e.g. 1 for MNIST, 3 for CIFAR).
    img_size : int
        Spatial side length of the input image, injected from the
        datamodule. Must be divisible by ``2 ** len(encoder_hidden_dims)``.
    latent_dim : int, optional
        Dimensionality of the bottleneck latent — what the sheaf
        orchestrator aligns across agents (default: 64).
    encoder_hidden_dims : list[int], optional
        Conv output channels per encoder block; the decoder mirrors this
        list in reverse (default: [32, 64]).
    dropout : float, optional
        Spatial dropout in the encoder (default: 0.0).
    activation : type[nn.Module], optional
        Activation class (default: nn.ReLU).
    use_batchnorm : bool, optional
        Whether to use BatchNorm2d in encoder/decoder blocks (default: False).
    weight_decay : float, optional
        L2 weight-decay strength applied as an explicit term in the loss
        (default: 0.0).
    l1_reg : float, optional
        Bottleneck sparsity strength (default: 0.0).
    sparsity_type : str, optional
        Sparsity penalty type (default: ``'l1'``).
    num_classes : int | None, optional
        Accepted for compatibility with the classifier wiring; ignored.
    """

    def __init__(
        self,
        in_features: int,
        img_size: int,
        latent_dim: int = 64,
        encoder_hidden_dims: list[int] | None = None,
        dropout: float = 0.0,
        activation: type[nn.Module] = nn.ReLU,
        use_batchnorm: bool = False,
        weight_decay: float = 0.0,
        l1_reg: float = 0.0,
        sparsity_type: str = 'l1',
        num_classes: int | None = None,
    ):
        if encoder_hidden_dims is None:
            encoder_hidden_dims = [32, 64]

        encoder = CNNAEEncoder(
            in_features=in_features,
            img_size=img_size,
            hidden_dims=encoder_hidden_dims,
            bottleneck_dim=latent_dim,
            dropout=dropout,
            activation=activation,
            use_batchnorm=use_batchnorm,
        )
        decoder = CNNAEDecoder(
            out_features=in_features,
            img_size=img_size,
            hidden_dims=encoder_hidden_dims,
            bottleneck_dim=latent_dim,
            activation=activation,
            use_batchnorm=use_batchnorm,
        )

        super().__init__(
            encoder=encoder,
            decoder=decoder,
            latent_dim=latent_dim,
            weight_decay=weight_decay,
            l1_reg=l1_reg,
            sparsity_type=sparsity_type,
        )
