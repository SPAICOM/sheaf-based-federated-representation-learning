"""Vision Transformer classifier agent for federated learning.

Composes a ViTEncoder (patch embedding + stacked attention blocks) with an
MLP decoder via PersonalizedClassifier.
"""

import torch.nn as nn

from .personalized_classifier import PersonalizedClassifier
from .utils import ViTEncoder


class TransformerClassifier(PersonalizedClassifier):
    """Vision Transformer classifier built on top of PersonalizedClassifier.

    Constructs a ViTEncoder from the given architecture parameters and
    delegates everything else to PersonalizedClassifier.

    Parameters
    ----------
    in_features : int
        Number of input channels, injected automatically by
        experiment.py from the datamodule.
    num_classes : int
        Number of output classes.
    img_size : int
        Spatial side length of the input image (assumed square). Must be
        evenly divisible by ``patch_size``. Injected from the datamodule.
    patch_size : int, optional
        Side length of each square patch (default: 4).
    embed_dim : int, optional
        Transformer token embedding dimension (default: 128).
    depth : int, optional
        Number of stacked TransformerEncoderLayer blocks (default: 4).
    num_heads : int, optional
        Number of self-attention heads (default: 4).
    mlp_ratio : float, optional
        Ratio of the FFN hidden dimension to ``embed_dim`` (default: 4.0).
    decoder_hidden_dims : list[int], optional
        Hidden layer dimensions for the decoder MLP (default: [256]).
    dropout : float, optional
        Dropout probability for encoder attention/FFN and decoder (default: 0.0).
    activation : type[nn.Module], optional
        Activation function class for the decoder MLP (default: nn.ReLU).
    use_batchnorm : bool, optional
        Whether to use batch normalisation in the decoder (default: False).
    weight_decay : float, optional
        L2 regularization strength (default: 0.0).
    l1_reg : float, optional
        L1 sparsity regularization strength on encoder latents (default: 0.0).
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        img_size: int,
        patch_size: int = 4,
        embed_dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        decoder_hidden_dims: list[int] | None = None,
        dropout: float = 0.0,
        activation: type[nn.Module] = nn.ReLU,
        use_batchnorm: bool = False,
        weight_decay: float = 0.0,
        l1_reg: float = 0.0,
    ):
        encoder = ViTEncoder(
            in_features=in_features,
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
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
            weight_decay=weight_decay,
            l1_reg=l1_reg,
        )
