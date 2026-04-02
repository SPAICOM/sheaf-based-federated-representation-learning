"""Timm-based classifier agent for federated learning."""

import torch.nn as nn

from .personalized_classifier import PersonalizedClassifier
from .utils import TimmEncoder


class TimmClassifier(PersonalizedClassifier):
    """Timm backbone classifier built on top of PersonalizedClassifier.

    Constructs a TimmEncoder from the given model name and delegates
    everything else to PersonalizedClassifier.

    Parameters
    ----------
    model_name : str
        timm model identifier (e.g. 'resnet50', 'vit_base_patch16_224').
    num_classes : int
        Number of output classes.
    pretrained : bool, optional
        Load pretrained backbone weights (default: True).
    freeze_encoder : bool, optional
        Freeze all backbone parameters; only the decoder trains
        (default: False).
    decoder_hidden_dims : list[int], optional
        Hidden layer dimensions for the decoder MLP (default: [256]).
    dropout : float, optional
        Dropout probability for the decoder (default: 0.0).
    activation : type[nn.Module], optional
        Activation function class (default: nn.ReLU).
    use_batchnorm : bool, optional
        Whether to use BatchNorm in the decoder (default: False).
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
        encoder = TimmEncoder(
            model_name=model_name,
            pretrained=pretrained,
            freeze_encoder=freeze_encoder,
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
