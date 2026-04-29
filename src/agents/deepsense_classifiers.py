"""DeepSense modality-specific classifier agents.

Each class wires a fixed modality encoder (RGB, LiDAR, mmWave) into
PersonalizedClassifier, following the same pattern as CNNClassifier.
The orchestrator can instantiate whichever classifier matches the modality
assigned to each federation node.
"""

import torch.nn as nn

from .deepsense_encoders import (
    DeepSenseLiDAREncoder,
    DeepSenseMMWaveEncoder,
    DeepSenseRGBEncoder,
)
from .personalized_classifier import PersonalizedClassifier


class DeepSenseRGBClassifier(PersonalizedClassifier):
    """Classifier for the DeepSense RGB camera modality.

    Builds a DeepSenseRGBEncoder and delegates training, encoding, and
    evaluation to PersonalizedClassifier.

    Parameters
    ----------
    num_classes : int
        Number of output classes.
    input_channels : int, optional
        Number of RGB channels (default: 3).
    output_dim : int, optional
        Encoder output / decoder input dimension (default: 64).
    decoder_hidden_dims : list[int] | None, optional
        Hidden layer dimensions for the MLP decoder (default: [256]).
    dropout : float, optional
        Dropout probability for the decoder (default: 0.0).
    activation : type[nn.Module], optional
        Activation function class (default: nn.ReLU).
    use_batchnorm : bool, optional
        Whether to use BatchNorm1d in the decoder (default: False).
    l1_reg : float, optional
        L1 regularization strength (default: 0.0) for embedding sparsification.

    Notes
    -----
    Expects spatial inputs of shape ``(B, input_channels, 64, 64)``.
    The encoder already returns a flat vector so PersonalizedClassifier
    skips the internal AdaptiveAvgPool2d step.
    """

    def __init__(
        self,
        num_classes: int,
        input_channels: int = 3,
        output_dim: int = 64,
        decoder_hidden_dims: list[int] | None = None,
        dropout: float = 0.0,
        activation: type[nn.Module] = nn.ReLU,
        use_batchnorm: bool = False,
        l1_reg: float = 0.0,
    ):
        encoder = DeepSenseRGBEncoder(
            input_channels=input_channels,
            output_dim=output_dim,
        )
        super().__init__(
            encoder=encoder,
            latent_dim=encoder.output_dim,
            num_classes=num_classes,
            decoder_hidden_dims=decoder_hidden_dims,
            dropout=dropout,
            activation=activation,
            use_batchnorm=use_batchnorm,
            l1_reg=l1_reg,
        )


class DeepSenseLiDARClassifier(PersonalizedClassifier):
    """Classifier for the DeepSense LiDAR polar-map modality.

    Builds a DeepSenseLiDAREncoder and delegates training, encoding, and
    evaluation to PersonalizedClassifier.

    Parameters
    ----------
    num_classes : int
        Number of output classes.
    input_channels : int, optional
        Number of input channels, e.g. 2 for (range, intensity) maps
        (default: 2).
    output_dim : int, optional
        Encoder output / decoder input dimension (default: 64).
    decoder_hidden_dims : list[int] | None, optional
        Hidden layer dimensions for the MLP decoder (default: [256]).
    dropout : float, optional
        Dropout probability for the decoder (default: 0.0).
    activation : type[nn.Module], optional
        Activation function class (default: nn.ReLU).
    use_batchnorm : bool, optional
        Whether to use BatchNorm1d in the decoder (default: False).
    l1_reg : float, optional
        L1 regularization strength (default: 0.0) for embedding sparsification.
    """

    def __init__(
        self,
        num_classes: int,
        input_channels: int = 2,
        output_dim: int = 64,
        decoder_hidden_dims: list[int] | None = None,
        dropout: float = 0.0,
        activation: type[nn.Module] = nn.ReLU,
        use_batchnorm: bool = False,
        l1_reg: float = 0.0,
    ):
        encoder = DeepSenseLiDAREncoder(
            input_channels=input_channels,
            output_dim=output_dim,
        )
        super().__init__(
            encoder=encoder,
            latent_dim=encoder.output_dim,
            num_classes=num_classes,
            decoder_hidden_dims=decoder_hidden_dims,
            dropout=dropout,
            activation=activation,
            use_batchnorm=use_batchnorm,
            l1_reg=l1_reg,
        )


class DeepSenseMMWaveClassifier(PersonalizedClassifier):
    """Classifier for the DeepSense mmWave beam-energy modality.

    Builds a DeepSenseMMWaveEncoder and delegates training, encoding, and
    evaluation to PersonalizedClassifier.

    Parameters
    ----------
    num_classes : int
        Number of output classes.
    input_channels : int, optional
        Number of input channels (default: 1).
    output_dim : int, optional
        Encoder output / decoder input dimension (default: 64).
    decoder_hidden_dims : list[int] | None, optional
        Hidden layer dimensions for the MLP decoder (default: [256]).
    dropout : float, optional
        Dropout probability for the decoder (default: 0.0).
    activation : type[nn.Module], optional
        Activation function class (default: nn.ReLU).
    use_batchnorm : bool, optional
        Whether to use BatchNorm1d in the decoder (default: False).
    l1_reg : float, optional
        L1 regularization strength (default: 0.0) for embedding sparsification.

    Notes
    -----
    Designed for inputs of shape ``(B, 1, 16, 64)``.
    """

    def __init__(
        self,
        num_classes: int,
        input_channels: int = 1,
        output_dim: int = 64,
        decoder_hidden_dims: list[int] | None = None,
        dropout: float = 0.0,
        activation: type[nn.Module] = nn.ReLU,
        use_batchnorm: bool = False,
        l1_reg: float = 0.0,
    ):
        encoder = DeepSenseMMWaveEncoder(
            input_channels=input_channels,
            output_dim=output_dim,
        )
        super().__init__(
            encoder=encoder,
            latent_dim=encoder.output_dim,
            num_classes=num_classes,
            decoder_hidden_dims=decoder_hidden_dims,
            dropout=dropout,
            activation=activation,
            use_batchnorm=use_batchnorm,
            l1_reg=l1_reg,
        )
