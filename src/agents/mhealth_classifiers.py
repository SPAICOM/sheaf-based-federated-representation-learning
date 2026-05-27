"""mHealth modality-specific classifier agents.

Each class wires a fixed modality encoder (accelerometer, gyroscope,
magnetometer, ECG) into PersonalizedClassifier, following the same pattern
as DeepSense classifiers. The orchestrator can instantiate whichever
classifier matches the modality assigned to each federation node.
"""

import torch.nn as nn

from .mhealth_encoders import (
    MHealthAccelerometerEncoder,
    MHealthECGEncoder,
    MHealthGyroscopeEncoder,
    MHealthMagnetometerEncoder,
)
from .personalized_classifier import PersonalizedClassifier


class MHealthAccelerometerClassifier(PersonalizedClassifier):
    """Classifier for the mHealth accelerometer modality.

    Builds a MHealthAccelerometerEncoder and delegates training, encoding, and
    evaluation to PersonalizedClassifier.

    Parameters
    ----------
    num_classes : int
        Number of output classes.
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
    weight_decay : float, optional
        L2 regularization strength (default: 0.0).
    l1_reg : float, optional
        L1 regularization strength (default: 0.0) for embedding sparsification.

    Notes
    -----
    Expects inputs of shape ``(B, 3)`` (point-wise) or ``(B, T, 3)`` (windowed).
    """

    def __init__(
        self,
        num_classes: int,
        output_dim: int = 64,
        decoder_hidden_dims: list[int] | None = None,
        dropout: float = 0.0,
        activation: type[nn.Module] = nn.ReLU,
        use_batchnorm: bool = False,
        weight_decay: float = 0.0,
        l1_reg: float = 0.0,
        sparsity_type: str = 'l1',
    ):
        encoder = MHealthAccelerometerEncoder(output_dim=output_dim)
        super().__init__(
            encoder=encoder,
            latent_dim=encoder.output_dim,
            num_classes=num_classes,
            decoder_hidden_dims=decoder_hidden_dims,
            dropout=dropout,
            activation=activation,
            use_batchnorm=use_batchnorm,
            weight_decay=weight_decay,
            l1_reg=l1_reg,
            sparsity_type=sparsity_type,
        )


class MHealthGyroscopeClassifier(PersonalizedClassifier):
    """Classifier for the mHealth gyroscope modality.

    Builds a MHealthGyroscopeEncoder and delegates training, encoding, and
    evaluation to PersonalizedClassifier.

    Parameters
    ----------
    num_classes : int
        Number of output classes.
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
    weight_decay : float, optional
        L2 regularization strength (default: 0.0).
    l1_reg : float, optional
        L1 regularization strength (default: 0.0) for embedding sparsification.
    Notes
    -----
    Expects inputs of shape ``(B, 3)`` (point-wise) or ``(B, T, 3)`` (windowed).
    """

    def __init__(
        self,
        num_classes: int,
        output_dim: int = 64,
        decoder_hidden_dims: list[int] | None = None,
        dropout: float = 0.0,
        activation: type[nn.Module] = nn.ReLU,
        use_batchnorm: bool = False,
        weight_decay: float = 0.0,
        l1_reg: float = 0.0,
        sparsity_type: str = 'l1',
    ):
        encoder = MHealthGyroscopeEncoder(output_dim=output_dim)
        super().__init__(
            encoder=encoder,
            latent_dim=encoder.output_dim,
            num_classes=num_classes,
            decoder_hidden_dims=decoder_hidden_dims,
            dropout=dropout,
            activation=activation,
            use_batchnorm=use_batchnorm,
            weight_decay=weight_decay,
            l1_reg=l1_reg,
            sparsity_type=sparsity_type,
        )


class MHealthMagnetometerClassifier(PersonalizedClassifier):
    """Classifier for the mHealth magnetometer modality.

    Builds a MHealthMagnetometerEncoder and delegates training, encoding, and
    evaluation to PersonalizedClassifier.

    Parameters
    ----------
    num_classes : int
        Number of output classes.
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
    weight_decay : float, optional
        L2 regularization strength (default: 0.0).
    l1_reg : float, optional
        L1 regularization strength (default: 0.0) for embedding sparsification.

    Notes
    -----
    Expects inputs of shape ``(B, 3)`` (point-wise) or ``(B, T, 3)`` (windowed).
    """

    def __init__(
        self,
        num_classes: int,
        output_dim: int = 64,
        decoder_hidden_dims: list[int] | None = None,
        dropout: float = 0.0,
        activation: type[nn.Module] = nn.ReLU,
        use_batchnorm: bool = False,
        weight_decay: float = 0.0,
        l1_reg: float = 0.0,
        sparsity_type: str = 'l1',
    ):
        encoder = MHealthMagnetometerEncoder(output_dim=output_dim)
        super().__init__(
            encoder=encoder,
            latent_dim=encoder.output_dim,
            num_classes=num_classes,
            decoder_hidden_dims=decoder_hidden_dims,
            dropout=dropout,
            activation=activation,
            use_batchnorm=use_batchnorm,
            weight_decay=weight_decay,
            l1_reg=l1_reg,
            sparsity_type=sparsity_type,
        )


class MHealthECGClassifier(PersonalizedClassifier):
    """Classifier for the mHealth ECG modality.

    Builds a MHealthECGEncoder and delegates training, encoding, and
    evaluation to PersonalizedClassifier.

    Parameters
    ----------
    num_classes : int
        Number of output classes.
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
    weight_decay : float, optional
        L2 regularization strength (default: 0.0).
    l1_reg : float, optional
        L1 regularization strength (default: 0.0) for embedding sparsification.

    Notes
    -----
    Expects inputs of shape ``(B, 2)`` (point-wise) or ``(B, T, 2)`` (windowed).
    """

    def __init__(
        self,
        num_classes: int,
        output_dim: int = 64,
        decoder_hidden_dims: list[int] | None = None,
        dropout: float = 0.0,
        activation: type[nn.Module] = nn.ReLU,
        use_batchnorm: bool = False,
        weight_decay: float = 0.0,
        l1_reg: float = 0.0,
        sparsity_type: str = 'l1',
    ):
        encoder = MHealthECGEncoder(output_dim=output_dim)
        super().__init__(
            encoder=encoder,
            latent_dim=encoder.output_dim,
            num_classes=num_classes,
            decoder_hidden_dims=decoder_hidden_dims,
            dropout=dropout,
            activation=activation,
            use_batchnorm=use_batchnorm,
            weight_decay=weight_decay,
            l1_reg=l1_reg,
            sparsity_type=sparsity_type,
        )
