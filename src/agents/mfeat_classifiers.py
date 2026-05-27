"""MFeat MLP classifier agent.

A single class handles all MFeat modalities; the only difference across
agents is the encoder's input dimension, which is passed at construction time
(typically from the per-agent config in the experiment yaml).
"""

import torch.nn as nn

from .mfeat_encoders import MFeatMLPEncoder
from .personalized_classifier import PersonalizedClassifier


class MFeatMLPClassifier(PersonalizedClassifier):
    """MLP classifier for any MFeat modality.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the raw feature vector for this agent's modality.
    num_classes : int
        Number of output classes (10 for digit recognition).
    output_dim : int, optional
        Encoder output / decoder input dimension (default: 64).
    encoder_hidden_dims : list[int] or None, optional
        Hidden layer widths for the MLP encoder (default: ``[256]``).
    encoder_dropout : float, optional
        Dropout inside the encoder (default: 0.3).
    decoder_hidden_dims : list[int] or None, optional
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
        L1 regularization strength (default: 0.0).
    sparsity_type : str, optional
        Sparsity loss variant: ``'l1'``, ``'sae-l1'``, or ``'l21'``.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        output_dim: int = 64,
        encoder_hidden_dims: list[int] | None = None,
        encoder_dropout: float = 0.3,
        decoder_hidden_dims: list[int] | None = None,
        dropout: float = 0.0,
        activation: type[nn.Module] = nn.ReLU,
        use_batchnorm: bool = False,
        weight_decay: float = 0.0,
        l1_reg: float = 0.0,
        sparsity_type: str = 'l1',
    ):
        encoder = MFeatMLPEncoder(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=encoder_hidden_dims,
            dropout=encoder_dropout,
        )
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
