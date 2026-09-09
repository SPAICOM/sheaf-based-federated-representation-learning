import torch.nn as nn

from .personalized_classifier import PersonalizedClassifier
from .utils import ResNetEncoder


class ResNetClassifier(PersonalizedClassifier):
    """Configurable ResNet classifier for federated learning.

    By default uses a CIFAR-style ResNet18 encoder.
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        encoder_hidden_dims: list[int] | None = None,
        blocks_per_stage: list[int] | None = None,
        stage_strides: list[int] | None = None,
        decoder_hidden_dims: list[int] | None = None,
        dropout: float = 0.0,
        activation: type[nn.Module] = nn.ReLU,
        use_batchnorm: bool = True,
        weight_decay: float = 0.0,
        l1_reg: float = 0.0,
        sparsity_type: str = "l1",
        stem_kernel_size: int = 3,
        stem_stride: int = 1,
        stem_pool: bool = False,
    ):
        encoder = ResNetEncoder(
            in_features=in_features,
            hidden_dims=encoder_hidden_dims,
            blocks_per_stage=blocks_per_stage,
            stage_strides=stage_strides,
            activation=activation,
            use_batchnorm=use_batchnorm,
            dropout=dropout,
            stem_kernel_size=stem_kernel_size,
            stem_stride=stem_stride,
            stem_pool=stem_pool,
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
            sparsity_type=sparsity_type,
        )