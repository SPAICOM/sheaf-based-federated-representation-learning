import torch
import torch.nn as nn
from .personalized_classifier import PersonalizedClassifier

class FMTLEncoder(nn.Module):
    """
    Exact replica of the Sheaf-FMTL authors' 2-layer convolutional feature extractor.
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        self.out_features = 64 * 6 * 6

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        
        # Because we return a 2D tensor (Batch, 2304), the PersonalizedClassifier 
        # will skip its AdaptiveAvgPool2d, matching the authors' logic.
        return x.view(-1, 64 * 6 * 6)


class FMTLCNNClassifier(PersonalizedClassifier):
    """
    Exact replica of the authors' CIFAR10CNN adapted to the BaseAgent interface.
    """
    def __init__(
        self,
        in_features: int = 3,  # Kept for compatibility, but hardcoded to 3 inside encoder
        num_classes: int = 10,
        l1_reg: float = 0.0,
        **kwargs
    ):
        encoder = FMTLEncoder()

        super().__init__(
            encoder=encoder,
            latent_dim=64 * 6 * 6,  # 2304
            num_classes=num_classes,
            # an empty list so that the MLP utility becomes a 
            # single nn.Linear(2304, 10) layer
            decoder_hidden_dims=[], 
            dropout=0.0,
            use_batchnorm=False,
            l1_reg=l1_reg,
        )