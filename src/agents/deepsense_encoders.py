"""Modality-specific encoder backbones for the DeepSense dataset.

Each encoder inherits from BaseEncoder (defined in utils), which standardises
the output dimensionality contract used by PersonalizedClassifier (via the
``output_dim`` attribute).

Modalities
----------
- RGB camera  : DeepSenseRGBEncoder   — 3-channel, 64×64 crop
- LiDAR       : DeepSenseLiDAREncoder — 2-channel polar range/intensity map
- mmWave radar: DeepSenseMMWaveEncoder — 1-channel beam-energy map (16×64)
"""

import torch.nn as nn

from .utils import BaseEncoder


class DeepSenseRGBEncoder(BaseEncoder):
    """RGB branch used for the DeepSense image modality.

    Architecture: three Conv-ReLU-Pool stages followed by dropout and a
    compact two-layer MLP projection into the shared local embedding size.

    Parameters
    ----------
    input_channels : int, optional
        Number of input channels (default: 3 for RGB).
    output_dim : int, optional
        Size of the output embedding (default: 64).

    Notes
    -----
    Expects spatial inputs of shape ``(B, C, 64, 64)``. After three
    stride-2 max-pool layers the spatial map reaches 8×8, giving the MLP
    a 128·8·8 = 8192-dimensional input.
    """

    def __init__(self, input_channels: int = 3, output_dim: int = 64):
        super().__init__(output_dim)

        self.stage1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.dropout = nn.Dropout(0.5)
        self.mlp = nn.Sequential(
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x):
        """Encode RGB inputs shaped like ``(B, C, 64, 64)``."""
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.mlp(x)


class DeepSenseLiDAREncoder(BaseEncoder):
    """LiDAR encoder for polar-map inputs.

    Uses repeated Conv-ReLU-Pool-BatchNorm blocks, then global average
    pooling and a linear projection to the node embedding dimension.

    Parameters
    ----------
    input_channels : int, optional
        Number of input channels, e.g. 2 for (range, intensity) polar maps
        (default: 2).
    output_dim : int, optional
        Size of the output embedding (default: 64).
    """

    def __init__(self, input_channels: int = 2, output_dim: int = 64):
        super().__init__(output_dim)

        def _block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.BatchNorm2d(out_c),
            )

        self.block1 = _block(input_channels, 32)
        self.block2 = _block(32, 64)
        self.block3 = _block(64, 128)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(128, output_dim)

    def forward(self, x):
        """Encode LiDAR tensors shaped like ``(B, C, H, W)``."""
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)


class DeepSenseMMWaveEncoder(BaseEncoder):
    """mmWave encoder for beam-energy maps.

    First compresses width with anisotropic convolutions, then applies
    standard 2-D CNN blocks to capture joint spatial/temporal structure.

    Parameters
    ----------
    input_channels : int, optional
        Number of input channels (default: 1).
    output_dim : int, optional
        Size of the output embedding (default: 64).

    Notes
    -----
    Designed for inputs of shape ``(B, 1, 16, 64)``.  The width reducer
    brings the 64-wide beam axis to 16 before the spatiotemporal blocks.
    """

    def __init__(self, input_channels: int = 1, output_dim: int = 64):
        super().__init__(output_dim)

        self.width_reducer = nn.Sequential(
            nn.Conv2d(
                input_channels,
                32,
                kernel_size=(1, 3),
                stride=(1, 2),
                padding=(0, 1),
            ),
            nn.ReLU(),
            nn.Conv2d(
                32, 64, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)
            ),
            nn.ReLU(),
        )

        self.spatiotemporal = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, output_dim)

    def forward(self, x):
        """Encode mmWave tensors expected in shape ``(B, 1, 16, 64)``."""
        x = self.width_reducer(x)
        x = self.spatiotemporal(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
