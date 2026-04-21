"""Modality-specific encoder backbones for the mHealth dataset.

Each encoder inherits from BaseEncoder (defined in utils), which standardises
the output dimensionality contract used by PersonalizedClassifier (via the
``output_dim`` attribute).

Modalities
----------
- Accelerometer: MHealthAccelerometerEncoder — 3-channel (x, y, z)
- Gyroscope: MHealthGyroscopeEncoder — 3-channel (x, y, z)
- Magnetometer: MHealthMagnetometerEncoder — 3-channel (x, y, z)
- ECG: MHealthECGEncoder — 2-channel (lead1, lead2)
"""

import torch
import torch.nn as nn

from .utils import BaseEncoder


class MHealthSensorEncoder(BaseEncoder):
    """Base 1D CNN encoder for mHealth sensor time-series data.

    Uses Conv1D blocks to extract temporal features from sliding-window
    sensor readings, followed by global average pooling and MLP projection.

    Parameters
    ----------
    input_channels : int
        Number of input channels (3 for acc/gyro/mag, 2 for ECG).
    output_dim : int, optional
        Size of the output embedding (default: 64).
    hidden_dims : list[int], optional
        Channel dimensions for Conv1D layers (default: [32, 64, 128]).
    """

    def __init__(
        self,
        input_channels: int,
        output_dim: int = 64,
        hidden_dims: list[int] | None = None,
    ):
        super().__init__(output_dim)

        if hidden_dims is None:
            hidden_dims = [32, 64, 128]

        self.input_channels = input_channels

        layers = []
        in_ch = input_channels
        for out_ch in hidden_dims:
            layers.extend(
                [
                    nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.BatchNorm1d(out_ch),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                ]
            )
            in_ch = out_ch

        self.conv_blocks = nn.Sequential(*layers)

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dims[-1], output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode sensor time-series.

        Expected input shapes:
        - Point-wise: (B, C) where C = input_channels
        - Windowed: (B, T, C) where T = window_size, C = input_channels

        Output: (B, output_dim)
        """
        if x.ndim == 2:
            x = x.unsqueeze(-1).repeat(1, 1, 1)
        elif x.ndim == 3:
            x = x.transpose(1, 2)

        x = self.conv_blocks(x)
        x = self.gap(x).squeeze(-1)
        x = self.dropout(x)
        x = self.fc(x)
        return self.layer_norm(x)


class MHealthAccelerometerEncoder(MHealthSensorEncoder):
    """Encoder for accelerometer sensor data (x, y, z axes).

    Parameters
    ----------
    output_dim : int, optional
        Size of the output embedding (default: 64).
    """

    def __init__(self, output_dim: int = 64):
        super().__init__(input_channels=3, output_dim=output_dim)


class MHealthGyroscopeEncoder(MHealthSensorEncoder):
    """Encoder for gyroscope sensor data (x, y, z axes).

    Parameters
    ----------
    output_dim : int, optional
        Size of the output embedding (default: 64).
    """

    def __init__(self, output_dim: int = 64):
        super().__init__(input_channels=3, output_dim=output_dim)


class MHealthMagnetometerEncoder(MHealthSensorEncoder):
    """Encoder for magnetometer sensor data (x, y, z axes).

    Parameters
    ----------
    output_dim : int, optional
        Size of the output embedding (default: 64).
    """

    def __init__(self, output_dim: int = 64):
        super().__init__(input_channels=3, output_dim=output_dim)


class MHealthECGEncoder(MHealthSensorEncoder):
    """Encoder for ECG sensor data (lead1, lead2).

    Parameters
    ----------
    output_dim : int, optional
        Size of the output embedding (default: 64).
    """

    def __init__(self, output_dim: int = 64):
        super().__init__(input_channels=2, output_dim=output_dim)
