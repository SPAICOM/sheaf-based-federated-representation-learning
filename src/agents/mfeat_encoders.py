"""MLP encoder backbone for the MFeat dataset.

MFeat provides six independent flat feature sets extracted from handwritten
digit images.  All modalities share the same MLP architecture; only the
input dimension differs and is passed at construction time.
"""

import torch
import torch.nn as nn

from .utils import BaseEncoder


class MFeatMLPEncoder(BaseEncoder):
    """MLP encoder for a flat MFeat feature vector.

    Architecture: one or more fully-connected hidden blocks followed by a
    linear projection and layer-norm to the shared embedding dimension.
    Each hidden block applies ``Linear → BatchNorm1d → ReLU → Dropout``.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the raw feature vector (modality-specific).
    output_dim : int, optional
        Size of the output embedding (default: 64).
    hidden_dims : list[int] or None, optional
        Widths of the hidden layers.  ``None`` → ``[256]``.
    dropout : float, optional
        Dropout probability after each hidden block (default: 0.3).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 64,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.3,
    ):
        super().__init__(output_dim)

        if hidden_dims is None:
            hidden_dims = [256]

        self.input_dim = input_dim

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.LayerNorm(output_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
