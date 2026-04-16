"""Utility modules for building neural network components.

This module provides reusable neural network building blocks including:
- BaseEncoder: Abstract base class for fixed-output-dim encoders
- CNN: Convolutional encoder with configurable blocks
- TimmEncoder: Wrapper for timm models with preprocessing
- MLP: Fully connected network with flexible configuration

These components are designed to be used as building blocks for federated
learning agents in the sheaf-based distributed representation
learning framework.
"""

import timm
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


class BaseEncoder(nn.Module):
    """Base class for modality encoders that emit fixed-size node embeddings.

    Subclasses must implement ``forward`` and set ``output_dim`` via
    ``super().__init__(output_dim)``.  PersonalizedClassifier reads
    ``encoder.output_dim`` to wire up the decoder automatically.

    Parameters
    ----------
    output_dim : int
        Dimensionality of the flat embedding produced by ``forward``.
    """

    def __init__(self, output_dim: int):
        super().__init__()
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input tensor ``x`` into a latent vector of size ``output_dim``."""
        raise NotImplementedError


class CNN(nn.Module):
    """Convolutional encoder: stacked Conv2d → [BN] → activation → MaxPool2d.

    This module implements a standard CNN architecture suitable for feature
    extraction from 2D inputs like images. It returns spatial feature maps
    that can be further processed (e.g., with global pooling) to obtain
    flat latent vectors.

    The architecture consists of repeated blocks of:
    Conv2d → [BatchNorm2d] → Activation → MaxPool2d → [Dropout2d]

    Parameters
    ----------
    in_features : int
        Number of input channels (e.g., 3 for RGB images, 1 for grayscale).
    hidden_dims : list[int], optional
        Output channels for each Conv2d block. The length of this list
        determines the number of blocks in the network.
        Default: [32, 64, 128].
    dropout : float, optional
        Spatial dropout probability after each block. Applied as Dropout2d.
        Default: 0.0 (no dropout).
    activation : type[nn.Module], optional
        Activation function class to instantiate (not an instance).
        Default: nn.ReLU.
    use_batchnorm : bool, optional
        Whether to add BatchNorm2d after each Conv2d layer.
        Default: False.

    Attributes
    ----------
    out_features : int
        Number of output channels from the final convolutional block.
        Equal to the last element of hidden_dims (or 128 if using defaults).
    layers : nn.Sequential
        Sequential container of all layers in the network.

    Example
    -------
    >>> # Create a CNN for 3-channel images with 3 blocks
    >>> cnn = CNN(in_features=3, hidden_dims=[16, 32, 64])
    >>> x = torch.randn(4, 3, 64, 64)  # Batch of 4 images
    >>> features = cnn(x)  # Shape: (4, 64, 16, 16)
    """

    def __init__(
        self,
        in_features: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.0,
        activation: type[nn.Module] = nn.ReLU,
        use_batchnorm: bool = False,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [32, 64, 128]

        layers = []
        in_ch = in_features
        for out_ch in hidden_dims:
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(
                activation() if isinstance(activation, type) else activation
            )
            layers.append(nn.MaxPool2d(2))
            if dropout > 0:
                layers.append(nn.Dropout2d(p=dropout))
            in_ch = out_ch

        self.out_features: int = hidden_dims[-1]
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply conv blocks and return spatial feature maps (B, C, H, W).

        No pooling or flattening is applied here. The caller (e.g.
        PersonalizedClassifier.encode) is responsible for reducing the
        spatial dimensions to a flat latent vector.
        """
        return self.layers(x)


class TimmEncoder(nn.Module):
    """Timm backbone encoder with built-in input preprocessing.

    Wraps a timm model (classification head removed) and handles all
    input normalization: PIL images, arbitrary spatial sizes, and
    single-channel (grayscale) inputs are all accepted.

    Returns raw backbone features — spatial (4-D) or flat (2-D)
    depending on the underlying model. Pair with
    PersonalizedClassifier, which applies AdaptiveAvgPool2d + flatten
    when the output is spatial.

    Parameters
    ----------
    model_name : str
        timm model identifier (e.g. 'resnet50', 'vit_base_patch16_224').
    pretrained : bool, optional
        Load pretrained weights (default: True).
    freeze_encoder : bool, optional
        Freeze all backbone parameters (default: False).
    """

    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        freeze_encoder: bool = False,
    ):
        super().__init__()

        self._backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0
        )

        if freeze_encoder:
            for param in self._backbone.parameters():
                param.requires_grad = False
            self._backbone.eval()

        # Probe the backbone's output dimensionality with a dummy forward
        # pass. This is necessary because timm models have heterogeneous
        # output shapes (some return 2-D features, others 4-D spatial maps)
        # and there is no unified API to query the feature dimension without
        # running a forward pass.
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            out = self._backbone(dummy)
            if out.ndim == 4:
                out = nn.AdaptiveAvgPool2d((1, 1))(out)
            self.out_features: int = out.flatten(1).shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Preprocess input and return raw backbone features.

        Handles three input formats before passing to the backbone:
        - PIL Image: converted to tensor and resized to 224×224.
        - 3-D tensor (C, H, W): unsqueezed to (1, C, H, W) and resized.
        - 4-D tensor (B, C, H, W): resized only if not already 224×224.

        Single-channel (grayscale) inputs are expanded to 3 channels by
        repeating the channel dimension, allowing RGB backbones to accept
        datasets such as MNIST without altering pretrained weights.

        Returns raw backbone features — spatial (B, C, H, W) or flat
        (B, D) depending on the model. PersonalizedClassifier.encode
        handles pooling and flattening of spatial outputs.
        """
        if isinstance(x, Image.Image):
            x = transforms.Compose(
                [transforms.Resize((224, 224)), transforms.ToTensor()]
            )(x).unsqueeze(0)
        elif x.ndim == 3:
            x = transforms.Resize((224, 224))(x.unsqueeze(0))
        elif x.ndim == 4 and (x.shape[-2] != 224 or x.shape[-1] != 224):
            x = transforms.Resize((224, 224))(x)

        # Expand grayscale (C=1) to pseudo-RGB (C=3)
        if x.ndim == 4 and x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        return self._backbone(x)


class MLP(nn.Module):
    """Fully connected neural network (multi-layer perceptron).

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input features.
    output_dim : int
        Dimensionality of the output features.
    hidden_dims : list[int] | None, optional
        List of hidden layer dimensions. If None or empty, the network
        reduces to a single linear layer.
    activation : type[nn.Module], optional
        Activation function class (e.g., nn.ReLU, nn.GELU).
        Default: nn.ReLU.
    dropout : float, optional
        Dropout probability applied after each hidden layer (default: 0.0).
    use_batchnorm : bool, optional
        Whether to include BatchNorm1d after each linear layer
        (default: False).

    Example
    -------
    >>> mlp = MLP(input_dim=128, output_dim=10, hidden_dims=[64, 32])
    >>> x = torch.randn(32, 128)
    >>> y = mlp(x)  # torch.Size([32, 10])
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list[int] | None = None,
        activation: type[nn.Module] = nn.ReLU,
        dropout: float = 0.0,
        use_batchnorm: bool = False,
    ):
        """Build the MLP network from input to output dimensions.

        Parameters
        ----------
        input_dim : int
            Dimensionality of the input features.
        output_dim : int
            Dimensionality of the output features.
        hidden_dims : list[int] | None, optional
            List of hidden layer dimensions. If None or empty, the network
            reduces to a single linear layer.
        activation : type[nn.Module], optional
            Activation function class (e.g., nn.ReLU, nn.GELU).
            Default: nn.ReLU.
        dropout : float, optional
            Dropout probability applied after each hidden layer (default: 0.0).
        use_batchnorm : bool, optional
            Whether to include BatchNorm1d after each linear layer
            (default: False).
        """
        super().__init__()

        # Default to empty list if no hidden layers specified
        if hidden_dims is None:
            hidden_dims = []

        layers = []
        prev_dim = input_dim

        # Build hidden layers sequentially
        # Each layer: Linear -> [BatchNorm] -> Activation -> [Dropout]
        for h_dim in hidden_dims:
            # Linear transformation from previous dimension to hidden dimension
            layers.append(nn.Linear(prev_dim, h_dim))

            # Optional batch normalization after linear transformation
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h_dim))

            # Activation function (handle both class and instance)
            layers.append(
                activation() if isinstance(activation, type) else activation
            )

            # Optional dropout for regularization
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            # Output of this layer becomes input to next
            prev_dim = h_dim

        # Final output layer (no activation, dropout, or batchnorm)
        layers.append(nn.Linear(prev_dim, output_dim))

        # Compose all layers into sequential network
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the MLP.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, output_dim).
        """
        return self.network(x)


# --------------------------------------------
# ----------- MODULES FOR HETEROFL -----------
# --------------------------------------------


class Scaler(nn.Module):
    """Scaler module from "HeteroFL [...]".
    It is required to keep the magnitude of the gradients
    comparable across clients according to the model
    inherent complexity width-wise.

    Parameters
    ----------
    rate : float
        Determines the scaling factor of the method (r ** p)
    """

    def __init__(self, rate: float) -> None:
        super().__init__()
        assert rate != 0.0, 'Rate must be non-zero'
        self.rate = rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return x / self.rate


class HeteroConvLayer(nn.Module):
    """Convolutional layer for HeteroFL
    Concatenates Conv, Scaler, statBN and ReLU

    Parameters
    ----------
    in_ch : int
        Number of input features
    out_ch : int
        Number of output features
    rate : float
        Shrinking rate (required to initialize the Scaler module)
    use_batchnorm : bool
        Flag for the (static) batch normalization layer
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        rate: float = 1.0,
        use_batchnorm: bool = True,
        **conv_kwargs,
    ) -> None:
        super().__init__()

        # Defining the blocks of the layer
        layers = [
            nn.Conv2d(in_ch, out_ch, **conv_kwargs),
            Scaler(rate),
        ]

        if use_batchnorm:
            # The static Batch Normalization layer requires
            # the statistics not to be tracked during training
            layers.append(nn.BatchNorm2d(out_ch, track_running_stats=False))

        layers.append(nn.ReLU(inplace=True))

        self.layer = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.layer(x)


class HeteroCNN(nn.Module):
    """Heterogeneous Convolutional encoder.

    This module implements a hetero CNN architecture suitable for feature
    extraction from 2D inputs like images. It returns spatial feature maps
    that can be further processed (e.g., with global pooling) to obtain
    flat latent vectors.

    The architecture consists of repeated blocks of HeteroConvLayer

    Parameters
    ----------
    in_features : int
        Number of input channels (e.g., 3 for RGB images, 1 for grayscale).
    hidden_dims : list[int], optional
        Output channels for each Conv2d block. The length of this list
        determines the number of blocks in the network.
        Default: [32, 64, 128].
    rate : float, optional
        Width shrinking ratio for channel-level heterogeneity
    dropout : float, optional
        Spatial dropout probability after each block. Applied as Dropout2d.
        Default: 0.0 (no dropout).
    activation : type[nn.Module], optional
        Activation function class to instantiate (not an instance).
        Default: nn.ReLU.
    use_batchnorm : bool, optional
        Whether to add BatchNorm2d after each Conv2d layer.
        Default: False.

    Attributes
    ----------
    out_features : int
        Number of output channels from the final convolutional block.
        Equal to the last element of hidden_dims (or 128 if using defaults).
    layers : nn.Sequential
        Sequential container of all layers in the network.
    """

    def __init__(
        self,
        in_features: int,
        hidden_dims: list[int] | None = None,
        rate: float = 1.0,
        dropout: float = 0.0,
        use_batchnorm: bool = False,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [32, 64, 128]

        layers = []
        in_ch = in_features
        for out_ch in hidden_dims:
            scaled_out_ch = int(out_ch * rate)
            layers.append(
                HeteroConvLayer(
                    in_ch,
                    scaled_out_ch,
                    rate,
                    use_batchnorm,
                    kernel_size=3,
                    padding=1,
                )
            )

            layers.append(nn.MaxPool2d(2))
            if dropout > 0:
                layers.append(nn.Dropout2d(p=dropout))
            in_ch = scaled_out_ch

        self.out_features: int = int(hidden_dims[-1] * rate)
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply conv blocks and return spatial feature maps (B, C, H, W).

        No pooling or flattening is applied here. The caller (e.g.
        PersonalizedClassifier.encode) is responsible for reducing the
        spatial dimensions to a flat latent vector.
        """
        return self.layers(x)


class HeteroMLP(nn.Module):
    """Fully connected neural network
    (multi-layer perceptron) for heterogenous settings.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input features.
    output_dim : int
        Dimensionality of the output features.
    hidden_dims : list[int] | None, optional
        List of hidden layer dimensions. If None or empty, the network
        reduces to a single linear layer.
    rate : float
        Width shrinking ratio for channel-level heterogeneity
    activation : type[nn.Module], optional
        Activation function class (e.g., nn.ReLU, nn.GELU).
        Default: nn.ReLU.
    dropout : float, optional
        Dropout probability applied after each hidden layer (default: 0.0).
    use_batchnorm : bool, optional
        Whether to include BatchNorm1d after each linear layer
        (default: False).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list[int] | None = None,
        rate: float = 1.0,
        activation: type[nn.Module] = nn.ReLU,
        dropout: float = 0.0,
        use_batchnorm: bool = False,
    ):
        """Build the MLP network from input to output dimensions.

        Parameters
        ----------
        input_dim : int
            Dimensionality of the input features.
        output_dim : int
            Dimensionality of the output features.
        hidden_dims : list[int] | None, optional
            List of hidden layer dimensions. If None or empty, the network
            reduces to a single linear layer.
        activation : type[nn.Module], optional
            Activation function class (e.g., nn.ReLU, nn.GELU).
            Default: nn.ReLU.
        dropout : float, optional
            Dropout probability applied after each hidden layer (default: 0.0).
        use_batchnorm : bool, optional
            Whether to include BatchNorm1d after each linear layer
            (default: False).
        """
        super().__init__()

        # Default to empty list if no hidden layers specified
        if hidden_dims is None:
            hidden_dims = []

        layers = []
        prev_dim = input_dim

        # Build hidden layers sequentially
        # Each layer: Linear -> [BatchNorm] -> Activation -> [Dropout]
        for h_dim in hidden_dims:
            scaled_h_dim = int(rate * h_dim)
            # Linear transformation from previous dimension to hidden dimension
            # with Scaler adapter
            layers.append(nn.Linear(prev_dim, scaled_h_dim))
            layers.append(Scaler(rate))

            # Optional batch normalization after linear transformation
            if use_batchnorm:
                layers.append(
                    nn.BatchNorm1d(scaled_h_dim, track_running_stats=False)
                )

            # Activation function (handle both class and instance)
            layers.append(
                activation() if isinstance(activation, type) else activation
            )

            # Optional dropout for regularization
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            # Output of this layer becomes input to next
            prev_dim = scaled_h_dim

        # Final output layer (no activation, dropout, or batchnorm)
        layers.append(nn.Linear(prev_dim, output_dim))

        # Compose all layers into sequential network
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the MLP.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, output_dim).
        """
        return self.network(x)
