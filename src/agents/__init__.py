"""
Federated learning agent implementations.

This package provides agent classes for federated learning scenarios:
- BaseAgent: Abstract base class defining the agent interface
- LatentClassifier: MLP-based classifier with encoder-decoder structure
- CNNClassifier: CNN-based image classifier
- TransformerClassifier: Vision Transformer image classifier (ViT-style)
- TimmClassifier: Vision transformer/CNN classifier using timm library
- DeepSenseRGBClassifier: RGB camera classifier for the DeepSense dataset
- DeepSenseLiDARClassifier: LiDAR classifier for the DeepSense dataset
- DeepSenseMMWaveClassifier: mmWave classifier for the DeepSense dataset
- MFeatMLPClassifier: single MLP classifier for all MFeat modalities

Each agent implements:
- forward: Standard forward pass returning predictions (logits)
- encode: Pass through encoder only, returning latent features
- compute_loss: Task-specific loss computation
- task_performance: Task-specific performance metric
"""

from .base_agent import BaseAgent
from .cnn_autoencoder import CNNAutoencoder
from .cnn_classifier import CNNClassifier
from .deepsense_classifiers import (
    DeepSenseLiDARClassifier,
    DeepSenseMMWaveClassifier,
    DeepSenseRGBClassifier,
)
from .deepsense_encoders import (
    DeepSenseLiDAREncoder,
    DeepSenseMMWaveEncoder,
    DeepSenseRGBEncoder,
)
from .fmtl_classifier import FMTLCNNClassifier
from .hetero_cnn_classifier import HeteroCNNClassifier
from .latent_classifier import LatentClassifier
from .mfeat_classifiers import MFeatMLPClassifier
from .mfeat_encoders import MFeatMLPEncoder
from .mhealth_classifiers import (
    MHealthAccelerometerClassifier,
    MHealthECGClassifier,
    MHealthGyroscopeClassifier,
    MHealthMagnetometerClassifier,
)
from .mhealth_encoders import (
    MHealthAccelerometerEncoder,
    MHealthECGEncoder,
    MHealthGyroscopeEncoder,
    MHealthMagnetometerEncoder,
)
from .personalized_ae import PersonalizedAE
from .personalized_classifier import PersonalizedClassifier
from .timm_classifier import TimmClassifier
from .transformer_classifier import TransformerClassifier
from .utils import (
    CNN,
    MLP,
    BaseEncoder,
    CNNAEDecoder,
    CNNAEEncoder,
    HeteroCNN,
    HeteroMLP,
    TimmEncoder,
    ViTEncoder,
)

__all__ = [
    'BaseAgent',
    'BaseEncoder',
    'CNN',
    'CNNAEDecoder',
    'CNNAEEncoder',
    'CNNAutoencoder',
    'CNNClassifier',
    'DeepSenseLiDARClassifier',
    'DeepSenseLiDAREncoder',
    'DeepSenseMMWaveClassifier',
    'DeepSenseMMWaveEncoder',
    'DeepSenseRGBClassifier',
    'DeepSenseRGBEncoder',
    'FMTLCNNClassifier',
    'HeteroCNN',
    'HeteroCNNClassifier',
    'HeteroMLP',
    'LatentClassifier',
    'MFeatMLPClassifier',
    'MFeatMLPEncoder',
    'MHealthAccelerometerClassifier',
    'MHealthAccelerometerEncoder',
    'MHealthECGClassifier',
    'MHealthECGEncoder',
    'MHealthGyroscopeClassifier',
    'MHealthGyroscopeEncoder',
    'MHealthMagnetometerClassifier',
    'MHealthMagnetometerEncoder',
    'MLP',
    'PersonalizedAE',
    'PersonalizedClassifier',
    'TimmClassifier',
    'TimmEncoder',
    'TransformerClassifier',
    'ViTEncoder',
]
