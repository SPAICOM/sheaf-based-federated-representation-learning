"""
Federated learning agent implementations.

This package provides agent classes for federated learning scenarios:
- BaseAgent: Abstract base class defining the agent interface
- LatentClassifier: MLP-based classifier with encoder-decoder structure
- TimmClassifier: Vision transformer/CNN classifier using timm library
- DeepSenseRGBClassifier: RGB camera classifier for the DeepSense dataset
- DeepSenseLiDARClassifier: LiDAR classifier for the DeepSense dataset
- DeepSenseMMWaveClassifier: mmWave classifier for the DeepSense dataset

Each agent implements:
- forward: Standard forward pass returning predictions (logits)
- encode: Pass through encoder only, returning latent features
- compute_loss: Task-specific loss computation
- task_performance: Task-specific performance metric
"""

from .base_agent import BaseAgent
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
from .fmtl_classifier import FMTLCNNClassifier
from .latent_classifier import LatentClassifier
from .personalized_classifier import PersonalizedClassifier
from .timm_classifier import TimmClassifier
from .utils import CNN, MLP, BaseEncoder, TimmEncoder

__all__ = [
    'BaseAgent',
    'BaseEncoder',
    'CNN',
    'CNNClassifier',
    'DeepSenseLiDARClassifier',
    'DeepSenseLiDAREncoder',
    'DeepSenseMMWaveClassifier',
    'DeepSenseMMWaveEncoder',
    'DeepSenseRGBClassifier',
    'DeepSenseRGBEncoder',
    'FMTLCNNClassifier',
    'FMTLCNNClassifier',
    'LatentClassifier',
    'MLP',
    'PersonalizedClassifier',
    'TimmClassifier',
    'TimmEncoder',
]
