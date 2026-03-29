"""
Federated learning agent implementations.

This package provides agent classes for federated learning scenarios:
- BaseAgent: Abstract base class defining the agent interface
- LatentClassifier: MLP-based classifier with encoder-decoder structure
- TimmClassifier: Vision transformer/CNN classifier using timm library

Each agent implements:
- forward: Standard forward pass returning predictions (logits)
- encode: Pass through encoder only, returning latent features
- compute_loss: Task-specific loss computation
- task_performance: Task-specific performance metric
"""

from .base_agent import BaseAgent
from .cnn_classifier import CNNClassifier
from .latent_classifier import LatentClassifier
from .timm_classifier import TimmClassifier

__all__ = [
    'BaseAgent',
    'CNNClassifier',
    'LatentClassifier',
    'TimmClassifier',
]
