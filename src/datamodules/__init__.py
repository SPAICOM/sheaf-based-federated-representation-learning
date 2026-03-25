"""
Data modules for loading and distributing datasets across federated agents.

This package provides PyTorch Lightning data modules for two types:

- ClassificationDataModule: For image classification (CIFAR, MNIST, etc.)
- SemanticDataModule: For pre-computed embeddings with semantic attributes

Both support:
- Loading data from HuggingFace datasets
- Splitting data across multiple agents
- Custom train/validation/test partitions
- Class-based filtering per agent
"""

from .classification_datamodule import ClassificationDataModule
from .semantic_datamodule import SemanticDataModule

__all__ = [
    'ClassificationDataModule',
    'SemanticDataModule',
]
