"""
Data modules for loading and distributing datasets across federated agents.

This package provides PyTorch Lightning data modules for three dataset types:

- ClassificationDataModule: For image classification (CIFAR, MNIST, etc.)
- HeteroClassificationDataModule: Client-first non-IID classification splits
- SemanticDataModule: For pre-computed embeddings with semantic attributes
- MHealthDataModule: For MHEALTH wearable-sensor activity recognition

All support:
- Splitting data across multiple agents
- Custom train/validation/test partitions
- Shared pilot datasets for federated calibration
- CombinedLoader with optional pilot loaders
"""

from .classification_datamodule import ClassificationDataModule
from .deepsense_datamodule import DeepSenseDataModule
from .hetero_datamodule import HeteroClassificationDataModule
from .mhealth_datamodule import MHealthDataModule
from .semantic_datamodule import SemanticDataModule

__all__ = [
    'ClassificationDataModule',
    'HeteroClassificationDataModule',
    'MHealthDataModule',
    'SemanticDataModule',
    'DeepSenseDataModule',
]
