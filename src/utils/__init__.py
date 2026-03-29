"""
Utility functions for federated learning experiments.

This package provides helper functions for:
- Data partitioning: Non-IID data splitting for federated learning
- Graph generation: Creating communication graphs for federated learning
- I/O operations: Managing experiment directories and checkpoints
"""

from .data_partitioner import partition_non_iid
from .graph_generator import generate_neighbors
from .io import remove_non_empty_dir

__all__ = ['generate_neighbors', 'partition_non_iid', 'remove_non_empty_dir']

