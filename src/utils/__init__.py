"""
Utility functions for federated learning experiments.

This package provides helper functions for:
- Graph generation: Creating communication graphs for federated learning
- I/O operations: Managing experiment directories and checkpoints
"""

from .graph_generator import generate_neighbors
from .io import remove_non_empty_dir

__all__ = ['generate_neighbors', 'remove_non_empty_dir']
