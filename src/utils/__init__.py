"""
Utility functions for federated learning experiments.

This package provides helper functions for:
- Data partitioning: Non-IID data splitting for federated learning
- Communication accounting: Counting transmitted bits/KB consistently
- Graph generation: Creating communication graphs for federated learning
- I/O operations: Managing experiment directories and checkpoints
"""

from .anchors import (
    AnchorConfig,
    build_anchor_bundles,
    build_semantic_pilot_bundles,
    shared_anchor_rows,
    supported_anchor_strategy,
)
from .communication import calculate_communication_cost
from .data_partitioner import partition_non_iid
from .graph_generator import generate_neighbors
from .io import remove_non_empty_dir

__all__ = [
    'AnchorConfig',
    'build_anchor_bundles',
    'build_semantic_pilot_bundles',
    'calculate_communication_cost',
    'generate_neighbors',
    'partition_non_iid',
    'remove_non_empty_dir',
    'shared_anchor_rows',
    'supported_anchor_strategy',
]
