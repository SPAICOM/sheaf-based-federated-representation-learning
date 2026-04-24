"""
Utility functions for federated learning experiments.

This package provides helper functions for:
- Data partitioning: Non-IID data splitting for federated learning
- Communication accounting: Counting transmitted bits/KB consistently
- Graph generation: Creating communication graphs for federated learning
- I/O operations: Managing experiment directories and checkpoints
- Logging: WandB logging, config persistence, and metric extraction
"""

from .anchors import (
    AnchorConfig,
    #build_anchor_bundles,
    #build_semantic_pilot_bundles,
    shared_anchor_rows,
    #supported_anchor_strategy,
)
from .communication import calculate_communication_cost
from .data_partitioner import (
    build_shared_class_partition,
    partition_by_agent_classes,
    partition_non_iid,
    partition_non_iid_with_margin,
)
from .graph_generator import generate_neighbors
from .io import remove_non_empty_dir
from .logging import (
    _extract_objective_metric,
    _filter_supported_init_kwargs,
    _finish_active_wandb_run,
    _json_ready,
    _persist_run_results,
    _resolve_agent_overrides,
    _sanitize_instantiation_config,
    _update_logger_config,
)

__all__ = [
    'AnchorConfig',
    #'build_anchor_bundles',
    #'build_semantic_pilot_bundles',
    'build_shared_class_partition',
    'calculate_communication_cost',
    'generate_neighbors',
    'partition_by_agent_classes',
    'partition_non_iid',
    'partition_non_iid_with_margin',
    'remove_non_empty_dir',
    'shared_anchor_rows',
    #'supported_anchor_strategy',
    '_extract_objective_metric',
    '_filter_supported_init_kwargs',
    '_finish_active_wandb_run',
    '_json_ready',
    '_persist_run_results',
    '_resolve_agent_overrides',
    '_sanitize_instantiation_config',
    '_update_logger_config',
]
