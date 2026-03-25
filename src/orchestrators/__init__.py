"""
Orchestrators for federated learning coordination.

This package provides orchestration classes that manage training across
multiple federated learning agents:
- FederatedLearning: Federated averaging with neighbor-restricted updates
- SheafFRL: Sheaf-based FRL with latent alignment

Each orchestrator:
- Coordinates multi-agent training in PyTorch Lightning
- Implements epoch-level aggregation strategies
- Handles validation and testing across agents
"""

from .federated import FederatedLearning
from .sheaf_frl import SheafFRL

__all__ = [
    'FederatedLearning',
    'SheafFRL',
]
