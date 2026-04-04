"""
Orchestrators for federated learning coordination.

This package provides orchestration classes that manage training across
multiple federated learning agents:
- FederatedLearning: Federated averaging with neighbor-restricted updates
- FedPer: centralized shared-base training with personalized heads
- DPSGD: decentralized parallel SGD with per-step model mixing
- DFedU: decentralized federated multitask learning with Laplacian updates
- SheafFMTL: sheaf multitask learning in parameter space
- SheafFRL: sheaf-based FRL with latent alignment
- NonCooperativeLearning: independent local training baseline

Each orchestrator:
- Coordinates multi-agent training in PyTorch Lightning
- Implements epoch-level aggregation strategies
- Handles validation and testing across agents
"""

from .dfedu import DFedU
from .dpsgd import DPSGD
from .fedper import FedPer
from .federated import FederatedLearning
from .non_cooperative import NonCooperativeLearning
from .sheaf_fmtl import SheafFMTL
from .sheaf_frl import SheafFRL

__all__ = [
    'DFedU',
    'DPSGD',
    'FedPer',
    'FederatedLearning',
    'NonCooperativeLearning',
    'SheafFMTL',
    'SheafFRL',
]
