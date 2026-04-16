"""
Orchestrators for federated learning coordination.

This package provides orchestration classes that manage training across
multiple federated learning agents:
- FederatedLearning: Federated averaging with neighbor-restricted updates
- FedPer: centralized shared-base training with personalized heads
- HeteroFL: heterogeneous-width federated learning (Diao et al., ICLR 2021)
- DPSGD: decentralized parallel SGD with per-step model mixing
- DFedU: decentralized federated multitask learning with Laplacian updates
- SheafFMTL: sheaf multitask learning in parameter space
- SheafFRL: sheaf-based FRL with latent alignment
- ComFed: learnable projection matrices aligning class-conditional means
- NonCooperativeLearning: independent local training baseline

Each orchestrator:
- Coordinates multi-agent training in PyTorch Lightning
- Implements epoch-level aggregation strategies
- Handles validation and testing across agents
"""

from .comfed import ComFed
from .dfedu import DFedU
from .dpsgd import DPSGD
from .federated import FederatedLearning
from .fedper import FedPer
from .heterofl import HeteroFL
from .non_cooperative import NonCooperativeLearning
from .sheaf_fmtl import SheafFMTL
from .sheaf_frl import SheafFRL

__all__ = [
    "ComFed",
    "DFedU",
    "DPSGD",
    "FedMuscle",
    "FedPer",
    "FedProto",
    "FederatedLearning",
    "HeteroFL",
    "NonCooperativeLearning",
    "SheafFMTL",
    "SheafFRL",
]
