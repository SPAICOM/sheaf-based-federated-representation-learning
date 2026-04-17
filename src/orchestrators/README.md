# Orchestrators Module

This module contains orchestrator implementations for federated learning that coordinate training across multiple agents. Orchestrators manage the federated learning process, including agent coordination, communication handling, and algorithm-specific logic for model updates and aggregation.

## Components

### Base Orchestrator
- [`base_orchestrator.py`](base_orchestrator.py): Abstract base orchestrator defining the interface for coordinating federated learning across multiple agents.
  - **Purpose**: Provides common infrastructure for all federated learning algorithms in the framework
  - **Key Responsibilities**:
    - Agent management: Stores and manages multiple agent models using PyTorch Lightning's ModuleDict
    - Communication tracking: Monitors and logs cumulative communication volume in kilobytes together with communication rounds
    - Metric logging: Handles per-agent and aggregate metric collection for training/validation/test
    - Optimization: Configures and manages optimizers for all agent parameters
    - Lightning integration: Implements training_step, validation_step, test_step, and predict_step interfaces
  - **Extension Points**: 
    - `on_train_epoch_end()`: Must be implemented by subclasses for epoch-level aggregation/updates
    - `_shared_eval()`: Must be implemented by subclasses for evaluation logic (shared between train/val/test)
  - **Communication Tracking**: 
    - Automatically tracks payload sizes when agents communicate
    - Supports different transmission patterns (unicast, broadcast)
    - Logs cumulative `communication_kilobytes` and `communication_rounds` for each stage

### Specific Orchestrators
- [`sheaf_frl.py`](sheaf_frl.py): Sheaf-based Federated Representation Learning orchestrator implementing the proposed framework with Sheaf regularization
  - **Algorithm**: 
    - Maintains aligned latent spaces across agents through Stiefel manifold optimization of cross-covariance matrices
    - Uses orthogonal Procrustes problem solution to compute optimal alignment matrices
    - Applies sheaf regularization penalty to encourage latent space alignment
  - **Key Features**:
    - Anchor-based alignment: Selects semantically meaningful anchors for alignment computation
    - Multiple anchor strategies:
      - `prototype`: one class prototype per observed class
      - `uniform`: Monte Carlo baseline using raw latent samples, with the global budget allocated proportionally to observed class mass
      - `geometric`: geometric-aware supervised anchors using farthest-point sampling within each class
      - `semantic_pilots`: anchors keyed by shared pilot sample ids from auxiliary `pilot_*` loaders
      - `clustering`: cluster-centroid anchors that maximize latent-space coverage within each class
    - Parseval normalization option for anchor features
    - Per-agent latent space dimension handling
    - Local SGD stays fully local within the epoch; train communication is charged only at epoch end when anchors are shared once for the Stiefel SVD update and the subsequent local sheaf-penalty gradient step
    - Stiefel matrices (orthogonal constraints) updated via closed-form SVD solutions
  - **Mathematical Foundation**:
    - Solves orthogonal Procrustes problem: min ||A_i - A_j * V||_F subject to V^T * V = I
    - Solution: V = U * W^T where U*Σ*W^T = SVD(A_i^T * A_j)
    - Sheaf penalty: Sum of Frobenius norms of aligned feature differences across edges
  - **Typical Use**: When you want to learn representations that are both task-relevant and aligned across agents

- [`sheaf_fmtl.py`](sheaf_fmtl.py): Sheaf-based Federated Multi-Task Learning orchestrator
  - **Algorithm**: Implements the Sheaf-FMTL formulation from "Tackling Feature and Sample Heterogeneity in Decentralized
  Multi-Task Learning: A Sheaf-Theoretic Approach", which introduces the sheaf framework in federated multitask learning by communicating and aligning agents in parameter space.
  - **Key Features**:
    - Trainable projection matrices `P_ij` map each agent's parameter vector into edge-specific shared latent subspaces
    - Sheaf Laplacian penalty is applied directly to local parameter gradients before the optimizer step
    - Projection matrices are updated manually after each training batch following the paper's update rule
    - Communication accounting charges only for exchanged projected vectors `P_ij θ_i`; raw parameter vectors `θ_i` and full matrices `P_ij` are never transmitted
    - One full alternating-gradient iteration incurs two communication rounds: one for the parameter update phase and one for the projection-matrix update phase
  - **Typical Use**: When you want the earlier sheaf-based multitask learning baseline that couples neighboring agents through parameter-space alignment

- [`federated.py`](federated.py): Standard Federated Learning orchestrator implementing Federated Averaging (FedAvg)
  - **Algorithm**: 
    - Agents train locally for multiple epochs
    - Periodic aggregation of model parameters via weighted averaging
    - Central server coordinates the aggregation process
  - **Key Features**:
    - Classical FedAvg approach with optional momentum
    - Flexible local update steps
    - Baseline for comparison with more advanced methods
  - **Typical Use**: As a baseline or when simplicity and broad compatibility are priorities

- [`dpsgd.py`](dpsgd.py): Decentralized Parallel SGD orchestrator
  - **Algorithm**: Implements D-PSGD from "Can Decentralized Algorithms Outperform Centralized Algorithms? A Case Study for Decentralized Parallel Stochastic Gradient Descent", where each agent mixes parameters with its neighbors and then takes its local SGD step
  - **Key Features**:
    - Peer-to-peer parameter mixing with a Metropolis-Hastings doubly stochastic weight matrix
    - Per-step communication performed in `on_before_optimizer_step`
    - Homogeneous-architecture requirement because agents exchange and overwrite flattened parameter vectors
    - No privacy-specific clipping, noise injection, or privacy budget accounting
  - **Typical Use**: When you want a decentralized data-parallel SGD baseline with direct neighbor averaging at every optimization step

- [`dfedu.py`](dfedu.py): d-FedU orchestrator
  - **Algorithm**: Implements the method from "A New Look and Convergence Rate of Federated Multitask Learning With Laplacian Regularization", performing local training followed by an epoch-end Laplacian consensus step in parameter space
  - **Key Features**:
    - End-of-epoch update `w_i <- w_i - eta * sum_j (w_i - w_j)` over graph neighbors
    - Synchronous parameter-space consensus computed from pre-update snapshots
    - Direct parameter mixing across neighbors, so all agents must share the same architecture
    - Fully decentralized communication without a central server
  - **Typical Use**: When you want decentralized federated multitask learning with Laplacian regularization over homogeneous models

- [`non_cooperative.py`](non_cooperative.py): Non-cooperative game-theoretic approach to federated learning
  - **Algorithm**: Models federated learning as a game where agents optimize individual objectives
  - **Key Features**:
    - Agents may have conflicting objectives
    - Equilibrium-seeking behavior rather than global optimization
    - Models strategic interactions between self-interested agents
    - Convergence to Nash equilibrium under certain conditions
  - **Typical Use**: When agents have genuinely different objectives and you want to model strategic behavior

## Usage Pattern

Orchestrators are typically instantiated through Hydra configuration files but can also be created programmatically:

```python
# Example usage (typically instantiated via Hydra config)
from src.orchestrators.sheaf_frl import SheafFRL
from src.agents.timm_classifier import TimmClassifier
import torch

# Create agent models
agents = {
    0: TimmClassifier(model_name="resnet18", pretrained=True, latent_dim=128, num_classes=10),
    1: TimmClassifier(model_name="resnet18", pretrained=True, latent_dim=128, num_classes=10),
    2: TimmClassifier(model_name="resnet18", pretrained=True, latent_dim=128, num_classes=10)
}

# Define communication topology (neighbors for each agent)
neighbors = {
    0: {1, 2},  # Agent 0 communicates with 1 and 2
    1: {0, 2},  # Agent 1 communicates with 0 and 2
    2: {0, 1}   # Agent 2 communicates with 0 and 1
}

# Create optimizer configuration (typically done via Hydra)
optimizer_config = {
    "_target_": "torch.optim.Adam",
    "lr": 0.001
}

# Create the orchestrator
orchestrator = SheafFRL(
    agents=agents,
    neighbors=neighbors,
    optimizer=optimizer_config,
    lambda_sheaf=0.1,           # Sheaf regularization weight
    latent_dims={0: 128, 1: 128, 2: 128},  # Latent dimensions per agent
    anchor_strategy="semantic_pilots",     # Anchor selection strategy
    num_anchors=10,             # Number of anchors per epoch
    parseval_normalization=True, # Whether to normalize anchor features
    l2_normalization=False,
)

# Standard PyTorch Lightning training loop
# trainer = pl.Trainer(max_epochs=100, ...)
# trainer.fit(orchestrator, datamodule)
```

Each orchestrator implements a specific federated learning algorithm while sharing common infrastructure for:
- Agent management and parameter tracking
- Communication cost monitoring
- Metric logging and reporting
- Optimization setup
- PyTorch Lightning integration

The choice of orchestrator depends on your specific federated learning requirements:
- Use `sheaf_frl` for learning aligned representations
- Use `sheaf_fmtl` for the prior sheaf-based multitask learning method with parameter-space alignment
- Use `federated` for standard FedAvg baseline
- Use `dpsgd` for decentralized parallel SGD with per-step neighbor mixing
- Use `dfedu` for decentralized federated multitask learning with Laplacian regularization
- Use `non_cooperative` when modeling strategic agent behavior
