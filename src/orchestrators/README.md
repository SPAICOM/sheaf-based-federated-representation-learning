# Orchestrators Module

This module contains orchestrator implementations for federated learning that coordinate training across multiple agents. Orchestrators manage the federated learning process, including agent coordination, communication handling, and algorithm-specific logic for model updates and aggregation.

## Components

### Base Orchestrator
- [`base_orchestrator.py`](base_orchestrator.py): Abstract base orchestrator defining the interface for coordinating federated learning across multiple agents.
  - **Purpose**: Provides common infrastructure for all federated learning algorithms in the framework
  - **Key Responsibilities**:
    - Agent management: Stores and manages multiple agent models using PyTorch Lightning's ModuleDict
    - Communication tracking: Monitors and logs communication costs (bits, bytes, kilobytes) and rounds
    - Metric logging: Handles per-agent and aggregate metric collection for training/validation/test
    - Optimization: Configures and manages optimizers for all agent parameters
    - Lightning integration: Implements training_step, validation_step, test_step, and predict_step interfaces
  - **Extension Points**: 
    - `on_train_epoch_end()`: Must be implemented by subclasses for epoch-level aggregation/updates
    - `_shared_eval()`: Must be implemented by subclasses for evaluation logic (shared between train/val/test)
  - **Communication Tracking**: 
    - Automatically tracks payload sizes when agents communicate
    - Supports different transmission patterns (unicast, broadcast)
    - Logs metrics like communication_bits, communication_bytes, communication_kilobytes, communication_rounds

### Specific Orchestrators
- [`sheaf_frl.py`](sheaf_frl.py): Sheaf-based Federated Representation Learning orchestrator implementing the proposed framework with Sheaf regularization
  - **Algorithm**: 
    - Maintains aligned latent spaces across agents through Stiefel manifold optimization of cross-covariance matrices
    - Uses orthogonal Procrustes problem solution to compute optimal alignment matrices
    - Applies sheaf regularization penalty to encourage latent space alignment
  - **Key Features**:
    - Anchor-based alignment: Selects semantically meaningful anchors for alignment computation
    - Multiple anchor strategies: prototype, random, balanced, semantic_pilots, clustered_pilots, dynamic
    - Parseval normalization option for anchor features
    - Per-agent latent space dimension handling
    - Stiefel matrices (orthogonal constraints) updated via closed-form SVD solutions
  - **Mathematical Foundation**:
    - Solves orthogonal Procrustes problem: min ||A_i - A_j * V||_F subject to V^T * V = I
    - Solution: V = U * W^T where U*Σ*W^T = SVD(A_i^T * A_j)
    - Sheaf penalty: Sum of Frobenius norms of aligned feature differences across edges
  - **Typical Use**: When you want to learn representations that are both task-relevant and aligned across agents

- [`sheaf_fmtl.py`](sheaf_fmtl.py): Sheaf-based Federated Multi-Task Learning orchestrator
  - **Algorithm**: Extends Sheaf FRL to multi-task learning scenarios where each agent may have different but related tasks
  - **Key Features**:
    - Task-specific heads while sharing aligned representations
    - Sheaf regularization applied to shared representation spaces
    - Handles heterogeneous label spaces across agents
  - **Typical Use**: When agents have different label sets but share underlying feature distributions (e.g., different medical conditions from same imaging data)

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

- [`dpsgd.py`](dpsgd.py): Differentially Private SGD orchestrator
  - **Algorithm**: Implements federated learning with differential privacy guarantees using DP-SGD
  - **Key Features**:
    - Gradient clipping to bound sensitivity
    - Gaussian noise addition for privacy
    - Privacy budget tracking (epsilon, delta)
    - Compatible with standard federated learning workflows
  - **Typical Use**: When privacy guarantees are required for sensitive data (medical, financial, personal)

- [`dfedu.py`](dfedu.py): Decentralized Federated Learning with Dual Encoders orchestrator
  - **Algorithm**: Decentralized approach where agents communicate directly with neighbors without central server
  - **Key Features**:
    - Dual encoder architecture for privacy-preserving communication
    - Peer-to-peer agent communication topology
    - No central coordinator required
    - Emergent consensus through local interactions
  - **Typical Use**: When decentralization is important or central server creates bottlenecks/trust issues

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
    parseval_normalization=True # Whether to normalize anchor features
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
- Use `sheaf_fmtl` for multi-task learning with shared representations
- Use `federated` for standard FedAvg baseline
- Use `dpsgd` when differential privacy is required
- Use `dfedu` for fully decentralized setups
- Use `non_cooperative` when modeling strategic agent behavior