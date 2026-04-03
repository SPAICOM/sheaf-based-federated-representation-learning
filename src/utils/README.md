# Utilities Module

This module contains various utility functions and classes used throughout the federated learning framework.

## Components

### Anchor Management
- [`anchors.py`](anchors.py): Functions for building anchor bundles, managing anchor strategies, and handling semantic pilot anchors for federated learning alignment.
  - **Purpose**: Implements mathematically distinct anchor selection strategies for computing alignment in sheaf-based federated learning
  - **Key Functions**:
    - `build_anchor_bundles`: Creates anchor bundles for class-keyed strategies
    - `build_semantic_pilot_bundles`: Creates anchor bundles for semantic pilot strategies
    - `filter_anchor_bundles`: Restricts candidate anchors to a chosen semantic subset
    - `shared_anchor_rows`: Finds shared anchors between agents for alignment computation
    - `supported_anchor_strategy`: Validates and returns normalized anchor strategy names
    - `AnchorConfig`: Configuration class for anchor selection parameters
  - **Strategies**:
    - `prototype`: one prototype per observed class
    - `uniform`: Monte Carlo baseline over raw latent samples, with budget allocated proportionally to class frequency
    - `diversity`: diversity-aware farthest-point anchors within each class
    - `clustering`: centroid anchors from latent-space clustering within each class
    - `semantic_pilots`: anchors keyed by shared pilot sample ids
    - `dynamic`: disagreement-driven based on residuals. Fixes some across batches via `dynamic_persistent_ratio` and refreshed the rest
  - **Usage**: Used by orchestrators to select anchors for computing cross-covariance matrices and alignment

### Communication Tracking
- [`communication.py`](communication.py): Utilities for calculating and tracking communication costs in federated learning systems.
  - **Purpose**: Monitors communication efficiency in federated learning
  - **Key Functions**:
    - `calculate_communication_cost`: Computes the communication cost (size in bits) of transmitting a payload
    - Handles various payload types (tensors, dictionaries, lists, etc.)
    - Accounts for precision (bits vs bytes) and transmission counts
  - **Usage**: Integrated into BaseOrchestrator to track communication rounds and bandwidth usage

### Data Partitioning
- [`data_partitioner.py`](data_partitioner.py): Functions for partitioning data across agents in federated learning scenarios, including label-based and Dirichlet partitioning.
  - **Purpose**: Creates non-IID data distributions for realistic federated learning evaluation
  - **Key Functions**:
    - `partition_by_labels`: Splits data by label classes for class-based partitioning
    - `dirichlet_partition`: Creates non-IID splits using Dirichlet distribution
    - `balance_partition`: Attempts to balance partition sizes
  - **Usage**: Used in experiment setups to create heterogeneous data distributions across agents

### Graph Generation
- [`graph_generator.py`](graph_generator.py): Functions for generating various graph structures (Erdos-Renyi, Barabasi-Albert, fully connected, manual) that define agent communication topologies.
  - **Purpose**: Creates network topologies for defining which agents can communicate in federated learning
  - **Key Functions**:
    - `generate_neighbors`: Main function for generating neighbor dictionaries from graph models
    - Supports multiple graph types:
      - Erdős-Rényi (random): G(n, p) model with edge probability p
      - Barabási-Albert (scale-free): Preferential attachment model
      - Fully connected: Complete graph where all agents communicate with all others
      - Manual: User-defined custom topology
  - **Usage**: Used to define the communication graph that determines agent interactions

### I/O Operations
- [`io.py`](io.py): Input/output utility functions for saving/loading models, results, and other federated learning artifacts.
  - **Purpose**: Handles persistence of federated learning experiments
  - **Key Functions**:
    - Model checkpointing and loading
    - Results saving (metrics, configurations, etc.)
    - Experiment tracking utilities
  - **Usage**: Used throughout training scripts to save and load experiment states

## Usage Pattern

Utilities are typically imported and used as needed throughout the codebase:

```python
# Example: Generating a communication graph
from src.utils.graph_generator import generate_neighbors

# Create a random Erdos-Renyi graph for 10 agents with connection probability 0.3
neighbors = generate_neighbors(
    mode='erdos_renyi',
    n_agents=10,
    p=0.3,
    seed=42
)
# Returns: {0: {1, 3, 4, 7, 8}, 1: {0, 2, 5}, ...}  # Agent indices and their neighbors

# Example: Computing communication cost
from src.utils.communication import calculate_communication_cost
import torch

# Calculate cost of transmitting a tensor to 3 neighbors
tensor = torch.randn(1000, 512)  # 1000 samples, 512 features (float32 = 4 bytes per element)
# Raw size: 1000 * 512 * 4 = 2,048,000 bytes
cost = calculate_communication_cost(tensor, n_transmissions=3)
# Returns: {'bits': 4915200.0, 'bytes': 614400.0, 'kilobytes': 600.0}
# (2,048,000 bytes * 3 transmissions = 6,144,000 bytes = 49,152,000 bits)

# Example: Creating anchor bundles for sheaf alignment
from src.utils.anchors import build_anchor_bundles, AnchorConfig
import torch

# Create anchor configuration
anchor_config = AnchorConfig(
    strategy='prototype',
    num_anchors=5,
    parseval_normalization=True,
    l2_normalization=False,
)

# Simulate agent features and labels
agent_features = {
    0: torch.randn(100, 128),  # 100 samples, 128 features
    1: torch.randn(100, 128)
}
agent_labels = {
    0: torch.randint(0, 10, (100,)),
    1: torch.randint(0, 10, (100,))
}

# Build anchor bundles
A_dict, anchor_keys = build_anchor_bundles(
    agent_features,
    agent_labels,
    anchor_config
)
# Returns selected anchor matrices plus semantic keys for each row

# Example: Partitioning data for non-IID federated learning
from src.utils.data_partitioner import dirichlet_partition
import numpy as np

# Simulate dataset labels
labels = np.array([0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3])

# Create non-IID distribution with alpha=0.5 (lower = more heterogeneous)
partition = dirichlet_partition(
    labels=labels,
    n_parties=3,
    alpha=0.5
)
# Returns: [[indices for party 0], [indices for party 1], [indices for party 2]]
# Each party gets a different distribution of classes

# Example: Saving and loading model checkpoints
from src.utils.io import save_checkpoint, load_checkpoint
import torch
import torch.nn as nn

model = nn.Linear(10, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Save checkpoint
save_checkpoint(
    model=model,
    optimizer=optimizer,
    epoch=42,
    path='./checkpoint_epoch_42.pt',
    extra={'loss': 0.5}
)

# Load checkpoint
checkpoint = load_checkpoint('./checkpoint_epoch_42.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
extra_info = checkpoint.get('extra', {})
```

## Module Relationships

These utilities are designed to work together in federated learning experiments:

1. **Graph Generation + Orchestrators**: Use `graph_generator.py` to create communication topologies that orchestrators use to determine agent interactions
2. **Data Partitioning + Datamodules**: Use `data_partitioner.py` to create non-IID data splits that are then loaded by datamodules
3. **Anchor Management + Orchestrators**: Use `anchors.py` in sheaf-based orchestrators to select anchors for alignment computation
4. **Communication Tracking + Orchestrators**: Orchestrators automatically use `communication.py` to monitor communication efficiency
5. **I/O + Experiment Scripts**: Use `io.py` in training scripts to save/load experiment states and results

Each utility module is focused on a specific concern but designed to integrate seamlessly with the rest of the federated learning framework.
