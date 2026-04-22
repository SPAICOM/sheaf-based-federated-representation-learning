# Utilities Module

This module contains various utility functions and classes used throughout the federated learning framework.

## Components

### Anchor Management
- [`anchors.py`](anchors.py): Utilities for normalizing, matching, and accounting for communicated anchor tensors in sheaf-based federated learning.
  - **Purpose**: Implements the current anchor-processing path used by `SheafFRL`
  - **Key Functions**:
    - `AnchorConfig`: Controls normalization, prototype compression, and unseen-class filtering
    - `normalize_anchor_matrix`: Applies Parseval or row-wise L2 normalization
    - `shared_anchor_rows`: Aligns two anchor matrices using both agents' class labels, optionally compressing each side to per-class prototypes first
    - `communication_anchor_payload`: Computes the actual transmitted anchor payload after any prototype compression
  - **Usage**: Used by `SheafFRL` to construct communication payloads, compute the per-step sheaf penalty, and align cached anchors for the epoch-end Stiefel update

### Communication Tracking
- [`communication.py`](communication.py): Utilities for calculating and tracking communication costs in federated learning systems.
  - **Purpose**: Monitors communication efficiency in federated learning
  - **Key Functions**:
    - `calculate_communication_cost`: Computes the communication cost (size in bits) of transmitting a payload
    - Handles various payload types (tensors, dictionaries, lists, etc.)
    - Accounts for precision (bits vs bytes) and transmission counts
  - **Usage**: Integrated into BaseOrchestrator to track cumulative communication rounds and exchanged kilobytes

### Data Partitioning
- [`data_partitioner.py`](data_partitioner.py): Functions for partitioning data across agents in federated learning scenarios, including class-partition, standard non-IID, and safety-margin non-IID splits.
  - **Purpose**: Creates non-IID data distributions for realistic federated learning evaluation
  - **Key Functions**:
    - `build_shared_class_partition`: Assigns a controlled number of globally shared classes across agents
    - `partition_by_agent_classes`: Builds disjoint per-agent splits from an explicit agent-to-class mapping
    - `partition_non_iid`: Standard non-IID splitter with reused or sampled class assignments and skew controlled by `alpha`
    - `partition_non_iid_with_margin`: Exact-`K` class assignment with full class coverage and a per-class safety margin reserved for every assigned agent before the skewed allocation
    - `partition_non_iid_fair`: Exact-`K` non-IID split with per-agent normalized class skew, safety margins, and equal total samples per agent
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

# Example: Matching anchors for sheaf alignment
from src.utils.anchors import AnchorConfig, shared_anchor_rows
import torch

# Create anchor configuration
anchor_config = AnchorConfig(
    parseval_normalization=True,
    l2_normalization=False,
    filter_unseen_classes=True,
    use_prototypes=True,
)

# Simulate agent anchor matrices and labels
A_i = torch.randn(32, 128)
A_j = torch.randn(24, 128)
labels_i = torch.randint(0, 10, (32,))
labels_j = torch.randint(0, 10, (24,))

# Match anchors shared across both agents
matched = shared_anchor_rows(
    A_i=A_i,
    A_j=A_j,
    labels_i=labels_i,
    labels_j=labels_j,
    seen_i=set(labels_i.tolist()),
    seen_j=set(labels_j.tolist()),
    config=anchor_config,
)
# Returns aligned anchor tensors or None if the two agents share no classes

# Example: Partitioning data for non-IID federated learning with a safety margin
from src.utils.data_partitioner import partition_non_iid_with_margin

# Simulate dataset labels
labels = [idx % 10 for idx in range(500)]

# Create an exact-K non-IID split with per-class safety margins
partition = partition_non_iid_with_margin(
    labels=labels,
    n_agents=5,
    classes_per_agent=3,
    alpha=-1.0,
    safety_margin=10,
)
# Returns: {0: [...], 1: [...], ...}
# Each agent gets exactly 3 classes and at least 10 reserved samples per assigned class

# Fair variant: per-agent label skew with uniform client volume
from src.utils.data_partitioner import partition_non_iid_fair

fair_partition = partition_non_iid_fair(
    labels=labels,
    n_agents=5,
    classes_per_agent=3,
    alpha=-1.0,
    safety_margin=10,
)
# Returns equal-size client partitions when len(labels) is divisible by n_agents

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
