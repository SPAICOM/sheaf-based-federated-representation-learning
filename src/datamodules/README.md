# Datamodules Module

This module contains data loading utilities for federated learning scenarios, particularly focused on semantic embedding datasets.

## Components

### Core Datamodules
- [`semantic_datamodule.py`](semantic_datamodule.py): DataModule for loading pre-computed embedding datasets with attributes. Designed for federated learning with multiple agents, supporting custom train/val/test splits per agent and class filtering.
- [`classification_datamodule.py`](classification_datamodule.py): Standard classification DataModule for image datasets with labels.

### Utilities
- [`utils.py`](utils.py): Helper functions for datamodule operations:
  - `compute_split_indices`: Computes deterministic pilot/train/val/test indices from one dataset
  - `repeat_dataset_to_num_samples`: Repeats a dataset until it has at least the requested size

## Usage

### SemanticDataModule

The `SemanticDataModule` is designed for federated learning with pre-computed embeddings:

```python
from src.datamodules.semantic_datamodule import SemanticDataModule

# Create a datamodule for semantic embedding data
datamodule = SemanticDataModule(
    repo="username/dataset-name",
    name="dataset-config",
    attributes=["label", "attribute1", "attribute2"],  # Columns to use as labels
    agents={
        0: {"model": "model_name_1"},  # Agent 0 uses model 1 embeddings
        1: {"model": "model_name_2"}   # Agent 1 uses model 2 embeddings
    },
    batch_size=32,
    num_workers=4
)

# Access dataloaders
train_loader = datamodule.train_dataloader()
val_loader = datamodule.val_dataloader()
test_loader = datamodule.test_dataloader()
```

Each agent gets its own dataloader with the specified embeddings, and all agents share the same underlying dataset splits for consistency.

### ClassificationDataModule

The `ClassificationDataModule` is designed for standard image classification tasks:

```python
from src.datamodules.classification_datamodule import ClassificationDataModule

# Create a datamodule for image classification
datamodule = ClassificationDataModule(
    dataset_name="cifar10",
    data_dir="./data",
    val_split=0.1,
    test_split=0.1,
    batch_size=32,
    num_workers=4
)

# Access dataloaders
train_loader = datamodule.train_dataloader()
val_loader = datamodule.val_dataloader()
test_loader = datamodule.test_dataloader()
```

### Utility Functions

#### compute_split_indices

Use this function to create reproducible data splits for federated learning:

```python
from src.datamodules.utils import compute_split_indices

# Create split indices for a dataset of 1000 samples
indices = compute_split_indices(
    total_size=1000,
    val_split=0.1,
    test_split=0.1,
    seed=42,
    pilot_split=0.05  # Optional pilot set for calibration
)

# Access the splits
train_indices = indices['train']
val_indices = indices['val']
test_indices = indices['test']
pilot_indices = indices['pilot']  # May be empty if pilot_split=0
```

#### repeat_dataset_to_num_samples

Use this function to create a larger dataset by repeating a smaller one:

```python
from src.datamodules.utils import repeat_dataset_to_num_samples
from torch.utils.data import Dataset

# Create a repeated dataset
original_dataset = Dataset(...)  # Some PyTorch dataset
repeated_dataset = repeat_dataset_to_num_samples(
    dataset=original_dataset,
    target_num_samples=10000  # Want at least 10k samples
)

# The repeated dataset will contain enough copies of the original
# to reach or exceed the target size
```

## Implementation Details

### SemanticDataModule Features

1. **Pre-computed Embeddings Support**: Loads datasets with pre-computed embeddings from HuggingFace Hub
2. **Multi-Model Support**: Different agents can use different embedding models from the same dataset
3. **Custom Splits**: Each agent gets custom train/validation/test splits while maintaining consistency across agents for calibration purposes
4. **Class Filtering**: Agents can be restricted to specific classes for non-IID federated learning scenarios
5. **Pilot Sets**: Optional pilot sets for calibration or inspection purposes
6. **Attribute Flexibility**: Supports both single-label and multi-label scenarios through dictionary return types

### Data Flow

1. **prepare_data()**: Downloads datasets from HuggingFace Hub (called once in distributed settings)
2. **setup()**: 
   - Loads datasets for each agent/model combination
   - Merges all available splits (train/validation/test) to maximize data
   - Computes deterministic split indices for reproducibility
   - Creates SemanticDataset instances for each split and agent
   - Infers input dimensions and number of classes from the data
3. **Dataloader Creation**: Returns CombinedLoader instances that iterate over all agent dataloaders in parallel

This design ensures that while each agent trains on its own custom split of the data, they all share the same underlying dataset characteristics, enabling meaningful comparison and aggregation in federated learning scenarios.