# Agents Module

This module contains implementations of federated learning agents that can be trained locally and potentially shared/aggregated with other agents in the federation. Each agent represents a model with explicit encoder-decoder architecture that can participate in federated learning scenarios.

## Components

### Base Classes
- [`base_agent.py`](base_agent.py): Abstract base class defining the interface that all federated learning agents must implement. 
  - **Purpose**: Establishes the contract that all agents must follow to ensure compatibility with the federated learning framework
  - **Key Requirements**: 
    - `forward()`: Standard forward pass returning predictions (logits)
    - `encode()`: Pass through encoder only, returning latent features
    - `compute_loss()`: Task-specific loss computation (e.g., CrossEntropy for classification)
    - `task_performance()`: Task-specific performance metric (e.g., accuracy)
    - Properties `encoder` and `decoder` for accessing the respective components
  - **Usage**: All specific agent implementations inherit from this base class

### Specific Agent Implementations
- [`cnn_classifier.py`](cnn_classifier.py): CNN-based classifier agent designed for image classification tasks
  - **Architecture**: Uses configurable CNN blocks (Conv2d → [BatchNorm] → Activation → MaxPool2d → [Dropout]) as feature extractor
  - **Best For**: Image data where spatial hierarchies are important
  - **Configuration Options**: 
    - Number and size of filters per layer
    - Activation functions (ReLU, GELU, etc.)
    - Batch normalization usage
    - Dropout rates for regularization
  - **Typical Use**: Paired with a final classification head for tasks like CIFAR-10, ImageNet subsets

- [`latent_classifier.py`](latent_classifier.py): MLP-based classifier that builds on top of PersonalizedClassifier
  - **Architecture**: Simple MLP encoder (input → hidden layers → latent space) combined with a classifier
  - **Best For**: Feature vectors or already-processed embeddings where a simple transformation is sufficient
  - **Configuration Options**:
    - Encoder hidden layer dimensions
    - Latent space dimensionality
    - Decoder configuration (inherited from PersonalizedClassifier)
  - **Typical Use**: When working with pre-computed embeddings or simple tabular data

- [`personalized_classifier.py`](personalized_classifier.py): Highly customizable classifier agent with flexible encoder-decoder architecture
  - **Architecture**: Modular design allowing independent configuration of encoder and decoder
  - **Best For**: Research scenarios requiring experimentation with different architectural choices
  - **Configuration Options**:
    - Encoder type (MLP, CNN, or custom)
    - Latent space dimensionality
    - Decoder hidden layer dimensions
    - Activation functions, dropout, batch normalization for both encoder and decoder
  - **Typical Use**: As a base class for specialized agents like LatentClassifier, or when maximum flexibility is needed

- [`timm_classifier.py`](timm_classifier.py): Agent using Timm library backbone encoders with preprocessing capabilities
  - **Architecture**: Wrapper around Timm models (ResNet, ViT, EfficientNet, etc.) with classification head removed
  - **Best For**: Leveraging state-of-the-art pretrained models with minimal configuration
  - **Key Features**:
    - Automatic input preprocessing (handles PIL images, arbitrary sizes, grayscale→RGB conversion)
    - Configurable pretrained weight loading
    - Option to freeze backbone parameters
    - Compatible with hundreds of architectures from the Timm library
  - **Typical Use**: When you want to use powerful pretrained vision models without dealing with preprocessing complexities

### Utilities
- [`utils.py`](utils.py): Collection of reusable neural network building blocks used across agent implementations
  - **CNN**: 
    - Configurable convolutional encoder with options for batch normalization, activation functions, and dropout
    - Returns spatial feature maps suitable for further processing (e.g., with global pooling)
    - Building block used in cnn_classifier.py
  - **TimmEncoder**:
    - Wrapper for Timm models with comprehensive input preprocessing
    - Handles various input formats (PIL images, tensors of different dimensions, grayscale images)
    - Automatically determines output feature dimensions through dummy forward pass
    - Building block used in timm_classifier.py
  - **MLP**:
    - Fully connected network with flexible hidden layer configuration
    - Options for activation functions, dropout, and batch normalization
    - Building block used in latent_classifier.py and as encoder in other agents
    - Can be configured as simple linear layer or deep network

## Usage Pattern

All agents follow a consistent interface that enables interchangeability in federated learning scenarios:

```python
# Import the base class for type hints or custom implementations
from src.agents.base_agent import BaseAgent

# Import specific agent implementations
from src.agents.cnn_classifier import CNNClassifier
from src.agents.latent_classifier import LatentClassifier
from src.agents.personalized_classifier import PersonalizedClassifier
from src.agents.timm_classifier import TimmClassifier

# All agents can be used interchangeably due to common interface
def create_agent(agent_type: str, **kwargs) -> BaseAgent:
    if agent_type == "cnn":
        return CNNClassifier(**kwargs)
    elif agent_type == "latent":
        return LatentClassifier(**kwargs)
    elif agent_type == "personalized":
        return PersonalizedClassifier(**kwargs)
    elif agent_type == "timm":
        return TimmClassifier(**kwargs)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

# Example usage in federated learning setup
agent = create_agent(
    agent_type="timm",
    model_name="vit_base_patch16_224",
    pretrained=True,
    latent_dim=256,
    num_classes=10
)

# Standard operations work identically across all agent types
logits = agent(input_batch)           # Forward pass
features = agent.encode(input_batch)  # Encoding only
loss = agent.compute_loss(logits, labels)  # Loss computation
accuracy = agent.task_performance(logits, labels)  # Performance metric
```

This common interface is what enables the federated learning framework to work with diverse model architectures while maintaining consistent training and evaluation procedures.