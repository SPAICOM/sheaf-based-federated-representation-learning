"""Base agent abstract class for federated learning.

This module defines the interface contract that all federated learning agents
must implement. Each agent represents a model that can be trained locally
and potentially shared/aggregated with other agents in the federation.

Subclass Requirements
---------------------
Derived agents must implement:
    - ``forward``: Standard forward pass returning predictions (logits)
    - ``encode``: Pass through encoder only, returning latent features
    - ``compute_loss``: Task-specific loss computation (e.g., CrossEntropy
      for classification)
    - ``task_performance``: Task-specific performance metric (e.g., accuracy)

Usage Example
-------------
    >>> from src.agents.base_agent import BaseAgent
    >>> from src.agents.latent_classifier import LatentClassifier
    >>>
    >>> # Create an agent for 10-class classification with 128-dim input
    >>> agent = LatentClassifier(
    ...     in_features=128, num_classes=10, latent_dim=64
    ... )
    >>>
    >>> # Standard forward pass returns logits
    >>> x = torch.randn(32, 128)
    >>> logits = agent(x)
    >>> print(logits.shape)  # torch.Size([32, 10])
    >>>
    >>> # Encode returns latent features
    >>> features = agent.encode(x)
    >>> print(features.shape)  # torch.Size([32, 64])
    >>>
    >>> # Compute loss and performance metric
    >>> y = torch.randint(0, 10, (32,))
    >>> loss = agent.compute_loss(logits, y)
    >>> accuracy = agent.task_performance(logits, y)
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseAgent(nn.Module, ABC):
    """Abstract base class for all federated learning agents.

    Defines the interface that all federated learning agents must implement.
    Subclasses must provide implementations for forward pass, loss computation,
    and performance evaluation.

    Architecture
    -------------
    All agents must have an explicit encoder-decoder structure:
    - encoder: Maps input to latent representation
    - decoder: Maps latent representation to output predictions

    Notes
    -----
    All derived agents must implement the abstract methods:
    - ``forward``: Standard forward pass returning predictions.
    - ``encode``: Pass through encoder only, returning latent features.
    - ``compute_loss``: Compute task-specific loss from predictions.
    - ``task_performance``: Compute task-specific performance metric.

    All derived agents must implement the properties:
    - ``encoder``: The encoder module.
    - ``decoder``: The decoder module.
    """

    @property
    @abstractmethod
    def encoder(self) -> nn.Module:
        """Encoder module that maps input to latent representation.

        Returns
        -------
        nn.Module
            The encoder network.
        """
        pass

    @property
    @abstractmethod
    def decoder(self) -> nn.Module:
        """Decoder module that maps latent representation to predictions.

        Returns
        -------
        nn.Module
            The decoder network.
        """
        pass

    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Pass through encoder only, returning latent features.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Latent representation from the encoder.
        """
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass of the agent.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output predictions or logits.
        """
        pass

    @abstractmethod
    def compute_loss(
        self, y_hat: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Compute the task-specific loss.

        Parameters
        ----------
        y_hat : torch.Tensor
            Model predictions.
        y : torch.Tensor
            Ground truth labels.

        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        pass

    @abstractmethod
    def task_performance(
        self, y_hat: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor | float:
        """Compute the task-specific performance metric.

        Parameters
        ----------
        y_hat : torch.Tensor
            Model predictions.
        y : torch.Tensor
            Ground truth labels.

        Returns
        -------
        float
            Performance metric value (e.g., accuracy, F1 score).
        """
        pass
