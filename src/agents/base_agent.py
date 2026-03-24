"""Base agent abstract class for federated learning.

This module defines the interface contract that all federated learning agents
must implement. Each agent represents a model that can be trained locally
and potentially shared/aggregated with other agents in the federation.

Subclass Requirements
---------------------
Derived agents must implement four abstract methods:
    - ``forward``: Standard forward pass returning predictions (logits)
    - ``forward_with_features``: Returns predictions AND intermediate
      latent features
    - ``compute_loss``: Task-specific loss computation (e.g., CrossEntropy
      for classification)
    - ``task_performance``: Task-specific metric (e.g., accuracy, F1, MSE)

Usage Example
-------------
    >>> from src.agents.base_agent import BaseAgent
    >>> from src.agents.latent_classifier import LatentClassifier
    >>>
    >>> # Create an agent for 10-class classification with 128-dim input
    >>> agent = LatentClassifier(in_features=128, num_classes=10)
    >>>
    >>> # Standard forward pass returns logits
    >>> x = torch.randn(32, 128)
    >>> logits = agent(x)
    >>> print(logits.shape)  # torch.Size([32, 10])
    >>>
    >>> # Forward with features returns logits and latent representation
    >>> logits, features = agent.forward_with_features(x)
    >>> print(features.shape)  # torch.Size([32, 64]) (last hidden layer)
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

    Notes
    -----
    All derived agents must implement the abstract methods:
    - ``forward``: Standard forward pass returning predictions.
    - ``forward_with_features``: Forward pass returning both predictions and
      intermediate features.
    - ``compute_loss``: Compute task-specific loss from predictions.
    - ``task_performance``: Compute task-specific performance metric.
    """

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
    def forward_with_features(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning predictions and intermediate features.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Tuple of (predictions, features) where features are the
            intermediate latent representation.
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
    def task_performance(self, y_hat: torch.Tensor, y: torch.Tensor) -> float:
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
