"""Base agent abstract class for federated learning."""

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
