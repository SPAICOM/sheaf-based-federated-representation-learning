"""
Base orchestrator for federated learning with multiple agents.

This module defines the abstract interface for orchestrating training
across multiple agents in a federated learning setting. Subclasses must
implement epoch-end aggregation logic and evaluation procedures.
"""

from abc import ABC, abstractmethod
from typing import Any

import lightning as l
import torch
import torch.nn as nn
from hydra.utils import instantiate


class BaseOrchestrator(l.LightningModule, ABC):
    """Abstract base orchestrator for federated learning with multiple agents.

    Base class that defines the interface for orchestrating training across
    multiple agents in a federated learning setting. Subclasses must implement
    epoch-end aggregation logic and evaluation procedures.

    Parameters
    ----------
    agents : dict[int, nn.Module]
        Dictionary mapping agent indices to their model instances. Must not be
        empty.
    neighbors : dict[int, set[int]]
        Dictionary mapping each agent index to the set of its neighbor indices.
    optimizer : hydra config
        Optimizer configuration for training.

    Notes
    -----
    - Agents are stored in a ``ModuleDict`` so Lightning can track parameters.
    - Subclasses must implement ``on_train_epoch_end`` for epoch-level ops.
    - Subclasses must implement ``_shared_eval`` for evaluation logic.
    """

    def __init__(
        self,
        agents: dict[int, nn.Module],
        neighbors: dict[int, set[int]],
        optimizer,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['agents', 'cfg'])

        assert len(agents) > 0, 'The "agents" dictionary must be not empty'

        self.agents = nn.ModuleDict(
            {str(idx): agent for idx, agent in agents.items()}
        )

    @abstractmethod
    def on_train_epoch_end(self):
        """Perform epoch-level aggregation or updates.

        Called at the end of each training epoch. Subclasses should implement
        this method to perform operations such as federated averaging,
        parameter synchronization, or model aggregation across agents.
        """
        pass

    def forward(
        self,
        batch: dict[int, list[torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        """Forward pass for multiple agents.

        Parameters
        ----------
        batch : dict[int, list[torch.Tensor]]
            Dictionary mapping agent indices to (input, label) pairs.

        Returns
        -------
        dict[str, tuple[torch.Tensor, torch.Tensor]]
            Dictionary mapping agent indices to (prediction, label) pairs.
        """
        # Handle tuple input (CombinedLoader returns tuple)
        if isinstance(batch, tuple):
            batch = batch[0]

        outputs = {}

        # Run forward pass for each agent on their respective data
        for idx, agent in self.agents.items():
            # Handle both string and int keys in batch dictionary
            key = str(idx) if str(idx) in batch else idx
            x, y = batch[key]

            # Get predictions (logits) from agent
            y_hat = agent(x)

            # Store predictions with labels for loss computation
            outputs[idx] = (y_hat, y)

        return outputs

    def _shared_eval(
        self,
        batch: dict[int, list[torch.Tensor]],
        batch_idx: int,
        prefix: str,
    ) -> tuple[dict[str, Any], torch.Tensor]:
        """Compute losses and metrics for validation/test steps.

        Parameters
        ----------
        batch : dict[int, list[torch.Tensor]]
            Dictionary mapping agent indices to (input, label) pairs.
        batch_idx : int
            Index of the current batch.
        prefix : str
            Prefix for logging (e.g., 'train', 'validation', 'test').

        Returns
        -------
        tuple[dict, torch.Tensor]
            Tuple of (outputs, total_loss) where outputs maps agent indices to
            (prediction, label) pairs and total_loss is the summed loss.

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses.
        """
        # Base implementation raises NotImplementedError
        # Subclasses must implement this method to compute losses and metrics
        raise NotImplementedError(
            f'{self.__class__.__name__} must implement _shared_eval'
        )

    def training_step(
        self,
        batch: dict[int, list[torch.Tensor]],
        batch_idx: int,
    ) -> torch.Tensor:
        """Execute a single training step.

        Parameters
        ----------
        batch : dict[int, list[torch.Tensor]]
            Dictionary mapping agent indices to (input, label) pairs.
        batch_idx : int
            Index of the current batch.

        Returns
        -------
        torch.Tensor
            Training loss for the step.
        """
        _, loss = self._shared_eval(
            batch=batch,
            batch_idx=batch_idx,
            prefix='train',
        )
        return loss

    def test_step(
        self,
        batch: dict[int, list[torch.Tensor]],
        batch_idx: int,
    ) -> None:
        """Execute a single test step.

        Parameters
        ----------
        batch : dict[int, list[torch.Tensor]]
            Dictionary mapping agent indices to (input, label) pairs.
        batch_idx : int
            Index of the current batch.
        """
        self._shared_eval(
            batch=batch,
            batch_idx=batch_idx,
            prefix='test',
        )

    def validation_step(
        self,
        batch: dict[int, list[torch.Tensor]],
        batch_idx: int,
    ) -> dict[int, torch.Tensor]:
        """Execute a single validation step.

        Parameters
        ----------
        batch : dict[int, list[torch.Tensor]]
            Dictionary mapping agent indices to (input, label) pairs.
        batch_idx : int
            Index of the current batch.

        Returns
        -------
        dict[int, tuple[torch.Tensor, torch.Tensor]]
            Dictionary mapping agent indices to (prediction, label) pairs.
        """
        output, _ = self._shared_eval(
            batch=batch,
            batch_idx=batch_idx,
            prefix='validation',
        )
        return output

    def predict_step(
        self,
        batch: dict[int, list[torch.Tensor]],
        batch_idx: int,
    ) -> dict[int, torch.Tensor]:
        """Execute a single prediction step.

        Parameters
        ----------
        batch : dict[int, list[torch.Tensor]]
            Dictionary mapping agent indices to (input, label) pairs.
        batch_idx : int
            Index of the current batch.

        Returns
        -------
        dict[int, tuple[torch.Tensor, torch.Tensor]]
            Dictionary mapping agent indices to (prediction, label) pairs.
        """
        return self(batch)

    def configure_optimizers(self) -> dict[str, object]:
        """Configure the optimizer for training.

        Returns
        -------
        dict[str, object]
            Dictionary containing the configured optimizer.
        """
        optimizer = instantiate(
            self.hparams.optimizer,
            params=self.parameters(),
        )
        return {
            'optimizer': optimizer,
        }


if __name__ == '__main__':
    pass
