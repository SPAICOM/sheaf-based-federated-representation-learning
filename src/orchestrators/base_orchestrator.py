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

from src.utils import calculate_communication_cost


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

    _COMMUNICATION_SPLITS = ('train', 'validation', 'test')

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
        self._reset_communication_state()

    def _empty_communication_state(self) -> dict[str, float]:
        """Return a zero-initialized communication state."""
        return {
            'kilobytes': 0.0,
            'rounds': 0.0,
        }

    def _reset_communication_state(self) -> None:
        """Reset cumulative communication accounting for all stages."""
        self._communication_by_split = {
            split: self._empty_communication_state()
            for split in self._COMMUNICATION_SPLITS
        }

    def on_train_start(self) -> None:
        """Reset communication accounting at the start of training."""
        self._reset_communication_state()

    def on_validation_start(self) -> None:
        """Reset validation communication accounting at validation start."""
        self._reset_split_communication('validation')

    def on_test_start(self) -> None:
        """Reset test communication accounting at test start."""
        self._reset_split_communication('test')

    def _reset_split_communication(self, prefix: str) -> None:
        """Reset communication accounting for a single stage."""
        self._communication_by_split[prefix] = (
            self._empty_communication_state()
        )

    def _metric_tensor(self, value: Any) -> torch.Tensor:
        """Ensure a metric value is a scalar tensor on the module device.

        Accepts either a ``torch.Tensor`` (returned as-is) or any numeric
        type that can be cast to ``float``. Used to normalise heterogeneous
        outputs from ``compute_loss`` and ``task_performance`` before
        stacking or logging.
        """
        if isinstance(value, torch.Tensor):
            return value
        return torch.tensor(float(value), device=self.device)

    def _record_communication(
        self,
        payload: Any,
        *,
        n_transmissions: int = 1,
        prefix: str = 'train',
    ) -> dict[str, float]:
        """Accumulate communication volume in kilobytes for a payload.

        Parameters
        ----------
        payload : Any
            The tensor or structure being transmitted.
        n_transmissions : int, optional
            Number of neighbours the payload is sent to. The total cost
            is ``payload_size × n_transmissions``. A value < 1 records
            zero cost and is silently ignored (default: 1).
        """
        if int(n_transmissions) < 1:
            return {'kilobytes': 0.0}

        cost = calculate_communication_cost(
            payload,
            n_transmissions=n_transmissions,
        )
        stage_state = self._communication_by_split[prefix]
        stage_state['kilobytes'] += float(cost['kilobytes'])
        return {'kilobytes': float(cost['kilobytes'])}

    def _record_communication_round(
        self,
        n_rounds: int = 1,
        *,
        prefix: str = 'train',
    ) -> None:
        """Accumulate protocol-level communication rounds."""
        if int(n_rounds) < 1:
            return None
        self._communication_by_split[prefix]['rounds'] += float(
            int(n_rounds)
        )
        return None

    def _communication_metrics(self, prefix: str) -> dict[str, float]:
        """Return cumulative communication metrics ready for ``log_dict``.

        Parameters
        ----------
        prefix : str
            Logging stage prefix (e.g. ``'train'``, ``'validation'``).
            Keys are formatted as ``'{prefix}/communication_*'``.
        """
        stage_state = self._communication_by_split[prefix]
        return {
            f'{prefix}/communication_kilobytes': stage_state['kilobytes'],
            f'{prefix}/communication_rounds': stage_state['rounds'],
        }

    def _finalize_train_epoch_communication(self) -> None:
        """Log cumulative train communication metrics once per epoch."""
        communication_logs = self._communication_metrics('train')
        self.log_dict(
            communication_logs,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

    def _finalize_stage_communication(self, prefix: str) -> None:
        """Log cumulative stage communication metrics once per stage epoch."""
        communication_logs = self._communication_metrics(prefix)
        self.log_dict(
            communication_logs,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

    def on_validation_epoch_end(self) -> None:
        """Log cumulative validation communication metrics."""
        self._finalize_stage_communication('validation')

    def on_test_epoch_end(self) -> None:
        """Log cumulative test communication metrics."""
        self._finalize_stage_communication('test')

    def _log_shared_metrics(
        self,
        *,
        prefix: str,
        agent_losses: dict[int, Any],
        agent_performances: dict[int, Any],
        batch_size: int,
        agent_sample_counts: dict[int, int] | None = None,
        total_loss: torch.Tensor | None = None,
        extra_metrics: dict[str, Any] | None = None,
        prog_bar: bool = True,
        per_agent_loss_name: str = 'loss',
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Log per-agent and aggregate metrics for val/test reporting."""
        normalized_losses = {
            idx: self._metric_tensor(loss)
            for idx, loss in agent_losses.items()
        }
        normalized_performances = {
            idx: self._metric_tensor(performance)
            for idx, performance in agent_performances.items()
        }

        per_agent_logs = {}
        for idx in sorted(normalized_losses):
            loss = normalized_losses[idx]
            performance = normalized_performances[idx]
            per_agent_logs[f'{prefix}/loss_agent_{idx}'] = loss
            if per_agent_loss_name != 'loss':
                per_agent_logs[
                    f'{prefix}/{per_agent_loss_name}_agent_{idx}'
                ] = loss
            per_agent_logs[f'{prefix}/task_performance_agent_{idx}'] = (
                performance
            )

        if per_agent_logs:
            self.log_dict(
                per_agent_logs,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
            )

        if normalized_losses:
            losses_tensor = torch.stack(list(normalized_losses.values()))
            total_task_loss = losses_tensor.sum()
        else:
            losses_tensor = torch.tensor([0.0], device=self.device)
            total_task_loss = torch.tensor(0.0, device=self.device)

        if normalized_performances:
            performances_tensor = torch.stack(
                list(normalized_performances.values())
            )
            avg_performance = performances_tensor.mean()
        else:
            performances_tensor = torch.tensor([0.0], device=self.device)
            avg_performance = torch.tensor(0.0, device=self.device)

        if agent_sample_counts:
            sample_weights = torch.tensor(
                [
                    float(max(agent_sample_counts.get(idx, 0), 0))
                    for idx in sorted(normalized_losses)
                ],
                device=self.device,
            )
        else:
            sample_weights = torch.ones(
                len(normalized_losses),
                device=self.device,
            )

        if len(normalized_losses) > 0 and float(sample_weights.sum()) > 0.0:
            global_task_loss = (
                losses_tensor * sample_weights
            ).sum() / sample_weights.sum()
            global_task_performance = (
                performances_tensor * sample_weights
            ).sum() / sample_weights.sum()
        else:
            global_task_loss = torch.tensor(0.0, device=self.device)
            global_task_performance = torch.tensor(0.0, device=self.device)

        resolved_total_loss = (
            total_task_loss
            if total_loss is None
            else self._metric_tensor(total_loss)
        )

        aggregate_logs = {
            f'{prefix}/task_loss_total_epoch': total_task_loss,
            f'{prefix}/global_task_loss_epoch': global_task_loss,
            f'{prefix}/total_loss_epoch': resolved_total_loss,
            f'{prefix}/avg_task_performance_epoch': avg_performance,
            f'{prefix}/global_task_performance_epoch': global_task_performance,
        }
        if extra_metrics is not None:
            aggregate_logs.update(extra_metrics)

        self.log_dict(
            aggregate_logs,
            on_step=False,
            on_epoch=True,
            prog_bar=prog_bar,
            batch_size=batch_size,
        )
        return total_task_loss, avg_performance

    def _resolve_batch_size(
        self,
        batch: dict[int, list[torch.Tensor]]
        | tuple[dict[int, list[torch.Tensor]]],
    ) -> int:
        """Infer an explicit batch size from a multi-agent combined batch."""
        if isinstance(batch, tuple):
            batch = batch[0]

        batch_sizes = []
        for key, values in batch.items():
            if isinstance(key, str) and key.startswith('pilot_'):
                continue
            if not isinstance(values, (list, tuple)) or not values:
                continue
            x = values[0]
            if isinstance(x, torch.Tensor) and x.ndim > 0:
                batch_sizes.append(int(x.shape[0]))

        if batch_sizes:
            return max(batch_sizes)
        return 1

    def _resolve_agent_sample_counts(
        self,
        batch: dict[int, list[torch.Tensor]]
        | tuple[dict[int, list[torch.Tensor]]],
    ) -> dict[int, int]:
        """Infer per-agent sample counts from a combined multi-agent batch."""
        if isinstance(batch, tuple):
            batch = batch[0]

        sample_counts: dict[int, int] = {}
        for idx_str in self.agents:
            idx = int(idx_str)
            key = idx if idx in batch else idx_str
            if key not in batch:
                continue

            values = batch[key]
            if not isinstance(values, (list, tuple)) or len(values) == 0:
                continue

            x = values[0]
            if isinstance(x, torch.Tensor) and x.ndim > 0:
                sample_counts[idx] = int(x.shape[0])

        return sample_counts

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
            String-keyed dict (agent index as str) mapping each agent to
            a ``(y_hat, y)`` pair where ``y_hat`` are raw logits and
            ``y`` are the ground-truth labels.
        """
        # Lightning's CombinedLoader wraps batches in a tuple; unwrap it
        # to get the underlying dict before dispatching to agents.
        if isinstance(batch, tuple):
            batch = batch[0]

        outputs = {}

        # Run forward pass for each agent on their respective data
        for idx, agent in self.agents.items():
            # Handle both string and int keys in batch dictionary
            key = idx if idx in batch else int(idx)
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
            Subclasses must override this method.
        """
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
        self.log(
            'train/total_loss_step',
            loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            batch_size=self._resolve_batch_size(batch),
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
