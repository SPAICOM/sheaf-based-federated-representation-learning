"""Non-cooperative multi-agent training baseline.

Each agent minimizes only its own task loss. No communication or model
alignment is performed during training, so cumulative communication remains
zero throughout the run.
"""

from typing import Any

import torch
import torch.nn as nn

from src.orchestrators.base_orchestrator import BaseOrchestrator


class NonCooperativeLearning(BaseOrchestrator):
    """Independent local-training baseline with shared evaluation logging."""

    def __init__(
        self,
        agents: dict[int, nn.Module],
        neighbors: dict[int, set[int]],
        optimizer: Any,
        **kwargs,
    ):
        super().__init__(
            agents=agents,
            neighbors=neighbors,
            optimizer=optimizer,
        )
        self.save_hyperparameters(ignore=['agents'])

    def on_train_epoch_end(self) -> None:
        """No communication or aggregation is performed."""
        self._finalize_train_epoch_communication()
        return None

    def _shared_eval(
        self,
        batch: dict[int, list[torch.Tensor]],
        batch_idx: int,
        prefix: str,
    ) -> tuple[dict[str, Any], torch.Tensor]:
        """Run each agent's forward pass and compute its own task loss.

        No inter-agent communication takes place: each agent sees only
        its own mini-batch and no gradients or parameters are exchanged.
        Communication counters therefore remain zero for the whole run.

        Parameters
        ----------
        batch : dict[int, list[torch.Tensor]]
            Mapping from agent index to ``(x, y)`` pairs.
        batch_idx : int
            Current batch index (unused, kept for interface compatibility).
        prefix : str
            Logging prefix (e.g. ``'train'``, ``'validation'``).

        Returns
        -------
        tuple[dict, torch.Tensor]
            ``(outputs, total_loss)`` where ``outputs`` maps each agent
            index to its ``(y_hat, y)`` pair and ``total_loss`` is the
            sum of per-agent losses.
        """
        outputs = self(batch)

        agent_losses = {}
        agent_performances = {}

        for idx, agent in self.agents.items():
            y_hat, y = outputs[idx]
            agent_losses[int(idx)] = agent.compute_loss(y_hat, y)
            agent_performances[int(idx)] = agent.task_performance(y_hat, y)

        total_loss, _avg_performance = self._log_shared_metrics(
            prefix=prefix,
            agent_losses=agent_losses,
            agent_performances=agent_performances,
            batch_size=self._resolve_batch_size(batch),
            agent_sample_counts=self._resolve_agent_sample_counts(batch),
        )

        return outputs, total_loss
