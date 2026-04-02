"""Non-cooperative multi-agent training baseline.

Each agent minimizes only its own task loss. No communication or model
alignment is performed during training, so cumulative communication remains
zero throughout the run.
"""

from __future__ import annotations

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
        return None

    def _shared_eval(
        self,
        batch: dict[int, list[torch.Tensor]],
        batch_idx: int,
        prefix: str,
    ) -> tuple[dict[str, Any], torch.Tensor]:
        """Compute local losses and task metrics without cooperation."""
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
        )

        return outputs, total_loss
