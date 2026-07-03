"""Non-cooperative multi-agent training baseline.

Each agent minimizes only its own task loss. No communication or model
alignment is performed during training, so cumulative communication remains
zero throughout the run.
"""

from typing import Any

import torch
import torch.nn as nn

from src.communication.alignment_mixin import (
    VALID_ALIGNMENT_METHODS,
    PostTrainingAlignmentMixin,
)
from src.orchestrators.base_orchestrator import BaseOrchestrator


class NonCooperativeLearning(PostTrainingAlignmentMixin, BaseOrchestrator):
    """Independent local-training baseline with shared evaluation logging."""

    def __init__(
        self,
        agents: dict[int, nn.Module],
        neighbors: dict[int, set[int]],
        optimizer: Any,
        alignment_method: str = 'general',
        **kwargs,
    ):
        super().__init__(
            agents=agents,
            neighbors=neighbors,
            optimizer=optimizer,
            **kwargs,
        )
        alignment_method = str(alignment_method)
        if alignment_method not in VALID_ALIGNMENT_METHODS:
            raise ValueError(
                f"Unknown alignment_method '{alignment_method}'. "
                f'Valid options: {VALID_ALIGNMENT_METHODS}'
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
            skip_task_performance=(prefix == 'test'),
        )

        return outputs, total_loss

    # ── Post-hoc alignment evaluation ─────────────────────────────────────────
    # _fit_alignment_maps, send_message, _cleanup_alignment and
    # evaluate_communication_accuracy (which skips edges without a fitted map)
    # all come from PostTrainingAlignmentMixin.  The 'general' alignment_method
    # default means maps are always fitted before evaluation.

    def on_test_epoch_end(self) -> None:
        super().on_test_epoch_end()
        dm = getattr(self.trainer, 'datamodule', None)
        if dm is None:
            return
        logs = self.evaluate_communication_accuracy(dm)
        if logs:
            self.log_dict(
                logs,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                add_dataloader_idx=False,
            )
