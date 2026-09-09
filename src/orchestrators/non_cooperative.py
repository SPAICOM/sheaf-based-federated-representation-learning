"""Non-cooperative multi-agent training baseline.

Each agent minimizes only its own task loss. No communication or model
alignment is performed during training, so cumulative communication remains
zero throughout the run.
"""

import inspect
from typing import Any

import torch
import torch.nn as nn

from src.communication.alignment_mixin import (
    VALID_ALIGNMENT_METHODS,
    PostTrainingAlignmentMixin,
)
from src.orchestrators.base_orchestrator import BaseOrchestrator
from src.utils.anchors import VALID_PAIRED_ANCHOR_SELECTIONS


class NonCooperativeLearning(PostTrainingAlignmentMixin, BaseOrchestrator):
    """Independent local-training baseline with shared evaluation logging.

    Post-hoc alignment for the communication-accuracy evaluation supports
    the mixin's three methods: 'general' (least-squares), 'procrustes'
    (semi-orthogonal), and 'relative' — zero-shot anchor frames, where the
    sender's anchors act as the analysis operator and the pseudo-inverse of
    the receiver's anchors as the synthesis operator (relative
    representations; no map is fitted).  All three fit on the same edge
    *anchors*, reduced from the matched pilot pool exactly as SheafFRL does
    at train time, so one override set sweeps the anchor budget/strategy
    across a SheafFRL run and this baseline alike: ``anchor_selection``
    ('all' | 'random' | 'proto_class' | 'proto_kmeans', one-shot here — see
    :func:`~src.utils.anchors.select_paired_anchors`), ``num_anchors``,
    ``protos_per_class``.  ``anchor_parseval_normalize`` left at its default
    (``None``) auto-picks True for 'relative' (the analysis/synthesis
    pairing is what Parseval frames are for, Fiorellino et al. 2025) and
    False for 'general'/'procrustes' (there the anchors only ever feed a
    least-squares/SVD fit — prewhitening an already-whitened, possibly
    small anchor subset is redundant at best); an explicit True/False
    overrides this for every method.
    """

    def __init__(
        self,
        agents: dict[int, nn.Module],
        neighbors: dict[int, set[int]],
        optimizer: Any,
        alignment_method: str = 'general',
        anchor_selection: str = 'all',
        num_anchors: int = 128,
        protos_per_class: int = 1,
        anchor_parseval_normalize: bool | None = None,
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
        if str(anchor_selection) not in VALID_PAIRED_ANCHOR_SELECTIONS:
            raise ValueError(
                f"Unknown anchor_selection '{anchor_selection}'. "
                f'Valid options: {list(VALID_PAIRED_ANCHOR_SELECTIONS)}'
            )
        if int(num_anchors) < 1:
            raise ValueError('num_anchors must be at least 1')
        if int(protos_per_class) < 1:
            raise ValueError('protos_per_class must be at least 1')
        self.save_hyperparameters(ignore=['agents'])

    def on_train_epoch_end(self) -> None:
        """No communication or aggregation is performed."""
        self._finalize_train_epoch_communication()
        if getattr(self, '_trainer', None) is not None:
            self._log_train_comm_task_perf()
        return None

    @staticmethod
    def _supports_eval_mask(method) -> bool:
        return 'eval_mask' in inspect.signature(method).parameters

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
            y_hat, y, *rest = outputs[idx]
            eval_mask = rest[0] if rest else None
            if self._supports_eval_mask(agent.compute_loss):
                agent_losses[int(idx)] = agent.compute_loss(
                    y_hat,
                    y,
                    eval_mask=eval_mask,
                )
            else:
                agent_losses[int(idx)] = agent.compute_loss(y_hat, y)

            if self._supports_eval_mask(agent.task_performance):
                agent_performances[int(idx)] = agent.task_performance(
                    y_hat,
                    y,
                    eval_mask=eval_mask,
                )
            else:
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
