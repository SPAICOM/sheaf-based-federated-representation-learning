"""Local-Global Federated Learning (LGFed) orchestrator.

This module implements a centralized LGFed-style training loop where all
agents keep their encoder/feature-extraction layers private while sharing
decoder/classification-head layers. Head layers are aggregated on a server
with sample-count weights, then broadcast back to every client on a
configurable global aggregation schedule.

This is the complement of FedPer: encoders are personalized, decoders are
shared.
"""

from typing import Any

import torch
import torch.nn as nn

from src.communication.alignment_mixin import (
    VALID_ALIGNMENT_METHODS,
    PostTrainingAlignmentMixin,
)
from src.orchestrators.base_orchestrator import BaseOrchestrator


class LGFed(PostTrainingAlignmentMixin, BaseOrchestrator):
    """Centralized LGFed orchestrator with personalized encoders.

    The implementation uses the agent ``decoder`` as the shared head network
    and the agent ``encoder`` as the personalized feature extractor. Only
    decoder parameters and buffers participate in server-side aggregation.

    Parameters
    ----------
    agents : dict[int, nn.Module]
        Client models. All agents must expose compatible ``decoder`` modules.
    neighbors : dict[int, set[int]]
        Unused by the aggregation rule, but kept for interface compatibility
        with the base orchestrator and experiment pipeline.
    optimizer : hydra config
        Optimizer used for local SGD updates.
    aggregation_interval : int | None, optional
        Number of local epochs or optimizer steps between successive global
        aggregations. When omitted, defaults to one aggregation per epoch.
    aggregation_unit : str | None, optional
        Unit used by ``aggregation_interval``. Supported values are
        ``'epoch'`` and ``'step'``. When omitted, defaults to ``'epoch'``.
    max_global_rounds : int | None, optional
        Optional cap on scheduled server aggregation rounds. If ``None`` or
        non-positive, training is not stopped by the orchestrator.
    local_steps : int | None, optional
        Legacy alias for step-based aggregation. When provided, it maps to
        ``aggregation_interval=local_steps`` and
        ``aggregation_unit='step'``.
    global_steps : int | None, optional
        Legacy alias for ``max_global_rounds``.
    sample_counts : dict[int, int] | None, optional
        Optional per-client training sample counts. When omitted, the
        orchestrator tries to infer them from ``trainer.datamodule`` and
        falls back to uniform weights if unavailable.
    sync_on_train_start : bool, optional
        If ``True``, synchronize decoder weights once before the first local
        update so all clients start from a common shared head (default: True).
    """

    def __init__(
        self,
        agents: dict[int, nn.Module],
        neighbors: dict[int, set[int]],
        optimizer: Any,
        aggregation_interval: int | None = None,
        aggregation_unit: str | None = None,
        max_global_rounds: int | None = None,
        local_steps: int | None = None,
        global_steps: int | None = None,
        sample_counts: dict[int, int] | None = None,
        sync_on_train_start: bool = True,
        alignment_method: str = 'general',
        **kwargs,
    ):
        (
            aggregation_interval,
            aggregation_unit,
            max_global_rounds,
        ) = self._resolve_aggregation_schedule(
            aggregation_interval=aggregation_interval,
            aggregation_unit=aggregation_unit,
            max_global_rounds=max_global_rounds,
            local_steps=local_steps,
            global_steps=global_steps,
        )
        local_steps = None
        global_steps = None

        alignment_method = str(alignment_method)
        if alignment_method not in VALID_ALIGNMENT_METHODS:
            raise ValueError(
                f"Unknown alignment_method '{alignment_method}'. "
                f'Valid options: {VALID_ALIGNMENT_METHODS}'
            )

        super().__init__(
            agents=agents,
            neighbors=neighbors,
            optimizer=optimizer,
            **kwargs,
        )
        self.save_hyperparameters(ignore=['agents'])

        self._validate_agents_for_lgfed()
        self._client_sample_counts = self._normalize_sample_counts(
            sample_counts
        )
        self._last_aggregated_step = 0
        self._last_aggregated_epoch = 0
        self._global_aggregation_count = 0

    @staticmethod
    def _resolve_aggregation_schedule(
        *,
        aggregation_interval: int | None,
        aggregation_unit: str | None,
        max_global_rounds: int | None,
        local_steps: int | None,
        global_steps: int | None,
    ) -> tuple[int, str, int | None]:
        """Resolve the configured LGFed aggregation schedule."""
        if local_steps is not None:
            if (
                aggregation_interval is not None
                or aggregation_unit is not None
            ):
                raise ValueError(
                    'Specify either aggregation_interval/aggregation_unit or '
                    'the legacy local_steps alias, not both.'
                )
            aggregation_interval = int(local_steps)
            aggregation_unit = 'step'

        if global_steps is not None:
            if max_global_rounds is not None:
                raise ValueError(
                    'Specify either max_global_rounds or the legacy '
                    'global_steps alias, not both.'
                )
            max_global_rounds = int(global_steps)

        resolved_interval = (
            1 if aggregation_interval is None else int(aggregation_interval)
        )
        resolved_unit = (
            'epoch' if aggregation_unit is None else str(aggregation_unit)
        ).lower()

        if resolved_unit not in {'epoch', 'step'}:
            raise ValueError(
                "aggregation_unit must be either 'epoch' or 'step'."
            )

        return resolved_interval, resolved_unit, max_global_rounds

    def _normalize_sample_counts(
        self,
        sample_counts: dict[int, int] | None,
    ) -> dict[int, int]:
        """Normalize sample-count keys to integer agent indices."""
        if sample_counts is None:
            return {}

        normalized = {}
        for key, value in sample_counts.items():
            normalized[int(key)] = max(int(value), 0)
        return normalized

    def _validate_agents_for_lgfed(self) -> None:
        """Validate that every agent exposes a compatible shared decoder."""
        agents = list(self.agents.values())
        ref = agents[0]

        if not hasattr(ref, 'encoder') or not hasattr(ref, 'decoder'):
            raise TypeError(
                'LGFed requires agents with encoder/decoder attributes.'
            )

        ref_decoder_state = ref.decoder.state_dict()
        ref_decoder_shapes = {
            name: tensor.shape for name, tensor in ref_decoder_state.items()
        }

        for i, agent in enumerate(agents[1:], start=1):
            if not hasattr(agent, 'encoder') or not hasattr(agent, 'decoder'):
                raise TypeError(
                    f'Agent {i} is missing encoder/decoder attributes.'
                )

            decoder_state = agent.decoder.state_dict()
            if decoder_state.keys() != ref_decoder_state.keys():
                raise ValueError(
                    f'Agent {i} decoder state keys mismatch with agent 0.'
                )

            for name, tensor in decoder_state.items():
                if tensor.shape != ref_decoder_shapes[name]:
                    raise ValueError(
                        f'Decoder shape mismatch in {name}:'
                        f' agent {i} vs agent 0.'
                    )

    def _infer_client_sample_counts(self) -> dict[int, int]:
        """Infer per-client training sample counts from the attached datamodule."""
        if self._client_sample_counts:
            return dict(self._client_sample_counts)

        trainer = getattr(self, '_trainer', None)
        datamodule = (
            None if trainer is None else getattr(trainer, 'datamodule', None)
        )
        train_datasets = (
            None
            if datamodule is None
            else getattr(datamodule, 'train_datasets', None)
        )

        counts = {}
        for idx_str in self.agents:
            idx = int(idx_str)
            count = None
            if train_datasets is not None:
                dataset = train_datasets.get(idx)
                if dataset is None:
                    dataset = train_datasets.get(str(idx))
                if dataset is not None:
                    count = len(dataset)

            counts[idx] = 1 if count is None else max(int(count), 0)

        if sum(counts.values()) <= 0:
            counts = {int(idx): 1 for idx in self.agents}

        self._client_sample_counts = counts
        return dict(counts)

    def _client_weights(self) -> dict[int, float]:
        """Return normalized sample-count weights for server aggregation."""
        counts = self._infer_client_sample_counts()
        total = float(sum(counts.values()))
        if total <= 0:
            uniform = 1.0 / max(len(counts), 1)
            return dict.fromkeys(counts, uniform)
        return {idx: float(count) / total for idx, count in counts.items()}

    @torch.no_grad()
    def _aggregate_head_layers(self) -> dict[str, torch.Tensor]:
        """Compute the centralized weighted average of decoder states."""
        agents = {int(k): v for k, v in self.agents.items()}
        weights = self._client_weights()

        ref_state = agents[min(agents)].decoder.state_dict()
        aggregated_state = {
            name: torch.zeros_like(tensor, dtype=torch.float32)
            for name, tensor in ref_state.items()
        }

        for idx, agent in agents.items():
            head_state = agent.decoder.state_dict()
            self._record_communication(
                head_state,
                n_transmissions=1,
                prefix='train',
            )
            for name, tensor in head_state.items():
                aggregated_state[name] += (
                    tensor.detach().to(dtype=torch.float32) * weights[idx]
                )

        for name, ref_tensor in ref_state.items():
            aggregated_state[name] = aggregated_state[name].to(
                dtype=ref_tensor.dtype
            )

        self._record_communication(
            aggregated_state,
            n_transmissions=len(agents),
            prefix='train',
        )
        return aggregated_state

    @torch.no_grad()
    def _broadcast_head_layers(
        self,
        aggregated_state: dict[str, torch.Tensor],
    ) -> None:
        """Load the server-aggregated decoder state into every agent."""
        for agent in self.agents.values():
            agent.decoder.load_state_dict(aggregated_state)

    @torch.no_grad()
    def _synchronize_head_layers(self) -> None:
        """Aggregate decoder parameters on the server and broadcast them."""
        aggregated_state = self._aggregate_head_layers()
        self._broadcast_head_layers(aggregated_state)
        self._record_communication_round(prefix='train')

    def on_train_start(self) -> None:
        """Initialize counters and optionally synchronize shared head layers."""
        super().on_train_start()
        self._infer_client_sample_counts()
        self._last_aggregated_step = 0
        self._last_aggregated_epoch = 0
        self._global_aggregation_count = 0

        if self.hparams.sync_on_train_start:
            self._synchronize_head_layers()

    def _maybe_stop_after_global_aggregation(self) -> None:
        """Stop training once the configured aggregation budget is reached."""
        max_global_rounds = self.hparams.max_global_rounds
        if (
            max_global_rounds is None
            or int(max_global_rounds) <= 0
            or self._global_aggregation_count < int(max_global_rounds)
        ):
            return None

        trainer = getattr(self, '_trainer', None)
        if trainer is not None:
            trainer.should_stop = True
        return None

    def on_train_epoch_end(self) -> None:
        """Aggregate shared decoders according to the configured schedule."""
        if self.hparams.aggregation_unit == 'epoch':
            completed_epochs = int(self.current_epoch) + 1
            interval = int(self.hparams.aggregation_interval)

            if (
                interval > 0
                and completed_epochs > 0
                and completed_epochs != self._last_aggregated_epoch
                and completed_epochs % interval == 0
            ):
                self._synchronize_head_layers()
                self._last_aggregated_epoch = completed_epochs
                self._global_aggregation_count += 1
                self._maybe_stop_after_global_aggregation()

        self._finalize_train_epoch_communication()
        self._log_train_comm_task_perf()
        return None

    def on_train_batch_end(
        self,
        outputs: Any,
        batch: dict[int, list[torch.Tensor]],
        batch_idx: int,
    ) -> None:
        """Run step-scheduled LGFed aggregation when configured."""
        if self.hparams.aggregation_unit != 'step':
            return None

        step = int(self.global_step)
        interval = int(self.hparams.aggregation_interval)
        if (
            interval <= 0
            or step <= 0
            or step == self._last_aggregated_step
            or step % interval != 0
        ):
            return None

        self._synchronize_head_layers()
        self._last_aggregated_step = step
        self._global_aggregation_count += 1
        self._maybe_stop_after_global_aggregation()

        return None

    def _shared_eval(
        self,
        batch: dict[int, list[torch.Tensor]],
        batch_idx: int,
        prefix: str,
    ) -> tuple[dict[str, Any], torch.Tensor]:
        """Compute per-agent losses and metrics for train/val/test steps."""
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

    # send_message, _fit_alignment_maps, _cleanup_alignment inherited from
    # PostTrainingAlignmentMixin: post-hoc whitening + alignment pipeline.

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


if __name__ == '__main__':
    pass
