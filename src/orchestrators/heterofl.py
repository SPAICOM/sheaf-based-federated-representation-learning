"""Heterogeneous Federated Learning orchestrator.

This module implements the HeteroFL algorithm (Diao et al., ICLR 2021) where
clients receive sub-models that are channel-width slices of a server-side
global full-width model. The server aggregates client updates using a nested
averaging scheme: each parameter position is averaged only over the clients
whose allocated sub-model covers that position.

Only encoder parameters participate in global aggregation; each agent's
decoder (classification head) remains local/personalized.

Reference
---------
Diao, E., Ding, J., & Tarokh, V. (2021). HeteroFL: Computation and
Communication Efficient Federated Learning for Heterogeneous Clients.
ICLR 2021. https://arxiv.org/abs/2010.01264
"""

from typing import Any

import torch
import torch.nn as nn

from src.orchestrators.base_orchestrator import BaseOrchestrator


class HeteroFL(BaseOrchestrator):
    """Centralized HeteroFL orchestrator with heterogeneous client widths.

    The server maintains a global full-width encoder. At each aggregation
    round it:

    1. Collects each client's locally-trained encoder sub-model.
    2. Reconstructs the global encoder via nested weighted averaging: for
       each output-channel position ``i`` in a parameter tensor of shape
       ``(C_full, ...)``, only clients whose sub-model covers position ``i``
       (i.e. whose rate satisfies ``int(rate * C_full) > i``) contribute
       to the average.  Weights are proportional to per-client sample counts.
    3. Broadcasts the appropriate leading-channel slice back to every client.

    Decoder parameters are never aggregated and remain local to each client.

    Parameters
    ----------
    agents : dict[int, nn.Module]
        Client models. Each agent must expose an ``encoder`` and a
        ``decoder`` attribute (e.g. ``HeteroCNNClassifier``).
    neighbors : dict[int, set[int]]
        Unused by the aggregation rule; kept for interface compatibility
        with the base orchestrator and the experiment pipeline.
    optimizer : hydra config
        Optimizer used for local SGD updates.
    rates : dict[int, float]
        Per-client width scaling factors in ``(0, 1]``. A factor of
        ``1.0`` means the client receives the full-width encoder;
        ``0.5`` means half the output channels per layer.  Keys must
        match the keys in ``agents``.
    aggregation_interval : int | None, optional
        Number of local epochs or optimizer steps between successive
        global aggregations.  Defaults to one aggregation per epoch.
    aggregation_unit : str | None, optional
        Unit for ``aggregation_interval``: ``'epoch'`` or ``'step'``.
        Defaults to ``'epoch'``.
    max_global_rounds : int | None, optional
        Optional cap on scheduled server aggregation rounds.  Training
        is stopped once the budget is exhausted.  ``None`` means no cap.
    local_steps : int | None, optional
        Legacy alias: maps to ``aggregation_interval=local_steps`` and
        ``aggregation_unit='step'``.
    global_steps : int | None, optional
        Legacy alias for ``max_global_rounds``.
    sample_counts : dict[int, int] | None, optional
        Per-client training sample counts used for weighted aggregation.
        When omitted the orchestrator tries to infer them from
        ``trainer.datamodule`` and falls back to uniform weights.
    sync_on_train_start : bool, optional
        If ``True``, distribute global encoder slices to all clients once
        before the first local update (default: ``True``).
    """

    def __init__(
        self,
        agents: dict[int, nn.Module],
        neighbors: dict[int, set[int]],
        optimizer: Any,
        rates: dict[int, float],
        aggregation_interval: int | None = None,
        aggregation_unit: str | None = None,
        max_global_rounds: int | None = None,
        local_steps: int | None = None,
        global_steps: int | None = None,
        sample_counts: dict[int, int] | None = None,
        sync_on_train_start: bool = True,
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

        super().__init__(
            agents=agents,
            neighbors=neighbors,
            optimizer=optimizer,
        )
        self.save_hyperparameters(ignore=['agents'])

        self._rates: dict[int, float] = {
            int(k): float(v) for k, v in rates.items()
        }
        self._validate_agents_for_heterofl()
        self._client_sample_counts = self._normalize_sample_counts(
            sample_counts
        )

        # Populated on_train_start from the highest-rate agent's encoder.
        self._global_encoder_state: dict[str, torch.Tensor] = {}

        self._last_aggregated_step = 0
        self._last_aggregated_epoch = 0
        self._global_aggregation_count = 0

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_aggregation_schedule(
        *,
        aggregation_interval: int | None,
        aggregation_unit: str | None,
        max_global_rounds: int | None,
        local_steps: int | None,
        global_steps: int | None,
    ) -> tuple[int, str, int | None]:
        """Resolve the configured HeteroFL aggregation schedule."""
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
        return {int(k): max(int(v), 0) for k, v in sample_counts.items()}

    def _validate_agents_for_heterofl(self) -> None:
        """Check enc-dec presence, rate validity, and name consistency."""
        agents = {int(k): v for k, v in self.agents.items()}

        for idx, agent in agents.items():
            if not hasattr(agent, 'encoder') or not hasattr(agent, 'decoder'):
                raise TypeError(
                    f'HeteroFL requires agents with enc-dec attributes:'
                    f'Agent {idx} is missing one or both.'
                )
            if idx not in self._rates:
                raise ValueError(
                    f'Agent {idx} has no entry in the rates dictionary.'
                )
            rate = self._rates[idx]
            if not (0.0 < rate <= 1.0):
                raise ValueError(
                    f'Rate for agent {idx} must be in (0, 1], got {rate}.'
                )

        # All encoder state dicts must share the same parameter names so that
        # the global state can be constructed and sliced unambiguously.
        ref_idx = min(agents.keys())
        ref_names = set(agents[ref_idx].encoder.state_dict().keys())
        for idx, agent in agents.items():
            names = set(agent.encoder.state_dict().keys())
            if names != ref_names:
                raise ValueError(
                    f'Agent {idx} encoder state dict keys differ from '
                    f'agent {ref_idx}.'
                )

    # ------------------------------------------------------------------
    # Sample count helpers
    # ------------------------------------------------------------------

    def _infer_client_sample_counts(self) -> dict[int, int]:
        """Infer per-client counts from the datamodule or fall back to unif."""
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

        counts: dict[int, int] = {}
        for idx_str in self.agents:
            idx = int(idx_str)
            count = None
            if train_datasets is not None:
                dataset = train_datasets.get(idx) or train_datasets.get(
                    str(idx)
                )
                if dataset is not None:
                    count = len(dataset)
            counts[idx] = 1 if count is None else max(int(count), 0)

        if sum(counts.values()) <= 0:
            counts = {int(idx): 1 for idx in self.agents}

        self._client_sample_counts = counts
        return dict(counts)

    # ------------------------------------------------------------------
    # Global state management
    # ------------------------------------------------------------------

    def _init_global_encoder_state(self) -> None:
        """Seed the global encoder from the highest-rate agent's encoder."""
        max_idx = max(self._rates, key=lambda k: self._rates[k])
        self._global_encoder_state = {
            name: tensor.clone().detach().float()
            for name, tensor in self.agents[str(max_idx)]
            .encoder.state_dict()
            .items()
        }

    # ------------------------------------------------------------------
    # Aggregation (upload: clients → server)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _aggregate_global_encoder(self) -> None:
        """Reconstruct the global encoder via nested sample-weighted averaging.

        For every named parameter tensor of global shape ``(C_full, ...)``:
        - Client with rate ``r`` covers positions ``0 … int(r*C_full)-1``
          along dimension 0 (output channels).
        - Each position is averaged only over the clients that cover it,
          using sample counts as unnormalised weights; the normalisation
          denominator therefore varies per position.

        This implements the set-difference aggregation described in §3 of
        the HeteroFL paper without explicitly enumerating rate levels.
        """
        agents = {int(k): v for k, v in self.agents.items()}
        counts = self._infer_client_sample_counts()

        new_state: dict[str, torch.Tensor] = {}

        for name, global_tensor in self._global_encoder_state.items():
            full_out = global_tensor.shape[0]
            extra_ndim = global_tensor.ndim - 1

            accum = torch.zeros_like(global_tensor, dtype=torch.float32)
            # One weight accumulator per output-channel position.
            weight_sum = torch.zeros(full_out, dtype=torch.float32)

            for idx, agent in agents.items():
                local_sd = agent.encoder.state_dict()
                self._record_communication(
                    local_sd,
                    n_transmissions=1,
                    prefix='train',
                )

                local_tensor = local_sd[name].detach().float()
                n_out = local_tensor.shape[0]  # int(rate * full_out)
                raw_w = float(counts.get(idx, 1))

                accum[:n_out] += local_tensor * raw_w
                weight_sum[:n_out] += raw_w

            # Normalise: broadcast weight_sum over the remaining dimensions.
            norm = weight_sum.view(-1, *([1] * extra_ndim)).clamp(min=1e-8)
            new_state[name] = (accum / norm).to(dtype=global_tensor.dtype)

        self._global_encoder_state = new_state

    # ------------------------------------------------------------------
    # Distribution (download: server → clients)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _distribute_global_encoder(self) -> None:
        """Send each client the leading-channel slice of the global encoder.

        A client with rate ``r`` receives the first ``int(r * C_full)``
        output channels of every parameter tensor, which forms a nested
        sub-model of the full-width global encoder.
        """
        for idx_str, agent in self.agents.items():
            idx = int(idx_str)
            rate = self._rates[idx]

            local_state: dict[str, torch.Tensor] = {}
            for name, global_tensor in self._global_encoder_state.items():
                n_out = int(rate * global_tensor.shape[0])
                local_state[name] = (
                    global_tensor[:n_out]
                    .clone()
                    .to(dtype=agent.encoder.state_dict()[name].dtype)
                )

            agent.encoder.load_state_dict(local_state)
            self._record_communication(
                local_state,
                n_transmissions=1,
                prefix='train',
            )

    # ------------------------------------------------------------------
    # Synchronisation round
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _synchronize(self) -> None:
        """Run one full HeteroFL communication round: aggregate-distribute."""
        self._aggregate_global_encoder()
        self._distribute_global_encoder()
        self._record_communication_round(prefix='train')

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def on_train_start(self) -> None:
        """Initialise counters and global state; optionally seed clients."""
        super().on_train_start()
        self._infer_client_sample_counts()
        self._init_global_encoder_state()
        self._last_aggregated_step = 0
        self._last_aggregated_epoch = 0
        self._global_aggregation_count = 0

        if self.hparams.sync_on_train_start:
            self._distribute_global_encoder()

    def _maybe_stop_after_global_aggregation(self) -> None:
        """Stop training once the configured aggregation budget is reached."""
        max_rounds = self.hparams.max_global_rounds
        if (
            max_rounds is None
            or int(max_rounds) <= 0
            or self._global_aggregation_count < int(max_rounds)
        ):
            return
        trainer = getattr(self, '_trainer', None)
        if trainer is not None:
            trainer.should_stop = True

    def on_train_epoch_end(self) -> None:
        """Aggregate and redistribute encoders on the epoch schedule."""
        if self.hparams.aggregation_unit == 'epoch':
            completed = int(self.current_epoch) + 1
            interval = int(self.hparams.aggregation_interval)

            if (
                interval > 0
                and completed > 0
                and completed != self._last_aggregated_epoch
                and completed % interval == 0
            ):
                self._synchronize()
                self._last_aggregated_epoch = completed
                self._global_aggregation_count += 1
                self._maybe_stop_after_global_aggregation()

        self._finalize_train_epoch_communication()

    def on_train_batch_end(
        self,
        outputs: Any,
        batch: dict[int, list[torch.Tensor]],
        batch_idx: int,
    ) -> None:
        """Run step-scheduled HeteroFL aggregation when configured."""
        if self.hparams.aggregation_unit != 'step':
            return

        step = int(self.global_step)
        interval = int(self.hparams.aggregation_interval)
        if (
            interval <= 0
            or step <= 0
            or step == self._last_aggregated_step
            or step % interval != 0
        ):
            return

        self._synchronize()
        self._last_aggregated_step = step
        self._global_aggregation_count += 1
        self._maybe_stop_after_global_aggregation()

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _shared_eval(
        self,
        batch: dict[int, list[torch.Tensor]],
        batch_idx: int,
        prefix: str,
    ) -> tuple[dict[str, Any], torch.Tensor]:
        """Compute per-agent losses and metrics for train/val/test steps."""
        outputs = self(batch)

        agent_losses: dict[int, Any] = {}
        agent_performances: dict[int, Any] = {}

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


if __name__ == '__main__':
    pass
