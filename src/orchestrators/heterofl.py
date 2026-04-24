"""Decentralized HeteroFL orchestrator.

This module implements a neighborhood-level variant of HeteroFL (Diao et al.,
ICLR 2021) where aggregation is restricted to each node's local neighborhood
rather than a central server.  Each node i averages encoder parameters only
with the sub-networks possessed by its neighbors, using the nested HeteroFL
scheme extended to all tensor dimensions:

    For every parameter tensor, node i and neighbor j contribute to the
    intersection of their shapes in every dimension.  Within that intersection
    each output-channel position is weighted by the neighbor's sample count,
    and the denominator accumulates the total weight per position.

Only encoder parameters participate in aggregation; decoder (classification
head) parameters remain local/personalized.

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
    """Decentralized HeteroFL orchestrator with neighbor-level aggregation.

    Each node maintains its own encoder sub-model whose channel width is
    determined by its rate.  At each aggregation round every node i:

    1. Snapshots all neighbors' (and its own) current encoder states.
    2. Builds a new encoder by nested weighted averaging restricted to its
       local neighborhood: for each parameter tensor, positions covered by
       neighbor j (i.e. within j's sub-model shape) contribute with weight
       proportional to j's sample count.  The intersection is computed in
       every tensor dimension so that multi-layer encoders where input
       channels also scale with rate are handled correctly.
    3. Loads the averaged state back into its own encoder.

    No central server or global encoder exists.  Decoders are never
    aggregated and remain local to each node.

    Per-agent metrics are suppressed (``_LOG_PER_AGENT_METRICS = False``);
    only global aggregates are emitted to avoid flooding WandB with one
    series per agent.

    Parameters
    ----------
    agents : dict[int, nn.Module]
        Client models.  Each agent must expose an ``encoder`` and a
        ``decoder`` attribute (e.g. ``HeteroCNNClassifier``).
    neighbors : dict[int, set[int]]
        Adjacency mapping from each agent index to the set of its neighbor
        indices.  Self-loops are added implicitly during aggregation so that
        every node always mixes in its own parameters.
    optimizer : hydra config
        Optimizer used for local SGD updates.
    rates : dict[int, float]
        Per-client width scaling factors in ``(0, 1]``.  A factor of
        ``1.0`` means the client uses the full-width encoder; ``0.5``
        means half the output channels per layer.  Keys must match the keys
        in ``agents``.
    aggregation_interval : int | None, optional
        Number of local epochs or optimizer steps between successive
        aggregation rounds.  Defaults to one aggregation per epoch.
    aggregation_unit : str | None, optional
        Unit for ``aggregation_interval``: ``'epoch'`` or ``'step'``.
        Defaults to ``'epoch'``.
    max_global_rounds : int | None, optional
        Optional cap on scheduled aggregation rounds.  Training is stopped
        once the budget is exhausted.  ``None`` means no cap.
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
        If ``True``, run one neighborhood aggregation round before the
        first local update so that neighbors start from a common basis
        (default: ``True``).
    """

    _LOG_PER_AGENT_METRICS: bool = False

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
        # the aggregation can be performed unambiguously.
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
    # Neighborhood helpers
    # ------------------------------------------------------------------

    def _build_neighbors(self) -> dict[int, set[int]]:
        """Return the adjacency map with integer keys and node sets."""
        return {
            int(k): {int(n) for n in v}
            for k, v in self.hparams.neighbors.items()
        }

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _neighbor_aggregate(self) -> None:
        """Run one neighborhood-level HeteroFL aggregation round.

        For each node i the method:

        1. Collects a snapshot of every neighbor's (including i's own)
           current encoder state — snapshotting before any mutation
           ensures order-independence.
        2. For each parameter tensor ``name`` of node i's encoder,
           accumulates contributions from neighbors using the nested
           weighted-average rule:

           - The overlap between node i and neighbor j in every tensor
             dimension d is ``min(shape_i[d], shape_j[d])``.
           - j's slice within that overlap is added to i's accumulator
             weighted by j's sample count.
           - The weight accumulator ``weight_sum`` is tracked per
             position (same shape as the tensor) so that the denominator
             at each position only counts neighbors that actually reached
             that position — critical for deeper Conv layers where both
             output and input channels scale with rate.

        3. Loads the normalised result back into node i's encoder.

        Communication cost: one upload per directed edge (j → i),
        counting j's full encoder sub-model payload.
        """
        agents = {int(k): v for k, v in self.agents.items()}
        counts = self._infer_client_sample_counts()
        neighbors = self._build_neighbors()

        # Parameter names from the highest-rate agent (superset of all names).
        max_rate_idx = max(self._rates, key=lambda k: self._rates[k])
        param_names = list(agents[max_rate_idx].encoder.state_dict().keys())

        # Snapshot all encoder states before any mutation.
        snapshots: dict[int, dict[str, torch.Tensor]] = {
            idx: {
                name: t.clone().detach().float()
                for name, t in agent.encoder.state_dict().items()
            }
            for idx, agent in agents.items()
        }

        # Record communication: one payload per directed edge j → i.
        communicated: set[tuple[int, int]] = set()
        for idx in agents:
            neighborhood = neighbors.get(idx, set()) | {idx}
            for j in neighborhood:
                if j != idx and j in agents and (idx, j) not in communicated:
                    self._record_communication(
                        snapshots[j],
                        n_transmissions=1,
                        prefix='train',
                    )
                    communicated.add((idx, j))

        # Compute and load the new encoder state for each node.
        for idx, agent in agents.items():
            neighborhood = neighbors.get(idx, set()) | {idx}
            snap_i = snapshots[idx]

            new_state: dict[str, torch.Tensor] = {}

            for name in param_names:
                tensor_i = snap_i[name]  # shape owned by node i
                accum = torch.zeros_like(tensor_i, dtype=torch.float32)
                # Per-position weight accumulator (same shape as tensor_i) so
                # that positions only reachable by high-rate neighbors are not
                # diluted by the weights of low-rate neighbors that didn't
                # contribute to those positions (e.g. in_channels > rate * C
                # in deeper Conv layers where both dims scale with rate).
                weight_sum = torch.zeros_like(tensor_i, dtype=torch.float32)

                for j in neighborhood:
                    if j not in agents:
                        continue
                    tensor_j = snapshots[j][name]

                    # Intersection in every dimension: min of the two shapes.
                    slices = tuple(
                        slice(min(s_i, s_j))
                        for s_i, s_j in zip(tensor_i.shape, tensor_j.shape)
                    )
                    if not slices or any(s.stop == 0 for s in slices):
                        continue

                    raw_w = float(counts.get(j, 1))
                    accum[slices] += tensor_j[slices] * raw_w
                    weight_sum[slices] += raw_w

                norm = weight_sum.clamp(min=1e-8)
                dtype = agent.encoder.state_dict()[name].dtype
                new_state[name] = (accum / norm).to(dtype=dtype)

            agent.encoder.load_state_dict(new_state)

    # ------------------------------------------------------------------
    # Synchronisation round
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _synchronize(self) -> None:
        """Run one full neighborhood aggregation round."""
        self._neighbor_aggregate()
        self._record_communication_round(prefix='train')

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def on_train_start(self) -> None:
        """Initialise counters; optionally seed from a neighborhood round."""
        super().on_train_start()
        self._infer_client_sample_counts()
        self._last_aggregated_step = 0
        self._last_aggregated_epoch = 0
        self._global_aggregation_count = 0

        if self.hparams.sync_on_train_start:
            self._neighbor_aggregate()

    def _maybe_stop_after_global_aggregation(self) -> None:
        """Stop training once the configured aggregation budget is reached."""
        max_rounds = self.hparams.max_global_rounds
        if (
            max_rounds is None
            or int(max_rounds) <= 0
            or self._global_aggregation_count < int(max_rounds)
        ):
            return
        self.trainer.should_stop = True

    def on_train_epoch_end(self) -> None:
        """Aggregate neighborhood encoders on the epoch schedule."""
        if self.hparams.aggregation_unit == 'epoch':
            completed = int(self.current_epoch) + 1
            interval = int(self.hparams.aggregation_interval)

            if (
                interval > 0
                and completed > 0
                and completed != self._last_aggregated_epoch
                and completed % interval == 0
            ):
                kb_before = self._communication_by_split['train']['kilobytes']
                self._synchronize()
                kb_this_round = (
                    self._communication_by_split['train']['kilobytes'] - kb_before
                )
                self._last_aggregated_epoch = completed
                self._global_aggregation_count += 1
                self.log(
                    'global_round',
                    float(self._global_aggregation_count),
                    logger=True,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                )
                self.log(
                    'train/communication_kb_per_round',
                    kb_this_round,
                    logger=True,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                )
                self._maybe_stop_after_global_aggregation()

        self._finalize_train_epoch_communication()

    def on_train_batch_end(
        self,
        outputs: Any,
        batch: dict[int, list[torch.Tensor]],
        batch_idx: int,
    ) -> None:
        """Run step-scheduled neighborhood aggregation when configured."""
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
