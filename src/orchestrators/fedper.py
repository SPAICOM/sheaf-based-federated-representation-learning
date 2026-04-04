"""Federated Personalization (FedPer) orchestrator.

This module implements a centralized FedPer-style training loop where all
agents share encoder/base layers while keeping their decoder/classification
head private. Base layers are aggregated on a server with sample-count
weights, then broadcast back to every client after a fixed number of local
SGD steps.
"""

from typing import Any

import torch
import torch.nn as nn

from src.orchestrators.base_orchestrator import BaseOrchestrator


class FedPer(BaseOrchestrator):
    """Centralized FedPer orchestrator with personalized heads.

    The implementation uses the agent ``encoder`` as the shared base network
    and the agent ``decoder`` as the personalized head. Only encoder
    parameters and buffers participate in server-side aggregation.

    Parameters
    ----------
    agents : dict[int, nn.Module]
        Client models. All agents must expose compatible ``encoder`` modules.
    neighbors : dict[int, set[int]]
        Unused by the aggregation rule, but kept for interface compatibility
        with the base orchestrator and experiment pipeline.
    optimizer : hydra config
        Optimizer used for local SGD updates.
    local_steps : int, optional
        Number of local optimizer steps between successive global
        aggregations (default: 4).
    global_steps : int | None, optional
        Number of server aggregation rounds to execute. If ``None`` or
        non-positive, training is not stopped by the orchestrator
        (default: 100).
    sample_counts : dict[int, int] | None, optional
        Optional per-client training sample counts. When omitted, the
        orchestrator tries to infer them from ``trainer.datamodule`` and
        falls back to uniform weights if unavailable.
    sync_on_train_start : bool, optional
        If ``True``, synchronize encoder weights once before the first local
        update so all clients start from a common shared base (default: True).
    """

    def __init__(
        self,
        agents: dict[int, nn.Module],
        neighbors: dict[int, set[int]],
        optimizer: Any,
        local_steps: int = 4,
        global_steps: int | None = 100,
        sample_counts: dict[int, int] | None = None,
        sync_on_train_start: bool = True,
        **kwargs,
    ):
        super().__init__(
            agents=agents,
            neighbors=neighbors,
            optimizer=optimizer,
        )
        self.save_hyperparameters(ignore=['agents'])

        self._validate_agents_for_fedper()
        self._client_sample_counts = self._normalize_sample_counts(
            sample_counts
        )
        self._last_aggregated_step = 0
        self._global_aggregation_count = 0

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

    def _validate_agents_for_fedper(self) -> None:
        """Validate that every agent exposes a compatible shared encoder."""
        agents = list(self.agents.values())
        ref = agents[0]

        if not hasattr(ref, 'encoder') or not hasattr(ref, 'decoder'):
            raise TypeError(
                'FedPer requires agents with encoder/decoder attributes.'
            )

        ref_encoder_state = ref.encoder.state_dict()
        ref_encoder_shapes = {
            name: tensor.shape
            for name, tensor in ref_encoder_state.items()
        }

        for i, agent in enumerate(agents[1:], start=1):
            if not hasattr(agent, 'encoder') or not hasattr(agent, 'decoder'):
                raise TypeError(
                    f'Agent {i} is missing encoder/decoder attributes.'
                )

            encoder_state = agent.encoder.state_dict()
            if encoder_state.keys() != ref_encoder_state.keys():
                raise ValueError(
                    f'Agent {i} encoder state keys mismatch with agent 0.'
                )

            for name, tensor in encoder_state.items():
                if tensor.shape != ref_encoder_shapes[name]:
                    raise ValueError(
                        f'Encoder shape mismatch in {name}:'
                        f' agent {i} vs agent 0.'
                    )

    def _infer_client_sample_counts(self) -> dict[int, int]:
        """Infer per-client training sample counts from the attached datamodule."""
        if self._client_sample_counts:
            return dict(self._client_sample_counts)

        trainer = getattr(self, '_trainer', None)
        datamodule = None if trainer is None else getattr(trainer, 'datamodule', None)
        train_datasets = (
            None if datamodule is None else getattr(datamodule, 'train_datasets', None)
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
            return {idx: uniform for idx in counts}
        return {idx: float(count) / total for idx, count in counts.items()}

    @torch.no_grad()
    def _aggregate_base_layers(self) -> dict[str, torch.Tensor]:
        """Compute the centralized weighted average of encoder states."""
        agents = {int(k): v for k, v in self.agents.items()}
        weights = self._client_weights()

        ref_state = agents[min(agents)].encoder.state_dict()
        aggregated_state = {
            name: torch.zeros_like(tensor, dtype=torch.float32)
            for name, tensor in ref_state.items()
        }

        for idx, agent in agents.items():
            base_state = agent.encoder.state_dict()
            self._record_communication(base_state, n_transmissions=1)
            for name, tensor in base_state.items():
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
        )
        return aggregated_state

    @torch.no_grad()
    def _broadcast_base_layers(
        self,
        aggregated_state: dict[str, torch.Tensor],
    ) -> None:
        """Load the server-aggregated encoder state into every agent."""
        for agent in self.agents.values():
            agent.encoder.load_state_dict(aggregated_state)

    @torch.no_grad()
    def _synchronize_base_layers(self) -> None:
        """Aggregate encoder parameters on the server and broadcast them."""
        aggregated_state = self._aggregate_base_layers()
        self._broadcast_base_layers(aggregated_state)

    def on_train_start(self) -> None:
        """Initialize counters and optionally synchronize shared base layers."""
        super().on_train_start()
        self._infer_client_sample_counts()
        self._last_aggregated_step = 0
        self._global_aggregation_count = 0

        if self.hparams.sync_on_train_start:
            self._synchronize_base_layers()

    def on_train_epoch_end(self) -> None:
        """No epoch-level action: FedPer aggregates on the step schedule."""
        return None

    def on_train_batch_end(
        self,
        outputs: Any,
        batch: dict[int, list[torch.Tensor]],
        batch_idx: int,
    ) -> None:
        """Run a server aggregation after every ``local_steps`` optimizer steps."""
        step = int(self.global_step)
        local_steps = int(self.hparams.local_steps)
        if (
            local_steps <= 0
            or step <= 0
            or step == self._last_aggregated_step
            or step % local_steps != 0
        ):
            return None

        self._synchronize_base_layers()
        self._last_aggregated_step = step
        self._global_aggregation_count += 1

        max_global_steps = self.hparams.global_steps
        if (
            max_global_steps is not None
            and int(max_global_steps) > 0
            and self._global_aggregation_count >= int(max_global_steps)
        ):
            trainer = getattr(self, '_trainer', None)
            if trainer is not None:
                trainer.should_stop = True

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
        )

        return outputs, total_loss


if __name__ == '__main__':
    pass
