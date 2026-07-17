"""
FedProto: Federated Prototype Learning Orchestrator.

This module implements FedProto, a federated learning algorithm that
aggregates per-class mean embeddings at the server into a global prototype
set and uses them as a regularization target to pull each client's local
class embeddings towards the global means.

Reference: Based on federated prototype learning methods for cross-client
representation alignment without weight sharing or a public dataset.
"""

from typing import Any

import torch
import torch.nn as nn

from src.communication.alignment_mixin import (
    VALID_ALIGNMENT_METHODS,
    PostTrainingAlignmentMixin,
)
from src.orchestrators.base_orchestrator import BaseOrchestrator


class FedProto(PostTrainingAlignmentMixin, BaseOrchestrator):
    """Federated learning with prototype-based representation alignment.

    Implements FedProto where:
    1. Each client computes per-class mean embeddings (prototypes) from local data
    2. Server aggregates prototypes across clients to form global prototypes
    3. Local training includes a regularization loss pulling embeddings
       towards global prototypes

    This provides cross-client representation alignment without weight
    sharing or requiring a public dataset.

    Parameters
    ----------
    agents : dict[int, nn.Module]
        Dictionary mapping agent indices to their model instances.
    neighbors : dict[int, set[int]]
        Dictionary mapping each agent index to the set of its neighbor indices.
    optimizer : hydra config
        Optimizer configuration for training.
    max_lmb : float, optional
        Maximum weight coefficient for the prototype regularization loss (default: 0.1).
    lambda_schedule : str or None, optional
        Scheduling strategy: ``None`` (constant), ``'cosine'``, or ``'exp'``.
    prototype_momentum : float, optional
        Momentum for updating global prototypes (default: 0.9).
    alignment_method : str, optional
        Post-training alignment used to measure cross-agent communication
        accuracy at test time: ``'general'`` (unconstrained least-squares,
        default) or ``'procrustes'`` (semi-orthogonal). Whitening + alignment
        maps are fitted post-hoc on pilot data via
        :class:`PostTrainingAlignmentMixin`; no training-time behaviour changes.

    Notes
    -----
    - Each agent must expose `encode(x)` method returning latent representation.
    - Each agent must expose `decoder(z)` method returning predictions.
    - Global prototypes are computed once per epoch via aggregation.
    - Regularization loss pulls local class embeddings towards global means.
    """

    def __init__(
        self,
        agents: dict[int, nn.Module],
        neighbors: dict[int, set[int]],
        optimizer: Any,
        max_lmb: float = 0.1,
        prototype_momentum: float = 0.9,
        lambda_schedule: str | None = None,
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

        self._validate_agents()
        self._global_prototypes: dict[int, torch.Tensor] = {}

    def _validate_agents(self) -> None:
        """Validate that all agents have encode/decoder interface."""
        agents = list(self.agents.values())
        if len(agents) <= 1:
            return

        ref = agents[0]
        if not hasattr(ref, 'encode') or not hasattr(ref, 'decoder'):
            raise TypeError(
                'FedProto requires agents with encode/decoder methods.'
            )

    def _compute_class_prototypes(
        self,
        representations: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict[int, torch.Tensor]:
        """Compute per-class mean embeddings (prototypes) from representations.

        Parameters
        ----------
        representations : torch.Tensor
            Latent representations of shape (N, D).
        labels : torch.Tensor
            Class labels of shape (N,).

        Returns
        -------
        dict[int, torch.Tensor]
            Dictionary mapping class indices to prototype (mean) tensors.
        """
        prototypes = {}
        unique_classes = torch.unique(labels)

        for c in unique_classes:
            class_idx = int(c.item())
            mask = labels == c
            prototypes[class_idx] = representations[mask].mean(dim=0)

        return prototypes

    def _update_global_prototypes(
        self,
        local_prototypes: dict[int, dict[int, torch.Tensor]],
    ) -> None:
        """Update global prototypes via EMA using local per-batch prototypes.

        For each class seen in this batch across all agents, the global prototype
        is updated as a momentum-weighted average. This mirrors the per-step
        aggregation described in the FedProto paper, where the server integrates
        client prototypes after every local update.

        Parameters
        ----------
        local_prototypes : dict[int, dict[int, torch.Tensor]]
            Mapping from agent index to per-class prototype tensors for this batch.
        """
        momentum = float(self.hparams.prototype_momentum)

        for local_proto in local_prototypes.values():
            for class_idx, proto in local_proto.items():
                p = proto.detach()
                if class_idx in self._global_prototypes:
                    self._global_prototypes[class_idx] = (
                        momentum * self._global_prototypes[class_idx]
                        + (1.0 - momentum) * p
                    )
                else:
                    self._global_prototypes[class_idx] = p

    def _compute_proto_regularization(
        self,
        local_prototypes: dict[int, torch.Tensor],
        global_prototypes: dict[int, torch.Tensor],
    ) -> torch.Tensor:
        """Compute prototype regularization loss.

        Pulls each client's local class embeddings towards global means.

        L_proto = sum_c ||mu_local_c - mu_global_c||^2

        Parameters
        ----------
        local_prototypes : dict[int, torch.Tensor]
            Local per-class prototypes.
        global_prototypes : dict[int, torch.Tensor]
            Global aggregated prototypes.

        Returns
        -------
        torch.Tensor
            Scalar prototype regularization loss.
        """
        if not global_prototypes or not local_prototypes:
            return torch.tensor(0.0, device=self.device)

        shared_classes = set(local_prototypes.keys()) & set(
            global_prototypes.keys()
        )
        if not shared_classes:
            return torch.tensor(0.0, device=self.device)

        loss = torch.tensor(0.0, device=self.device)
        for c in shared_classes:
            diff = local_prototypes[c] - global_prototypes[c]
            loss = loss + (diff**2).sum()

        return loss / len(shared_classes)

    def _extract_local_batch(
        self,
        batch: dict[int, list[torch.Tensor]],
    ) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
        """Extract local data for each agent from the batch.

        Parameters
        ----------
        batch : dict[int, list[torch.Tensor]]
            Dictionary mapping agent indices to (input, label) pairs.

        Returns
        -------
        tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]
            Tuple of (inputs dict, labels dict) for each agent.
        """
        inputs = {}
        labels = {}

        for idx_str in self.agents:
            idx = int(idx_str)
            key = idx if idx in batch else idx_str

            if key in batch:
                x, y = batch[key]
                inputs[idx] = x
                labels[idx] = y

        return inputs, labels

    def on_train_epoch_end(self) -> None:
        self._finalize_train_epoch_communication()
        self._log_train_comm_task_perf()

    def _shared_eval(
        self,
        batch: dict[int, list[torch.Tensor]],
        batch_idx: int,
        prefix: str,
    ) -> tuple[dict[str, Any], torch.Tensor]:
        """Compute losses and metrics for train/validation/test steps.

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
            (prediction, label) pairs and total_loss includes task and
            prototype regularization.
        """
        if isinstance(batch, tuple):
            batch = batch[0]

        outputs = {}
        agent_losses = {}
        agent_performances = {}
        latents_per_agent: dict[int, torch.Tensor] = {}

        inputs, labels = self._extract_local_batch(batch)

        for idx_str, agent in self.agents.items():
            idx = int(idx_str)
            x = inputs.get(idx)
            y = labels.get(idx)

            if x is None or y is None:
                continue

            z = agent.encode(x)
            y_hat = agent.decoder(z)

            outputs[idx] = (y_hat.detach(), y)
            latents_per_agent[idx] = z

            agent_losses[idx] = agent.compute_loss(y_hat, y)
            agent_performances[idx] = agent.task_performance(y_hat, y)

        proto_loss: torch.Tensor = torch.tensor(0.0, device=self.device)

        if prefix == 'train':
            local_prototypes = {
                idx: self._compute_class_prototypes(z, labels[idx])
                for idx, z in latents_per_agent.items()
                if labels.get(idx) is not None
            }

            # Per the FedProto paper: after each local update clients upload their
            # prototypes, the server aggregates and broadcasts global prototypes.
            # We model this as one communication round per batch.
            if local_prototypes:
                # Each agent sends its local class prototypes to the server (1 hop).
                for idx, local_proto in local_prototypes.items():
                    if local_proto:
                        proto_tensor = torch.stack(
                            list(local_proto.values())
                        ).detach()
                        self._record_communication(
                            proto_tensor, n_transmissions=1
                        )

                self._record_communication_round(n_rounds=1, prefix='train')

                # EMA update of global prototypes with the current batch's locals.
                self._update_global_prototypes(local_prototypes)

            if self._global_prototypes:
                client_proto_losses = {}
                for idx, local_proto in local_prototypes.items():
                    loss = self._compute_proto_regularization(
                        local_proto, self._global_prototypes
                    )
                    client_proto_losses[idx] = loss

                if client_proto_losses:
                    proto_loss = torch.stack(
                        list(client_proto_losses.values())
                    ).mean()

        total_task_loss = torch.stack(list(agent_losses.values())).sum()
        total_loss = (
            total_task_loss + self._effective_lambda_reg() * proto_loss
        )

        self._log_shared_metrics(
            prefix=prefix,
            agent_losses=agent_losses,
            agent_performances=agent_performances,
            batch_size=self._resolve_batch_size(batch),
            agent_sample_counts=self._resolve_agent_sample_counts(batch),
            total_loss=total_loss,
            skip_task_performance=(prefix == 'test'),
            extra_metrics={
                f'{prefix}/proto_loss': proto_loss,
                f'{prefix}/num_global_prototypes': torch.tensor(
                    len(self._global_prototypes), device=self.device
                ),
            },
            prog_bar=False,
            per_agent_loss_name='task_loss',
        )

        return outputs, total_loss

    # send_message, _fit_alignment_maps, _cleanup_alignment and
    # evaluate_communication_accuracy are inherited from
    # PostTrainingAlignmentMixin.  Post-hoc whitening + alignment maps are
    # fitted on pilot data before evaluation so cross-agent communication task
    # metrics can be reported, mirroring FederatedLearning / NonCooperative.

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
