"""
Sheaf-based Federated Representation Learning orchestrator.

This module implements the proposed federated learning framework with
Sheaf regularization that maintains aligned latent spaces across agents
through Stiefel manifold optimization of cross-covariance matrices. 
Anchor strategies are implemented with explicit semantic correspondence 
keys or shared pilot batches so neighboring agents are aligned accurately.
"""

from typing import Any
from dataclasses import replace

import torch
import torch.nn as nn

from src.orchestrators.base_orchestrator import BaseOrchestrator
from src.utils.anchors import (
    AnchorConfig,
    communication_anchor_payload,
    normalize_anchor_matrix,
    shared_anchor_rows,
)


class SheafFRL(BaseOrchestrator):
    """Sheaf-based Federated Representation Learning orchestrator."""

    def __init__(
        self,
        agents: dict[int, nn.Module],
        neighbors: dict[int, set[int]],
        optimizer,
        max_lmb: float,
        latent_dims: dict,
        parseval_normalization: bool,
        l2_normalization: bool,
        parseval_eps: float = 1e-4,
        local_steps: int = 1,
        anchor_strategy: str = 'pilots',
        filter_unseen_classes: bool = True,
        use_prototypes: bool = False,
        lambda_schedule: str | None = None,
        **kwargs,
    ):
        super().__init__(
            agents=agents,
            neighbors=neighbors,
            optimizer=optimizer,
        )

        if anchor_strategy not in {'pilots', 'batch_anchors'}:
            raise ValueError(
                f'Unknown anchor_strategy: {anchor_strategy}. '
                "Valid options: ['batch_anchors', 'pilots']"
            )

        self.save_hyperparameters()
        self.anchor_config = AnchorConfig(
            parseval_normalization=bool(parseval_normalization),
            l2_normalization=bool(l2_normalization),
            parseval_eps=float(parseval_eps),
            filter_unseen_classes=bool(filter_unseen_classes),
            use_prototypes=bool(use_prototypes),
        )

        self.seen_classes: dict[int, set[int]] = {int(k): set() for k in agents.keys()}

        self.epoch_latents_cache: dict[int, list[torch.Tensor]] = {int(k): [] for k in agents.keys()}
        self.epoch_labels_cache: dict[int, list[torch.Tensor]] = {int(k): [] for k in agents.keys()}

        self.stiefel_matrices = nn.ParameterDict()
        latent_dims_int = {int(k): int(v) for k, v in latent_dims.items()}

        for i_raw, neighborset in neighbors.items():
            for j_raw in neighborset:
                i, j = int(i_raw), int(j_raw)

                if latent_dims_int[i] > latent_dims_int[j]:
                    node_i, node_j = i, j
                elif latent_dims_int[i] < latent_dims_int[j]:
                    node_i, node_j = j, i
                else:
                    node_i, node_j = max(i, j), min(i, j)

                edge_key = f'{node_i}_{node_j}'

                if edge_key not in self.stiefel_matrices:
                    d_i = latent_dims_int[node_i]
                    d_j = latent_dims_int[node_j]
                    stiefel_matrix = torch.eye(d_i, d_j)
                    self.stiefel_matrices[edge_key] = nn.Parameter(
                        stiefel_matrix, requires_grad=False
                    )

    def _resolve_key(self, batch: dict, idx: int) -> int | str:
        str_key = str(idx)
        return str_key if str_key in batch else idx
    
    def _extract_pilot_batch(self, batch: dict, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Extract pilot data for an agent, handling global, private, or pairwise keys."""
        if f'global_pilot_{idx}' in batch:
            return batch[f'global_pilot_{idx}'][0], batch[f'global_pilot_{idx}'][1], batch[f'global_pilot_{idx}'][2]
            
        if f'pilot_{idx}' in batch:
            return batch[f'pilot_{idx}'][0], batch[f'pilot_{idx}'][1], batch[f'pilot_{idx}'][2]
            
        for key, value in batch.items():
            if isinstance(key, str) and key.startswith('pilot_'):
                parts = key.split('_')
                if len(parts) == 3:
                    i, j = int(parts[1]), int(parts[2])
                    if i == idx:
                        return value[0], value[1], value[2]
                    elif j == idx:
                        return value[3], value[4], value[5]
                        
        raise ValueError(f"Pilot batch missing for agent {idx}.")

    def _effective_anchor_config(self) -> AnchorConfig:
        return replace(
            self.anchor_config,
            use_prototypes=bool(
                self.anchor_config.use_prototypes
                or self.hparams.anchor_strategy == 'batch_anchors'
            ),
        )

    def _compute_class_prototypes(
        self,
        anchor_matrix: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if anchor_matrix.numel() == 0 or labels.numel() == 0:
            return anchor_matrix[:0], labels[:0]

        prototypes, prototype_labels = [], []
        for class_label in torch.unique(labels, sorted=True).tolist():
            mask = labels == class_label
            if mask.any():
                prototypes.append(anchor_matrix[mask].mean(dim=0))
                prototype_labels.append(class_label)

        if not prototypes:
            return anchor_matrix[:0], labels[:0]

        return torch.stack(prototypes, dim=0), torch.tensor(
            prototype_labels, device=labels.device, dtype=labels.dtype,
        )

    def _match_cached_anchors(
        self, A_i: torch.Tensor, A_j: torch.Tensor, y_i: torch.Tensor, y_j: torch.Tensor, seen_i: set[int], seen_j: set[int]
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Safely pairs up accumulated epoch rows chronologically without re-averaging them."""
        target_classes = set(y_i.tolist()) & set(y_j.tolist())
        if self.anchor_config.filter_unseen_classes:
            target_classes &= seen_i & seen_j

        if not target_classes:
            return None

        # Chronological matching perfectly aligns step trajectories for BOTH pilots and batch_anchors
        matched_i, matched_j = [], []
        for c in sorted(target_classes):
            idx_i = torch.where(y_i == c)[0]
            idx_j = torch.where(y_j == c)[0]
            k = min(len(idx_i), len(idx_j))
            if k > 0:
                matched_i.append(A_i[idx_i[:k]])
                matched_j.append(A_j[idx_j[:k]])

        if not matched_i:
            return None
        return torch.cat(matched_i, dim=0), torch.cat(matched_j, dim=0)

    @torch.no_grad()
    def _update_stiefel_matrices(
        self,
        latents_per_agent: dict[int, torch.Tensor],
        labels_per_agent: dict[int, torch.Tensor]
    ) -> None:
        if not self.stiefel_matrices:
            return

        param_device = next(iter(self.stiefel_matrices.values())).device

        for edge_key, V_param in self.stiefel_matrices.items():
            node_i, node_j = map(int, edge_key.split('_'))

            if node_i not in latents_per_agent or node_j not in latents_per_agent:
                continue

            shared_rows = self._match_cached_anchors(
                A_i=latents_per_agent[node_i],
                A_j=latents_per_agent[node_j],
                y_i=labels_per_agent[node_i],
                y_j=labels_per_agent[node_j],
                seen_i=self.seen_classes[node_i],
                seen_j=self.seen_classes[node_j],
            )
            if shared_rows is None:
                continue

            A_i, A_j = shared_rows
            A_i = A_i - A_i.mean(dim=0, keepdim=True)
            A_j = A_j - A_j.mean(dim=0, keepdim=True)

            C = torch.matmul(A_i.T, A_j)
            U, _S, W_T = torch.linalg.svd(C, full_matrices=False)
            
            V_new = torch.matmul(U, W_T).to(dtype=V_param.dtype, device=param_device)
            V_param.copy_(V_new)

    def on_train_start(self) -> None:
        super().on_train_start()
        self._train_local_step_count = 0
        for idx in self.seen_classes:
            self.seen_classes[idx].clear()
        for idx in self.epoch_latents_cache:
            self.epoch_latents_cache[idx].clear()
            self.epoch_labels_cache[idx].clear()

    @torch.no_grad()
    def on_train_epoch_end(self) -> None:
        epoch_latents: dict[int, torch.Tensor] = {}
        epoch_labels: dict[int, torch.Tensor] = {}

        for idx_str in self.agents.keys():
            idx = int(idx_str)
            if self.epoch_latents_cache.get(idx):
                epoch_latents[idx] = torch.cat(self.epoch_latents_cache[idx], dim=0).to(self.device)
                epoch_labels[idx] = torch.cat(self.epoch_labels_cache[idx], dim=0).to(self.device)

        if epoch_latents:
            self._update_stiefel_matrices(epoch_latents, epoch_labels)

        for idx in self.epoch_latents_cache:
            self.epoch_latents_cache[idx].clear()
            self.epoch_labels_cache[idx].clear()

        self._finalize_train_epoch_communication()

    def _shared_eval(
        self,
        batch: dict[int, list[torch.Tensor]],
        batch_idx: int,
        prefix: str,
    ):
        if isinstance(batch, tuple):
            batch = batch[0]

        outputs = {}
        agent_losses = {}
        agent_performances = {}

        latents_per_agent: dict[int, torch.Tensor] = {}
        labels_per_agent: dict[int, torch.Tensor] = {}
        task_latents_cache: dict[int, torch.Tensor] = {}
        task_labels_cache: dict[int, torch.Tensor] = {}
        anchor_config = self._effective_anchor_config()

        # Compute task loss
        for idx_str, agent in self.agents.items():
            idx = int(idx_str)
            x_task, y_task = batch[self._resolve_key(batch, idx)]

            if prefix == 'train':
                self.seen_classes[idx].update(y_task.detach().cpu().tolist())

            latent_task = agent.encode(x_task)
            y_hat = agent.decoder(latent_task)

            task_latents_cache[idx] = latent_task
            task_labels_cache[idx] = y_task

            outputs[idx_str] = (y_hat.detach(), y_task)
            agent_losses[idx] = agent.compute_loss(y_hat, y_task)
            agent_performances[idx] = agent.task_performance(y_hat, y_task)

        total_task_loss = torch.stack(list(agent_losses.values())).sum()

        is_communication_step = True
        if prefix == 'train':
            self._train_local_step_count += 1
            is_communication_step = (
                self._train_local_step_count % int(self.hparams.local_steps) == 0
            )

        # Extract anchors and compute prototypes if configured, then cache for SVD at epoch end
        for idx_str, agent in self.agents.items():
            idx = int(idx_str)
            if self.hparams.anchor_strategy == 'batch_anchors':
                raw_latents = task_latents_cache[idx]
                anchor_labels = task_labels_cache[idx]
                raw_latents, anchor_labels = self._compute_class_prototypes(raw_latents, anchor_labels)
            elif self.hparams.anchor_strategy == 'pilots':
                x_pilot, y_pilot, _ = self._extract_pilot_batch(batch, idx)
                raw_latents = agent.encode(x_pilot)
                anchor_labels = y_pilot
                if anchor_config.use_prototypes:
                    raw_latents, anchor_labels = self._compute_class_prototypes(raw_latents, anchor_labels)
            else:
                raise ValueError(f'Unknown anchor_strategy: {self.hparams.anchor_strategy}')

            latents_per_agent[idx] = normalize_anchor_matrix(raw_latents, anchor_config)
            labels_per_agent[idx] = anchor_labels

        if prefix == 'train':
            for idx in latents_per_agent:
                self.epoch_latents_cache[idx].append(latents_per_agent[idx].detach().cpu())
                self.epoch_labels_cache[idx].append(labels_per_agent[idx].detach().cpu())

        # Early exit for local training steps
        if prefix == 'train' and not is_communication_step:
            sheaf_penalty = torch.tensor(0.0, device=self.device)
            self._log_shared_metrics(
                prefix=prefix,
                agent_losses=agent_losses,
                agent_performances=agent_performances,
                batch_size=self._resolve_batch_size(batch),
                agent_sample_counts=self._resolve_agent_sample_counts(batch),
                total_loss=total_task_loss,
                extra_metrics={f'{prefix}/sheaf_penalty': sheaf_penalty},
                prog_bar=False,
                per_agent_loss_name='task_loss',
            )
            return outputs, total_loss

        # Pilot batch
        for idx_str, agent in self.agents.items():
            idx = int(idx_str)
            x_pilot, y_pilot, _ = self._extract_pilot_batch(batch, idx)

            raw_latents = agent.encode(x_pilot)
            latents_per_agent[idx] = normalize_anchor_matrix(raw_latents, self.anchor_config)
            labels_per_agent[idx] = y_pilot

        if prefix in {'train', 'test', 'test_monitor'} or (prefix == 'train' and is_communication_step):
            self._record_communication_round(n_rounds=1, prefix=prefix)
            for idx, latents in latents_per_agent.items():
                n_neighbors = len(self.hparams.neighbors.get(idx, self.hparams.neighbors.get(str(idx), set())))
                if n_neighbors > 0:
                    payload = communication_anchor_payload(
                        anchor_matrix=latents,
                        labels=labels_per_agent[idx],
                        config=anchor_config,
                    )
                    self._record_communication(
                        payload, n_transmissions=n_neighbors, prefix=prefix,
                    )

        sheaf_penalty = torch.tensor(0.0, device=self.device)
        
        # Disable re-averaging in step penalty because we already compressed them in step 2
        step_penalty_config = replace(anchor_config, use_prototypes=False)

        for edge_key, V in self.stiefel_matrices.items():
            node_i, node_j = map(int, edge_key.split('_'))

            if node_i not in latents_per_agent or node_j not in latents_per_agent:
                continue

            shared_rows = shared_anchor_rows(
                A_i=latents_per_agent[node_i],
                A_j=latents_per_agent[node_j],
                labels_i=labels_per_agent[node_i],
                labels_j=labels_per_agent[node_j],
                seen_i=self.seen_classes[node_i],
                seen_j=self.seen_classes[node_j],
                config=step_penalty_config,
            )
            if shared_rows is None:
                continue

            A_i_shared, A_j_shared = shared_rows
            diff = A_i_shared - torch.matmul(A_j_shared, V.T)
            sheaf_penalty += (diff**2).sum(dim=1).mean()

        total_loss = total_task_loss + self._effective_lambda_reg() * sheaf_penalty

        self._log_shared_metrics(
            prefix=prefix,
            agent_losses=agent_losses,
            agent_performances=agent_performances,
            batch_size=self._resolve_batch_size(batch),
            agent_sample_counts=self._resolve_agent_sample_counts(batch),
            total_loss=total_loss,
            extra_metrics={f'{prefix}/sheaf_penalty': sheaf_penalty},
            prog_bar=False,
            per_agent_loss_name='task_loss',
        )

        return outputs, total_loss