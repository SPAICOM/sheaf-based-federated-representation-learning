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
)


class SheafFRL(BaseOrchestrator):
    """Sheaf-based Federated Representation Learning orchestrator."""

    def __init__(
        self,
        agents: dict[int, nn.Module],
        neighbors: dict[int, set[int]],
        optimizer,
        lambda_sheaf: float,
        latent_dims: dict,
        parseval_normalization: bool,
        l2_normalization: bool,
        parseval_eps: float = 1e-4,
        local_steps: int = 1,
        anchor_strategy: str = 'pilots',
        num_anchors: int = 64,
        use_prototypes: bool = False,
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
            use_prototypes=bool(use_prototypes),
        )

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
        if f'pilot_{idx}' in batch:
            return batch[f'pilot_{idx}'][0], batch[f'pilot_{idx}'][1], batch[f'pilot_{idx}'][2]
        if f'global_pilot_{idx}' in batch:
            return batch[f'global_pilot_{idx}'][0], batch[f'global_pilot_{idx}'][1], batch[f'global_pilot_{idx}'][2]
        if 'global_pilot' in batch:
            return batch['global_pilot'][0], batch['global_pilot'][1], batch['global_pilot'][2]
            
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
        return replace(self.anchor_config)

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

    def _match_keys(
        self,
        A_i: torch.Tensor,
        A_j: torch.Tensor,
        keys_i: torch.Tensor,
        keys_j: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Universal geometric matcher: pairs up matrices via Sample IDs or Classes."""
        target_keys = set(keys_i.tolist()) & set(keys_j.tolist())

        if not target_keys:
            return None

        matched_i, matched_j = [], []
        for k in sorted(target_keys):
            idx_i = torch.where(keys_i == k)[0]
            idx_j = torch.where(keys_j == k)[0]
            matched_count = min(len(idx_i), len(idx_j))
            if matched_count > 0:
                matched_i.append(A_i[idx_i[:matched_count]])
                matched_j.append(A_j[idx_j[:matched_count]])

        if not matched_i:
            return None
        return torch.cat(matched_i, dim=0), torch.cat(matched_j, dim=0)

    @torch.no_grad()
    def _update_stiefel_matrices(
        self,
        latents_per_agent: dict[int, torch.Tensor],
        keys_per_agent: dict[int, torch.Tensor]
    ) -> None:
        if not self.stiefel_matrices:
            return

        param_device = next(iter(self.stiefel_matrices.values())).device

        for edge_key, V_param in self.stiefel_matrices.items():
            node_i, node_j = map(int, edge_key.split('_'))

            if node_i not in latents_per_agent or node_j not in latents_per_agent:
                continue

            shared_rows = self._match_keys(
                A_i=latents_per_agent[node_i],
                A_j=latents_per_agent[node_j],
                keys_i=keys_per_agent[node_i],
                keys_j=keys_per_agent[node_j],
            )
            if shared_rows is None:
                continue

            A_i, A_j = shared_rows
            #A_i = A_i - A_i.mean(dim=0, keepdim=True)
            #A_j = A_j - A_j.mean(dim=0, keepdim=True)

            C = torch.matmul(A_i.T, A_j)
            C = C + torch.randn_like(C) * 1e-6

            try:
                U, _S, W_T = torch.linalg.svd(C, full_matrices=False)
            except RuntimeError:
                C_cpu = C.cpu()
                U_cpu, _, W_T_cpu = torch.linalg.svd(C_cpu, full_matrices=False)
                U, W_T = U_cpu.to(param_device), W_T_cpu.to(param_device)
            
            V_new = torch.matmul(U, W_T).to(dtype=V_param.dtype, device=param_device)
            V_param.copy_(V_new)

    def on_train_start(self) -> None:
        super().on_train_start()
        self._train_local_step_count = 0
        for idx in self.epoch_latents_cache:
            self.epoch_latents_cache[idx].clear()
            self.epoch_labels_cache[idx].clear()

    @torch.no_grad()
    def on_train_epoch_end(self) -> None:
        epoch_latents: dict[int, torch.Tensor] = {}
        epoch_keys: dict[int, torch.Tensor] = {}

        for idx_str in self.agents.keys():
            idx = int(idx_str)
            if self.epoch_latents_cache.get(idx):
                raw_A = torch.cat(self.epoch_latents_cache[idx], dim=0).to(self.device)
                raw_keys = torch.cat(self.epoch_labels_cache[idx], dim=0).to(self.device)

                if self.anchor_config.use_prototypes:
                    final_A, final_keys = self._compute_class_prototypes(raw_A, raw_keys)
                elif self.hparams.num_anchors < raw_A.shape[0]:
                    # Global Uniform Subsampling
                    g = torch.Generator(device=self.device).manual_seed(self.current_epoch)
                    indices = torch.randperm(
                        raw_A.shape[0],
                        generator=g,
                        device=raw_A.device,
                    )[: self.hparams.num_anchors]
                    final_A = raw_A[indices]
                    final_keys = raw_keys[indices]
                else:
                    final_A, final_keys = raw_A, raw_keys

                epoch_latents[idx] = normalize_anchor_matrix(final_A, self.anchor_config)
                epoch_keys[idx] = final_keys

        if epoch_latents:
            self._update_stiefel_matrices(epoch_latents, epoch_keys)

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
        keys_per_agent: dict[int, torch.Tensor] = {}
        task_latents_cache: dict[int, torch.Tensor] = {}
        task_labels_cache: dict[int, torch.Tensor] = {}
        anchor_config = self._effective_anchor_config()

        # Compute task loss
        for idx_str, agent in self.agents.items():
            idx = int(idx_str)
            x_task, y_task = batch[self._resolve_key(batch, idx)]

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

        for idx_str, agent in self.agents.items():
            idx = int(idx_str)
            
            # Extract anchors
            if self.hparams.anchor_strategy == 'batch_anchors':
                raw_latents = task_latents_cache[idx]
                raw_keys = task_labels_cache[idx]
                class_labels = task_labels_cache[idx]
            elif self.hparams.anchor_strategy == 'pilots':
                x_pilot, y_pilot, sample_ids = self._extract_pilot_batch(batch, idx)
                raw_latents = agent.encode(x_pilot)
                raw_keys = sample_ids if sample_ids is not None else y_pilot
                class_labels = y_pilot
            else:
                raise ValueError(f'Unknown anchor_strategy: {self.hparams.anchor_strategy}')

            # Caching (ensuring SVD groups by class if prototype, else by sample ID for pilots)
            if prefix == 'train':
                self.epoch_latents_cache[idx].append(raw_latents.detach().cpu())
                keys_to_cache = class_labels if anchor_config.use_prototypes else raw_keys
                self.epoch_labels_cache[idx].append(keys_to_cache.detach().cpu())

            # Apply strategy to current step
            if anchor_config.use_prototypes:
                step_latents, step_keys = self._compute_class_prototypes(raw_latents, class_labels)
            elif self.hparams.num_anchors < raw_latents.shape[0]:
                step_latents = raw_latents[:self.hparams.num_anchors]
                step_keys = raw_keys[:self.hparams.num_anchors]
            else:
                step_latents, step_keys = raw_latents, raw_keys

            latents_per_agent[idx] = normalize_anchor_matrix(step_latents, anchor_config)
            keys_per_agent[idx] = step_keys

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
            return outputs, total_task_loss

        # Communication Payload
        if prefix in {'train', 'test', 'test_monitor'}:
            self._record_communication_round(n_rounds=1, prefix=prefix)
            for idx, latents in latents_per_agent.items():
                n_neighbors = len(self.hparams.neighbors.get(idx, self.hparams.neighbors.get(str(idx), set())))
                if n_neighbors > 0:
                    payload = communication_anchor_payload(
                        anchor_matrix=latents,
                        labels=keys_per_agent[idx],
                        config=anchor_config,
                    )
                    self._record_communication(
                        payload, n_transmissions=n_neighbors, prefix=prefix,
                    )

        sheaf_penalty = torch.tensor(0.0, device=self.device)

        for edge_key, V in self.stiefel_matrices.items():
            node_i, node_j = map(int, edge_key.split('_'))

            if node_i not in latents_per_agent or node_j not in latents_per_agent:
                continue

            shared_rows = self._match_keys(
                A_i=latents_per_agent[node_i],
                A_j=latents_per_agent[node_j],
                keys_i=keys_per_agent[node_i],
                keys_j=keys_per_agent[node_j],
            )
            if shared_rows is None:
                continue

            A_i_shared, A_j_shared = shared_rows
            diff = A_i_shared - torch.matmul(A_j_shared, V.T)
            
            sheaf_penalty += (diff**2).sum(dim=1).mean()

        total_loss = total_task_loss + self.hparams.lambda_sheaf * sheaf_penalty

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
