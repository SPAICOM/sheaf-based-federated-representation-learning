"""
Sheaf-based Federated Representation Learning orchestrator.

This module implements the proposed federated learning framework with
Sheaf regularization that maintains aligned latent spaces across agents
through Stiefel manifold optimization of cross-covariance matrices.
Shared pilot batches provide the semantic correspondence needed to align
neighboring agents accurately.
"""

from dataclasses import replace
from typing import Any

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
        max_lmb: float,
        latent_dims: dict,
        parseval_normalization: bool,
        l2_normalization: bool,
        parseval_eps: float = 1e-4,
        # local_steps: int = 1,
        anchor_strategy: str = 'pilots',
        num_anchors: int = 128,
        use_prototypes: bool = False,
        lambda_schedule: str | None = None,
        sparse_communication: bool = False,
        sparse_epsilon: float = 1e-2,
        update_v_every_n_epochs: int = 1,
        log_latent_diagnostics: bool = False,
        **kwargs,
    ):
        super().__init__(
            agents=agents,
            neighbors=neighbors,
            optimizer=optimizer,
            log_latent_diagnostics=log_latent_diagnostics,
        )

        anchor_strategy = str(anchor_strategy)
        update_v_every_n_epochs = int(update_v_every_n_epochs)

        if anchor_strategy != 'pilots':
            raise ValueError(
                f'Unknown anchor_strategy: {anchor_strategy}. '
                "Valid options: ['pilots']"
            )
        if update_v_every_n_epochs < 1:
            raise ValueError('update_v_every_n_epochs must be at least 1')

        self.save_hyperparameters()
        self.anchor_config = AnchorConfig(
            parseval_normalization=bool(parseval_normalization),
            l2_normalization=bool(l2_normalization),
            parseval_eps=float(parseval_eps),
            use_prototypes=bool(use_prototypes),
            sparse_communication=bool(sparse_communication),
            sparse_epsilon=float(sparse_epsilon),
        )
        self._latest_pilots: dict[
            int, tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]
        ] = {}
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

    def _extract_pilot_batch(
        self, batch: dict, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Extract pilot data for an agent, handling global, private, or pairwise keys."""
        if f'pilot_{idx}' in batch:
            return (
                batch[f'pilot_{idx}'][0],
                batch[f'pilot_{idx}'][1],
                batch[f'pilot_{idx}'][2],
            )
        if f'global_pilot_{idx}' in batch:
            return (
                batch[f'global_pilot_{idx}'][0],
                batch[f'global_pilot_{idx}'][1],
                batch[f'global_pilot_{idx}'][2],
            )
        if 'global_pilot' in batch:
            return (
                batch['global_pilot'][0],
                batch['global_pilot'][1],
                batch['global_pilot'][2],
            )

        for key, value in batch.items():
            if isinstance(key, str) and key.startswith('pilot_'):
                parts = key.split('_')
                if len(parts) == 3:
                    i, j = int(parts[1]), int(parts[2])
                    if i == idx:
                        return value[0], value[1], value[2]
                    if j == idx:
                        return value[3], value[4], value[5]

        raise ValueError(f'Pilot batch missing for agent {idx}.')

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
            prototype_labels,
            device=labels.device,
            dtype=labels.dtype,
        )

    def _pilot_match_keys(
        self, y_pilot: torch.Tensor, sample_ids: torch.Tensor | None
    ) -> torch.Tensor:
        if sample_ids is not None:
            return sample_ids
        return y_pilot

    def _normalize_pilot_latents(
        self,
        pilot_latents: torch.Tensor,
        y_pilot: torch.Tensor,
        sample_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.anchor_config.use_prototypes:
            step_latents, step_keys = self._compute_class_prototypes(
                pilot_latents,
                y_pilot,
            )
        else:
            step_latents = pilot_latents
            step_keys = self._pilot_match_keys(y_pilot, sample_ids)

        return normalize_anchor_matrix(
            step_latents, self.anchor_config
        ), step_keys

    @torch.no_grad()
    def _encode_pilots_eval(
        self, agent: nn.Module, x_pilot: torch.Tensor
    ) -> torch.Tensor:
        was_training = agent.training
        agent.eval()
        try:
            return agent.encode(x_pilot)
        finally:
            agent.train(was_training)

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
        keys_per_agent: dict[int, torch.Tensor],
    ) -> None:
        if not self.stiefel_matrices:
            return

        param_device = next(iter(self.stiefel_matrices.values())).device

        for edge_key, V_param in self.stiefel_matrices.items():
            node_i, node_j = map(int, edge_key.split('_'))

            if (
                node_i not in latents_per_agent
                or node_j not in latents_per_agent
            ):
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
            # A_i = A_i - A_i.mean(dim=0, keepdim=True)
            # A_j = A_j - A_j.mean(dim=0, keepdim=True)

            C = torch.matmul(A_i.T, A_j)
            C = C + torch.randn_like(C) * 1e-6

            try:
                U, _S, W_T = torch.linalg.svd(C, full_matrices=False)
            except RuntimeError:
                C_cpu = C.cpu()
                U_cpu, _, W_T_cpu = torch.linalg.svd(
                    C_cpu, full_matrices=False
                )
                U, W_T = U_cpu.to(param_device), W_T_cpu.to(param_device)

            V_new = torch.matmul(U, W_T).to(
                dtype=V_param.dtype, device=param_device
            )
            V_param.copy_(V_new)

    def on_train_start(self) -> None:
        super().on_train_start()
        self._latest_pilots.clear()

    def on_train_epoch_start(self) -> None:
        self._latest_pilots.clear()

    @torch.no_grad()
    def on_train_epoch_end(self) -> None:
        if self.current_epoch % self.hparams.update_v_every_n_epochs != 0:
            self._latest_pilots.clear()
            self._finalize_train_epoch_communication()
            return

        epoch_latents: dict[int, torch.Tensor] = {}
        epoch_keys: dict[int, torch.Tensor] = {}
        payloads_per_agent: dict[int, Any] = {}

        for idx_str in self.agents:
            idx = int(idx_str)
            latest_pilots = self._latest_pilots.get(idx)
            if latest_pilots is None:
                continue

            x_pilot, y_pilot, sample_ids = latest_pilots
            x_pilot = x_pilot.to(self.device)
            y_pilot = y_pilot.to(self.device)
            if sample_ids is not None:
                sample_ids = sample_ids.to(self.device)

            raw_A = self._encode_pilots_eval(self.agents[idx_str], x_pilot)
            final_A, final_keys = self._normalize_pilot_latents(
                raw_A,
                y_pilot,
                sample_ids,
            )

            epoch_latents[idx] = final_A
            epoch_keys[idx] = final_keys
            payloads_per_agent[idx] = communication_anchor_payload(
                anchor_matrix=final_A,
                labels=final_keys,
                config=self.anchor_config,
            )

        if epoch_latents:
            self._record_communication_round(n_rounds=1, prefix='train')
            for idx, payload in payloads_per_agent.items():
                n_neighbors = len(
                    self.hparams.neighbors.get(
                        idx,
                        self.hparams.neighbors.get(str(idx), set()),
                    )
                )
                if n_neighbors > 0:
                    self._record_communication(
                        payload,
                        n_transmissions=n_neighbors,
                        prefix='train',
                    )
            self._update_stiefel_matrices(epoch_latents, epoch_keys)

        self._latest_pilots.clear()
        self._finalize_train_epoch_communication()

    def on_validation_epoch_end(self) -> None:
        super().on_validation_epoch_end()
        if not getattr(self.hparams, 'log_latent_diagnostics', False):
            return
        logs: dict[str, float] = {}
        for edge_key, V in self.stiefel_matrices.items():
            node_i, node_j = edge_key.split('_')
            v = V.detach().float().cpu()
            logs[f'validation/alignment_rank_edge_{node_i}_{node_j}'] = float(
                torch.linalg.matrix_rank(v).item()
            )
            logs[
                f'validation/alignment_effective_rank_edge_{node_i}_{node_j}'
            ] = self._effective_rank(v)
        if logs:
            self.log_dict(
                logs,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                add_dataloader_idx=False,
            )

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
        payloads_per_agent: dict[int, Any] = {}
        keys_per_agent: dict[int, torch.Tensor] = {}

        # Compute task loss
        for idx_str, agent in self.agents.items():
            idx = int(idx_str)
            x_task, y_task = batch[self._resolve_key(batch, idx)]

            latent_task = agent.encode(x_task)
            y_hat = agent.decoder(latent_task)

            outputs[idx_str] = (y_hat.detach(), y_task)
            agent_losses[idx] = agent.compute_loss(y_hat, y_task)
            agent_performances[idx] = agent.task_performance(y_hat, y_task)

        total_task_loss = torch.stack(list(agent_losses.values())).sum()

        for idx_str, agent in self.agents.items():
            idx = int(idx_str)
            x_pilot, y_pilot, sample_ids = self._extract_pilot_batch(
                batch, idx
            )
            self._latest_pilots[idx] = (x_pilot, y_pilot, sample_ids)

            pilot_latents = agent.encode(x_pilot)
            step_latents, step_keys = self._normalize_pilot_latents(
                pilot_latents,
                y_pilot,
                sample_ids,
            )

            latents_per_agent[idx] = step_latents
            keys_per_agent[idx] = step_keys

            payloads_per_agent[idx] = communication_anchor_payload(
                anchor_matrix=step_latents,
                labels=step_keys,
                config=self.anchor_config,
            )

        if prefix in self._COMMUNICATION_SPLITS:
            self._record_communication_round(n_rounds=1, prefix=prefix)
            for idx, payload in payloads_per_agent.items():
                n_neighbors = len(
                    self.hparams.neighbors.get(
                        idx,
                        self.hparams.neighbors.get(str(idx), set()),
                    )
                )
                if n_neighbors > 0:
                    self._record_communication(
                        payload,
                        n_transmissions=n_neighbors,
                        prefix=prefix,
                    )

        sheaf_penalty = torch.tensor(0.0, device=self.device)

        for edge_key, V in self.stiefel_matrices.items():
            node_i, node_j = map(int, edge_key.split('_'))

            if (
                node_i not in latents_per_agent
                or node_j not in latents_per_agent
            ):
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

        total_loss = (
            total_task_loss + self._effective_lambda_reg() * sheaf_penalty
        )

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
