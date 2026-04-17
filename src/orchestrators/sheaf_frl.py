"""
Sheaf-based Federated Representation Learning orchestrator.

This module implements the proposed federated learning framework with
Sheaf regularization that maintains aligned latent spaces across agents
through Stiefel manifold optimization of cross-covariance matrices. 
Anchor strategies are implemented with explicit semantic correspondence 
keys or shared pilot batches so neighboring agents are aligned accurately.
"""

from typing import Any

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
    """Sheaf-based Federated Representation Learning orchestrator.

    Implements a federated learning framework with Sheaf regularization that
    maintains aligned latent spaces across agents through Stiefel manifold
    optimization of cross-covariance matrices.

    Parameters
    ----------
    agents : dict[int, nn.Module]
        Dictionary mapping agent indices to their model instances.
    neighbors : dict[int, set[int]]
        Dictionary mapping each agent index to the set of its neighbor indices.
    optimizer : hydra config
        Optimizer configuration for training.
    lambda_sheaf : float
        Weight coefficient for the sheaf regularization penalty.
    latent_dims : dict
        Dictionary mapping agent indices to their latent space dimensions.
    parseval_normalization : bool
        Whether to apply Parseval normalization to anchor features.
    local_steps : int
        Number of local training steps between communication rounds.
    filter_unseen_classes : bool
        If True, restricts alignment strictly to mutually shared classes.
    use_prototypes : bool
        If True, aligns class means instead of raw individual samples.

    Notes
    -----
    - The Stiefel matrices are frozen and updated in closed form each epoch.
    - Parseval normalization applies local whitening to anchor features.
    """

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
        filter_unseen_classes: bool = True,
        use_prototypes: bool = False,
        **kwargs,
    ):
        super().__init__(
            agents=agents,
            neighbors=neighbors,
            optimizer=optimizer,
        )

        self.save_hyperparameters()
        self.anchor_config = AnchorConfig(
            parseval_normalization=bool(parseval_normalization),
            l2_normalization=bool(l2_normalization),
            parseval_eps=float(parseval_eps),
            filter_unseen_classes=bool(filter_unseen_classes),
            use_prototypes=bool(use_prototypes),
        )

        # Track the classes each agent has seen locally
        self.seen_classes: dict[int, set[int]] = {int(k): set() for k in agents.keys()}

        self.stiefel_matrices = nn.ParameterDict()
        latent_dims_int = {int(k): int(v) for k, v in latent_dims.items()}

        # Create Stiefel matrices for each edge in the neighbor graph
        # Stiefel matrices V_ij map latent space of agent j to agent i
        for i_raw, neighborset in neighbors.items():
            for j_raw in neighborset:
                i = int(i_raw)
                j = int(j_raw)

                # Sort nodes by dimension to ensure consistent edge keys
                # Higher-dimensional node comes first
                if latent_dims_int[i] > latent_dims_int[j]:
                    node_i, node_j = i, j
                elif latent_dims_int[i] < latent_dims_int[j]:
                    node_i, node_j = j, i
                else:
                    # Equal dimensions: use max/min for deterministic ordering
                    node_i, node_j = max(i, j), min(i, j)

                edge_key = f'{node_i}_{node_j}'

                # Only create matrix once per edge (avoid duplicates)
                if edge_key not in self.stiefel_matrices:
                    d_i = latent_dims_int[node_i]
                    d_j = latent_dims_int[node_j]

                    # Initialize as identity mapping
                    # requires_grad=False because we use closed-form SVD updates
                    stiefel_matrix = torch.eye(d_i, d_j)

                    self.stiefel_matrices[edge_key] = nn.Parameter(
                        stiefel_matrix, requires_grad=False
                    )

    def _resolve_key(self, batch: dict, idx: int) -> int | str:
        str_key = str(idx)
        return str_key if str_key in batch else idx
    
    def _extract_pilot_batch(self, batch: dict, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Extract pilot data for an agent, handling global, private, or pairwise keys."""
        if 'global_pilot' in batch:
            return batch['global_pilot'][0], batch['global_pilot'][1], batch['global_pilot'][2]
            
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

    @torch.no_grad()
    def _update_stiefel_matrices(
        self,
        latents_per_agent: dict[int, torch.Tensor],
        labels_per_agent: dict[int, torch.Tensor]
    ) -> None:
        """Exact minimization of the V-block (Stiefel Matrices) via SVD."""
        if not self.stiefel_matrices:
            return

        param_device = next(iter(self.stiefel_matrices.values())).device

        for edge_key, V_param in self.stiefel_matrices.items():
            node_i, node_j = map(int, edge_key.split('_'))

            if node_i not in latents_per_agent or node_j not in latents_per_agent:
                continue

            shared_rows = shared_anchor_rows(
                A_i=latents_per_agent[node_i],
                A_j=latents_per_agent[node_j],
                labels=labels_per_agent[node_i], 
                seen_i=self.seen_classes[node_i],
                seen_j=self.seen_classes[node_j],
                config=self.anchor_config
            )
            if shared_rows is None:
                continue

            A_i, A_j = shared_rows

            # Center for unbiased cross-covariance estimation
            # Subtract mean along feature dimension to center the data
            A_i = A_i - A_i.mean(dim=0, keepdim=True)
            A_j = A_j - A_j.mean(dim=0, keepdim=True)

            # Compute cross-covariance C = A_i^T * A_j
            # Measures linear relationship between agents' features
            C = torch.matmul(A_i.T, A_j)

            # Compute thin SVD of cross-covariance: C = U * S * W^T
            # We only need U and W (left/right singular vectors)
            # for the orthogonal Procrustes solution
            U, _S, W_T = torch.linalg.svd(C, full_matrices=False)

            # Compute optimal rotation matrix V = U * W^T
            # that minimizes ||A_i - A_j * V||_F
            # This solves the orthogonal Procrustes problem for
            # aligning latent spaces
            V_new = torch.matmul(U, W_T).to(
                dtype=V_param.dtype, device=param_device
            )
            V_param.copy_(V_new)

    def on_train_start(self) -> None:
        """Initialize/reset per-agent anchor and label buffers."""
        super().on_train_start()
        self._train_local_step_count = 0
        for idx in self.seen_classes:
            self.seen_classes[idx].clear()

    @torch.no_grad()
    def on_train_epoch_end(self) -> None:
        """Communication is logged per step; finalize the totals here."""
        self._finalize_train_epoch_communication()

    def _shared_eval(
        self,
        batch: dict[int, list[torch.Tensor]],
        batch_idx: int,
        prefix: str,
    ):
        """Compute losses and metrics for validation/test steps.

        Task loss is evaluated on the full mini-batch; the sheaf regularization
        penalty is evaluated strictly on the Pilot Set.
        """
        if isinstance(batch, tuple):
            batch = batch[0]

        outputs = {}
        agent_losses = {}
        agent_performances = {}

        latents_per_agent: dict[int, torch.Tensor] = {}
        labels_per_agent: dict[int, torch.Tensor] = {}

        # Task loss and performance (full batch)
        for idx_str, agent in self.agents.items():
            idx = int(idx_str)
            x_task, y_task = batch[self._resolve_key(batch, idx)]

            # track seen classes dynamically
            if prefix == 'train':
                self.seen_classes[idx].update(y_task.detach().cpu().tolist())

            latent_task = agent.encode(x_task)
            y_hat = agent.decoder(latent_task)

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

        if prefix == 'train' and not is_communication_step:
            sheaf_penalty = torch.tensor(0.0, device=self.device)
            total_loss = total_task_loss

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
                        config=self.anchor_config,
                    )
                    self._record_communication(
                        payload,
                        n_transmissions=n_neighbors,
                        prefix=prefix,
                    )

        # Exact Minimization of V (Stiefel Matrices)
        if prefix == 'train':
            self._update_stiefel_matrices(latents_per_agent, labels_per_agent)
        
        # Sheaf regularization penalty (evaluated on anchor set)
        sheaf_penalty = torch.tensor(0.0, device=self.device)

        for edge_key, V in self.stiefel_matrices.items():
            node_i, node_j = map(int, edge_key.split('_'))

            if node_i not in latents_per_agent or node_j not in latents_per_agent:
                continue

            shared_rows = shared_anchor_rows(
                A_i=latents_per_agent[node_i],
                A_j=latents_per_agent[node_j],
                labels=labels_per_agent[node_i],
                seen_i=self.seen_classes[node_i],
                seen_j=self.seen_classes[node_j],
                config=self.anchor_config
            )
            if shared_rows is None:
                continue

            A_i_shared, A_j_shared = shared_rows
            diff = A_i_shared - torch.matmul(A_j_shared, V.T)
            frob_dist = (diff**2).sum(dim=1).mean()

            sheaf_penalty += frob_dist

        # Total loss: task loss (full batch) + sheaf penalty (anchor set)
        total_loss = (
            total_task_loss + self.hparams.lambda_sheaf * sheaf_penalty
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


if __name__ == '__main__':
    pass
