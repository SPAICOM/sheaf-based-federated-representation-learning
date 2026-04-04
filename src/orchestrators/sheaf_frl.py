"""
Sheaf-based Federated Representation Learning orchestrator.

This module implements the proposed federated learning framework with
Sheaf regularization
that maintains aligned latent spaces across agents through Stiefel manifold
optimization of cross-covariance matrices. Anchor strategies are implemented
with explicit semantic correspondence keys so neighboring agents are aligned
only on class-consistent anchors.
"""

from dataclasses import replace

import torch
import torch.nn as nn

from src.orchestrators.base_orchestrator import BaseOrchestrator
from src.utils.anchors import (
    AnchorConfig,
    build_anchor_bundles,
    build_semantic_pilot_bundles,
    filter_anchor_bundles,
    shared_anchor_rows,
    supported_anchor_strategy,
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
    anchor_strategy : str
        Strategy for selecting anchors. Supported values:
        'prototype', 'uniform', 'diversity', 'semantic_pilots',
        'clustering', and 'dynamic'.
    num_anchors : int
        Number of anchors to select per epoch.
    parseval_normalization : bool
        Whether to apply Parseval normalization to anchor features.

    Notes
    -----
    - The Stiefel matrices are frozen and updated in closed form each epoch.
    - Anchor selection is performed per epoch using the specified strategy.
    - Parseval normalization applies local whitening to anchor features.
    - Epoch anchor buffers are stored on CPU to avoid GPU OOM.
    """

    def __init__(
        self,
        agents: dict[int, nn.Module],
        neighbors: dict[int, set[int]],
        optimizer,
        lambda_sheaf: float,
        latent_dims: dict,
        anchor_strategy: str,
        num_anchors: int,
        parseval_normalization: bool,
        l2_normalization: bool,
        parseval_eps: float = 1e-4,
        dynamic_persistent_ratio: float = 0.25,
        dynamic_candidate_multiplier: int = 2,
        dynamic_refresh_interval: int = 10,
        **kwargs,
    ):
        super().__init__(
            agents=agents,
            neighbors=neighbors,
            optimizer=optimizer,
        )

        self.save_hyperparameters()
        self.anchor_config = AnchorConfig(
            strategy=str(anchor_strategy),
            num_anchors=int(num_anchors),
            parseval_normalization=bool(parseval_normalization),
            l2_normalization=bool(l2_normalization),
            parseval_eps=float(parseval_eps),
        )

        self.stiefel_matrices = nn.ParameterDict()
        self.epoch_anchors = {}
        # Per-agent semantic anchor identifiers. For class-based strategies
        # these are class labels; for semantic pilots: shared sample ids.
        self.epoch_anchor_ids: dict[int, list[torch.Tensor]] = {}
        self.dynamic_persistent_keys: set[tuple[int, int]] = set()

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
                    # requires_grad=False because we use closed-form SVD
                    # updates, TODO: ablation about initialization later
                    stiefel_matrix = torch.eye(d_i, d_j)

                    self.stiefel_matrices[edge_key] = nn.Parameter(
                        stiefel_matrix, requires_grad=False
                    )

    def _dynamic_candidate_config(self) -> AnchorConfig:
        """Build the larger candidate budget used by dynamic anchors."""
        candidate_budget = max(
            int(self.hparams.num_anchors),
            int(self.hparams.num_anchors)
            * max(1, int(self.hparams.dynamic_candidate_multiplier)),
        )
        return replace(
            self.anchor_config,
            strategy='clustering',
            num_anchors=candidate_budget,
        )

    def _dynamic_key_scores(
        self,
        candidate_tensors: dict[int, torch.Tensor],
        candidate_keys: dict[int, list[tuple[int, int]]],
    ) -> dict[tuple[int, int], float]:
        """Score semantic anchor keys by current cross-client disagreement."""
        scores: dict[tuple[int, int], float] = {}

        for edge_key, V in self.stiefel_matrices.items():
            node_i, node_j = map(int, edge_key.split('_'))
            if node_i not in candidate_tensors or node_j not in candidate_tensors:
                continue

            keys_i = candidate_keys.get(node_i, [])
            keys_j = candidate_keys.get(node_j, [])
            if not keys_i or not keys_j:
                continue

            key_to_idx_i = {key: pos for pos, key in enumerate(keys_i)}
            key_to_idx_j = {key: pos for pos, key in enumerate(keys_j)}
            shared_keys = sorted(set(key_to_idx_i) & set(key_to_idx_j))

            for key in shared_keys:
                row_i = candidate_tensors[node_i][key_to_idx_i[key]]
                row_j = candidate_tensors[node_j][key_to_idx_j[key]]
                residual = row_i - torch.matmul(row_j, V.T)
                score = float(residual.pow(2).sum().detach().cpu().item())
                scores[key] = scores.get(key, 0.0) + score

        return scores

    def _select_dynamic_keys(
        self,
        candidate_keys: dict[int, list[tuple[int, int]]],
        scores: dict[tuple[int, int], float],
        *,
        update_persistent: bool,
        batch_idx: int,
    ) -> set[tuple[int, int]]:
        """Combine a persistent subset with refreshed disagreement anchors."""
        available_keys = set()
        for keys in candidate_keys.values():
            available_keys.update(keys)
        if not available_keys:
            return set()

        ranked_keys = sorted(
            available_keys,
            key=lambda key: (-scores.get(key, 0.0), key),
        )

        total_budget = max(1, int(self.hparams.num_anchors))
        persistent_budget = min(
            total_budget,
            max(
                1,
                int(
                    round(
                        total_budget
                        * float(self.hparams.dynamic_persistent_ratio)
                    )
                ),
            ),
        )
        refresh_budget = max(0, total_budget - persistent_budget)

        persistent_keys = [
            key
            for key in ranked_keys
            if key in self.dynamic_persistent_keys
        ][:persistent_budget]

        refreshed_keys = [
            key for key in ranked_keys if key not in set(persistent_keys)
        ][:refresh_budget]

        selected_keys = list(dict.fromkeys(persistent_keys + refreshed_keys))
        if len(selected_keys) < total_budget:
            for key in ranked_keys:
                if key not in selected_keys:
                    selected_keys.append(key)
                if len(selected_keys) == total_budget:
                    break

        if (
            update_persistent
            and batch_idx % max(1, int(self.hparams.dynamic_refresh_interval))
            == 0
        ):
            self.dynamic_persistent_keys = set(ranked_keys[:persistent_budget])

        return set(selected_keys)

    def _build_dynamic_anchor_bundles(
        self,
        latents: dict[int, torch.Tensor],
        anchor_ids_per_agent: dict[int, torch.Tensor],
        *,
        update_persistent: bool,
        batch_idx: int,
    ) -> tuple[dict[int, torch.Tensor], dict[int, list[tuple[int, int]]]]:
        """Build dynamic anchors from a persistent plus refreshed subset."""
        candidate_tensors, candidate_keys = build_anchor_bundles(
            latents,
            anchor_ids_per_agent,
            self._dynamic_candidate_config(),
        )
        if not candidate_tensors:
            return candidate_tensors, candidate_keys

        scores = self._dynamic_key_scores(candidate_tensors, candidate_keys)
        selected_keys = self._select_dynamic_keys(
            candidate_keys,
            scores,
            update_persistent=update_persistent,
            batch_idx=batch_idx,
        )
        return filter_anchor_bundles(
            candidate_tensors,
            candidate_keys,
            selected_keys,
        )

    def on_train_epoch_start(self) -> None:
        """Initialize/reset per-agent anchor and label buffers."""
        for idx_str in self.agents:
            idx = int(idx_str)
            self.epoch_anchors[idx] = []
            self.epoch_anchor_ids[idx] = []

    @torch.no_grad()
    def on_train_epoch_end(self) -> None:
        """Update Stiefel matrices using semantically shared anchors only."""
        if not self.epoch_anchors or not any(self.epoch_anchors.values()):
            return

        A_dict_raw: dict[int, torch.Tensor] = {}
        anchor_ids_per_agent: dict[int, torch.Tensor] = {}

        for idx_str in self.agents:
            idx = int(idx_str)
            if self.epoch_anchors[idx]:
                A_dict_raw[idx] = torch.cat(self.epoch_anchors[idx], dim=0)
            if self.epoch_anchor_ids[idx]:
                anchor_ids_per_agent[idx] = torch.cat(
                    self.epoch_anchor_ids[idx], dim=0
                )

        if not A_dict_raw:
            return

        strategy = supported_anchor_strategy(self.anchor_config.strategy)

        if strategy == 'semantic_pilots':
            A_dict, anchor_keys = build_semantic_pilot_bundles(
                A_dict_raw,
                anchor_ids_per_agent,
                self.anchor_config,
            )
        elif strategy == 'dynamic':
            A_dict, anchor_keys = self._build_dynamic_anchor_bundles(
                A_dict_raw,
                anchor_ids_per_agent,
                update_persistent=False,
                batch_idx=0,
            )
        else:
            A_dict, anchor_keys = build_anchor_bundles(
                A_dict_raw,
                anchor_ids_per_agent,
                self.anchor_config,
            )
        if not A_dict:
            return

        for idx, anchors in A_dict.items():
            self._record_communication(
                {
                    'anchors': anchors,
                    'anchor_keys': anchor_keys.get(idx, []),
                },
                n_transmissions=len(self.hparams.neighbors.get(idx, set())),
            )

        param_device = next(iter(self.stiefel_matrices.values())).device

        for edge_key, V_param in self.stiefel_matrices.items():
            node_i, node_j = map(int, edge_key.split('_'))

            if node_i not in A_dict or node_j not in A_dict:
                continue

            shared_rows = shared_anchor_rows(
                A_i=A_dict[node_i],
                keys_i=anchor_keys.get(node_i, []),
                A_j=A_dict[node_j],
                keys_j=anchor_keys.get(node_j, []),
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

        # Clear epoch buffers
        for idx in self.epoch_anchors:
            self.epoch_anchors[idx].clear()
        for idx in self.epoch_anchor_ids:
            self.epoch_anchor_ids[idx].clear()

    def _shared_eval(
        self,
        batch: dict[int, list[torch.Tensor]],
        batch_idx: int,
        prefix: str,
    ):
        """Compute losses and metrics for validation/test steps.

        Task loss is evaluated on the full mini-batch; the sheaf regularization
        penalty is evaluated strictly on the Anchor Set A, defined dynamically
        per mini-batch according to self.hparams.anchor_strategy.

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
            (prediction, label) pairs and total_loss includes both task and
            sheaf regularization terms.
        """
        if isinstance(batch, tuple):
            batch = batch[0]

        outputs = {}

        agent_losses = {}
        agent_performances = {}

        def _resolve_key(i: int):
            """Return the key used for agent i in the batch dict.

            CombinedLoader key type depends on the datamodule:
            - ClassificationDataModule  -> int keys  (0, 1, …)
            - SemanticDataModule        -> str keys ('0', '1', …)
            """
            str_key = str(i)
            return str_key if str_key in batch else i

        def _resolve_pilot_key(i: int) -> str:
            return f'pilot_{i}'

        strategy = supported_anchor_strategy(self.anchor_config.strategy)
        raw_latents: dict[int, torch.Tensor] = {}
        labels_per_agent: dict[int, torch.Tensor] = {}

        for idx_str, agent in self.agents.items():
            idx = int(idx_str)
            x, y = batch[_resolve_key(idx)]

            # Single encoder forward pass reused for both the task loss and
            # the sheaf penalty.
            A_batch_raw = agent.encode(x)
            y_hat = agent.decoder(A_batch_raw)

            outputs[idx] = (y_hat.detach(), y)
            raw_latents[idx] = A_batch_raw
            labels_per_agent[idx] = y

            # Task-specific loss and performance on the mini-batch
            task_loss = agent.compute_loss(y_hat, y)
            task_performance = agent.task_performance(y_hat, y)

            agent_losses[idx] = task_loss
            agent_performances[idx] = task_performance

        if strategy == 'semantic_pilots':
            pilot_latents: dict[int, torch.Tensor] = {}
            pilot_ids_per_agent: dict[int, torch.Tensor] = {}

            for idx_str, agent in self.agents.items():
                idx = int(idx_str)
                pilot_key = _resolve_pilot_key(idx)
                if pilot_key not in batch:
                    raise ValueError(
                        'semantic_pilots requires datamodule pilot loaders. '
                        'Configure pilot_split or pilot_num_samples.'
                    )

                x_pilot, _unused_y, sample_ids = batch[pilot_key]
                pilot_latents[idx] = agent.encode(x_pilot)
                pilot_ids_per_agent[idx] = sample_ids

            batch_latents, batch_anchor_keys = build_semantic_pilot_bundles(
                pilot_latents,
                pilot_ids_per_agent,
                self.anchor_config,
            )
            if prefix == 'train':
                for idx, A_pilot in pilot_latents.items():
                    self.epoch_anchors[idx].append(A_pilot.detach().cpu())
                    self.epoch_anchor_ids[idx].append(
                        pilot_ids_per_agent[idx].detach().cpu()
                    )
        elif strategy == 'dynamic':
            batch_latents, batch_anchor_keys = self._build_dynamic_anchor_bundles(
                raw_latents,
                labels_per_agent,
                update_persistent=(prefix == 'train'),
                batch_idx=batch_idx,
            )
            if prefix == 'train':
                for idx, A_batch_raw in raw_latents.items():
                    self.epoch_anchors[idx].append(A_batch_raw.detach().cpu())
                    self.epoch_anchor_ids[idx].append(
                        labels_per_agent[idx].detach().cpu()
                    )
        else:
            batch_latents, batch_anchor_keys = build_anchor_bundles(
                raw_latents,
                labels_per_agent,
                self.anchor_config,
            )
            if prefix == 'train':
                for idx, A_batch_raw in raw_latents.items():
                    self.epoch_anchors[idx].append(A_batch_raw.detach().cpu())
                    self.epoch_anchor_ids[idx].append(
                        labels_per_agent[idx].detach().cpu()
                    )

        # Sheaf regularization penalty (evaluated on anchor set)
        sheaf_penalty = 0.0

        for edge_key, V in self.stiefel_matrices.items():
            node_i, node_j = map(int, edge_key.split('_'))

            if node_i not in batch_latents or node_j not in batch_latents:
                continue

            shared_rows = shared_anchor_rows(
                A_i=batch_latents[node_i],
                keys_i=batch_anchor_keys.get(node_i, []),
                A_j=batch_latents[node_j],
                keys_j=batch_anchor_keys.get(node_j, []),
            )
            if shared_rows is None:
                continue

            A_i_shared, A_j_shared = shared_rows
            diff = A_i_shared - torch.matmul(A_j_shared, V.T)
            frob_dist = (diff**2).sum(dim=1).mean()

            sheaf_penalty += frob_dist

        # Total loss: task loss (full batch) + sheaf penalty (anchor set)
        total_task_loss = torch.stack(list(agent_losses.values())).sum()
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
