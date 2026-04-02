"""
Sheaf-based Federated Representation Learning orchestrator.

This module implements the proposed federated learning framework with Sheaf regularization
that maintains aligned latent spaces across agents through Stiefel manifold
optimization of cross-covariance matrices. Anchor strategies are implemented
with explicit semantic correspondence keys so neighboring agents are aligned
only on class-consistent anchors.
"""

import torch
import torch.nn as nn

from src.orchestrators.base_orchestrator import BaseOrchestrator
from src.utils.anchors import (
    AnchorConfig,
    build_anchor_bundles,
    build_semantic_pilot_bundles,
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
        'prototype', 'random', 'balanced', 'semantic_pilots',
        'clustered_pilots', and 'dynamic'.
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
        # these are class labels; for semantic pilots they are shared sample ids.
        self.epoch_anchor_ids: dict[int, list[torch.Tensor]] = {}

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

    def on_train_epoch_start(self) -> None:
        """Initialize/reset per-agent anchor and label buffers at epoch start."""
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

        if (
            supported_anchor_strategy(self.anchor_config.strategy)
            == 'semantic_pilots'
        ):
            A_dict, anchor_keys = build_semantic_pilot_bundles(
                A_dict_raw,
                anchor_ids_per_agent,
                self.anchor_config,
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
            A_i = A_i - A_i.mean(dim=0, keepdim=True)
            A_j = A_j - A_j.mean(dim=0, keepdim=True)

            # Cross-covariance thin-SVD
            C = torch.matmul(A_i.T, A_j)
            U, _S, W_T = torch.linalg.svd(C, full_matrices=False)
            V_new = torch.matmul(U, W_T).to(dtype=V_param.dtype, device=param_device)
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
        total_loss = total_task_loss + self.hparams.lambda_sheaf * sheaf_penalty

        self._log_shared_metrics(
            prefix=prefix,
            agent_losses=agent_losses,
            agent_performances=agent_performances,
            total_loss=total_loss,
            extra_metrics={f'{prefix}/sheaf_penalty': sheaf_penalty},
            prog_bar=False,
            per_agent_loss_name='task_loss',
        )

        return outputs, total_loss


if __name__ == '__main__':
    pass
