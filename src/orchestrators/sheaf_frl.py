"""
Sheaf-based Federated Representation Learning orchestrator.

This module implements the proposed federated learning framework with Sheaf regularization
that maintains aligned latent spaces across agents through Stiefel manifold
optimization of cross-covariance matrices.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.orchestrators.base_orchestrator import BaseOrchestrator


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
        Strategy for selecting anchors: 'prototype', 'balanced', or 'random'.
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

        self.stiefel_matrices = nn.ParameterDict()
        self.epoch_anchors = {}
        # Per-agent label buffers: supports Non-IID distributions where
        # different agents may see completely different label subsets.
        self.epoch_labels: dict[int, list[torch.Tensor]] = {}

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

                    # Initialize as identity (identity mapping)
                    # requires_grad=False because we use closed-form SVD
                    # updates, might do an ablation about initialization later
                    stiefel_matrix = torch.eye(d_i, d_j)

                    self.stiefel_matrices[edge_key] = nn.Parameter(
                        stiefel_matrix, requires_grad=False
                    )

    def _compute_anchors(
        self,
        latents: dict[int, torch.Tensor],
        labels_per_agent: dict[int, torch.Tensor],
    ) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
        """Compute anchor representations and validity masks per agent.

        Compatible with Non-IID settings where agents may observe a different
        subset of classes. Returns both anchor tensors and boolean validity
        masks so that downstream SVD updates can intersect semantics.

        Parameters
        ----------
        latents : dict[int, torch.Tensor]
            Dictionary mapping agent indices to their accumulated latent
            feature tensors (one tensor per agent, shape (N_i, d_i)).
        labels_per_agent : dict[int, torch.Tensor]
            Per-agent label tensors (shape (N_i,)). Labels may differ across
            agents under Non-IID data distributions.

        Returns
        -------
        A_dict : dict[int, torch.Tensor]
            Anchor tensors, one per agent, on CPU.
            - 'prototype': shape (num_global_classes, latent_dim). Rows for
              classes not seen by an agent are zero-filled (never used in SVD
              because the corresponding valid_mask entry is False).
            - 'balanced'/'random': shape (k, latent_dim) agent-local subset.
            - default: original latents unchanged.
        valid_masks : dict[int, torch.Tensor]
            Boolean mask per agent (shape (num_global_classes,) for prototype;
            empty dict for non-prototype strategies where all rows are valid).
        """
        match self.hparams.anchor_strategy:
            # Strategy: Prototype
            case 'prototype':
                all_labels = torch.cat(list(labels_per_agent.values()))
                global_classes = torch.unique(all_labels) 
                num_global = global_classes.shape[0]

                A_dict: dict[int, torch.Tensor] = {}
                valid_masks: dict[int, torch.Tensor] = {}

                for idx, A in latents.items():
                    y_i = labels_per_agent[idx]
                    d = A.shape[1]

                    protos = torch.zeros(num_global, d, dtype=A.dtype)
                    valid_mask = torch.zeros(num_global, dtype=torch.bool)

                    for c_pos, c in enumerate(global_classes):
                        c_mask = y_i == c
                        if c_mask.any():
                            protos[c_pos] = A[c_mask].mean(dim=0)
                            valid_mask[c_pos] = True
                        # Rows for missing classes stay zero

                    A_dict[idx] = protos
                    valid_masks[idx] = valid_mask

                return A_dict, valid_masks

            # Strategy: Balanced
            case 'balanced':
                A_dict = {}
                for idx, A in latents.items():
                    y_i = labels_per_agent[idx]
                    tot = len(y_i)
                    uniques = torch.unique(y_i)
                    k = min(self.hparams.num_anchors, tot)
                    anchors_per_class = max(1, k // len(uniques))

                    selected: list[torch.Tensor] = []
                    for c in uniques:
                        c_idx = torch.where(y_i == c)[0]
                        perm = torch.randperm(len(c_idx))
                        selected.append(c_idx[perm[:anchors_per_class]])

                    sel_idx = torch.cat(selected)
                    remaining = k - len(sel_idx)
                    if remaining > 0:
                        all_idx = torch.arange(tot)
                        mask = torch.ones(tot, dtype=torch.bool)
                        mask[sel_idx] = False
                        avail = all_idx[mask]
                        extra = avail[torch.randperm(len(avail))[:remaining]]
                        sel_idx = torch.cat([sel_idx, extra])

                    final = sel_idx[torch.randperm(len(sel_idx))]
                    A_dict[idx] = A[final]

                return A_dict, {}

            # Strategy: Random
            case 'random':
                A_dict = {}
                for idx, A in latents.items():
                    tot = len(A)
                    k = min(self.hparams.num_anchors, tot)
                    perm = torch.randperm(tot)
                    A_dict[idx] = A[perm[:k]]
                return A_dict, {}

            # Default: return original latents
            case _:
                return latents, {}

    def on_train_epoch_start(self) -> None:
        """Initialize/reset per-agent anchor and label buffers at epoch start."""
        for idx_str in self.agents:
            idx = int(idx_str)
            self.epoch_anchors[idx] = []
            self.epoch_labels[idx] = []

    def parseval_normalize(self, A: torch.Tensor) -> torch.Tensor:
        """Apply Parseval normalization to anchor features.

        Performs local whitening by computing the inverse square root of the
        covariance matrix C = A^T A and applying the transformation.

        Parameters
        ----------
        A : torch.Tensor
            Input anchor feature matrix of shape (n_samples, n_features).

        Returns
        -------
        torch.Tensor
            Normalized feature matrix with orthonormal columns.
        """
        C = torch.matmul(A.T, A)
        eps = float(getattr(self.hparams, 'parseval_eps', 1e-4))
        C = C + eps * torch.eye(C.size(0), device=C.device, dtype=C.dtype)

        # Cast to double for stable eigendecomposition
        original_dtype = C.dtype
        C_double = C.to(torch.float64)

        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(C_double)
        except torch._C._LinAlgError:
            # Fallback if double precision still fails: increase eps significantly
            C_double = C_double + (eps * 10) * torch.eye(
                C.size(0), device=C.device, dtype=torch.float64
            )
            eigenvalues, eigenvectors = torch.linalg.eigh(C_double)

        eigenvalues = eigenvalues.to(original_dtype)
        eigenvectors = eigenvectors.to(original_dtype)

        inv_sqrt_eigenvalues = torch.rsqrt(eigenvalues.clamp(min=eps))

        C_inv = torch.matmul(
            eigenvectors * inv_sqrt_eigenvalues.unsqueeze(0), eigenvectors.T
        )

        return torch.matmul(A, C_inv)
    
    def l2_normalize(self, A: torch.Tensor) -> torch.Tensor:
        """Apply L2 normalization to anchor features.

        Normalizes each row of A to have unit L2 norm.

        Parameters
        ----------
        A : torch.Tensor
            Input anchor feature matrix of shape (n_samples, n_features).

        Returns
        -------
        torch.Tensor
            Row-wise L2-normalized feature matrix.
        """
        return F.normalize(A, p=2, dim=1)

    @torch.no_grad()
    def on_train_epoch_end(self) -> None:
        """Update Stiefel matrices using Intersection SVD.

        Concatenates per-agent epoch buffers, computes anchor
        representations per agent (with per-class validity masks if non-IID),
        and updates each Stiefel matrix via a closed-form Procrustes
        SVD computed only on the semantically shared anchor rows (intersection
        of valid classes for each edge).

        V* = argmin_V ||A_i[shared] - A_j[shared] V^T||_F^2

        The solution is V = U W^T where C = A_i^T A_j = U Σ W^T.
        Skips any edge where the intersection is empty.
        """
        if not self.epoch_anchors or not any(self.epoch_anchors.values()):
            return

        A_dict: dict[int, torch.Tensor] = {}
        labels_per_agent: dict[int, torch.Tensor] = {}

        for idx_str in self.agents:
            idx = int(idx_str)
            if self.epoch_anchors[idx]:
                A_dict[idx] = torch.cat(self.epoch_anchors[idx], dim=0)
            if self.epoch_labels[idx]:
                labels_per_agent[idx] = torch.cat(
                    self.epoch_labels[idx], dim=0
                )

        if not A_dict:
            return

        # Compute anchors + validity masks
        A_dict, valid_masks = self._compute_anchors(A_dict, labels_per_agent)

        param_device = next(iter(self.stiefel_matrices.values())).device

        # Per-edge Intersection SVD
        for edge_key, V_param in self.stiefel_matrices.items():
            node_i, node_j = map(int, edge_key.split('_'))

            if node_i not in A_dict or node_j not in A_dict:
                continue

            if valid_masks:
                # Prototype: only use rows present in BOTH agents
                mask_i = valid_masks.get(node_i)
                mask_j = valid_masks.get(node_j)
                if mask_i is None or mask_j is None:
                    continue
                shared = mask_i & mask_j
                if not shared.any():
                    continue
                A_i = A_dict[node_i][shared]
                A_j = A_dict[node_j][shared]
            else:
                # Balanced / random / dynamic: all rows are valid
                A_i = A_dict[node_i]
                A_j = A_dict[node_j]

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
        for idx in self.epoch_labels:
            self.epoch_labels[idx].clear()

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
        outputs = {}

        # Track task metrics across agents
        total_task_loss = 0.0
        total_task_performance = 0.0

        losses = []
        performances = []

        # Anchor tensors for the sheaf penalty (one entry per agent).
        batch_latents: dict[int, torch.Tensor] = {}

        valid_classes_masks: dict[int, torch.Tensor] = {}

        def _resolve_key(i: int):
            """Return the key used for agent i in the batch dict.

            CombinedLoader key type depends on the datamodule:
            - ClassificationDataModule  -> int keys  (0, 1, …)
            - SemanticDataModule        -> str keys ('0', '1', …)
            """
            str_key = str(i)
            return str_key if str_key in batch else i

        # Build global sorted class list for the 'prototype' strategy once,
        # using the union of labels seen across all agents in this mini-batch.
        global_classes: torch.Tensor | None = None
        if self.hparams.anchor_strategy == 'prototype':
            all_y = torch.cat(
                [batch[_resolve_key(int(k))][1] for k in self.agents]
            )
            global_classes = torch.unique(all_y)  

        # Per-agent latent extraction, anchor routing, and Parseval normalization
        for idx_str, agent in self.agents.items():
            idx = int(idx_str)
            x, y = batch[_resolve_key(idx)]

            # Single encoder forward pass: reused for both task loss and the sheaf penalty
            A_batch_raw = agent.encode(x)
            y_hat = agent.decoder(A_batch_raw)

            outputs[idx] = (y_hat.detach(), y)

            # Anchor routing
            match self.hparams.anchor_strategy:
                case 'prototype':
                    # Compute per class prototype anchors 
                    assert global_classes is not None
                    num_global = global_classes.shape[0]
                    d = A_batch_raw.shape[1]
                    device = A_batch_raw.device

                    # Map each sample label to its position in global_classes.
                    row_idx = torch.searchsorted(
                        global_classes, y
                    )  #values in [0, num_global)

                    # Accumulate feature sums and per-class counts
                    proto_sum = torch.zeros(
                        num_global, d, device=device, dtype=A_batch_raw.dtype
                    )
                    proto_sum.index_add_(0, row_idx, A_batch_raw)

                    counts = torch.zeros(
                        num_global, device=device, dtype=A_batch_raw.dtype
                    )
                    counts.index_add_(
                        0, row_idx, torch.ones(len(y), device=device, dtype=A_batch_raw.dtype)
                    )

                    # valid_mask: classes with at least one sample in this batch
                    valid_mask = counts > 0
                    # Divide only where valid to avoid /0; invalid rows stay 0
                    protos = proto_sum / counts.clamp(min=1).unsqueeze(1)
                    protos[~valid_mask] = 0.0

                    A_i_raw = protos
                    valid_classes_masks[idx] = valid_mask

                case _:
                    # 'dynamic' or any unrecognised key: stochastic approx
                    # using the full mini-batch latents
                    A_i_raw = A_batch_raw

            # Parseval normalization
            if self.hparams.parseval_normalization:
                A_i_tilde = self.parseval_normalize(A_i_raw)
            elif self.hparams.l2_normalization:
                A_i_tilde = self.l2_normalize(A_i_raw)
            else:
                A_i_tilde = A_i_raw

            batch_latents[idx] = A_i_tilde

            # Collect raw batch latents during training for end-of-epoch Stiefel matrix updates
            if prefix == 'train':
                self.epoch_anchors[idx].append(A_batch_raw.detach().cpu())
                # Every agent tracks its own labels to support Non-IID splits
                self.epoch_labels[idx].append(y.detach().cpu())

            # Task-specific loss and performance on the mini-batch
            task_loss = agent.compute_loss(y_hat, y)
            task_performance = agent.task_performance(y_hat, y)

            self.log_dict(
                {
                    f'{prefix}/task_loss_agent_{idx}': task_loss,
                    f'{prefix}/task_performance_agent_{idx}': task_performance,
                },
                on_step=False,
                on_epoch=True,
            )

            total_task_loss += task_loss
            total_task_performance += task_performance
            losses.append(task_loss)
            performances.append(task_performance)

        # Sheaf regularization penalty (evaluated on anchor set)
        sheaf_penalty = 0.0

        for edge_key, V in self.stiefel_matrices.items():
            node_i, node_j = map(int, edge_key.split('_'))

            if node_i not in batch_latents or node_j not in batch_latents:
                continue

            A_i = batch_latents[node_i]
            A_j = batch_latents[node_j]

            match self.hparams.anchor_strategy:
                case 'prototype':
                    mask_i = valid_classes_masks.get(node_i)
                    mask_j = valid_classes_masks.get(node_j)

                    if mask_i is None or mask_j is None:
                        continue

                    # Only compute penalty on classes present in BOTH agents
                    shared_mask = mask_i & mask_j
                    if not shared_mask.any():
                        continue

                    A_i_shared = A_i[shared_mask]
                    A_j_shared = A_j[shared_mask]
                    diff = A_i_shared - torch.matmul(A_j_shared, V.T)
                    # Mean over shared class rows
                    frob_dist = (diff**2).sum(dim=1).mean()

                case _:
                    # 'dynamic' / default: penalty over the full batch
                    diff = A_i - torch.matmul(A_j, V.T)
                    frob_dist = (diff**2).sum(dim=1).mean()

            sheaf_penalty += frob_dist

        # Total loss: task loss (full batch) + sheaf penalty (anchor set)
        total_loss = (
            total_task_loss + self.hparams.lambda_sheaf * sheaf_penalty
        )
        avg_performance = total_task_performance / len(self.agents)

        losses_tensor = torch.stack(losses) if losses else torch.tensor([0.0], device=self.device)
        perfs_tensor = torch.stack(performances) if performances else torch.tensor([0.0], device=self.device)

        self.log_dict(
            {
                f'{prefix}/sheaf_penalty': sheaf_penalty,
                f'{prefix}/total_loss_epoch': total_loss,
                f'{prefix}/avg_task_performance_epoch': avg_performance,
                f'{prefix}/loss_std': losses_tensor.std(unbiased=False) if len(losses) > 1 else 0.0,
                f'{prefix}/task_performance_std': perfs_tensor.std(unbiased=False) if len(performances) > 1 else 0.0,
            },
            on_step=False,
            on_epoch=True,
        )

        return outputs, total_loss


if __name__ == '__main__':
    pass
