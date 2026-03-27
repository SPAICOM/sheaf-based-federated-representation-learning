"""
Sheaf-based Federated Representation Learning orchestrator.

This module implements a federated learning framework with Sheaf regularization
that maintains aligned latent spaces across agents through Stiefel manifold
optimization of cross-covariance matrices.

!!to be checked with the new specifications in the paper
"""

import torch
import torch.nn as nn

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
        self.epoch_labels = []

        latent_dims_int = {int(k): int(v) for k, v in latent_dims.items()}

        # Create Stiefel matrices for each edge in the neighbor graph
        # Stiefel matrices V_ij map latent space of agent j to agent i
        for i_raw, neighborset in neighbors.items():
            for j_raw in neighborset:
                i = int(i_raw)
                j = int(j_raw)

                # Sort nodes by dimension to ensure consistent edge keys
                # Higher-dimensional node comes first for consistent key
                # generation
                if latent_dims_int[i] > latent_dims_int[j]:
                    node_i, node_j = i, j
                elif latent_dims_int[i] < latent_dims_int[j]:
                    node_i, node_j = j, i
                else:
                    # Equal dimensions: use max/min for deterministic ordering
                    node_i, node_j = max(i, j), min(i, j)

                edge_key = f'{node_i}_{node_j}'

                # Only create matrix once per edge (avoid duplicates from
                # both directions)
                if edge_key not in self.stiefel_matrices:
                    d_i = latent_dims_int[node_i]
                    d_j = latent_dims_int[node_j]

                    # Initialize as identity (identity mapping)
                    # requires_grad=False because we use closed-form SVD
                    # updates
                    stiefel_matrix = torch.eye(d_i, d_j)

                    self.stiefel_matrices[edge_key] = nn.Parameter(
                        stiefel_matrix, requires_grad=False
                    )

    def _compute_anchors(
        self, latents: dict[int, torch.Tensor], labels: torch.Tensor
    ) -> dict[int, torch.Tensor]:
        """Compute or select anchor representations based on strategy.

        Anchors are used for computing cross-covariance matrices between
        neighboring agents. The strategy determines how anchors are selected.

        Parameters
        ----------
        latents : dict[int, torch.Tensor]
            Dictionary mapping agent indices to their latent feature tensors.
        labels : torch.Tensor
            Ground truth labels for all samples.

        Returns
        -------
        dict[int, torch.Tensor]
            Dictionary mapping agent indices to their selected anchor tensors.

        Notes
        -----
        Supported strategies:
        - 'prototype': Compute class-wise mean anchors.
        - 'balanced': Sample equal number of anchors per class.
        - 'random': Randomly sample anchors.
        - other: Return original latents unchanged.
        """
        # Get total samples and unique class labels
        tot = len(labels)
        uniques = torch.unique(labels)

        match self.hparams.anchor_strategy:
            # Strategy 1: Prototype - use class centroids as anchors
            # Each class contributes one anchor (mean of samples in class)
            case 'prototype':
                A_proto = {int(idx): [] for idx in self.agents}

                # Compute mean anchor for each class
                for c in uniques:
                    c_mask = labels == c
                    for idx in self.agents:
                        idx = int(idx)
                        # Compute mean of latent features for this class
                        mean_anchor = latents[idx][c_mask].mean(dim=0)
                        A_proto[idx].append(mean_anchor)

                # Stack anchors for each agent: (num_classes, latent_dim)
                return {
                    idx: torch.stack(protos, dim=0)
                    for idx, protos in A_proto.items()
                }

            # Strategy 2: Balanced - sample equal number from each class
            # Ensures all classes are represented in anchor set
            case 'balanced':
                # Number of anchors to select per class
                k = min(self.hparams.num_anchors, tot)
                anchors_per_class = k // len(uniques)

                selected_indices = []
                # Sample from each class
                for c in uniques:
                    c_idx = torch.where(labels == c)[0]
                    perm = torch.randperm(len(c_idx), device=labels.device)
                    chosen = c_idx[perm[:anchors_per_class]]
                    selected_indices.append(chosen)

                selected_indices = torch.cat(selected_indices)

                # Handle remainder: distribute remaining slots randomly
                remaining = k - len(selected_indices)
                if remaining > 0:
                    all_idx = torch.arange(tot, device=labels.device)
                    # Create mask to exclude already selected indices
                    mask = torch.ones(
                        tot, dtype=torch.bool, device=labels.device
                    )
                    mask[selected_indices] = False
                    avail_idx = all_idx[mask]

                    extra_perm = torch.randperm(
                        len(avail_idx), device=labels.device
                    )
                    extra = avail_idx[extra_perm[:remaining]]
                    selected_indices = torch.cat([selected_indices, extra])

                # Final shuffle to randomize order
                final_perm = torch.randperm(
                    len(selected_indices), device=labels.device
                )
                anchor_indices = selected_indices[final_perm]

                return {idx: A[anchor_indices] for idx, A in latents.items()}

            # Strategy 3: Random - simply sample k random samples
            case 'random':
                k = min(self.hparams.num_anchors, tot)
                perm = torch.randperm(tot, device=labels.device)
                anchor_indices = perm[:k]
                return {idx: A[anchor_indices] for idx, A in latents.items()}

            # Default: return original latents unchanged
            case _:
                return latents

    def on_train_epoch_start(self) -> None:
        """Initialize/reset anchor and label lists at epoch start."""
        for idx_str in self.agents:
            self.epoch_anchors[int(idx_str)] = []
        self.epoch_labels = []

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
        C = C + eps * torch.eye(C.size(0), device=C.device)
        
        # Cast to double for more stable eigendecomposition
        original_dtype = C.dtype
        C_double = C.to(torch.float64)
        
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(C_double)
        except torch._C._LinAlgError:
            # Fallback if double precision still fails: increase eps significantly
            C_double = C_double + (eps * 10) * torch.eye(C.size(0), device=C.device, dtype=torch.float64)
            eigenvalues, eigenvectors = torch.linalg.eigh(C_double)
            
        eigenvalues = eigenvalues.to(original_dtype)
        eigenvectors = eigenvectors.to(original_dtype)

        inv_sqrt_eigenvalues = torch.rsqrt(eigenvalues.clamp(min=eps))

        C_inv = torch.matmul(
            eigenvectors, inv_sqrt_eigenvalues.unsqueeze(1) * eigenvectors.T
        )

        A_normalized = torch.matmul(A, C_inv)

        return A_normalized

    @torch.no_grad()
    def on_train_epoch_end(self) -> None:
        """Update Stiefel matrices using cross-covariance SVD.

        Computes unbiased anchor representations, calculates cross-covariance
        matrices between neighboring agents, and updates Stiefel matrices via
        closed-form SVD-based optimization.

        The Stiefel manifold optimization finds the optimal orthogonal
        projection that aligns latent spaces between neighboring agents.
        This is solved via:

        V* = argmin_V ||A_i - A_j V^T||_F^2

        The solution is given by V = UV^T where C = A_i^T A_j = UΣV^T is
        the SVD. This enforces V^T V = I (orthonormality constraint).
        """
        if not self.epoch_anchors or not any(self.epoch_anchors.values()):
            return

        A_dict = {}
        for idx_str in self.agents:
            idx = int(idx_str)
            if self.epoch_anchors[idx]:
                A_dict[idx] = torch.cat(self.epoch_anchors[idx], dim=0)

        labels = torch.cat(self.epoch_labels, dim=0)
        A_dict = self._compute_anchors(A_dict, labels)

        for edge_key, V_param in self.stiefel_matrices.items():
            node_i, node_j = map(int, edge_key.split('_'))

            if node_i not in A_dict or node_j not in A_dict:
                continue

            # Center the data (subtract mean) for unbiased covariance
            # estimation
            A_i_unbiased = A_dict[node_i] - A_dict[node_i].mean(
                dim=0, keepdim=True
            )
            A_j_unbiased = A_dict[node_j] - A_dict[node_j].mean(
                dim=0, keepdim=True
            )

            # Compute cross-covariance matrix between agents i and j
            # C = A_i^T A_j (no division by n since we're using SVD anyway)
            C = torch.matmul(A_i_unbiased.T, A_j_unbiased)

            # SVD gives C = UΣV^T; optimal Stiefel matrix is V = U @ V^T
            # This solves the Procrustes problem for orthogonal matrices
            U, S, W_T = torch.linalg.svd(C, full_matrices=False)
            V_new = torch.matmul(U, W_T)

            V_param.copy_(V_new.to(V_param.device))

        for idx in self.epoch_anchors:
            self.epoch_anchors[idx].clear()

        self.epoch_labels.clear()

    def _shared_eval(
        self,
        batch: dict[int, list[torch.Tensor]],
        batch_idx: int,
        prefix: str,
    ):
        """Compute losses and metrics for validation/test steps.

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

        # Store latent representations for sheaf regularization
        batch_latents = {}

        # Process each agent's batch
        for idx_str, agent in self.agents.items():
            idx = int(idx_str)
            x_key = str(idx) if str(idx) in batch else idx
            x, y = batch[x_key]

            # Forward pass through agent
            y_hat = agent.forward(x)
            # Get latent features for sheaf regularization
            A_i_raw = agent.encode(x)
            outputs[idx] = (y_hat, y)

            # Apply Parseval normalization if enabled
            # This whitens the anchor features for better covariance estimation
            if self.hparams.parseval_normalization:
                A_i_tilde = self.parseval_normalize(A_i_raw)
            else:
                A_i_tilde = A_i_raw

            batch_latents[idx] = A_i_tilde

            # Collect anchors during training for Stiefel matrix updates
            # Only agent 0 stores labels (they're the same for all agents)
            if prefix == 'train':
                self.epoch_anchors[idx].append(A_i_tilde.detach())
                if idx == 0:
                    self.epoch_labels.append(y.detach())

            # Compute task-specific loss and performance
            task_loss = agent.compute_loss(y_hat, y)
            task_performance = agent.task_performance(y_hat, y)

            # Log per-agent metrics
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

        # Compute sheaf regularization penalty
        # This penalizes discrepancy between neighboring latent spaces
        # The penalty encourages: A_i ≈ A_j @ V_ij^T
        # Where V_ij is the Stiefel matrix for edge (i,j)
        sheaf_penalty = 0.0

        for edge_key, V in self.stiefel_matrices.items():
            node_i, node_j = map(int, edge_key.split('_'))
            node_i, node_j = int(node_i), int(node_j)

            # Skip if either node has no data in this batch
            if node_i not in batch_latents or node_j not in batch_latents:
                continue

            # Project node_j features through Stiefel matrix to node_i's space
            # A_j @ V^T maps from j's latent space to i's latent space
            diff = batch_latents[node_i] - torch.matmul(
                batch_latents[node_j], V.T
            )
            # Frobenius norm squared measures reconstruction error
            frob_dist = (diff**2).sum(dim=1).mean()
            sheaf_penalty += frob_dist

        # Total loss = task loss + sheaf regularization
        total_loss = (
            total_task_loss + self.hparams.lambda_sheaf * sheaf_penalty
        )
        avg_performance = total_task_performance / len(self.agents)

        # Log aggregate metrics
        self.log_dict(
            {
                f'{prefix}/sheaf_penalty': sheaf_penalty,
                f'{prefix}/total_loss_epoch': total_loss,
                f'{prefix}/avg_task_performance_epoch': avg_performance,
            },
            on_step=False,
            on_epoch=True,
        )

        return outputs, total_loss


if __name__ == '__main__':
    pass
