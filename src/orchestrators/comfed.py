"""
ComFed: Communication-efficient Federated Representation Learning orchestrator.

This module implements the ComFed framework where learnable projection matrices
align class-conditional mean latent representations between neighboring agents.
Unlike SheafFRL, which updates Stiefel matrices in closed form at epoch end,
ComFed learns its projection matrices end-to-end via backpropagation, jointly
with all other agent parameters.

Reference: https://arxiv.org/pdf/2603.19067
"""

import torch
import torch.nn as nn

from src.orchestrators.base_orchestrator import BaseOrchestrator


class ComFed(BaseOrchestrator):
    """ComFed: Federated learning with learnable projection matrices.

    Implements the ComFed framework where per-edge projection matrices P_{ij}
    align class-conditional mean latent representations between neighboring
    agents. The projection matrices are standard nn.Parameters optimized
    jointly with the encoders and decoders via the same optimizer.

    For each edge (i, j) and each class c observed by both agents:

        L_align += || mu_i^c - mu_j^c @ P_{ij}.T ||_F^2

    where mu_i^c is the mean latent representation of class c at agent i.

    Parameters
    ----------
    agents : dict[int, nn.Module]
        Dictionary mapping agent indices to their model instances.
    neighbors : dict[int, set[int]]
        Dictionary mapping each agent index to the set of its neighbor indices.
    optimizer : hydra config
        Optimizer configuration for training.
    lambda_comfed : float
        Weight coefficient for the alignment penalty in the total loss.
    latent_dims : dict
        Dictionary mapping agent indices to their latent space dimensions.

    Notes
    -----
    - Projection matrices are initialized as (rectangular) identity matrices
      and updated via autodiff — no closed-form epoch-end step is needed.
    - Edge ordering follows the same convention as SheafFRL: the agent with
      the higher latent dimension is designated node_i; ties are broken by
      taking the larger index as node_i.
    - Communication cost is recorded per training batch (class-conditional
      mean vectors sent to each neighbor).
    """

    def __init__(
        self,
        agents: dict[int, nn.Module],
        neighbors: dict[int, set[int]],
        optimizer,
        lambda_comfed: float,
        latent_dims: dict,
        **kwargs,
    ):
        super().__init__(
            agents=agents,
            neighbors=neighbors,
            optimizer=optimizer,
        )
        self.save_hyperparameters()

        self.projection_matrices = nn.ParameterDict()
        latent_dims_int = {int(k): int(v) for k, v in latent_dims.items()}

        for i_raw, neighborset in neighbors.items():
            for j_raw in neighborset:
                i = int(i_raw)
                j = int(j_raw)

                # Consistent edge ordering: higher-dimensional node is node_i.
                # Ties broken by max index to match SheafFRL convention.
                if latent_dims_int[i] > latent_dims_int[j]:
                    node_i, node_j = i, j
                elif latent_dims_int[i] < latent_dims_int[j]:
                    node_i, node_j = j, i
                else:
                    node_i, node_j = max(i, j), min(i, j)

                edge_key = f'{node_i}_{node_j}'

                if edge_key not in self.projection_matrices:
                    d_i = latent_dims_int[node_i]
                    d_j = latent_dims_int[node_j]

                    # Initialize as rectangular identity; requires_grad=True so
                    # the matrix is updated jointly via autodiff.
                    proj_matrix = torch.eye(d_i, d_j)
                    self.projection_matrices[edge_key] = nn.Parameter(
                        proj_matrix, requires_grad=True
                    )

    def on_train_epoch_end(self) -> None:
        """No-op: projection matrices are updated via autodiff each step."""
        pass

    def _compute_class_means(
        self,
        latents: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict[int, torch.Tensor]:
        """Compute class-conditional mean latent representations.

        Parameters
        ----------
        latents : torch.Tensor
            Latent representations of shape (N, D).
        labels : torch.Tensor
            Class labels of shape (N,).

        Returns
        -------
        dict[int, torch.Tensor]
            Mapping from class index to mean latent vector of shape (D,).
            Gradients are preserved for backpropagation through the alignment
            loss.
        """
        class_means: dict[int, torch.Tensor] = {}
        for c in torch.unique(labels):
            mask = labels == c
            class_means[int(c.item())] = latents[mask].mean(dim=0)
        return class_means

    def _shared_eval(
        self,
        batch: dict[int, list[torch.Tensor]],
        batch_idx: int,
        prefix: str,
    ):
        """Compute losses and metrics for train/validation/test steps.

        Task loss is evaluated on the full mini-batch. The ComFed alignment
        penalty is computed from class-conditional mean latent representations
        and backpropagated through both the encoders and the projection
        matrices.

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
            alignment terms.
        """
        if isinstance(batch, tuple):
            batch = batch[0]

        outputs = {}
        agent_losses = {}
        agent_performances = {}
        raw_latents: dict[int, torch.Tensor] = {}
        labels_per_agent: dict[int, torch.Tensor] = {}

        def _resolve_key(i: int):
            """Return the key used for agent i in the batch dict."""
            str_key = str(i)
            return str_key if str_key in batch else i

        for idx_str, agent in self.agents.items():
            idx = int(idx_str)
            x, y = batch[_resolve_key(idx)]

            # Single encoder forward pass reused for task loss and alignment.
            z = agent.encode(x)
            y_hat = agent.decoder(z)

            outputs[idx] = (y_hat.detach(), y)
            raw_latents[idx] = z
            labels_per_agent[idx] = y

            task_loss = agent.compute_loss(y_hat, y)
            task_performance = agent.task_performance(y_hat, y)

            agent_losses[idx] = task_loss
            agent_performances[idx] = task_performance

        # Compute class-conditional means once per agent (reused below).
        class_means_per_agent: dict[int, dict[int, torch.Tensor]] = {}
        for idx, z in raw_latents.items():
            class_means_per_agent[idx] = self._compute_class_means(
                z, labels_per_agent[idx]
            )

        # Record communication: each agent broadcasts its class means to
        # all neighbors once per batch.
        if prefix == 'train':
            for idx_str in self.agents:
                idx = int(idx_str)
                means = class_means_per_agent.get(idx, {})
                if not means:
                    continue
                means_tensor = torch.stack(list(means.values())).detach()
                n_neighbors = len(
                    self.hparams.neighbors.get(
                        idx, self.hparams.neighbors.get(str(idx), set())
                    )
                )
                self._record_communication(
                    means_tensor,
                    n_transmissions=n_neighbors,
                )

        # ComFed alignment penalty over all edges.
        alignment_penalty: torch.Tensor | float = 0.0

        for edge_key, P in self.projection_matrices.items():
            node_i, node_j = map(int, edge_key.split('_'))

            if (
                node_i not in class_means_per_agent
                or node_j not in class_means_per_agent
            ):
                continue

            means_i = class_means_per_agent[node_i]
            means_j = class_means_per_agent[node_j]

            shared_classes = sorted(set(means_i) & set(means_j))
            if not shared_classes:
                continue

            # Shape: (C, d_i) and (C, d_j) respectively.
            mu_i = torch.stack([means_i[c] for c in shared_classes])
            mu_j = torch.stack([means_j[c] for c in shared_classes])

            # Project j's means into i's latent space via P (d_i x d_j).
            # mu_j @ P.T has shape (C, d_i), same as mu_i.
            diff = mu_i - torch.matmul(mu_j, P.T)
            alignment_penalty = alignment_penalty + (diff**2).sum(dim=1).mean()

        total_task_loss = torch.stack(list(agent_losses.values())).sum()
        total_loss = (
            total_task_loss + self.hparams.lambda_comfed * alignment_penalty
        )

        self._log_shared_metrics(
            prefix=prefix,
            agent_losses=agent_losses,
            agent_performances=agent_performances,
            batch_size=self._resolve_batch_size(batch),
            agent_sample_counts=self._resolve_agent_sample_counts(batch),
            total_loss=total_loss,
            extra_metrics={f'{prefix}/alignment_penalty': alignment_penalty},
            prog_bar=False,
            per_agent_loss_name='task_loss',
        )

        return outputs, total_loss


if __name__ == '__main__':
    pass
