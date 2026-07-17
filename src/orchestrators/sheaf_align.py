"""
SheafAlign: Learnable projection-based federated representation alignment.

This module implements the SheafAlign framework where learnable projection
matrices align class-conditional mean latent representations between
neighboring agents. Unlike SheafFRL, which updates Stiefel matrices in
closed form at epoch end, SheafAlign learns its projection matrices
end-to-end via backpropagation, jointly with all other agent parameters.
"""

import torch
import torch.nn as nn

from src.orchestrators.base_orchestrator import BaseOrchestrator


class SheafAlign(BaseOrchestrator):
    """SheafAlign: Federated learning with learnable projection matrices.

    Implements the SheafAlign framework where per-edge projection matrices
    P_{ij} align class-conditional mean latent representations between
    neighboring agents. The projection matrices are standard nn.Parameters
    optimized jointly with the encoders and decoders via the same optimizer.

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
    max_lmb : float
        Maximum weight coefficient for the alignment penalty in the total loss.
    lambda_schedule : str or None
        Scheduling strategy: ``None`` (constant), ``'cosine'``, or ``'exp'``.
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
        max_lmb: float,
        latent_dims: dict,
        lambda_schedule: str | None = None,
        **kwargs,
    ):
        super().__init__(
            agents=agents,
            neighbors=neighbors,
            optimizer=optimizer,
            **kwargs,
        )
        self.save_hyperparameters()

        self.projection_matrices = nn.ParameterDict()
        latent_dims_int = {int(k): int(v) for k, v in latent_dims.items()}

        for i_raw, neighborset in neighbors.items():
            for j_raw in neighborset:
                i = int(i_raw)
                j = int(j_raw)

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

                    proj_matrix = torch.eye(d_i, d_j)
                    self.projection_matrices[edge_key] = nn.Parameter(
                        proj_matrix, requires_grad=True
                    )

    def on_train_epoch_end(self) -> None:
        self._finalize_train_epoch_communication()

    def _compute_class_means(
        self,
        latents: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict[int, torch.Tensor]:
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
        if isinstance(batch, tuple):
            batch = batch[0]

        outputs = {}
        agent_losses = {}
        agent_performances = {}
        raw_latents: dict[int, torch.Tensor] = {}
        labels_per_agent: dict[int, torch.Tensor] = {}

        def _resolve_key(i: int):
            str_key = str(i)
            return str_key if str_key in batch else i

        for idx_str, agent in self.agents.items():
            idx = int(idx_str)
            x, y = batch[_resolve_key(idx)]

            z = agent.encode(x)
            y_hat = agent.decoder(z)

            outputs[idx] = (y_hat.detach(), y)
            raw_latents[idx] = z
            labels_per_agent[idx] = y

            task_loss = agent.compute_loss(y_hat, y)
            task_performance = agent.task_performance(y_hat, y)

            agent_losses[idx] = task_loss
            agent_performances[idx] = task_performance

        class_means_per_agent: dict[int, dict[int, torch.Tensor]] = {}
        for idx, z in raw_latents.items():
            class_means_per_agent[idx] = self._compute_class_means(
                z, labels_per_agent[idx]
            )

        if prefix == 'train':
            self._record_communication_round(n_rounds=1, prefix=prefix)
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

            mu_i = torch.stack([means_i[c] for c in shared_classes])
            mu_j = torch.stack([means_j[c] for c in shared_classes])

            diff = mu_i - torch.matmul(mu_j, P.T)
            alignment_penalty = alignment_penalty + (diff**2).sum(dim=1).mean()

        total_task_loss = torch.stack(list(agent_losses.values())).sum()
        total_loss = (
            total_task_loss + self._effective_lambda_reg() * alignment_penalty
        )

        self._log_shared_metrics(
            prefix=prefix,
            agent_losses=agent_losses,
            agent_performances=agent_performances,
            batch_size=self._resolve_batch_size(batch),
            agent_sample_counts=self._resolve_agent_sample_counts(batch),
            total_loss=total_loss,
            extra_metrics={
                f'{prefix}/alignment_penalty': alignment_penalty,
                # Standardised alias tracked across all orchestrators (see
                # BaseOrchestrator.evaluate_misalignment_loss). SheafAlign has
                # no send_message/post-training alignment, so this is simply
                # its own online-learned projection-based penalty.
                f'{prefix}/misalignment_loss': alignment_penalty,
            },
            prog_bar=False,
            per_agent_loss_name='task_loss',
        )

        return outputs, total_loss


    def send_message(
        self,
        sender_idx: int,
        receiver_idx: int,
        Z_sender: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError(
            f'{self.__class__.__name__} does not implement send_message.'
        )


if __name__ == '__main__':
    pass
