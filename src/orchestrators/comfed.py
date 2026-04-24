"""
CoMFed: Communication-Efficient Multi-Modal Federated Learning via Latent-Space Consensus.

Reference: Badi et al., "Communication-Efficient and Robust Multi-Modal Federated
Learning via Latent-Space Consensus", IEEE Wireless Communications Letters, 2026.
"""

import torch
import torch.nn as nn
from hydra.utils import instantiate

from src.orchestrators.base_orchestrator import BaseOrchestrator


class ComFed(BaseOrchestrator):
    """CoMFed: per-agent projection matrices align class prototypes in a shared latent space.

    Each agent i owns Pi ∈ R^{proj_dim × d_i} that maps local features to a common
    d-dimensional space. Training alternates between two steps per batch:

      1. wi-step  — minimize  L_task(wi) + λ · L_align(wi; Pi fixed)
      2. Pi-step  — minimize  L_align(Pi; wi fixed)  ×  proj_steps times

    Alignment loss for agent i (squared Frobenius / L2 consensus from the paper):

        L_align_i = Σ_m  ‖Pi v̄_{i,m}  −  mean_{j∈N_i}(Pj v̄_{j,m})‖²_F

    where v̄_{i,m} is the class-m conditional mean of agent i's encoder output.
    Neighbor projected means are treated as fixed targets (detached) during both steps.

    Communication payload per round: one proj_dim-dimensional vector per class per agent.

    Parameters
    ----------
    agents : dict[int, nn.Module]
    neighbors : dict[int, set[int]]
    optimizer : hydra config
        Shared optimizer config; instantiated separately for wi and Pi.
    max_lmb : float
        Regularization weight λ.
    latent_dims : dict
        Mapping from agent index to local encoder output dimension d_i.
    proj_dim : int
        Shared latent space dimension d.
    proj_steps : int
        Number of Pi gradient steps per training batch (paper uses 10).
    lambda_schedule : str or None
        Scheduling strategy: None (constant), 'cosine', or 'exp'.
    """

    def __init__(
        self,
        agents: dict[int, nn.Module],
        neighbors: dict[int, set[int]],
        optimizer,
        max_lmb: float,
        latent_dims: dict,
        proj_dim: int,
        proj_steps: int = 10,
        lambda_schedule: str | None = None,
        **kwargs,
    ):
        super().__init__(agents=agents, neighbors=neighbors, optimizer=optimizer)
        self.save_hyperparameters()
        self.automatic_optimization = False

        latent_dims_int = {int(k): int(v) for k, v in latent_dims.items()}

        # Per-agent projection matrices Pi: R^{proj_dim × d_i}
        self.projection_matrices = nn.ParameterDict()
        for i, d_i in latent_dims_int.items():
            Pi = torch.empty(proj_dim, d_i)
            nn.init.xavier_uniform_(Pi)
            self.projection_matrices[str(i)] = nn.Parameter(Pi)

    # ------------------------------------------------------------------
    # Optimizers — separate for wi and Pi so each step only touches one
    # ------------------------------------------------------------------

    def configure_optimizers(self) -> list:
        model_opt = instantiate(
            self.hparams.optimizer,
            params=self.agents.parameters(),
        )
        proj_opt = instantiate(
            self.hparams.optimizer,
            params=self.projection_matrices.parameters(),
        )
        return [model_opt, proj_opt]

    def on_train_epoch_end(self) -> None:
        self._finalize_train_epoch_communication()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_neighbor_indices(self, idx: int) -> set[int]:
        neighbors = self.hparams.neighbors
        raw = neighbors.get(idx, neighbors.get(str(idx), set()))
        return {int(j) for j in raw}

    def _compute_class_means(
        self,
        latents: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict[int, torch.Tensor]:
        """Class-conditional mean latent vectors."""
        class_means: dict[int, torch.Tensor] = {}
        for c in torch.unique(labels):
            mask = labels == c
            class_means[int(c.item())] = latents[mask].mean(dim=0)
        return class_means

    def _project_class_means(
        self,
        class_means: dict[int, torch.Tensor],
        Pi: nn.Parameter,
    ) -> dict[int, torch.Tensor]:
        """Compute û_{i,m} = Pi @ v̄_{i,m} for each class m."""
        return {c: Pi @ mu for c, mu in class_means.items()}

    def _alignment_loss(
        self,
        proj_means: dict[int, dict[int, torch.Tensor]],
    ) -> torch.Tensor:
        """
        Squared Frobenius alignment loss.

        For each (agent i, class m) pair where at least one neighbor also
        observed class m, accumulates:
            ‖û_{i,m} − mean_{j∈N_i}(û_{j,m}.detach())‖²
        """
        loss = torch.tensor(0.0, device=self.device)
        n_terms = 0

        for idx_str in self.agents:
            idx = int(idx_str)
            proj_i = proj_means.get(idx, {})
            if not proj_i:
                continue

            neighbor_indices = self._get_neighbor_indices(idx)

            for c, u_i_c in proj_i.items():
                neighbor_projs = [
                    proj_means[j][c].detach()
                    for j in neighbor_indices
                    if j in proj_means and c in proj_means[j]
                ]
                if not neighbor_projs:
                    continue

                # Arithmetic mean of neighbors' projected prototypes (L2 consensus)
                target = torch.stack(neighbor_projs).mean(dim=0)
                diff = u_i_c - target
                loss = loss + (diff ** 2).sum()
                n_terms += 1

        return loss / max(n_terms, 1)

    # ------------------------------------------------------------------
    # Pi gradient step: encoder weights frozen, only Pi receives gradient
    # ------------------------------------------------------------------

    def _proj_loss(self, batch: dict) -> torch.Tensor:
        """Recompute projected class means with frozen encoder; gradient flows to Pi only."""
        class_means_per_agent: dict[int, dict[int, torch.Tensor]] = {}

        with torch.no_grad():
            for idx_str, agent in self.agents.items():
                idx = int(idx_str)
                key = str(idx) if str(idx) in batch else idx
                x, y = batch[key]
                z = agent.encode(x)
                class_means_per_agent[idx] = self._compute_class_means(z, y)

        proj_means = {
            idx: self._project_class_means(
                class_means, self.projection_matrices[str(idx)]
            )
            for idx, class_means in class_means_per_agent.items()
        }
        return self._alignment_loss(proj_means)

    # ------------------------------------------------------------------
    # Training step: alternating wi / Pi updates
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        model_opt, proj_opt = self.optimizers()

        if isinstance(batch, tuple):
            batch = batch[0]

        # Step 1 — wi update (task loss + alignment loss; Pi untouched by model_opt)
        model_opt.zero_grad()
        proj_opt.zero_grad()
        outputs, total_loss = self._shared_eval(batch, batch_idx, 'train')
        self.manual_backward(total_loss)
        model_opt.step()

        # Step 2 — Pi update (alignment loss only, encoder weights frozen)
        for _ in range(self.hparams.proj_steps):
            proj_opt.zero_grad()
            proj_loss = self._proj_loss(batch)
            self.manual_backward(proj_loss)
            proj_opt.step()

        self.log(
            'train/total_loss_step',
            total_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            batch_size=self._resolve_batch_size(batch),
        )
        return total_loss

    # ------------------------------------------------------------------
    # Shared evaluation (train / val / test)
    # ------------------------------------------------------------------

    def _shared_eval(self, batch, batch_idx: int, prefix: str):
        if isinstance(batch, tuple):
            batch = batch[0]

        outputs: dict = {}
        agent_losses: dict[int, torch.Tensor] = {}
        agent_performances: dict[int, torch.Tensor] = {}
        class_means_per_agent: dict[int, dict[int, torch.Tensor]] = {}

        def _resolve_key(i: int):
            str_key = str(i)
            return str_key if str_key in batch else i

        for idx_str, agent in self.agents.items():
            idx = int(idx_str)
            x, y = batch[_resolve_key(idx)]

            z = agent.encode(x)
            y_hat = agent.decoder(z)

            outputs[idx] = (y_hat.detach(), y)
            # class means preserve gradient so alignment loss reaches encoder (wi step)
            class_means_per_agent[idx] = self._compute_class_means(z, y)

            agent_losses[idx] = agent.compute_loss(y_hat, y)
            agent_performances[idx] = agent.task_performance(y_hat, y)

        proj_means = {
            idx: self._project_class_means(
                class_means, self.projection_matrices[str(idx)]
            )
            for idx, class_means in class_means_per_agent.items()
        }

        alignment_loss = self._alignment_loss(proj_means)

        if prefix == 'train':
            self._record_communication_round(n_rounds=1, prefix=prefix)
            for idx_str in self.agents:
                idx = int(idx_str)
                p_means = proj_means.get(idx, {})
                if not p_means:
                    continue
                means_tensor = torch.stack(list(p_means.values())).detach()
                n_neighbors = len(self._get_neighbor_indices(idx))
                self._record_communication(
                    means_tensor, n_transmissions=n_neighbors
                )

        total_task_loss = torch.stack(list(agent_losses.values())).sum()
        total_loss = total_task_loss + self._effective_lambda_reg() * alignment_loss

        self._log_shared_metrics(
            prefix=prefix,
            agent_losses=agent_losses,
            agent_performances=agent_performances,
            batch_size=self._resolve_batch_size(batch),
            agent_sample_counts=self._resolve_agent_sample_counts(batch),
            total_loss=total_loss,
            extra_metrics={f'{prefix}/alignment_loss': alignment_loss},
            prog_bar=False,
            per_agent_loss_name='task_loss',
        )

        return outputs, total_loss


if __name__ == '__main__':
    pass
