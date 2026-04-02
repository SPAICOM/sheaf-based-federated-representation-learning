"""
Sheaf-FMTL Orchestrator.

Implements Algorithm 1 from the Sheaf-FMTL paper using PyTorch Lightning hooks.
This orchestrator utilizes trainable projection matrices P_ij to map
agent local parameters into a shared latent space where a Laplacian
penalty enforces alignment
among neighboring agents in a communication graph.

Implementation Details:
-----------------------
1. Projection Matrices (P_ij) of size (d_ij x d_i)
   - d_i = the total number of trainable parameters for agent i
   - d_ij = max(1, int(gamma * min(d_i, d_j))) shared latent space dim

2. Local Parameter Regularization (`on_before_optimizer_step` hook):
   - Modifies the local gradients before the optimizer steps with the
     Sheaf Laplacian penalty

3. Projection Matrix Update (`on_train_batch_end` hook):
   - P_ij matrices are updated manually as
     P_ij = P_ij - eta * lambda_reg
          * (P_ij * theta_i - P_ji * theta_j) * theta_i^T
"""

from typing import Any

import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector

from src.orchestrators.base_orchestrator import BaseOrchestrator


class SheafFMTL(BaseOrchestrator):
    """
    Sheaf-FMTL Orchestrator.

    Parameters
    ----------
    agents : dict[int, nn.Module]
        Dictionary mapping agent indices to their model instances.
    neighbors : dict[int, set[int]]
        Dictionary mapping each agent index to the set of its neighbor indices.
    optimizer : Any
        Optimizer configuration for training.
    gamma : float
        Ratio for determining the projection dimension (d_ij).
    lambda_reg : float
        Regularization strength for the Sheaf Laplacian penalty.
    eta : float
        Learning rate for the manual P_ij matrix update step.
    """

    def __init__(
        self,
        agents: dict[int, nn.Module],
        neighbors: dict[int, set[int]],
        optimizer: Any,
        gamma: float = 0.01,
        lambda_reg: float = 0.001,
        eta: float = 0.01,
        **kwargs,
    ):
        super().__init__(
            agents=agents,
            neighbors=neighbors,
            optimizer=optimizer,
        )
        self.save_hyperparameters(ignore=['agents'])

        self.projection_matrices = nn.ParameterDict()

        # Count trainable parameters for each agent (d_i)
        d = {}
        for idx_str, agent in self.agents.items():
            idx = int(idx_str)
            d[idx] = sum(
                p.numel() for p in agent.parameters() if p.requires_grad
            )

        # Create Projection Matrices P_ij
        for idx_str in self.agents:
            i = int(idx_str)
            for j in self.hparams.neighbors.get(i, set()):
                d_i = d[i]
                d_j = d[j]

                # Projection dimension d_ij
                d_ij = max(1, int(self.hparams.gamma * min(d_i, d_j)))

                # Rectangular projection matrix P_ij (d_ij x d_i)
                P_ij = nn.Parameter(torch.empty(d_ij, d_i), requires_grad=True)
                nn.init.uniform_(P_ij, a=-0.01, b=0.01)

                self.projection_matrices[f'{i}_{j}'] = P_ij

    def on_train_epoch_end(self) -> None:
        """No epoch-level aggregation needed for Sheaf-FMTL.

        Both the Laplacian gradient penalty and the projection-matrix
        update are applied at finer granularity (per-step and per-batch
        respectively via ``on_before_optimizer_step`` and
        ``on_train_batch_end``), so nothing is required here.
        """
        pass

    def on_before_optimizer_step(self, optimizer: Any) -> None:
        """
        Calculate the sheaf Laplacian penalty and add it directly to
        parameter.grad.

        Penalty on theta_i:
        lambda * sum(P_ij^T * (P_ij*theta_i - P_ji*theta_j))
        """
        agent_vectors = {}
        for idx_str, agent in self.agents.items():
            idx = int(idx_str)
            # Ensure we only use trainable parameters
            trainable_params = [
                p for p in agent.parameters() if p.requires_grad
            ]
            vec = parameters_to_vector(trainable_params)
            agent_vectors[idx] = vec
            self._record_communication(
                vec,
                n_transmissions=len(self.hparams.neighbors.get(idx, set())),
            )

        for P_ij in self.projection_matrices.values():
            self._record_communication(P_ij)

        # Compute gradients manually without autograd to avoid graph issues
        with torch.no_grad():
            for idx_str, agent in self.agents.items():
                i = int(idx_str)
                theta_i = agent_vectors[i]

                grad_penalty_i = torch.zeros_like(theta_i)

                neighbors_i = self.hparams.neighbors.get(i, set())
                for j in neighbors_i:
                    theta_j = agent_vectors[j]

                    P_ij = self.projection_matrices[f'{i}_{j}']
                    P_ji = self.projection_matrices[f'{j}_{i}']

                    # diff_ij = P_ij * theta_i - P_ji * theta_j
                    diff_ij = torch.matmul(P_ij, theta_i) - torch.matmul(
                        P_ji, theta_j
                    )

                    # grad_contribution = lambda * P_ij^T * diff_ij
                    grad_penalty_i += self.hparams.lambda_reg * torch.matmul(
                        P_ij.t(), diff_ij
                    )

                # Add the penalty directly to the parameters' gradients
                pointer = 0
                for p in agent.parameters():
                    if p.requires_grad:
                        numel = p.numel()
                        # If param unused in forward pass, initialize grad
                        if p.grad is None:
                            p.grad = torch.zeros_like(p)

                        # Add penalty to existing gradient
                        p.grad += grad_penalty_i[
                            pointer : pointer + numel
                        ].view_as(p)
                        pointer += numel

    def on_train_batch_end(
        self, outputs: Any, batch: Any, batch_idx: int
    ) -> None:
        """
        Execute the matrix update for P_ij.

        P_ij = P_ij - eta * lambda * (P_ij*theta_i - P_ji*theta_j) * theta_i^T
        """
        agent_vectors = {}
        for idx_str, agent in self.agents.items():
            idx = int(idx_str)
            trainable_params = [
                p for p in agent.parameters() if p.requires_grad
            ]
            with torch.no_grad():
                vec = parameters_to_vector(trainable_params)
            agent_vectors[idx] = vec

        with torch.no_grad():
            # We must compute all differences synchronously before updating
            updates = {}
            for edge_key, P_ij in self.projection_matrices.items():
                i_str, j_str = edge_key.split('_')
                i, j = int(i_str), int(j_str)

                theta_i = agent_vectors[i]
                theta_j = agent_vectors[j]

                P_ji = self.projection_matrices[f'{j}_{i}']

                # diff = P_ij * theta_i - P_ji * theta_j  (shape: d_ij)
                diff = torch.matmul(P_ij, theta_i) - torch.matmul(
                    P_ji, theta_j
                )

                # update: outer product diff * theta_i^T (shape: d_ij x d_i)
                grad_P_ij = torch.outer(diff, theta_i)
                updates[edge_key] = (
                    self.hparams.eta * self.hparams.lambda_reg * grad_P_ij
                )

            # Apply updates
            for edge_key, update_matrix in updates.items():
                self.projection_matrices[edge_key].data -= update_matrix

    def _shared_eval(
        self,
        batch: dict[int, list[torch.Tensor]],
        batch_idx: int,
        prefix: str,
    ) -> tuple[dict[str, Any], torch.Tensor]:
        """Compute losses and metrics for validation/test steps."""
        outputs = self(batch)

        agent_losses = {}
        agent_performances = {}

        for idx, agent in self.agents.items():
            y_hat, y = outputs[idx]

            loss = agent.compute_loss(y_hat, y)
            performance = agent.task_performance(y_hat, y)

            agent_losses[int(idx)] = loss
            agent_performances[int(idx)] = performance

        total_loss, _avg_performance = self._log_shared_metrics(
            prefix=prefix,
            agent_losses=agent_losses,
            agent_performances=agent_performances,
            batch_size=self._resolve_batch_size(batch),
        )

        return outputs, total_loss


if __name__ == '__main__':
    pass
