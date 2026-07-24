"""
Sheaf-FMTL Orchestrator.

Implements the Sheaf-FMTL training logic with epoch-end synchronization in
PyTorch Lightning. This orchestrator uses trainable projection matrices
P_ij to map agent-local parameters into a shared latent space where a
Sheaf Laplacian penalty enforces alignment among neighboring agents in the
communication graph.

Implementation Details:
-----------------------
1. Projection Matrices (P_ij) of size (d_ij x d_i)
   - d_i = the total number of trainable parameters for agent i
   - d_ij = max(1, int(gamma * min(d_i, d_j))) shared latent space dim

2. Epoch-End Local Parameter Regularization (`on_train_epoch_end` hook):
    - At the end of each training epoch, the orchestrator collects the
      current flattened parameter vector theta_i for every agent
    - It computes the Sheaf Laplacian penalty
      lambda * sum_j P_ij^T * (P_ij * theta_i - P_ji * theta_j)
    - and applies one manual parameter update
      theta_i <- theta_i - lr * penalty_i

3. Epoch-End Projection Matrix Update:
    - After the epoch-end parameter update, the orchestrator recomputes the
      projected edge vectors using the updated theta_i values
    - Each restriction map is then updated manually as
     P_ij <- P_ij - eta * lambda * (P_ij * theta_i - P_ji * theta_j) * theta_i^T
    - One epoch-end synchronization therefore incurs two projected-vector
      exchange rounds: one before the agent update and one before the
      restriction-map update

Memory note:
    d_i is the *full* flattened parameter count of an agent's model, so
    P_ij is O(gamma * d^2) per edge. For many agents/large models this does
    not fit in accelerator memory even at small gamma. The equations above
    are unchanged, but purely for memory reasons P_ij lives permanently on
    CPU in a reduced-precision dtype (``matrix_dtype``), is upcast to fp32
    only transiently (one edge at a time) for each matmul/outer-product,
    and edges are updated in place one at a time rather than staging every
    edge's update in memory simultaneously.
"""

from typing import Any

import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from src.communication.alignment_mixin import (
    VALID_ALIGNMENT_METHODS,
    PostTrainingAlignmentMixin,
)
from src.orchestrators.base_orchestrator import BaseOrchestrator


class SheafFMTL(PostTrainingAlignmentMixin, BaseOrchestrator):
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
    max_lmb : float
        Maximum regularization coefficient for the Sheaf Laplacian penalty.
    lambda_schedule : str or None
        Scheduling strategy for lambda: ``None`` (constant), ``'cosine'``, or ``'exp'``.
    eta : float
        Learning rate for the manual P_ij matrix update step.
    matrix_dtype : str
        Storage dtype for the (potentially huge) P_ij matrices: ``'float16'``
        (default), ``'bfloat16'``, or ``'float32'``. Matrices always live on
        CPU regardless of accelerator, and are upcast to fp32 transiently
        (one edge at a time) for each matmul/outer-product, so this only
        controls resting memory, not numerical behaviour of the matmuls.
    """

    _MATRIX_DTYPES = {
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
        'float32': torch.float32,
    }

    def __init__(
        self,
        agents: dict[int, nn.Module],
        neighbors: dict[int, set[int]],
        optimizer: Any,
        gamma: float = 0.01,
        max_lmb: float = 0.001,
        eta: float = 0.01,
        lambda_schedule: str | None = None,
        alignment_method: str = 'general',
        matrix_dtype: str = 'float16',
        **kwargs,
    ):
        alignment_method = str(alignment_method)
        if alignment_method not in VALID_ALIGNMENT_METHODS:
            raise ValueError(
                f"Unknown alignment_method '{alignment_method}'. "
                f'Valid options: {VALID_ALIGNMENT_METHODS}'
            )
        if matrix_dtype not in self._MATRIX_DTYPES:
            raise ValueError(
                f"Unknown matrix_dtype '{matrix_dtype}'. "
                f'Valid options: {sorted(self._MATRIX_DTYPES)}'
            )

        super().__init__(
            agents=agents,
            neighbors=neighbors,
            optimizer=optimizer,
            **kwargs,
        )
        self.save_hyperparameters(ignore=['agents'])

        # Plain dict (not nn.ParameterDict): P_ij is never touched by
        # autograd (manual .data-style updates only, see on_train_epoch_end)
        # and must stay pinned to CPU regardless of accelerator, so it must
        # not be swept up by Lightning's automatic module.to(device) at
        # setup time.
        storage_dtype = self._MATRIX_DTYPES[matrix_dtype]
        self.projection_matrices: dict[str, torch.Tensor] = {}

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

                # Rectangular projection matrix P_ij (d_ij x d_i), CPU-resident
                P_ij = torch.empty(
                    d_ij, d_i, dtype=torch.float32, device='cpu'
                )
                nn.init.normal_(P_ij, mean=0.0, std=0.01)

                self.projection_matrices[f'{i}_{j}'] = P_ij.to(storage_dtype)

    def on_train_epoch_end(self) -> None:
        """Synchronize agent parameters and restriction maps once per epoch.

        The P_ij matrices are CPU-resident (see __init__), so this whole
        synchronization runs on CPU: agent parameter vectors are copied to
        CPU once per epoch (cheap relative to a P_ij matrix), and each P_ij
        is upcast to fp32 transiently, one edge at a time, rather than all
        at once, to keep peak memory to O(one edge) instead of O(all edges).
        """
        agent_vectors = self._collect_agent_vectors()
        projected_vectors = self._projected_edge_vectors(agent_vectors)
        self._record_projected_vector_exchange(
            projected_vectors,
            prefix='train',
        )

        lr = self._current_learning_rate()

        with torch.no_grad():
            updated_agent_vectors = {}
            for idx_str, agent in self.agents.items():
                i = int(idx_str)
                theta_i = agent_vectors[i]
                sheaf_penalty_i = torch.zeros_like(theta_i)

                for j in self.hparams.neighbors.get(i, set()):
                    P_ij = self.projection_matrices[f'{i}_{j}']
                    diff_ij = (
                        projected_vectors[f'{i}_{j}']
                        - projected_vectors[f'{j}_{i}']
                    )
                    sheaf_penalty_i += (
                        self._effective_lambda_reg()
                        * torch.matmul(P_ij.float().t(), diff_ij)
                    )

                updated_agent_vectors[i] = theta_i - lr * sheaf_penalty_i

            for idx_str, agent in self.agents.items():
                i = int(idx_str)
                trainable_params = [
                    p for p in agent.parameters() if p.requires_grad
                ]
                reference_param = trainable_params[0]
                updated_vector = updated_agent_vectors[i].to(
                    device=reference_param.device,
                    dtype=reference_param.dtype,
                )
                vector_to_parameters(updated_vector, trainable_params)

            updated_projected_vectors = self._projected_edge_vectors(
                updated_agent_vectors
            )
            self._record_projected_vector_exchange(
                updated_projected_vectors,
                prefix='train',
            )

            # Update each restriction map in place, one edge at a time:
            # never stage every edge's update matrix in memory at once.
            for edge_key in list(self.projection_matrices.keys()):
                i_str, j_str = edge_key.split('_')
                i, j = int(i_str), int(j_str)
                theta_i = updated_agent_vectors[i]
                diff = (
                    updated_projected_vectors[edge_key]
                    - updated_projected_vectors[f'{j}_{i}']
                )
                update_matrix = (
                    self.hparams.eta
                    * self._effective_lambda_reg()
                    * torch.outer(diff, theta_i)
                )
                P_ij = self.projection_matrices[edge_key]
                P_ij -= update_matrix.to(P_ij.dtype)

        self._finalize_train_epoch_communication()
        self._log_train_comm_task_perf()

    def _collect_agent_vectors(self) -> dict[int, torch.Tensor]:
        """Return flattened trainable parameter vectors for each agent.

        Vectors are moved to CPU (fp32) since they are combined with the
        CPU-resident P_ij matrices for the sheaf computation below.
        """
        agent_vectors = {}
        with torch.no_grad():
            for idx_str, agent in self.agents.items():
                idx = int(idx_str)
                trainable_params = [
                    p for p in agent.parameters() if p.requires_grad
                ]
                agent_vectors[idx] = (
                    parameters_to_vector(trainable_params)
                    .detach()
                    .to(device='cpu', dtype=torch.float32)
                )
        return agent_vectors

    def _projected_edge_vectors(
        self, agent_vectors: dict[int, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Return the communicated projected vector for each directed edge."""
        projected_vectors = {}
        with torch.no_grad():
            for edge_key, P_ij in self.projection_matrices.items():
                i_str, _j_str = edge_key.split('_')
                i = int(i_str)
                projected_vectors[edge_key] = torch.matmul(
                    P_ij.float(), agent_vectors[i]
                )
        return projected_vectors

    def _record_projected_vector_exchange(
        self,
        projected_vectors: dict[str, torch.Tensor],
        *,
        prefix: str = 'train',
    ) -> None:
        """Record one exchange round of transmitted projected vectors."""
        if not projected_vectors:
            return

        self._record_communication_round(prefix=prefix)
        for projected_vector in projected_vectors.values():
            self._record_communication(projected_vector, prefix=prefix)

    def _current_learning_rate(self) -> float:
        """Resolve the optimizer learning rate for epoch-level sheaf updates."""
        trainer = getattr(self, '_trainer', None)
        if trainer is not None and getattr(trainer, 'optimizers', None):
            optimizers = trainer.optimizers
            if optimizers and optimizers[0].param_groups:
                return float(optimizers[0].param_groups[0].get('lr', 0.0))

        optimizer_cfg = getattr(self.hparams, 'optimizer', None)
        if isinstance(optimizer_cfg, dict) and 'lr' in optimizer_cfg:
            return float(optimizer_cfg['lr'])
        if hasattr(optimizer_cfg, 'lr'):
            return float(optimizer_cfg.lr)
        return 0.0

    # def on_before_optimizer_step(self, optimizer: Any) -> None:
    #     """
    #     Calculate the sheaf Laplacian penalty and add it directly to
    #     parameter.grad.
    #
    #     Penalty on theta_i:
    #     lambda * sum(P_ij^T * (P_ij*theta_i - P_ji*theta_j))
    #     """
    #     agent_vectors = self._collect_agent_vectors()
    #     projected_vectors = self._projected_edge_vectors(agent_vectors)
    #     self._record_projected_vector_exchange(
    #         projected_vectors,
    #         prefix='train',
    #     )
    #
    #     # Compute gradients manually without autograd to avoid graph issues
    #     with torch.no_grad():
    #         for idx_str, agent in self.agents.items():
    #             i = int(idx_str)
    #             theta_i = agent_vectors[i]
    #
    #             grad_penalty_i = torch.zeros_like(theta_i)
    #
    #             neighbors_i = self.hparams.neighbors.get(i, set())
    #             for j in neighbors_i:
    #                 theta_j = agent_vectors[j]
    #
    #                 P_ij = self.projection_matrices[f'{i}_{j}']
    #
    #                 # diff_ij = P_ij * theta_i - P_ji * theta_j
    #                 diff_ij = projected_vectors[f'{i}_{j}'] - projected_vectors[
    #                     f'{j}_{i}'
    #                 ]
    #
    #                 # grad_contribution = lambda * P_ij^T * diff_ij
    #                 grad_penalty_i += self._effective_lambda_reg() * torch.matmul(
    #                     P_ij.t(), diff_ij
    #                 )
    #
    #             # Add the penalty directly to the parameters' gradients
    #             pointer = 0
    #             for p in agent.parameters():
    #                 if p.requires_grad:
    #                     numel = p.numel()
    #                     # If param unused in forward pass, initialize grad
    #                     if p.grad is None:
    #                         p.grad = torch.zeros_like(p)
    #
    #                     # Add penalty to existing gradient
    #                     p.grad += grad_penalty_i[
    #                         pointer : pointer + numel
    #                     ].view_as(p)
    #                     pointer += numel

    # def on_train_batch_end(
    #     self, outputs: Any, batch: Any, batch_idx: int
    # ) -> None:
    #     """
    #     Execute the matrix update for P_ij.
    #
    #     P_ij = P_ij - eta * lambda * (P_ij*theta_i - P_ji*theta_j) * theta_i^T
    #     """
    #     agent_vectors = self._collect_agent_vectors()
    #     projected_vectors = self._projected_edge_vectors(agent_vectors)
    #     self._record_projected_vector_exchange(
    #         projected_vectors,
    #         prefix='train',
    #     )
    #
    #     with torch.no_grad():
    #         # We must compute all differences synchronously before updating
    #         updates = {}
    #         for edge_key, P_ij in self.projection_matrices.items():
    #             i_str, j_str = edge_key.split('_')
    #             i, j = int(i_str), int(j_str)
    #
    #             theta_i = agent_vectors[i]
    #
    #             # diff = P_ij * theta_i - P_ji * theta_j  (shape: d_ij)
    #             diff = projected_vectors[edge_key] - projected_vectors[f'{j}_{i}']
    #
    #             # update: outer product diff * theta_i^T (shape: d_ij x d_i)
    #             grad_P_ij = torch.outer(diff, theta_i)
    #             updates[edge_key] = (
    #                 self.hparams.eta * self._effective_lambda_reg() * grad_P_ij
    #             )
    #
    #         # Apply updates
    #         for edge_key, update_matrix in updates.items():
    #             self.projection_matrices[edge_key].data -= update_matrix

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
            agent_sample_counts=self._resolve_agent_sample_counts(batch),
            skip_task_performance=(prefix == 'test'),
        )

        return outputs, total_loss


    # send_message, _fit_alignment_maps, _cleanup_alignment inherited from
    # PostTrainingAlignmentMixin: post-hoc whitening + alignment pipeline.

    def on_validation_epoch_end(self) -> None:
        # The projected-vector exchanges happen in on_train_epoch_end, which
        # fires right before validation. Pair validation with the cumulative
        # train comm budget so accuracy-vs-budget plots aren't stuck on a zero
        # x-axis.
        super().on_validation_epoch_end()
        self._finalize_paired_communication('validation', source_prefix='train')

    def on_test_epoch_end(self) -> None:
        super().on_test_epoch_end()
        dm = getattr(self.trainer, 'datamodule', None)
        if dm is None:
            return
        logs = self.evaluate_communication_accuracy(dm)
        if logs:
            self.log_dict(
                logs,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                add_dataloader_idx=False,
            )


if __name__ == '__main__':
    pass
