"""
Decentralized Parallel Stochastic Gradient Descent (D-PSGD) orchestrator.

This module implements the mathematical formulation of D-PSGD:
1. Compute local stochastic gradients based on mini-batch data.
2. Compute neighborhood weighted average by fetching optimization
   variables from neighbors.
3. Update the local optimization variable.

Requires same architecture across agents
"""

from typing import Any

import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from src.orchestrators.base_orchestrator import BaseOrchestrator


class DPSGD(BaseOrchestrator):
    """
    Decentralized Parallel Stochastic Gradient Descent (D-PSGD) orchestrator.

    This class strictly adheres to the decentralized data-parallel setup for
    homogeneous architectures.

    Implements:
        x_{k+1/2, i} = sum_{j} W_{ij} x_{k,j}
        x_{k+1, i} = x_{k+1/2, i} - γ ∇F_i(x_{k,i})

    The mixing is performed right before the optimizer step using the
    Lightning `on_before_optimizer_step` hook. This allows standard
    Lightning optimization to complete the gradient descent step
    automatically.

    Parameters
    ----------
    agents : dict[int, nn.Module]
        Dictionary mapping agent indices to their model instances.
    neighbors : dict[int, set[int]]
        Dictionary mapping each agent index to the set of its neighbor indices.
    optimizer : hydra config
        Optimizer configuration for training.
    """

    def __init__(
        self,
        agents: dict[int, nn.Module],
        neighbors: dict[int, set[int]],
        optimizer: Any,
        **kwargs,
    ):
        super().__init__(
            agents=agents,
            neighbors=neighbors,
            optimizer=optimizer,
        )

        # Build a doubly-stochastic mixing matrix W via the
        # Metropolis-Hastings rule.  For each edge (i, j):
        #
        #   W_{ij} = 1 / (max(d_i, d_j) + 1)
        #
        # and the self-weight W_{ii} = 1 - Σ_{j∈N(i)} W_{ij} ensures
        # each row sums to 1.  The +1 in the denominator avoids division
        # by zero for isolated nodes and guarantees W_{ij} < 1.
        self.mixing_weights = {}

        # Degree of each agent (number of neighbours)
        degrees = {}
        for idx_str in self.agents:
            idx = int(idx_str)
            degrees[idx] = len(neighbors.get(idx, set()))

        for idx_str in self.agents:
            i = int(idx_str)
            i_neighbors = neighbors.get(i, set())

            weight_sum = 0.0
            for j in i_neighbors:
                # Metropolis-Hastings off-diagonal weight
                w_ij = 1.0 / (max(degrees[i], degrees.get(j, 0)) + 1.0)
                self.mixing_weights[(i, j)] = w_ij
                weight_sum += w_ij

            # Self-weight ensures the row sums to 1
            self.mixing_weights[(i, i)] = 1.0 - weight_sum

        self._validate_agents()

    def _validate_agents(self) -> None:
        """Validate that all agents have identical architectures.

        D-PSGD directly mixes parameter vectors, which requires all
        agents to have the exact same parameter shapes and order.
        """
        agents_list = list(self.agents.values())
        if len(agents_list) <= 1:
            return

        ref = agents_list[0]
        ref_params = dict(ref.named_parameters())

        for i, agent in enumerate(agents_list[1:], start=1):
            if type(agent) is not type(ref):
                raise TypeError(
                    f'Agent {i} has different class than reference agent 0.'
                )

            params = dict(agent.named_parameters())
            if params.keys() != ref_params.keys():
                raise ValueError(
                    f'Agent {i} parameter names mismatch with agent 0.'
                )

            for k in ref_params:
                if params[k].shape != ref_params[k].shape:
                    raise ValueError(
                        f'Shape mismatch in parameter {k}:'
                        f' agent {i} vs agent 0.'
                    )

    def on_train_epoch_end(self) -> None:
        """No epoch-level aggregation is required for D-PSGD.

        Communication happens every step in `on_before_optimizer_step`.
        """
        pass

    def on_before_optimizer_step(self, optimizer: Any) -> None:
        """Compute the neighbourhood-weighted parameter average (x_{k+1/2}).

        Called after ``loss.backward()`` (gradients are in ``.grad``) but
        before ``optimizer.step()``.  The two-step update is:

            x_{k+1/2, i} = Σ_j W_{ij} · x_{k, j}   ← done here
            x_{k+1,   i} = x_{k+1/2, i} − γ ∇F_i   ← done by Lightning

        Because the mixed weights are written back into the model's
        parameters *in place*, the subsequent optimizer step applies the
        locally-computed gradients to the already-mixed weights, which
        is the intended decentralised SGD behaviour.
        """
        # Original parameter vectors for each agent before mixing
        agent_vectors = {}

        # Current parameter vectors for all agents (trainable only)
        for idx_str, agent in self.agents.items():
            idx = int(idx_str)
            trainable_params = [
                p for p in agent.parameters() if p.requires_grad
            ]
            vec = parameters_to_vector(trainable_params)
            agent_vectors[idx] = vec
            self._record_communication(
                vec,
                n_transmissions=len(self.hparams.neighbors.get(idx, set())),
            )

        # Dictionary to store the mixed parameter vectors
        mixed_vectors = {}

        # Compute the neighborhood weighted average
        for idx_str in self.agents:
            i = int(idx_str)

            # Start with the self-weight contribution: W_{ii} * x_{k,i}
            mixed_vec = self.mixing_weights[(i, i)] * agent_vectors[i]

            # Add neighbor contributions: sum_{j} W_{ij} * x_{k,j}
            i_neighbors = self.hparams.neighbors.get(i, set())
            for j in i_neighbors:
                mixed_vec += self.mixing_weights[(i, j)] * agent_vectors[j]

            mixed_vectors[i] = mixed_vec

        # Re-inject the mixed weights into the models
        for idx_str, agent in self.agents.items():
            idx = int(idx_str)
            trainable_params = [
                p for p in agent.parameters() if p.requires_grad
            ]
            vector_to_parameters(mixed_vectors[idx], trainable_params)

    def _shared_eval(
        self,
        batch: dict[int, list[torch.Tensor]],
        batch_idx: int,
        prefix: str,
    ) -> tuple[dict[str, Any], torch.Tensor]:
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
        tuple[dict[str, Any], torch.Tensor]
            Tuple of (outputs, total_loss) where outputs maps agent indices to
            (prediction, label) pairs and total_loss is the summed loss.
        """
        # Get predictions for all agents via forward pass
        outputs = self(batch)

        agent_losses = {}
        agent_performances = {}

        # Compute loss and performance for each agent
        for idx, agent in self.agents.items():
            y_hat, y = outputs[idx]

            # Compute task-specific loss
            loss = agent.compute_loss(y_hat, y)
            # Compute task-specific metric
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
