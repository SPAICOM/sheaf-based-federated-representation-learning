"""
Decentralized Parallel Stochastic Gradient Descent (D-PSGD) orchestrator.

This module implements the mathematical formulation of D-PSGD:
1. Compute local stochastic gradients based on mini-batch data.
2. Compute neighborhood weighted average by fetching optimization variables from neighbors.
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

    The mixing is performed right before the optimizer step using the Lightning
    `on_before_optimizer_step` hook. This allows standard Lightning optimization
    to complete the gradient descent step automatically.

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
        
        # Calculate Doubly Stochastic Mixing Matrix W using Metropolis-Hastings rule
        self.mixing_weights = {}
        
        # Calculate degree for each agent
        # We process keys as integers since neighbors dict typically uses ints
        degrees = {}
        for idx_str in self.agents.keys():
            idx = int(idx_str)
            # if a node doesn't have neighbors, default to empty set
            degrees[idx] = len(neighbors.get(idx, set()))
            
        # Compute Metropolis-Hastings weights
        for idx_str in self.agents.keys():
            i = int(idx_str)
            i_neighbors = neighbors.get(i, set())
            
            weight_sum = 0.0
            for j in i_neighbors:
                # W_{ij} = 1 / (max(d_i, d_j) + 1)
                w_ij = 1.0 / (max(degrees[i], degrees.get(j, 0)) + 1.0)
                self.mixing_weights[(i, j)] = w_ij
                weight_sum += w_ij
                
            # Self-weight: W_{ii} = 1 - sum_{j in neighbors} W_{ij}
            self.mixing_weights[(i, i)] = 1.0 - weight_sum
            
        self._validate_agents()

    def _validate_agents(self) -> None:
        """Validate that all agents have identical architectures.
        
        D-PSGD directly mixes parameter vectors, which requires all agents to have 
        the exact same parameter shapes and order.
        """
        agents_list = list(self.agents.values())
        if len(agents_list) <= 1:
            return
            
        ref = agents_list[0]
        ref_params = dict(ref.named_parameters())
        
        for i, agent in enumerate(agents_list[1:], start=1):
            if type(agent) is not type(ref):
                raise TypeError(f'Agent {i} has different class than reference agent 0.')
                
            params = dict(agent.named_parameters())
            if params.keys() != ref_params.keys():
                raise ValueError(f'Agent {i} parameter names mismatch with agent 0.')
                
            for k in ref_params:
                if params[k].shape != ref_params[k].shape:
                    raise ValueError(f'Shape mismatch in parameter {k}: agent {i} vs agent 0.')

    def on_train_epoch_end(self) -> None:
        """No epoch-level aggregation is required for D-PSGD.
        
        Communication happens every step in `on_before_optimizer_step`.
        """
        pass

    def on_before_optimizer_step(self, optimizer: Any) -> None:
        """
        Mix the optimization variables with neighbors.
        
        This hook is called after loss.backward() (so gradients ∇F_i are stored 
        in .grad) but before optimizer.step().
        We compute the neighborhood weighted average of the weights (x_{k+1/2, i})
        using only trainable parameters.
        We then overwrite the models' parameters. The subsequent optimizer.step() 
        will apply the local gradients to these mixed weights.
        """
        # Dictionary to store the original parameter vectors for each agent before mixing
        agent_vectors = {}
        
        # Extract the current parameter vectors for all agents (only trainable parameters)
        for idx_str, agent in self.agents.items():
            idx = int(idx_str)
            trainable_params = [p for p in agent.parameters() if p.requires_grad]
            vec = parameters_to_vector(trainable_params)
            agent_vectors[idx] = vec
            
        # Dictionary to store the mixed parameter vectors
        mixed_vectors = {}
        
        # Compute the neighborhood weighted average
        for idx_str in self.agents.keys():
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
            trainable_params = [p for p in agent.parameters() if p.requires_grad]
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
        # Get predictions for all agents via forward pass defined in BaseOrchestrator
        outputs = self(batch)

        # Accumulate metrics across all agents
        total_loss = 0.0
        total_performance = 0.0

        # Compute loss and performance for each agent
        for idx, agent in self.agents.items():
            y_hat, y = outputs[idx]

            # Compute task-specific loss 
            loss = agent.compute_loss(y_hat, y)
            # Compute task-specific metric
            performance = agent.task_performance(y_hat, y)

            # Log per-agent metrics
            self.log_dict(
                {
                    f'{prefix}/loss_agent_{idx}': loss,
                    f'{prefix}/task_performance_agent_{idx}': performance,
                },
                on_step=False,
                on_epoch=True,
            )

            # Accumulate for aggregate metrics
            total_loss += loss
            total_performance += performance

        # Compute average performance across all agents
        avg_performance = total_performance / len(self.agents)

        # Log aggregate metrics
        self.log_dict(
            {
                f'{prefix}/total_loss_epoch': total_loss,
                f'{prefix}/avg_task_performance_epoch': avg_performance,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return outputs, total_loss


if __name__ == '__main__':
    pass
