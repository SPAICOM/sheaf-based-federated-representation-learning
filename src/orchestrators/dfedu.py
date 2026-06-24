"""
d-FedU Orchestrator.

Implements Algorithm 2 from the d-FedU paper: Decentralized Federated
Multi-Task Learning with Laplacian Regularization.
"""

from typing import Any

import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from src.orchestrators.base_orchestrator import BaseOrchestrator


class DFedU(BaseOrchestrator):
    """
    Decentralized Federated Multi-Task Learning (d-FedU) orchestrator.

    Allows standard local training to occur automatically via Lightning
    and applies a Laplacian penalty at the end of the epoch to pull neighboring
    weights together.

    Parameters
    ----------
    agents : dict[int, nn.Module]
        Dictionary mapping agent indices to their model instances.
    neighbors : dict[int, set[int]]
        Dictionary mapping each agent index to the set of its neighbor indices.
    optimizer : Any
        Optimizer configuration for training.
    eta : float
        Laplacian step size (learning rate for the consensus step).
    """

    def __init__(
        self,
        agents: dict[int, nn.Module],
        neighbors: dict[int, set[int]],
        optimizer: Any,
        eta: float = 0.05,
        **kwargs,
    ):
        super().__init__(
            agents=agents,
            neighbors=neighbors,
            optimizer=optimizer,
            **kwargs,
        )
        self.save_hyperparameters(ignore=['agents'])
        self._validate_agents()

    def _validate_agents(self) -> None:
        """Validate that all agents have identical architectures.

        d-FedU directly mixes parameter vectors, which requires all
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

    @torch.no_grad()
    def on_train_epoch_end(self) -> None:
        """Apply Laplacian consensus step after each epoch of local SGD.

        Implements the weight-averaging rule:

            w_i ← w_i - η · Σ_j a_ij (w_i − w_j)

        where η is ``self.hparams.eta`` and a_ij = 1 for all edges
        (uniform, unweighted adjacency). This is equivalent to one step
        of gradient descent on the Laplacian regulariser
        λ Σ_{(i,j)∈E} ‖w_i − w_j‖².

        All updated vectors are computed from the *pre-update* snapshots
        before any model is mutated, so the order of iteration does not
        affect the result.
        """
        total_transmissions = sum(
            len(self.hparams.neighbors.get(int(idx_str), set()))
            for idx_str in self.agents
        )
        if total_transmissions > 0:
            self._record_communication_round(prefix='train')
        agent_vectors = {}
        # Extract flattened parameter vectors for all agents (trainable only)
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
                prefix='train',
            )

        new_vectors = {}

        # Synchronously compute updates
        for idx_str in self.agents:
            i = int(idx_str)
            w_i = agent_vectors[i]

            neighbors_i = self.hparams.neighbors.get(i, set())
            d_i = len(neighbors_i)

            # laplacian regularization step
            if d_i > 0:
                sum_diff = torch.zeros_like(w_i)
                for j in neighbors_i:
                    w_j = agent_vectors[j]
                    sum_diff += w_i - w_j

                # Uniform weights: a_ij = 1 for all edges.
                # Normalising by degree (/ d_i) is an alternative but is
                # not used here - eta implicitly controls step magnitude.
                w_i_new = w_i - self.hparams.eta * sum_diff
                new_vectors[i] = w_i_new
            else:
                # No neighbors, keep original weights
                new_vectors[i] = w_i

        # Re-inject updated weights into the models
        for idx_str, agent in self.agents.items():
            idx = int(idx_str)
            trainable_params = [
                p for p in agent.parameters() if p.requires_grad
            ]
            vector_to_parameters(new_vectors[idx], trainable_params)

        self._finalize_train_epoch_communication()

    def _shared_eval(
        self,
        batch: dict[int, list[torch.Tensor]],
        batch_idx: int,
        prefix: str,
    ) -> tuple[dict[str, Any], torch.Tensor]:
        """Forward pass and per-agent loss/performance computation.

        Parameters
        ----------
        batch : dict[int, list[torch.Tensor]]
            Mapping from agent index to ``(x, y)`` pairs.
        batch_idx : int
            Current batch index.
        prefix : str
            Logging prefix (e.g. ``'train'``, ``'validation'``).

        Returns
        -------
        tuple[dict, torch.Tensor]
            ``(outputs, total_loss)`` where ``outputs`` maps agent indices
            to ``(y_hat, y)`` pairs and ``total_loss`` is the summed loss.
        """
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
