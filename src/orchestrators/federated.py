"""
Federated learning orchestrator with neighbor-restricted parameter averaging.

This module implements a variant of Federated Averaging where each agent
updates its parameters by averaging only with neighboring agents in a
predefined communication graph.
"""

import torch
import torch.nn as nn
import torch.utils.data

from src.communication.alignment_mixin import (
    VALID_ALIGNMENT_METHODS,
    PostTrainingAlignmentMixin,
)
from src.orchestrators.base_orchestrator import BaseOrchestrator


class FederatedLearning(PostTrainingAlignmentMixin, BaseOrchestrator):
    """Federated learning orchestrator with neighbor-restricted averaging.

    Implements a localized variant of Federated Averaging where each agent
    updates its parameters by averaging only over its local neighborhood in a
    graph, rather than aggregating across all agents.

    In this framework, the communication graph defines which agents can share
    their model updates. For each agent i, only agents in the set
    {i} ∪ neighbors[i] contribute to the aggregation.

    Parameters
    ----------
    agents : dict[int, nn.Module]
        Dictionary mapping agent indices to their model instances. All agents
        must share the same architecture and parameter structure.
    neighbors : dict[int, set[int]]
        Dictionary mapping each agent index to the set of its neighbor indices.
    optimizer : hydra config
        Optimizer configuration for training.

    Example
    -------
    Given 3 agents with neighbors {0: {1}, 1: {0, 2}, 2: {1}}:
    - Agent 0 aggregates: (agent_0 + agent_1) / 2
    - Agent 1 aggregates: (agent_0 + agent_1 + agent_2) / 3
    - Agent 2 aggregates: (agent_1 + agent_2) / 2

    Notes
    -----
    - All agents must have identical architecture (validated at init).
    - Both model parameters and buffers (e.g., BatchNorm) are
      aggregated via ``state_dict()``.
    - Each agent is always included in its own aggregation set.
    - Aggregation is performed synchronously to avoid update order bias.
    """

    def __init__(
        self,
        agents: dict[int, nn.Module],
        neighbors: dict[int, set[int]],
        optimizer,
        alignment_method: str | None = None,
        **kwargs,
    ):
        if alignment_method is not None:
            alignment_method = str(alignment_method)
            if alignment_method not in VALID_ALIGNMENT_METHODS:
                raise ValueError(
                    f"Unknown alignment_method '{alignment_method}'. "
                    f'Valid options: {VALID_ALIGNMENT_METHODS}'
                )

        super().__init__(
            agents=agents,
            neighbors=neighbors,
            optimizer=optimizer,
            **kwargs,
        )
        self.save_hyperparameters(ignore=['agents'])
        self._validate_agents_for_fedavg()

    def _validate_agents_for_fedavg(self):
        """Validate that all agents have compatible architectures.

        Checks that all agents share the same class type, parameter names,
        parameter shapes, and buffer definitions.

        Raises
        ------
        TypeError
            If any agent has a different class than the reference agent.
        ValueError
            If parameter names, shapes, or buffers mismatch between agents.
        """
        agents = list(self.agents.values())
        ref = agents[0]

        ref_params = dict(ref.named_parameters())
        ref_buffers = dict(ref.named_buffers())

        for i, agent in enumerate(agents[1:], start=1):
            if type(agent) is not type(ref):
                raise TypeError(f'Agent {i} has different class.')

            params = dict(agent.named_parameters())
            if params.keys() != ref_params.keys():
                raise ValueError(f'Agent {i} param names mismatch.')

            for k in ref_params:
                if params[k].shape != ref_params[k].shape:
                    raise ValueError(f'Shape mismatch in {k}')

            buffers = dict(agent.named_buffers())
            if buffers.keys() != ref_buffers.keys():
                raise ValueError(f'Agent {i} buffer mismatch.')

    @torch.no_grad()
    def on_train_epoch_end(self) -> None:
        """Perform neighbor-restricted Federated Averaging.

        For each agent ``i``, the updated parameters are computed as the
        average of the parameters of the agents in ``{i} ∪ neighbors[i]``.

        The aggregation is performed synchronously:
        1. All new averaged states are computed and stored.
        2. The updated states are then loaded into the agents.

        This avoids in-place updates that would otherwise bias the aggregation.

        The algorithm:
        1. Iterate over each agent and its neighborhood (including itself)
        2. Initialize accumulator with zeros matching agent's state_dict
        3. Sum state_dicts from all neighbors
        4. Divide by number of participants to get the average
        5. Store averaged state and apply after loop (synchronous update)

        Raises
        ------
        KeyError
            If an agent index is missing from the neighbor dictionary.
        """
        # Convert agent keys from string to int for consistent indexing
        agents = {int(k): v for k, v in self.agents.items()}
        total_transmissions = sum(
            len(self.hparams.neighbors[idx_i]) for idx_i in agents
        )
        if total_transmissions > 0:
            self._record_communication_round(prefix='train')

        # Store new states before applying (synchronous update)
        new_states = {}

        # Process each agent and compute averaged parameters
        for idx_i, agent_i in agents.items():
            self._record_communication(
                agent_i.state_dict(),
                n_transmissions=len(self.hparams.neighbors[idx_i]),
                prefix='train',
            )

            # Include agent itself in aggregation set (neighbors | {self})
            # This ensures each agent always contributes to its own update
            neigh = self.hparams.neighbors[idx_i] | {idx_i}

            # Initialize accumulator with zero tensors matching agent_i's
            # state_dict structure (parameters and buffers)
            avg_state = {
                k: torch.zeros_like(v) for k, v in agent_i.state_dict().items()
            }

            # Accumulate state_dicts from all neighbors (including self)
            for idx_j in neigh:
                state_j = agents[idx_j].state_dict()

                # Sum each parameter/buffer across neighbors
                for k in avg_state:
                    avg_state[k] += state_j[k]

            # Compute average: divide sum by number of participants
            # Using float division then converting back handles mixed precision
            for k in avg_state:
                avg_state[k] = (avg_state[k].float() / len(neigh)).to(
                    avg_state[k].dtype
                )

            # Store averaged state for this agent
            new_states[idx_i] = avg_state

        # Apply all averaged states AFTER computing all updates
        # This synchronous approach avoids update order bias
        for idx_i, agent in agents.items():
            agent.load_state_dict(new_states[idx_i])

        self._finalize_train_epoch_communication()
        self._log_train_comm_task_perf()

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
        tuple[dict[int, tuple[torch.Tensor, torch.Tensor]], torch.Tensor]
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

            # Compute task-specific loss (e.g., cross-entropy)
            loss = agent.compute_loss(y_hat, y)
            # Compute task-specific metric (e.g., accuracy)
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

    # send_message and evaluate_communication_accuracy are inherited from
    # PostTrainingAlignmentMixin.  When alignment_method is None (default),
    # send_message acts as identity (no maps fitted) and evaluation delegates
    # directly to the base class.  When alignment_method is set, post-hoc
    # whitening + alignment maps are fitted before evaluation.

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
