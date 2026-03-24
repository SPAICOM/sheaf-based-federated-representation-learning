import torch
import torch.nn as nn

from src.orchestrators.base_orchestrator import BaseOrchestrator


class FederatedLearning(BaseOrchestrator):
    """Federated learning orchestrator with neighbor-restricted averaging.

    Implements a localized variant of Federated Averaging where each agent
    updates its parameters by averaging only over its local neighborhood in a
    graph, rather than aggregating across all agents.

    Parameters
    ----------
    agents : dict[int, nn.Module]
        Dictionary mapping agent indices to their model instances. All agents
        must share the same architecture and parameter structure.
    neighbors : dict[int, set[int]]
        Dictionary mapping each agent index to the set of its neighbor indices.
    optimizer : hydra config
        Optimizer configuration for training.

    Notes
    -----
    - All agents must have identical architecture (validated at init).
    - Both model parameters and buffers (e.g., BatchNorm) are
      aggregated via ``state_dict()``.
    - Each agent is always included in its own aggregation set.
    """

    def __init__(
        self,
        agents: dict[int, nn.Module],
        neighbors: dict[int, set[int]],
        optimizer,
    ):
        super().__init__(
            agents=agents,
            neighbors=neighbors,
            optimizer=optimizer,
        )

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

        Raises
        ------
        KeyError
            If an agent index is missing from the neighbor dictionary.
        """
        agents = {int(k): v for k, v in self.agents.items()}

        new_states = {}

        for idx_i, agent_i in agents.items():
            neigh = self.hparams.neighbors[idx_i] | {idx_i}

            avg_state = {
                k: torch.zeros_like(v) for k, v in agent_i.state_dict().items()
            }

            for idx_j in neigh:
                state_j = agents[idx_j].state_dict()

                for k in avg_state:
                    avg_state[k] += state_j[k]

            for k in avg_state:
                avg_state[k] = (avg_state[k].float() / len(neigh)).to(
                    avg_state[k].dtype
                )

            new_states[idx_i] = avg_state

        for idx_i, agent in agents.items():
            agent.load_state_dict(new_states[idx_i])

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
        outputs = self(batch)

        total_loss = 0
        total_performance = 0

        for idx, agent in self.agents.items():
            y_hat, y = outputs[idx]

            loss = agent.compute_loss(y_hat, y)
            performance = agent.task_performance(y_hat, y)

            self.log(
                f'{prefix}/loss_agent_{idx}',
                loss,
                on_step=False,
                on_epoch=True,
            )

            self.log(
                f'{prefix}/task_performance_agent_{idx}',
                performance,
                on_step=False,
                on_epoch=True,
            )

            total_loss += loss
            total_performance += performance

        avg_performance = total_performance / len(self.agents)

        self.log(
            f'{prefix}/total_loss_epoch',
            total_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        self.log(
            f'{prefix}/avg_task_performance_epoch',
            avg_performance,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return outputs, total_loss


if __name__ == '__main__':
    pass
