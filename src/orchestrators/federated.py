import torch
import torch.nn as nn

from src.orchestrators.base_orchestrator import BaseOrchestrator


class FederatedLearning(BaseOrchestrator):
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

        This method implements a localized variant of Federated Averaging
        where, instead of aggregating parameters across all agents, each agent
        updates its parameters by averaging only over its local neighborhood
        in a graph.

        For each agent ``i``, the updated parameters are computed as the
        average of the parameters of the agents in the set:

            {i} ∪ neighbors[i]

        The aggregation is performed synchronously:
        1. All new averaged states are computed and stored.
        2. The updated states are then loaded into the agents.

        This avoids in-place updates that would otherwise bias the aggregation.

        Notes:
            - The method assumes that all agents share the same architecture
              and parameter structure (validated beforehand).
            - Both model parameters and buffers (e.g., BatchNorm statistics or
              registered buffers) are included via ``state_dict()``.
            - The neighborhood structure must be provided in
              ``self.hparams.neighbors`` as a mapping:
                  dict[int, set[int]]
            - Each agent is always included in its own aggregation.

        Raises:
            KeyError: If an agent index is missing from the neighbor
                      dictionary.

        Returns:
            None
        """
        # Convert ModuleDict → int-indexed dict
        agents = {int(k): v for k, v in self.agents.items()}

        # Store new states (avoid in-place updates)
        new_states = {}

        for idx_i, agent_i in agents.items():
            # Include self in neighborhood
            neigh = self.hparams.neighbors[idx_i] | {idx_i}

            # Initialize accumulator
            avg_state = {
                k: torch.zeros_like(v) for k, v in agent_i.state_dict().items()
            }

            # Aggregate neighbors
            for idx_j in neigh:
                state_j = agents[idx_j].state_dict()

                for k in avg_state:
                    avg_state[k] += state_j[k]

            # Average
            for k in avg_state:
                avg_state[k] /= len(neigh)

            # Store result (do NOT load yet)
            new_states[idx_i] = avg_state

        # Apply updates synchronously
        for idx_i, agent in agents.items():
            agent.load_state_dict(new_states[idx_i])

        return None

    def _shared_eval(
        self,
        batch: dict[int, list[torch.Tensor]],
        batch_idx: int,
        prefix: str,
    ):
        """A common step performed in the test and validation step.

        Args:
            batch : dict[int, list[torch.Tensor]]
                The current batch.
            batch_idx : int
                The batch index.
            prefix : str
                The step type for logging purposes.

        Returns:
            (outputs, total_loss) : tuple[
                                        dict[int, torch.Tensor],
                                        torch.Tensor,
                                    ]
                The tuple with the output of the network and the epoch loss.
        """
        outputs = self(batch)

        total_loss = 0
        total_performance = 0

        # Compute the personalized loss for each agent
        for idx, agent in self.agents.items():
            y_hat, y = outputs[idx]

            loss = agent.compute_loss(y_hat, y)
            performance = agent.task_performance(y_hat, y)

            # per-agent loss
            self.log(
                f'{prefix}/loss_agent_{idx}',
                loss,
                on_step=False,
                on_epoch=True,
            )

            # per-agent task
            self.log(
                f'{prefix}/task_performance_agent_{idx}',
                performance,
                on_step=False,
                on_epoch=True,
            )

            total_loss += loss
            total_performance += performance

        # Average task across agents
        avg_performance = total_performance / len(self.agents)

        # Log the total
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
