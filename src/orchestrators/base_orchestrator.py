from abc import ABC, abstractmethod

import lightning as l
import torch
import torch.nn as nn
from hydra.utils import instantiate


class BaseOrchestrator(l.LightningModule, ABC):
    def __init__(
        self,
        agents: dict[int, nn.Module],
        neighbors: dict[int, set[int]],
        optimizer,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['agents', 'cfg'])

        assert len(agents) > 0, 'The "agents" dictionary must be not empty'

        # Store agents in a ModuleDict so Lightning tracks parameters
        self.agents = nn.ModuleDict(
            {str(idx): agent for idx, agent in agents.items()}
        )

    @abstractmethod
    def on_train_epoch_end(self):
        pass

    def forward(
        self,
        batch: dict[int, list[torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass for multiple agents.

        Args:
            batch : dict[int, torch.Tensor]
                Dictionary mapping agent names to their input tensors.
                   Example:
                   {
                       "agent1": x1,
                       "agent2": x2,
                   }

        Returns:
            outputs : dict[str, torch.Tensor]
                Dictionary mapping agent names to their latent representations.
        """
        if isinstance(batch, tuple):
            batch = batch[0]

        outputs = {}

        for idx, agent in self.agents.items():
            key = str(idx) if str(idx) in batch else idx
            x, y = batch[key]  # (embedding, label)

            y_hat = agent(x)

            outputs[idx] = (y_hat, y)

        return outputs

    @abstractmethod
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
        pass

    def training_step(
        self,
        batch: dict[int, list[torch.Tensor]],
        batch_idx: int,
    ) -> torch.Tensor:
        """The training step.

        Args:
            batch : dict[int, list[torch.Tensor]]
                The current batch.
            batch_idx : int
                The batch index.

        Returns:
            loss : torch.Tensor
                The epoch loss.
        """
        _, loss = self._shared_eval(
            batch=batch,
            batch_idx=batch_idx,
            prefix='train',
        )
        return loss

    def test_step(
        self,
        batch: dict[int, list[torch.Tensor]],
        batch_idx: int,
    ) -> None:
        """The test step.

        Args:
            batch : dict[int, list[torch.Tensor]]
                The current batch.
            batch_idx : int
                The batch index.

        Returns:
            None
        """
        _ = self._shared_eval(
            batch=batch,
            batch_idx=batch_idx,
            prefix='test',
        )
        return None

    def validation_step(
        self,
        batch: dict[int, list[torch.Tensor]],
        batch_idx: int,
    ) -> dict[int, torch.Tensor]:
        """The validation step.

        Args:
            batch : dict[int, list[torch.Tensor]]
                The current batch.
            batch_idx : int
                The batch index.

        Returns:
            output : dict[int, torch.Tensor]
                The output of the network.
        """
        output, _ = self._shared_eval(
            batch=batch,
            batch_idx=batch_idx,
            prefix='validation',
        )
        return output

    def predict_step(
        self,
        batch: dict[int, list[torch.Tensor]],
        batch_idx: int,
    ) -> dict[int, torch.Tensor]:
        """The predict step.

        Args:
            batch : dict[int, list[torch.Tensor]]
                The current batch.
            batch_idx : int
                The batch index.
            dataloader_idx : int
                The dataloader idx.

        Returns:
            dict[int, torch.Tensor]
                The output of the network.
        """
        return self(batch)

    def configure_optimizers(self) -> dict[str, object]:
        """Configure the optimizer used for training.

        Uses an optimizer with the learning rate defined in
        ``self.hparams.lr``.

        Returns:
            dict[str, object]: A dictionary containing the optimizer used by
            the training loop. The dictionary has the following key:

            - "optimizer": The instantiated optimizer.
        """
        optimizer = instantiate(
            self.hparams.optimizer,
            params=self.parameters(),
        )
        return {
            'optimizer': optimizer,
        }


if __name__ == '__main__':
    pass
