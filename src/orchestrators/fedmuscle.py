"""
FedMuscle: Federated Multi-Task Learning with Muscle Loss.

This module implements the FedMuscle algorithm from the paper:
"Toward Enhancing Representation Learning in Federated Multi-Task Settings"
(https://arxiv.org/pdf/2602.01626)

Muscle (Multi-task/modal Systematic Contrastive Learning) loss is a novel
contrastive learning objective that simultaneously aligns representations from
all participating models. Unlike pairwise contrastive methods, Muscle loss
captures dependencies across all tasks by maximizing mutual information.

Reference: arXiv:2602.01626 (ICLR 2026)
"""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.orchestrators.base_orchestrator import BaseOrchestrator


class FedMuscle(BaseOrchestrator):
    """Federated learning with Muscle loss for representation alignment.

    Implements the FedMuscle algorithm where:
    1. Each communication round consists of E local epochs (task loss) and
       T CL epochs (Muscle loss) for representation alignment.
    2. Muscle loss compares each agent's representations on shared pilot data
       against the network average representation.
    3. No weight averaging is performed - representation alignment only.

    Parameters
    ----------
    agents : dict[int, nn.Module]
        Dictionary mapping agent indices to their model instances.
    neighbors : dict[int, set[int]]
        Dictionary mapping each agent index to the set of its neighbor indices.
    optimizer : hydra config
        Optimizer configuration for training.
    max_lmb : float, optional
        Maximum weight coefficient for the Muscle loss (default: 0.1).
    lambda_schedule : str or None, optional
        Scheduling strategy: ``None`` (constant), ``'cosine'``, or ``'exp'``.
    temperature : float, optional
        Temperature parameter for contrastive loss (default: 0.1).
    E : int, optional
        Number of local epochs per communication round (default: 4).
    T : int, optional
        Number of CL alignment epochs per communication round (default: 1).
    alignment_mode : str, optional
        Loss mode: "contrastive" (default) or "alignment".

    Notes
    -----
    - Each agent must expose `encode(x)` method returning latent representation.
    - Each agent must expose `decoder(z)` method returning predictions.
    - Pilot data is expected to be available via `pilot_{i}` keys in batch.
    - This implementation does NOT perform weight averaging - only representation
      alignment via Muscle loss as per the heterogeneous model setup in the paper.
    """

    def __init__(
        self,
        agents: dict[int, nn.Module],
        neighbors: dict[int, set[int]],
        optimizer: Any,
        max_lmb: float = 0.1,
        temperature: float = 0.1,
        E: int = 4,
        T: int = 1,
        alignment_mode: str = 'contrastive',
        lambda_schedule: str | None = None,
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
        self._cl_epoch_count = 0
        self._local_epoch_count = 0
        self._round_complete = False
        self._in_cl_epoch = False

    def _validate_agents(self) -> None:
        """Validate that all agents have encode/decoder interface."""
        agents = list(self.agents.values())
        if len(agents) <= 1:
            return

        ref = agents[0]
        if not hasattr(ref, 'encode') or not hasattr(ref, 'decoder'):
            raise TypeError(
                'FedMuscle requires agents with encode/decoder methods.'
            )

    def _compute_alignment_loss(
        self,
        representations: dict[int, torch.Tensor],
    ) -> torch.Tensor:
        """Compute alignment loss: MSE between each representation and network average.

        L_alignment = (1/N) * sum_i ||z_i - z_bar||^2

        where z_bar = (1/N) * sum_k z_k is the network average.

        Parameters
        ----------
        representations : dict[int, torch.Tensor]
            Dictionary mapping agent indices to their representation tensors.

        Returns
        -------
        torch.Tensor
            Scalar alignment loss.
        """
        agent_indices = list(representations.keys())
        if len(agent_indices) <= 1:
            return torch.tensor(0.0, device=self.device)

        reps = torch.stack([representations[i] for i in agent_indices], dim=0)
        avg_representation = reps.mean(dim=0, keepdim=True)
        squared_diffs = (reps - avg_representation) ** 2
        loss = squared_diffs.sum(dim=1).mean()
        return loss

    def _compute_contrastive_loss(
        self,
        representations: dict[int, torch.Tensor],
    ) -> torch.Tensor:
        """Compute contrastive Muscle loss.

        For each sample i from each agent k:
        - Positive: average representation over ALL OTHER agents for sample i
        - Negatives: average representations for all OTHER samples j != i

        Parameters
        ----------
        representations : dict[int, torch.Tensor]
            Dictionary mapping agent indices to their representation tensors.

        Returns
        -------
        torch.Tensor
            Scalar contrastive loss.
        """
        agent_indices = list(representations.keys())
        n_agents = len(agent_indices)

        if n_agents <= 1:
            return torch.tensor(0.0, device=self.device)

        tau = self.hparams.temperature

        all_reps = [representations[i] for i in agent_indices]
        batch_size = all_reps[0].shape[0]

        for reps in all_reps:
            if reps.shape[0] != batch_size:
                return torch.tensor(0.0, device=self.device)

        all_reps_stacked = torch.stack(all_reps, dim=0)
        sample_means = all_reps_stacked.mean(dim=0)

        total_loss = torch.tensor(0.0, device=self.device)

        for k_idx in range(n_agents):
            z_k = all_reps[k_idx]

            sample_losses = []

            for i in range(batch_size):
                z_k_i = z_k[i]

                pos_sim = torch.tensor(0.0, device=self.device)
                other_agent_reps = [
                    all_reps[j][i] for j in range(n_agents) if j != k_idx
                ]
                if other_agent_reps:
                    other_mean = torch.stack(other_agent_reps).mean(dim=0)
                    pos_sim = F.cosine_similarity(
                        z_k_i.unsqueeze(0), other_mean.unsqueeze(0), dim=1
                    )

                neg_sim_sum = torch.tensor(0.0, device=self.device)
                neg_count = 0
                for j in range(batch_size):
                    if j != i:
                        neg_mean = sample_means[j]
                        neg_sim = F.cosine_similarity(
                            z_k_i.unsqueeze(0), neg_mean.unsqueeze(0), dim=1
                        )
                        neg_sim_sum = neg_sim_sum + neg_sim
                        neg_count += 1

                if neg_count > 0:
                    neg_sim = neg_sim_sum / neg_count
                else:
                    neg_sim = torch.tensor(-1.0, device=self.device)

                pos_exp = torch.exp(pos_sim / tau)
                neg_exp = torch.exp(neg_sim / tau)

                denom = pos_exp + neg_exp
                sample_loss = -torch.log(pos_exp / denom)
                sample_losses.append(sample_loss)

            if sample_losses:
                agent_loss = torch.stack(sample_losses).mean()
                total_loss = total_loss + agent_loss

        total_loss = total_loss / n_agents

        return total_loss

    def _compute_muscle_loss(
        self,
        representations: dict[int, torch.Tensor],
    ) -> torch.Tensor:
        """Compute Muscle loss based on configured alignment mode.

        Parameters
        ----------
        representations : dict[int, torch.Tensor]
            Dictionary mapping agent indices to their representation tensors.

        Returns
        -------
        torch.Tensor
            Scalar muscle loss.
        """
        mode = self.hparams.alignment_mode

        if mode == 'contrastive':
            return self._compute_contrastive_loss(representations)
        elif mode == 'alignment':
            return self._compute_alignment_loss(representations)
        else:
            raise ValueError(f'Unknown alignment_mode: {mode}')

    def _extract_pilot_batch(
        self,
        batch: dict[int, list[torch.Tensor]],
    ) -> dict[int, torch.Tensor]:
        """Extract pilot data for each agent from the batch.

        Parameters
        ----------
        batch : dict[int, list[torch.Tensor]]
            Dictionary mapping agent indices to (input, label) pairs.
            Pilot data is accessed via 'pilot_{i}' or 'global_pilot_{i}' keys.

        Returns
        -------
        dict[int, torch.Tensor]
            Dictionary mapping agent indices to pilot input tensors.
        """
        pilot_data = {}

        for idx_str in self.agents:
            idx = int(idx_str)
            x_pilot = None

            for key in [f'pilot_{idx}', f'global_pilot_{idx}']:
                if key in batch:
                    x_pilot = batch[key][0]
                    break

            if x_pilot is None:
                if idx_str in batch:
                    x_pilot = batch[idx_str][0]
                elif idx in batch:
                    x_pilot = batch[idx][0]

            if x_pilot is not None:
                pilot_data[idx] = x_pilot

        return pilot_data

    def _get_batch_representations(
        self,
        batch: dict[int, list[torch.Tensor]],
    ) -> dict[int, torch.Tensor]:
        """Extract representations from local batch data.

        Parameters
        ----------
        batch : dict[int, list[torch.Tensor]]
            Dictionary mapping agent indices to (input, label) pairs.

        Returns
        -------
        dict[int, torch.Tensor]
            Dictionary mapping agent indices to representation tensors.
        """
        representations = {}

        for idx_str, agent in self.agents.items():
            idx = int(idx_str)
            key = idx if idx in batch else idx_str

            if key in batch:
                x, _ = batch[key]
                representations[idx] = agent.encode(x)

        return representations

    def on_train_start(self) -> None:
        """Initialize counters at the start of training."""
        super().on_train_start()
        self._cl_epoch_count = 0
        self._local_epoch_count = 0
        self._round_complete = False
        self._in_cl_epoch = False

    def on_train_epoch_end(self) -> None:
        E = int(self.hparams.E)
        T = int(self.hparams.T)
        round_length = E + T
        epoch_in_round = int(self.current_epoch) % round_length

        if epoch_in_round < E:
            self._local_epoch_count += 1
            self._in_cl_epoch = False
        else:
            self._cl_epoch_count += 1
            self._in_cl_epoch = True

        if epoch_in_round == (round_length - 1):
            self._round_complete = True
            self._local_epoch_count = 0
            self._cl_epoch_count = 0

        self._finalize_train_epoch_communication()

    def _shared_eval(
        self,
        batch: dict[int, list[torch.Tensor]],
        batch_idx: int,
        prefix: str,
    ) -> tuple[dict[str, Any], torch.Tensor]:
        """Compute losses and metrics for train/validation/test steps.

        Parameters
        ----------
        batch : dict[int, list[torch.Tensor]]
            Dictionary mapping agent indices to (input, label) pairs.
            Pilot data expected via 'pilot_{i}' or 'global_pilot_{i}' keys.
        batch_idx : int
            Index of the current batch.
        prefix : str
            Prefix for logging (e.g., 'train', 'validation', 'test').

        Returns
        -------
        tuple[dict, torch.Tensor]
            Tuple of (outputs, total_loss) where outputs maps agent indices to
            (prediction, label) pairs and total_loss includes task and muscle loss.
        """
        if isinstance(batch, tuple):
            batch = batch[0]

        outputs = {}
        agent_losses = {}
        agent_performances = {}

        def _resolve_key(i: int):
            str_key = str(i)
            return str_key if str_key in batch else i

        for idx_str, agent in self.agents.items():
            idx = int(idx_str)
            x, y = batch[_resolve_key(idx)]

            z = agent.encode(x)
            y_hat = agent.decoder(z)

            outputs[idx] = (y_hat.detach(), y)

            task_loss = agent.compute_loss(y_hat, y)
            task_performance = agent.task_performance(y_hat, y)

            agent_losses[idx] = task_loss
            agent_performances[idx] = task_performance

        pilot_data = self._extract_pilot_batch(batch)
        muscle_loss: torch.Tensor = torch.tensor(0.0, device=self.device)

        effective_lambda = 0.0

        # Compute whether the current epoch is a CL epoch inline, so the check
        # is correct for all batches in the epoch (on_train_epoch_end runs too
        # late to set a flag that is accurate within the same epoch).
        E = int(self.hparams.E)
        T = int(self.hparams.T)
        in_cl_epoch = (int(self.current_epoch) % (E + T)) >= E

        if pilot_data and prefix == 'train' and in_cl_epoch:
            pilot_representations = {}
            for idx_str, agent in self.agents.items():
                idx = int(idx_str)
                if idx in pilot_data:
                    x_pilot = pilot_data[idx]
                    pilot_representations[idx] = agent.encode(x_pilot)

            if pilot_representations:
                muscle_loss = self._compute_muscle_loss(pilot_representations)
                effective_lambda = self._effective_lambda_reg()

                for idx in pilot_representations:
                    self._record_communication(
                        pilot_representations[idx],
                        n_transmissions=len(self.agents) - 1,
                    )
                self._record_communication_round(n_rounds=1, prefix='train')

        total_task_loss = torch.stack(list(agent_losses.values())).sum()
        total_loss = total_task_loss + effective_lambda * muscle_loss

        self._log_shared_metrics(
            prefix=prefix,
            agent_losses=agent_losses,
            agent_performances=agent_performances,
            batch_size=self._resolve_batch_size(batch),
            agent_sample_counts=self._resolve_agent_sample_counts(batch),
            total_loss=total_loss,
            extra_metrics={
                f'{prefix}/muscle_loss': muscle_loss,
                f'{prefix}/effective_lambda': torch.tensor(
                    effective_lambda, device=self.device
                ),
                f'{prefix}/local_epoch_count': torch.tensor(
                    self._local_epoch_count, device=self.device
                ),
                f'{prefix}/cl_epoch_count': torch.tensor(
                    self._cl_epoch_count, device=self.device
                ),
            },
            prog_bar=False,
            per_agent_loss_name='task_loss',
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
