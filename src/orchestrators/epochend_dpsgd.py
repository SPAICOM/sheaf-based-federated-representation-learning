"""Epoch-end D-PSGD orchestrator.

This module implements a periodic variant of D-PSGD that uses the exact same
parameter-space Metropolis-Hastings mixing rule as :mod:`dpsgd`, but applies
that mixing once at the end of each training epoch instead of before every
optimizer step.
"""

from typing import Any

import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from src.orchestrators.dpsgd import DPSGD


class EpochEndDPSGD(DPSGD):
    """D-PSGD with epoch-end parameter mixing instead of per-step mixing.

    Local SGD updates proceed normally throughout the epoch. At
    ``on_train_epoch_end``, each agent's trainable parameter vector is mixed
    with its neighbors using the same Metropolis-Hastings weights as
    :class:`src.orchestrators.dpsgd.DPSGD`.

    Relative to stepwise D-PSGD, the only intended behavioral change is the
    communication cadence: one neighborhood-mixing round per epoch rather than
    one round per optimization step.
    """

    def on_before_optimizer_step(self, optimizer: Any) -> None:
        """Disable per-step mixing so local optimizer steps remain untouched."""
        return None

    @torch.no_grad()
    def on_train_epoch_end(self) -> None:
        """Apply one D-PSGD mixing step to trainable parameters at epoch end."""
        total_transmissions = sum(
            len(self.hparams.neighbors.get(int(idx_str), set()))
            for idx_str in self.agents
        )
        if total_transmissions > 0:
            self._record_communication_round(prefix='train')

        agent_vectors = {}
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

        mixed_vectors = {}
        for idx_str in self.agents:
            i = int(idx_str)
            mixed_vec = self.mixing_weights[(i, i)] * agent_vectors[i]

            for j in self.hparams.neighbors.get(i, set()):
                mixed_vec += self.mixing_weights[(i, j)] * agent_vectors[j]

            mixed_vectors[i] = mixed_vec

        for idx_str, agent in self.agents.items():
            idx = int(idx_str)
            trainable_params = [
                p for p in agent.parameters() if p.requires_grad
            ]
            vector_to_parameters(mixed_vectors[idx], trainable_params)

        self._finalize_train_epoch_communication()
