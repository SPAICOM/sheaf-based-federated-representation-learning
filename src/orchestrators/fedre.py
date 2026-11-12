"""FedRE: representation entanglement with a server-trained shared classifier.

Each round every client uploads ONE random convex combination of its class
prototypes plus the matching soft label; the server fits a shared classifier on
those K pairs and broadcasts it, replacing the local decoders.

See refs/notes/yao2026fedre.md, eq. 1-3 and Alg. 1. Where the paper is silent
(appendices missing from our PDF) this follows the official code,
``HtFLlib/system/flcore/{servers/serverre,clients/clientre}.py``.
"""

import copy
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.communication.alignment_mixin import (
    VALID_ALIGNMENT_METHODS,
    PostTrainingAlignmentMixin,
)
from src.communication.projectors import (
    common_dim,
    latent_dim,
    retarget_input,
)
from src.orchestrators.base_orchestrator import BaseOrchestrator


class FedRE(PostTrainingAlignmentMixin, BaseOrchestrator):
    """FedRE with the Random Average Prototype (RAP) entanglement.

    One round is one local epoch (eq. 1, plain task loss through the Trainer),
    then ``_server_round`` at epoch end: entangle (eq. 2), train the server
    head (eq. 3), broadcast. Evaluation therefore always sees extractor +
    global head, as in the official code, which evaluates right after
    ``send_parameters``.

    Pilots play no role: FedRE learns from private labels only. With
    ``alignment_method=None`` the message is the identity, so cross-agent
    accuracy is sender extractor + the shared head.

    Parameters
    ----------
    agents : dict[int, nn.Module]
        Classifiers exposing ``encode``/``decoder``; decoders must match.
    neighbors : dict[int, set[int]]
        Used only by the communication evaluation; the server sees every agent.
    optimizer : hydra config
        Local optimizer.
    server_lr : float, optional
        Server SGD learning rate (official ``-slr``, default 0.01).
    head_batch_size : int, optional
        Server batch size (official ``-hbs``, default 10); one server epoch per
        round, so K <= head_batch_size means a single SGD step.
    rm_dim : int | None, optional
        Representation mapping (RM, the paper's FC variant): each agent's
        decoder becomes ``Sequential(RM_i, head)``, with
        ``RM_i = Linear(d_i -> rm_dim)`` private and trained by the local CE,
        and ``head`` (the agent's decoder retargeted to ``rm_dim`` inputs) the
        shared classifier. ``None`` (default) uses the largest latent dim in
        the network, and no RM when every agent has the same dim (the decoder
        is shared as it is). Prototypes are taken after RM. The latent that
        alignment and the comm metric see is still the raw ``z_i``, so this
        needs a post-hoc ``alignment_method``.
    alignment_method : str | None, optional
        ``None`` is native FedRE. A mixin method adds post-hoc alignment on
        pilots at evaluation time; it is not part of the paper.
    """

    def __init__(
        self,
        agents: dict[int, nn.Module],
        neighbors: dict[int, set[int]],
        optimizer: Any,
        server_lr: float = 0.01,
        head_batch_size: int = 10,
        rm_dim: int | None = None,
        alignment_method: str | None = None,
        **kwargs,
    ):
        super().__init__(
            agents=agents,
            neighbors=neighbors,
            optimizer=optimizer,
            **kwargs,
        )
        if (
            alignment_method is not None
            and str(alignment_method) not in VALID_ALIGNMENT_METHODS
        ):
            raise ValueError(
                f"Unknown alignment_method '{alignment_method}'. "
                f'Valid options: {VALID_ALIGNMENT_METHODS}'
            )
        if float(server_lr) <= 0.0:
            raise ValueError('server_lr must be positive')
        if int(head_batch_size) < 1:
            raise ValueError('head_batch_size must be at least 1')
        self.save_hyperparameters(ignore=['agents'])
        self._rm_dim = common_dim(self.agents, rm_dim)
        if self._rm_dim is not None:
            self._add_representation_maps(self._rm_dim, alignment_method)
        self._validate_shared_decoders()

    def _add_representation_maps(
        self, rm_dim: int, alignment_method: str | None
    ) -> None:
        """Replace each decoder with ``Sequential(RM_i, head)`` (see ``rm_dim``)."""
        if alignment_method is None:
            # Identity would feed z_i to RM_j, a map trained for another agent.
            raise ValueError(
                'Representation maps (rm_dim, or different latent dims) need a '
                'post-hoc alignment_method: the comm metric sends raw latents, '
                'which identity cannot carry across RMs.'
            )
        for key, agent in self.agents.items():
            d = latent_dim(agent)
            if d is None or not hasattr(agent, '_decoder'):
                raise TypeError(
                    f'Agent {key}: rm_dim needs a decoder that starts with a '
                    'linear layer, stored as agent._decoder.'
                )
            agent._decoder = nn.Sequential(
                nn.Linear(d, rm_dim),
                retarget_input(agent.decoder, rm_dim),
            )

    def _shared_decoder(self, agent: nn.Module) -> nn.Module:
        """Return the part of the decoder the server owns: all of it, or the head after RM."""
        if self._rm_dim is None:
            return agent.decoder
        return agent.decoder[1]

    def _representation_map(self, idx: int, Z: torch.Tensor) -> torch.Tensor:
        """Agent ``idx``'s RM applied to raw latents (identity without RM)."""
        if self._rm_dim is None:
            return Z
        rm = self.agents[str(idx)].decoder[0]
        return rm(Z.to(rm.weight.device))

    def _validate_shared_decoders(self) -> None:
        """Every agent must be able to load the single global classifier."""
        shapes = [
            {
                k: v.shape
                for k, v in self._shared_decoder(agent).state_dict().items()
            }
            for agent in self.agents.values()
        ]
        if any(s != shapes[0] for s in shapes):
            raise ValueError('FedRE requires identical decoder architectures.')

    def _broadcast_head(self) -> None:
        """Replace every local decoder with the server head (Alg. 1, line 3)."""
        state = self._server_head.state_dict()
        for agent in self.agents.values():
            self._shared_decoder(agent).load_state_dict(state)
        self._record_communication(state, n_transmissions=len(self.agents))

    def on_train_start(self) -> None:
        """Start from one shared head: agent 0's, as in the official server."""
        super().on_train_start()
        first = min(self.agents, key=int)
        # object.__setattr__ keeps the head out of self.parameters(): assigned
        # normally, the local optimizer would step it using the .grad left by
        # the server update.
        object.__setattr__(
            self,
            '_server_head',
            copy.deepcopy(self._shared_decoder(self.agents[first])),
        )
        self._broadcast_head()

    @staticmethod
    def _entangle(
        Z: torch.Tensor,
        y: torch.Tensor,
        num_classes: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build (r~, y~): eq. 2 over class prototypes (RAP), fresh weights.

        Weights are drawn over the classes present at the client, so y~ puts
        mass on those classes only and sums to 1.
        """
        classes = y.unique()
        protos = torch.stack([Z[y == c].mean(dim=0) for c in classes])
        weights = torch.rand(len(classes), device=Z.device)
        weights = weights / weights.sum()
        soft_label = torch.zeros(num_classes, device=Z.device)
        soft_label[classes] = weights
        return weights @ protos, soft_label

    def _train_server_head(
        self,
        reps: torch.Tensor,
        soft_labels: torch.Tensor,
    ) -> torch.Tensor:
        """One server epoch of SGD on the K uploaded pairs (eq. 3, summed)."""
        head = self._server_head
        head.train()
        optimizer = torch.optim.SGD(
            head.parameters(), lr=float(self.hparams.server_lr)
        )
        batches = torch.randperm(len(reps), device=reps.device).split(
            int(self.hparams.head_batch_size)
        )
        total = reps.new_zeros(())
        with torch.enable_grad():
            for rows in batches:
                log_probs = F.log_softmax(head(reps[rows]), dim=1)
                loss = -(soft_labels[rows] * log_probs).sum()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total = total + loss.detach()
        return total / len(batches)

    def _server_round(self, dm) -> torch.Tensor:
        """Entangle, train the server head, broadcast. Returns the server loss.

        Prototypes use the whole local training set in eval mode, as the
        official ``collect_protos``.
        """
        with torch.no_grad():
            # Workers: per-sample input corruption is too slow in-process for
            # a pass over the whole training set every round.
            encoded = self._encode_full_training_set(
                dm, num_workers=getattr(dm, 'num_workers', 0)
            )
        missing = {int(k) for k in self.agents} - set(encoded)
        if missing:
            raise ValueError(f'No training data for agents {sorted(missing)}.')

        num_classes = next(iter(self.agents.values())).accuracy.num_classes
        reps, soft_labels = [], []
        for idx, (Z, y, _) in encoded.items():
            with torch.no_grad():
                Z = self._representation_map(idx, Z).cpu()
            rep, soft_label = self._entangle(Z, y, num_classes)
            self._record_communication(
                torch.cat([rep, soft_label]), n_transmissions=1
            )
            reps.append(rep)
            soft_labels.append(soft_label)

        loss = self._train_server_head(
            torch.stack(reps).to(self.device),
            torch.stack(soft_labels).to(self.device),
        )
        self._broadcast_head()
        self._record_communication_round(prefix='train')
        return loss

    def on_train_epoch_end(self) -> None:
        server_loss = self._server_round(self.trainer.datamodule)
        self.log('train/server_head_loss', server_loss, on_epoch=True)
        self._finalize_train_epoch_communication()
        self._log_train_comm_task_perf()

    def _shared_eval(
        self,
        batch: dict[int, list[torch.Tensor]],
        batch_idx: int,
        prefix: str,
    ) -> tuple[dict[str, Any], torch.Tensor]:
        """Local CE on each agent's own batch (eq. 1)."""
        outputs = self(batch)

        agent_losses = {}
        agent_performances = {}
        for idx, agent in self.agents.items():
            y_hat, y = outputs[idx][:2]
            agent_losses[int(idx)] = agent.compute_loss(y_hat, y)
            agent_performances[int(idx)] = agent.task_performance(y_hat, y)

        total_loss, _ = self._log_shared_metrics(
            prefix=prefix,
            agent_losses=agent_losses,
            agent_performances=agent_performances,
            batch_size=self._resolve_batch_size(batch),
            agent_sample_counts=self._resolve_agent_sample_counts(batch),
            skip_task_performance=(prefix == 'test'),
        )
        return outputs, total_loss

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
