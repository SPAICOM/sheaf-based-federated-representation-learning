"""FedGH: heterogeneous FL with a generalized global header (Yi et al., MM '23).

See refs/notes/yi2023fedgh.md (Alg. 1, eqs. 3-6).  Every agent keeps a private
extractor and all share one header, the last linear layer.  One round:

1. the server broadcasts the header, which replaces the local one (eq. 5);
2. each agent trains its whole model on its own data for E epochs (eq. 6);
3. each agent uploads one averaged representation per held class (LAR, eq. 3);
4. the server takes one CE step on the header per LAR, sequentially over
   agents and unweighted by sample count (eq. 4).

Choices the paper leaves open or that differ from it:

- The extractor is everything before the decoder's last ``nn.Linear``, so with
  ``decoder_hidden_dims: [256]`` the representation is that hidden layer
  (d_R=256) and the agent architecture stays identical to the baselines; only
  with ``model.decoder_hidden_dims=[]`` is the representation the latent z.
- LARs come from un-augmented images in eval mode (the paper does not say):
  the repo's convention for offline encoding; it leaves BatchNorm stats alone.
- Every agent joins every round (C=1): the setups here have no client sampling.
- The header is broadcast at round *start*, as in Alg. 1, so the model scored
  at the end is omega_k^{T-1}: extractor + header after local training.

Metrics are NonCooperativeLearning's: cross-agent accuracy goes through the
same post-hoc pilot alignment as every baseline, which FedGH does not replace
(it shares a header, not a latent space).  Its training traffic is charged
into the 'train' bucket on top of the one-off post-hoc pilot exchange.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.orchestrators.non_cooperative import NonCooperativeLearning


class FedGH(NonCooperativeLearning):
    """Private extractors + server-trained shared header.

    Parameters
    ----------
    header_lr : float, optional
        Server SGD step size eta_theta in eq. 4 (default: 0.01, as the paper).
    local_epochs : int, optional
        Local epochs E per round (default: 1).
    **kwargs
        Forwarded to :class:`NonCooperativeLearning`.
    """

    def __init__(
        self,
        *,
        header_lr: float = 0.01,
        local_epochs: int = 1,
        **kwargs,
    ):
        if float(header_lr) <= 0.0:
            raise ValueError(f'header_lr must be > 0, got {header_lr}')
        if int(local_epochs) < 1:
            raise ValueError(
                f'local_epochs must be at least 1, got {local_epochs}'
            )
        super().__init__(**kwargs)
        self._header_lr = float(header_lr)
        self._local_epochs = int(local_epochs)
        self.hparams['header_lr'] = self._header_lr
        self.hparams['local_epochs'] = self._local_epochs

        heads = [self._shared_head(agent) for agent in self.agents.values()]
        if len({(h.in_features, h.out_features) for h in heads}) > 1:
            raise ValueError(
                'FedGH needs one header shape (same d_R and classes) across '
                f'agents, got {[(h.in_features, h.out_features) for h in heads]}'
            )
        # theta.  Stays a registered module for device moves and checkpoints.
        # It is in self.parameters() but its .grad is never set (eq. 4 uses
        # autograd.grad), so the agents' optimizer skips it.
        self.global_header = copy.deepcopy(heads[0])

    @staticmethod
    def _shared_head(agent: nn.Module) -> nn.Linear:
        """The header of ``agent``: the last linear layer of its decoder."""
        if getattr(agent, 'task_type', 'classification') != 'classification':
            raise TypeError('FedGH requires classification agents.')
        linears = [
            m for m in agent.decoder.modules() if isinstance(m, nn.Linear)
        ]
        if not linears:
            raise TypeError('FedGH requires a decoder with a linear layer.')
        return linears[-1]

    @torch.no_grad()
    def _representations(
        self, agent: nn.Module, Z: torch.Tensor
    ) -> torch.Tensor:
        """Map latents to the header's input, in eval mode (BatchNorm frozen)."""
        reps: list[torch.Tensor] = []
        handle = self._shared_head(agent).register_forward_pre_hook(
            lambda _module, args: reps.append(args[0].detach().cpu())
        )
        was_training = agent.training
        agent.eval()
        try:
            for chunk in Z.split(1024):
                agent.decoder(chunk.to(self.device))
        finally:
            handle.remove()
            agent.train(was_training)
        return torch.cat(reps, dim=0)

    @torch.no_grad()
    def _compute_lars(
        self, dm
    ) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
        """Per-agent LARs (eq. 3) as ``(reps [S, d_R], classes [S])``."""
        datasets = list(getattr(dm, 'train_datasets', {}).values())
        saved = [getattr(d, 'transform', None) for d in datasets]
        for dataset, transform in zip(datasets, saved):
            if transform is not None:
                dataset.transform = None
        try:
            encoded = self._encode_full_training_set(dm)
        finally:
            for dataset, transform in zip(datasets, saved):
                if transform is not None:
                    dataset.transform = transform
        if not encoded:
            raise RuntimeError('FedGH needs dm.train_datasets to compute LARs')

        lars = {}
        for idx, (Z, y, _sample_ids) in encoded.items():
            reps = self._representations(self.agents[str(idx)], Z)
            classes = y.unique()
            lars[idx] = (
                torch.stack([reps[y == c].mean(dim=0) for c in classes]),
                classes,
            )
        return lars

    def _update_global_header(
        self, lars: dict[int, tuple[torch.Tensor, torch.Tensor]]
    ) -> float:
        """Eq. 4: one SGD step per LAR, in agent-id order; returns the mean loss."""
        params = list(self.global_header.parameters())
        losses: list[float] = []
        with torch.enable_grad():
            for idx in sorted(lars):
                reps, classes = lars[idx]
                for rep, target in zip(
                    reps.to(self.device), classes.to(self.device)
                ):
                    loss = F.cross_entropy(
                        self.global_header(rep.unsqueeze(0)),
                        target.unsqueeze(0),
                    )
                    grads = torch.autograd.grad(loss, params)
                    with torch.no_grad():
                        for param, grad in zip(params, grads):
                            param.sub_(self._header_lr * grad)
                    losses.append(float(loss.detach()))
        return sum(losses) / len(losses)

    def _server_round(self, dm) -> None:
        """Agents upload LARs, the server updates the header (steps 3-4)."""
        lars = self._compute_lars(dm)
        for reps, classes in lars.values():
            # Uplink (S + S*r) * 32 bits, p. 8689.
            self._record_communication((reps, classes.to(torch.int32)))
        self._record_communication_round()
        self.log(
            'train/server_header_loss',
            self._update_global_header(lars),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

    def _broadcast_header(self) -> None:
        """Replace every local header with the global one (step 1, eq. 5)."""
        state = self.global_header.state_dict()
        for agent in self.agents.values():
            self._shared_head(agent).load_state_dict(state)
        self._record_communication(state, n_transmissions=len(self.agents))

    def on_train_epoch_start(self) -> None:
        if int(self.current_epoch) % self._local_epochs == 0:
            self._broadcast_header()

    def on_train_epoch_end(self) -> None:
        # Before the parent's logging so the epoch's cumulative comm includes
        # this round.
        if (int(self.current_epoch) + 1) % self._local_epochs == 0:
            self._server_round(self.trainer.datamodule)
        return super().on_train_epoch_end()
