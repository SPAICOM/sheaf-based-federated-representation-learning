"""Communication-efficient Sheaf-FRL: the agents communicate on a share of the epochs only.

``CESheafFRL`` is :class:`SheafFRL` whose epochs are of two kinds:

    collaborative epoch : a SheafFRL epoch. Per step, the pilot minibatch is exchanged and the
        sheaf penalty applied; at the epoch end each edge's anchors are re-selected
        (``_rebuild_edge_anchor_caches``) and the restriction maps refit.
    local epoch : no new pilot exchange. By default the sheaf penalty is dropped and every
        agent trains on its task loss alone. With ``local_reg=True`` a communication-free
        penalty pulls each agent's current representation of the anchor cache toward the
        rotated representations its neighbours sent at the last collaborative epoch
        (``self._frozen_edge_anchors``); :class:`SheafCFRL` builds on it.

**Schedule.** ``comm_percentage`` in ``(0, 100]`` sets ``K = max(1, round(comm_percentage / 100 * N))``
collaborative epochs out of the ``N = trainer.max_epochs`` (or ``total_epochs``) epochs, spread as
evenly as the integers allow and anchored to the end: epoch ``e`` communicates iff
``floor((e + 1) K / N) > floor(e K / N)``, so the last epoch always communicates and the maps
refit on the final encoders. ``0%`` is rejected (the maps would never be fit): the
zero-communication baseline is the ``non_cooperative`` orchestrator.

:class:`CEDiSheafFRL` puts this schedule on top of DiSheafFRL (the paper's ablation on the number
of communication epochs).
"""

from __future__ import annotations

import warnings

import torch

from src.orchestrators.sheaf_frl import SheafFRL


class CESheafFRL(SheafFRL):
    """Decoupled, communication-efficient Sheaf-FRL (uniformly-scheduled collaborative epochs)."""

    def __init__(
        self,
        *,
        comm_percentage: float = 100.0,
        total_epochs: int | None = None,
        local_reg: bool = False,
        **kwargs,
    ):
        if not 0.0 < float(comm_percentage) <= 100.0:
            raise ValueError(
                'comm_percentage must lie in (0, 100]. For 0% communication use the '
                'non_cooperative orchestrator: without a collaborative epoch the maps are '
                'never fit, and a 0% run would transport through the identity init.'
            )
        super().__init__(**kwargs)
        self._comm_percentage = float(comm_percentage)
        self._local_reg = bool(local_reg)
        # N (total epochs) and K (collaborative epochs): resolved now when total_epochs is
        # given, else from trainer.max_epochs at train start.
        self._total_epochs_cfg = None if total_epochs is None else int(total_epochs)
        self._n_epochs: int | None = self._total_epochs_cfg
        self._comm_epochs: int | None = (
            self._comm_epochs_for(self._n_epochs) if self._n_epochs else None
        )
        # Local-epoch minibatch rotation offsets per edge (anchor_selection='all' only, where
        # the frozen snapshot spans the full pilot pool).
        self._local_offsets: dict[tuple[int, int], int] = {}
        # Logged with the run (save_hyperparameters only sees SheafFRL's signature).
        self.hparams['comm_percentage'] = self._comm_percentage
        self.hparams['local_reg'] = self._local_reg
        if self._total_epochs_cfg is not None:
            self.hparams['total_epochs'] = self._total_epochs_cfg

    # ── Uniform communication schedule ─────────────────────────────────────────

    def _comm_epochs_for(self, n_epochs: int) -> int:
        """Exact number of collaborative epochs for an ``n_epochs`` horizon.

        ``max(1, ·)`` guarantees at least one collaborative epoch for any
        positive ``comm_percentage`` (a run with none would never fit its maps).
        """
        return max(1, round(self._comm_percentage / 100.0 * int(n_epochs)))

    def on_train_start(self) -> None:
        """Resolve the schedule horizon N from ``trainer.max_epochs``."""
        super().on_train_start()
        # ``total_epochs`` (the schedule horizon) is the actual training length.
        # Prefer the trainer's value; warn if an explicit total_epochs disagrees.
        n_trainer = getattr(self.trainer, 'max_epochs', None)
        if isinstance(n_trainer, int) and n_trainer > 0:
            if (
                self._total_epochs_cfg is not None
                and self._total_epochs_cfg != n_trainer
            ):
                warnings.warn(
                    f'CESheafFRL: total_epochs={self._total_epochs_cfg} '
                    f'disagrees with trainer.max_epochs={n_trainer}; using '
                    'trainer.max_epochs for the communication schedule.',
                    UserWarning,
                    stacklevel=2,
                )
            self._n_epochs = n_trainer
        if not self._n_epochs:
            raise RuntimeError(
                'CESheafFRL: cannot resolve total_epochs for the communication '
                'schedule — pass total_epochs or set trainer.max_epochs.'
            )
        self._comm_epochs = self._comm_epochs_for(self._n_epochs)

    def _is_collab_epoch(self, epoch: int) -> bool:
        """Tell whether ``epoch`` is a collaborative epoch of the uniform schedule.

        With ``K`` collaborative epochs over ``N`` total, epoch ``e`` communicates
        iff ``floor((e+1)·K/N) > floor(e·K/N)`` — exactly ``K`` epochs, spread as
        evenly as the integers allow, and anchored so the FINAL epoch (``N-1``) is
        always included: the maps refit on the last epoch's (most up-to-date)
        representations before testing.  (Epoch 0 communicates only at 100%.)
        Until the first collaborative epoch the maps sit at their init and, with
        ``local_reg=True``, there is no neighbour snapshot yet — those leading
        local epochs therefore train purely locally, see
        :meth:`_frozen_alignment_losses`.
        """
        n, k = self._n_epochs, self._comm_epochs
        if not n or not k:
            return False
        if k >= n:
            return True
        return (epoch + 1) * k // n > epoch * k // n

    # ── Overridden hooks ──────────────────────────────────────────────────────

    def _should_update_maps_at_epoch_end(self) -> bool:
        """Refresh anchor caches + maps at every collaborative epoch end; never on local epochs."""
        return self._is_collab_epoch(self.current_epoch)

    def _compute_alignment_losses(
        self, comm_weight: float, skip: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Local epoch (training only): either train purely locally (no penalty)
        # or apply the frozen-neighbour local regularizer.  Everything else
        # (collaborative epochs, validation, test) uses the classical both-live
        # penalty (embedding for CESheafFRL, compressed when SheafCFRL overrides
        # the geometry).
        if self.training:
            local = not self._is_collab_epoch(self.current_epoch)
            self.log(
                'train/local_epoch',
                1.0 if local else 0.0,
                on_step=False, on_epoch=True, prog_bar=False,
                add_dataloader_idx=False,
            )
            if local:
                if not self._local_reg:
                    return self._zero_losses()
                return self._frozen_alignment_losses(skip)
        return self._both_live_alignment_losses(comm_weight, skip)

    def _zero_losses(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a zero (sheaf_penalty, after_comm) pair: local epoch, ``local_reg=False``."""
        return (
            torch.zeros((), device=self.device),
            torch.zeros((), device=self.device),
        )

    def _frozen_alignment_losses(
        self, skip: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Local-epoch regularizer: pull each live node toward its frozen neighbours.

        With ``local_reg=True`` this is the sheaf penalty used during local
        epochs.  For every edge it contributes two terms, one per endpoint: each
        pairs one side's *fresh* re-encoding of the fixed anchor cache
        (:meth:`~src.orchestrators.sheaf_frl.SheafFRL._edge_live_anchors`)
        against the OTHER side's whitened snapshot as of the last collaborative
        refresh (``self._frozen_edge_anchors``, populated by
        ``_rebuild_edge_anchor_caches``), rotated through the edge's restriction
        map.  Summed over each node's incident edges this pulls its current
        representation toward the rotated, whitened representations its
        neighbours last communicated — a consensus that needs NO new exchange,
        since the snapshot is already locally known.

        With ``anchor_selection='all'`` the snapshot spans the full pilot pool,
        so each step re-encodes only a rotating ``pilot_batch_size``-sized row
        window of it (the frozen side is sliced with the same rows, keeping the
        pairing aligned); an epoch of local steps therefore cycles the whole
        snapshot.  Before the first collaborative epoch has populated a snapshot
        there is nothing to regularize toward, so these leading local epochs
        train purely locally (zero penalty, no communication).
        """
        sheaf_penalty = torch.tensor(0.0, device=self.device)
        after_comm = torch.tensor(0.0, device=self.device)
        if skip:
            return sheaf_penalty, after_comm

        if not self._frozen_edge_anchors:
            return sheaf_penalty, after_comm

        minibatch = self.hparams.anchor_selection == 'all'
        budget = self._row_budget() if minibatch else None

        for edge_key, V in self.stiefel_matrices.items():
            node_i, node_j = map(int, edge_key.split('_'))
            frozen = self._frozen_edge_anchors.get((node_i, node_j))
            if frozen is None:
                continue
            Z_i_frozen, _y_i_frozen, Z_j_frozen, _y_j_frozen = frozen

            rows = None
            n_frozen = int(Z_i_frozen.shape[0])
            if minibatch and budget and n_frozen > budget:
                off = self._local_offsets.get((node_i, node_j), 0)
                rows = (torch.arange(budget) + off) % n_frozen
                self._local_offsets[(node_i, node_j)] = (
                    off + budget
                ) % n_frozen

            live = self._edge_live_anchors(node_i, node_j, rows=rows)
            if live is None:
                continue
            Z_i_live, _y_i_live, Z_j_live, _y_j_live = live
            if rows is not None:
                frozen_rows = rows.to(Z_i_frozen.device)
                Z_i_frozen = Z_i_frozen[frozen_rows]
                Z_j_frozen = Z_j_frozen[frozen_rows]
            # Snapshots are stored on CPU; move only the consumed rows.
            Z_i_frozen = Z_i_frozen.to(self.device)
            Z_j_frozen = Z_j_frozen.to(self.device)

            # node_i live, pulled toward node_j's last-known (frozen) state.
            X_live, AY_frozen = self._edge_sides(Z_i_live, Z_j_frozen, V)
            diff_i = X_live - AY_frozen
            sheaf_penalty += self._edge_penalty_term(
                edge_key, diff_i, sides=(X_live, AY_frozen)
            )
            # node_j live, pulled toward node_i's last-known (frozen) state.
            X_frozen, AY_live = self._edge_sides(Z_i_frozen, Z_j_live, V)
            diff_j = X_frozen - AY_live
            sheaf_penalty += self._edge_penalty_term(
                edge_key, diff_j, sides=(X_frozen, AY_live)
            )

        return sheaf_penalty, after_comm
