"""Communication-Efficient Sheaf-FRL orchestrator (uniformly-scheduled communication).

``CESheafFRL`` is :class:`SheafFRL` with a decoupled training schedule that trades
some pilot communication for cheaper rounds.  Each of the ``total_epochs`` training
epochs is one of two kinds:

    Collaborative epoch : *classical* SheafFRL — the alignment maps refit, each
        edge's anchor set is re-selected (``_rebuild_edge_anchor_caches``) at the
        epoch end, and a fresh pilot exchange is recorded.  (≡ SheafFRL.)
    Local epoch : AVOIDS any new pilot exchange.  By default the sheaf
        regularization simply *disappears* here and each node trains its task
        loss alone.  With ``local_reg=True`` a communication-free local
        regularizer is instead applied: each node's current (freshly re-encoded)
        representation is pulled toward the rotated, whitened representations its
        neighbours sent during the last collaborative epoch
        (``self._frozen_edge_anchors``) — a consensus toward the last-known
        neighbour state that needs no new communication.

**Schedule.** The user sets a single knob, ``comm_percentage`` (in ``(0, 100]``),
together with the training horizon ``total_epochs`` (``None`` → read from
``trainer.max_epochs``).  This fixes the *exact* number of collaborative epochs
``K = max(1, round(comm_percentage/100 · N))`` out of ``N = total_epochs``, and
those ``K`` epochs are spread **as uniformly as possible** by a largest-remainder
(Bresenham) rule anchored to the END: the final epoch ``N-1`` always communicates,
so the alignment maps refit on the last epoch's (most up-to-date) representations
before testing.  ``comm_percentage`` replaces the old
``collab_epochs`` / ``local_epochs`` cyclic split (kept as deprecated aliases,
now mapped onto the uniform schedule); ``0%`` is intentionally *not* allowed —
with no collaborative epoch the maps never refit, so use the ``non_cooperative``
orchestrator (post-hoc alignment) for the zero-communication baseline instead.

``SheafFRL`` itself stays the simple version with no schedule.
"""

from __future__ import annotations

import warnings

import torch

from src.orchestrators.sheaf_frl import SheafFRL


class CESheafFRL(SheafFRL):
    """Decoupled, communication-efficient Sheaf-FRL (uniformly-scheduled collaborative epochs)."""

    # Deprecated cyclic-split knobs, now folded into ``comm_percentage``.
    _LEGACY_SCHEDULE_KEYS = (
        'collab_epochs', 'local_epochs', 'phase_c_epochs', 'phase_a_epochs',
    )

    def __init__(
        self,
        *,
        comm_percentage: float | None = None,
        total_epochs: int | None = None,
        local_reg: bool = False,
        **kwargs,
    ):
        comm_percentage = self._resolve_comm_percentage(comm_percentage, kwargs)
        if not 0.0 < float(comm_percentage) <= 100.0:
            raise ValueError(
                'comm_percentage must lie in (0, 100]. For 0% communication use '
                'the non_cooperative orchestrator: CESheafFRL never fits its '
                'alignment maps without at least one collaborative epoch, so a '
                '0% run would transport through the untrained (identity) maps.'
            )
        super().__init__(**kwargs)
        self._comm_percentage = float(comm_percentage)
        self._local_reg = bool(local_reg)
        # ``N`` (total epochs) and ``K`` (collaborative epochs) — resolved now if
        # total_epochs was given, else finalised from the trainer at train start.
        self._total_epochs_cfg = (
            None if total_epochs is None else int(total_epochs)
        )
        self._n_epochs: int | None = self._total_epochs_cfg
        self._comm_epochs: int | None = (
            self._comm_epochs_for(self._n_epochs) if self._n_epochs else None
        )
        # Local-epoch minibatch rotation offsets per edge (anchor_selection='all'
        # only, where the frozen snapshot spans the full pilot pool).
        self._local_offsets: dict[tuple[int, int], int] = {}
        # Expose for logging (not captured by the parent's save_hyperparameters,
        # which only sees SheafFRL's signature).
        self.hparams['comm_percentage'] = self._comm_percentage
        self.hparams['local_reg'] = self._local_reg
        if self._total_epochs_cfg is not None:
            self.hparams['total_epochs'] = self._total_epochs_cfg

    def _resolve_comm_percentage(
        self, comm_percentage: float | None, kwargs: dict
    ) -> float:
        """Resolve ``comm_percentage``, folding in the deprecated cyclic knobs.

        The old ``collab_epochs`` / ``local_epochs`` split (and their
        ``phase_c_epochs`` / ``phase_a_epochs`` aliases) are popped from
        ``kwargs`` (so they never reach the base orchestrator).  When present
        they **take precedence** and are converted to the equivalent percentage
        ``100·collab/(collab+local)`` — the same communication budget, now
        distributed uniformly rather than clustered.  This deliberately wins
        over ``comm_percentage`` so that an experiment config still expressing
        its schedule as a phase split keeps that intent even though the
        ``ce_sheaf_frl`` group now carries a ``comm_percentage`` default.
        """
        legacy = {
            k: kwargs.pop(k)
            for k in self._LEGACY_SCHEDULE_KEYS
            if k in kwargs
        }
        if legacy:
            collab = legacy.get('collab_epochs', legacy.get('phase_c_epochs'))
            local = legacy.get('local_epochs', legacy.get('phase_a_epochs'))
            warnings.warn(
                'collab_epochs/local_epochs (and the phase_c/phase_a aliases) '
                'are deprecated; CESheafFRL now takes comm_percentage + '
                'total_epochs and spreads the collaborative epochs uniformly. '
                'Mapping the given split to an equivalent comm_percentage.',
                DeprecationWarning,
                stacklevel=3,
            )
            if collab is not None and local is not None:
                total = int(collab) + int(local)
                if total > 0:
                    return 100.0 * int(collab) / total
        if comm_percentage is not None:
            return float(comm_percentage)
        # Nothing usable supplied — a sensible middle default.
        return 50.0

    # ── Uniform communication schedule ─────────────────────────────────────────

    def _comm_epochs_for(self, n_epochs: int) -> int:
        """Exact number of collaborative epochs for an ``n_epochs`` horizon.

        ``max(1, ·)`` guarantees at least one collaborative epoch for any
        positive ``comm_percentage`` (a run with none would never fit its maps).
        """
        return max(1, round(self._comm_percentage / 100.0 * int(n_epochs)))

    def on_train_start(self) -> None:
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
        """True on the collaborative epochs of the uniform schedule.

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
        """A (sheaf_penalty, after_comm) pair of zeros — local epoch, ``local_reg=False``."""
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
            diff_i = torch.matmul(Z_i_live, V) - Z_j_frozen
            sheaf_penalty += self._edge_penalty_term(edge_key, diff_i)
            # node_j live, pulled toward node_i's last-known (frozen) state.
            diff_j = torch.matmul(Z_i_frozen, V) - Z_j_live
            sheaf_penalty += self._edge_penalty_term(edge_key, diff_j)

        return sheaf_penalty, after_comm
