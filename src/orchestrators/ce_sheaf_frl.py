"""Communication-Efficient Sheaf-FRL orchestrator (decoupled local/collaborative schedule).

``CESheafFRL`` is :class:`SheafFRL` with a decoupled training schedule that trades
some pilot communication for cheaper rounds.  Training epochs are split into two
kinds, interleaved in a fixed repeating cycle (collaborative epochs lead):

    Collaborative epoch : ``collab_epochs`` epochs of *classical* SheafFRL — the
        alignment maps refit and each edge's anchor set is re-selected
        (``_rebuild_edge_anchor_caches``) at the end of EVERY such epoch, and a
        fresh pilot exchange is recorded.  (Collaborative epoch ≡ SheafFRL.)
    Local epoch : ``local_epochs`` epochs that AVOID any new pilot exchange.
        By default the sheaf regularization simply *disappears* here and each
        node trains its task loss alone.  With ``local_reg=True`` a
        communication-free local regularizer is instead applied: each node's
        current (freshly re-encoded) representation is pulled toward the rotated,
        whitened representations its neighbours sent during the last
        collaborative epoch (``self._frozen_edge_anchors``) — a consensus toward
        the last-known neighbour state that needs no new communication.

So the maps and anchor caches refresh exactly as in plain SheafFRL, but only on
collaborative epochs; the ``local_reg`` flag controls whether the intervening
local epochs regularize toward the frozen neighbours or train purely locally.
The fraction of collaborative (communication) epochs is set by the
``collab_epochs`` / ``local_epochs`` split, which is what the communication
ablation sweeps.  ``SheafFRL`` itself stays the simple version with no schedule.
"""

from __future__ import annotations

import warnings

import torch

from src.orchestrators.sheaf_frl import SheafFRL


class CESheafFRL(SheafFRL):
    """Decoupled, communication-efficient Sheaf-FRL (collaborative ≡ SheafFRL + local)."""

    def __init__(
        self,
        *,
        collab_epochs: int = 1,
        local_epochs: int = 1,
        local_reg: bool = False,
        **kwargs,
    ):
        # Back-compat: phase_c_epochs / phase_a_epochs were renamed to
        # collab_epochs / local_epochs.  Some experiment configs still use the
        # old names, so accept them (popped from kwargs so they never reach the
        # base orchestrator) as deprecated aliases.
        collab_epochs = self._pop_deprecated(
            kwargs, 'phase_c_epochs', 'collab_epochs', collab_epochs
        )
        local_epochs = self._pop_deprecated(
            kwargs, 'phase_a_epochs', 'local_epochs', local_epochs
        )

        ce, le = int(collab_epochs), int(local_epochs)
        if ce < 1:
            raise ValueError(
                'collab_epochs (collaborative epochs per cycle) must be >= 1: '
                'the alignment maps only ever refit on collaborative epochs.'
            )
        if le < 0:
            raise ValueError('local_epochs must be >= 0.')
        super().__init__(**kwargs)
        self._collab_epochs = ce
        self._local_epochs = le
        self._local_reg = bool(local_reg)
        # Local-epoch minibatch rotation offsets per edge (anchor_selection='all'
        # only, where the frozen snapshot spans the full pilot pool).
        self._local_offsets: dict[tuple[int, int], int] = {}
        # Expose for logging (not captured by the parent's save_hyperparameters,
        # which only sees SheafFRL's signature).
        self.hparams['collab_epochs'] = ce
        self.hparams['local_epochs'] = le
        self.hparams['local_reg'] = self._local_reg

    @staticmethod
    def _pop_deprecated(kwargs: dict, old: str, new: str, current):
        """Return the value of a renamed kwarg, honouring the deprecated alias.

        If ``old`` is present in ``kwargs`` it is popped and used (with a
        warning); otherwise ``current`` (the value bound to the new name) is
        returned unchanged.
        """
        if old in kwargs:
            value = kwargs.pop(old)
            warnings.warn(
                f'{old} is deprecated; use {new} instead.',
                DeprecationWarning,
                stacklevel=3,
            )
            return value
        return current

    # ── Local/collaborative schedule (collaborative epochs lead each cycle) ────

    def _cycle_len(self) -> int:
        return self._collab_epochs + self._local_epochs

    def _is_collab_epoch(self, epoch: int) -> bool:
        """True for the first ``collab_epochs`` epochs of each cycle.

        Those lead epochs communicate (refit maps, re-select anchors, exchange
        pilots); the trailing ``local_epochs`` are communication-free.  With
        ``local_epochs == 0`` every epoch is collaborative (≡ SheafFRL).
        """
        if self._local_epochs == 0:
            return True
        return (epoch % self._cycle_len()) < self._collab_epochs

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
        snapshot.  Falls back to the both-live penalty (no comm recorded —
        ``comm_weight=0.0``) before the first collaborative refresh has ever
        populated a snapshot.
        """
        sheaf_penalty = torch.tensor(0.0, device=self.device)
        after_comm = torch.tensor(0.0, device=self.device)
        if skip:
            return sheaf_penalty, after_comm

        if not self._frozen_edge_anchors:
            return self._both_live_alignment_losses(0.0, skip)

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
