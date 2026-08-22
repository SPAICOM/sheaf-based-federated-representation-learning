"""Communication-Efficient Sheaf-FRL orchestrator (decoupled three-phase schedule).

``CESheafFRL`` is :class:`SheafFRL` with a decoupled training schedule that trades
some pilot communication for cheaper rounds.  Each outer iteration is::

    Phase C : T_C epochs of *classical* SheafFRL — the alignment maps refit,
              and each edge's fixed anchor set is re-selected
              (``_rebuild_edge_anchor_caches``), at the end of EVERY Phase-C
              epoch.  (Phase C ≡ SheafFRL.)
    Phase A : T_A epochs that AVOID any new pilot exchange — each node keeps
              re-encoding its own half of the *same* fixed anchor cache
              (fresh values, same identities) while pulling toward the
              OTHER node's snapshot as of the last Phase-C refresh
              (``self._frozen_edge_anchors``), trains task + λ·TV, and does
              NOT refresh the maps or the anchor cache.

The maps and anchor caches are therefore refreshed exactly as in plain
SheafFRL during Phase C; the only addition is the communication-free Phase A,
which reuses whatever was last communicated instead of selecting anything new.
``SheafFRL`` itself stays the simple version with no phase logic.
"""

from __future__ import annotations

import torch

from src.orchestrators.sheaf_frl import SheafFRL


class CESheafFRL(SheafFRL):
    """Decoupled, communication-efficient Sheaf-FRL (Phase C ≡ SheafFRL + Phase A)."""

    def __init__(
        self,
        *,
        phase_a_epochs: int = 1,
        phase_c_epochs: int = 1,
        **kwargs,
    ):
        pa, pc = int(phase_a_epochs), int(phase_c_epochs)
        if pa < 1:
            raise ValueError('phase_a_epochs (T_A) must be >= 1.')
        if pc < 1:
            raise ValueError('phase_c_epochs (T_C) must be >= 1.')
        super().__init__(**kwargs)
        self._phase_a_epochs = pa
        self._phase_c_epochs = pc
        # Expose for logging (not captured by the parent's save_hyperparameters,
        # which only sees SheafFRL's signature).
        self.hparams['phase_a_epochs'] = pa
        self.hparams['phase_c_epochs'] = pc

    # ── Phase schedule (Phase C leads each outer iteration) ───────────────────

    def _cycle_len(self) -> int:
        return self._phase_c_epochs + self._phase_a_epochs

    def _phase_for_epoch(self, epoch: int) -> str:
        """'C' for the first T_C epochs of each cycle, 'A' for the next T_A."""
        return 'C' if (epoch % self._cycle_len()) < self._phase_c_epochs else 'A'

    # ── Overridden hooks ──────────────────────────────────────────────────────

    def _should_update_maps_at_epoch_end(self) -> bool:
        """Refresh the anchor caches + maps at the end of every Phase-C epoch; never in Phase A."""
        return self._phase_for_epoch(self.current_epoch) == 'C'

    def _compute_alignment_losses(
        self, comm_weight: float, skip: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Phase A (training only) → frozen-neighbour coboundary; otherwise the
        # classical both-live SheafFRL penalty (Phase C, validation, test).
        if self.training:
            phase = self._phase_for_epoch(self.current_epoch)
            self.log(
                'train/phase_A',
                1.0 if phase == 'A' else 0.0,
                on_step=False, on_epoch=True, prog_bar=False,
                add_dataloader_idx=False,
            )
            if phase == 'A':
                return self._frozen_alignment_losses(skip)
        # Phase C / validation / test: the classical both-live penalty (embedding
        # for CESheafFRL, compressed when SheafCFRL overrides the hook).
        return self._both_live_alignment_losses(comm_weight, skip)

    def _frozen_alignment_losses(
        self, skip: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Phase-A coboundary: each live node pulled toward the OTHER's frozen snapshot.

        Two terms per edge, each pairing one side's *fresh* re-encoding of the
        fixed anchor cache (:meth:`~src.orchestrators.sheaf_frl.SheafFRL._edge_live_anchors`,
        same cache and grouping used every Phase-C step too) against the other
        side's snapshot as of the last refresh (``self._frozen_edge_anchors``,
        populated by ``_rebuild_edge_anchor_caches``) — no new pilot exchange
        needed, since the snapshot is already locally known from the last
        Phase-C communication.  Falls back to the both-live penalty (no comm
        recorded — ``comm_weight=0.0``) before the first Phase-C refresh has
        ever populated a snapshot.
        """
        sheaf_penalty = torch.tensor(0.0, device=self.device)
        after_comm = torch.tensor(0.0, device=self.device)
        if skip:
            return sheaf_penalty, after_comm

        if not self._frozen_edge_anchors:
            return self._both_live_alignment_losses(0.0, skip)

        for edge_key, V in self.stiefel_matrices.items():
            node_i, node_j = map(int, edge_key.split('_'))
            frozen = self._frozen_edge_anchors.get((node_i, node_j))
            if frozen is None:
                continue
            live = self._edge_live_anchors(node_i, node_j)
            if live is None:
                continue
            Z_i_live, _y_i_live, Z_j_live, _y_j_live = live
            Z_i_frozen, _y_i_frozen, Z_j_frozen, _y_j_frozen = frozen

            # node_i live, pulled toward node_j's last-known (frozen) state.
            diff_i = torch.matmul(Z_i_live, V) - Z_j_frozen
            sheaf_penalty += self._edge_penalty_term(edge_key, diff_i)
            # node_j live, pulled toward node_i's last-known (frozen) state.
            diff_j = torch.matmul(Z_i_frozen, V) - Z_j_live
            sheaf_penalty += self._edge_penalty_term(edge_key, diff_j)

        return sheaf_penalty, after_comm
