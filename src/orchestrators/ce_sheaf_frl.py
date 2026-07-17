"""Communication-Efficient Sheaf-FRL orchestrator (decoupled three-phase schedule).

``CESheafFRL`` is :class:`SheafFRL` with a decoupled training schedule that trades
some pilot communication for cheaper rounds.  Each outer iteration is::

    Phase C : T_C epochs of *classical* SheafFRL — exchange pilots every step,
              optimise task + λ·TV (+ μ·after-comm), and refresh the alignment
              maps at the end of EVERY Phase-C epoch.  (Phase C ≡ SheafFRL.)
    Phase A : T_A epochs that AVOID pilot communication — each node reuses a
              frozen snapshot of its neighbours' whitened pilots (taken at the
              end of the last Phase-C epoch) as a fixed alignment target, trains
              task + λ·TV against it, and does NOT update the maps.

The maps are therefore updated exactly as in plain SheafFRL during Phase C; the
only addition is the communication-free Phase A.  ``SheafFRL`` itself stays the
simple per-step version with no phase logic.
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

    def _is_phase_c_to_a_boundary(self, epoch: int) -> bool:
        """True at the last Phase-C epoch (where neighbour pilots are snapshotted)."""
        return (epoch % self._cycle_len()) == self._phase_c_epochs - 1

    # ── Overridden hooks ──────────────────────────────────────────────────────

    def on_train_start(self) -> None:
        super().on_train_start()
        # Frozen neighbour pilots, finalised at each C→A boundary (whole pool,
        # keyed by sample-id); and the per-id cache filled over the Phase-C epoch.
        self._frozen_pilots: dict[
            int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] = {}
        self._phase_a_cache: dict[
            int, dict[int, tuple[torch.Tensor, torch.Tensor]]
        ] = {}

    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()
        # Start accumulating neighbour pilots fresh when the last Phase-C epoch
        # begins, so the snapshot reflects the most recent encoders.
        if self._is_phase_c_to_a_boundary(self.current_epoch):
            self._phase_a_cache = {}

    def _should_update_maps_at_epoch_end(self) -> bool:
        """Refresh the maps at the end of every Phase-C epoch; never in Phase A."""
        return self._phase_for_epoch(self.current_epoch) == 'C'

    def _on_maps_refreshed(self, did_update: bool) -> None:
        """At the C→A boundary, freeze the cached full-pool neighbour pilots."""
        if did_update and self._is_phase_c_to_a_boundary(self.current_epoch):
            self._finalize_frozen_pilots()

    def _exchanges_pilots(self, prefix: str) -> bool:
        """No pilot exchange (or comm recording) during Phase-A training steps."""
        return not (
            prefix == 'train'
            and self._phase_for_epoch(self.current_epoch) == 'A'
        )

    def _on_pilots_whitened(
        self,
        whitened_per_agent: dict[int, torch.Tensor],
        keys_per_agent: dict[int, torch.Tensor],
        labels_per_agent: dict[int, torch.Tensor],
        prefix: str,
    ) -> None:
        """Cache each node's whitened pilots over the last Phase-C epoch.

        Keyed by sample-id (latest wins), so by the C→A boundary the cache spans
        the whole global-pilot pool even when ``pilot_batch_size`` < pool size and
        each step only carries a subset.  This is the set of neighbour reps a node
        "receives" during Phase C and reuses, fixed, throughout Phase A.
        """
        if prefix != 'train' or not self._is_phase_c_to_a_boundary(self.current_epoch):
            return
        for idx, Z in whitened_per_agent.items():
            keys = keys_per_agent[idx]
            labels = labels_per_agent[idx]
            Zc = Z.detach()
            per_id = self._phase_a_cache.setdefault(idx, {})
            for r in range(Zc.shape[0]):
                per_id[int(keys[r])] = (Zc[r], labels[r].detach())

    # ── Phase-A frozen-neighbour penalty ──────────────────────────────────────

    def _compute_alignment_losses(
        self,
        whitened_per_agent: dict[int, torch.Tensor],
        keys_per_agent: dict[int, torch.Tensor],
        labels_per_agent: dict[int, torch.Tensor],
        comm_weight: float,
        skip: bool,
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
                return self._frozen_alignment_losses(
                    whitened_per_agent, keys_per_agent, labels_per_agent, skip
                )
        # Phase C / validation / test: the classical both-live penalty (embedding
        # for CESheafFRL, compressed when SheafCFRL overrides the hook).
        return self._both_live_alignment_losses(
            whitened_per_agent, keys_per_agent, labels_per_agent, comm_weight, skip
        )

    def _frozen_alignment_losses(
        self,
        whitened_per_agent: dict[int, torch.Tensor],
        keys_per_agent: dict[int, torch.Tensor],
        labels_per_agent: dict[int, torch.Tensor],
        skip: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Phase-A coboundary: each live node pulled toward FROZEN neighbours.

        Two terms per edge, each freezing one side (the detached snapshot), so
        gradient reaches only the live node — no pilot is communicated this step.
        """
        sheaf_penalty = torch.tensor(0.0, device=self.device)
        after_comm = torch.tensor(0.0, device=self.device)
        if skip:
            return sheaf_penalty, after_comm

        frozen = getattr(self, '_frozen_pilots', None)
        if not frozen:
            # No snapshot yet (no Phase C has run) → fall back to both-live, no comm.
            return self._both_live_alignment_losses(
                whitened_per_agent, keys_per_agent, labels_per_agent, 0.0, skip
            )

        for edge_key, V in self.stiefel_matrices.items():
            node_i, node_j = map(int, edge_key.split('_'))
            if (
                node_i not in whitened_per_agent
                or node_j not in whitened_per_agent
            ):
                continue
            src_i = (
                whitened_per_agent[node_i], keys_per_agent[node_i], labels_per_agent[node_i]
            )
            src_j = (
                whitened_per_agent[node_j], keys_per_agent[node_j], labels_per_agent[node_j]
            )
            if node_j in frozen:
                m = self._match_edge(node_i, node_j, src_i, frozen[node_j])
                if m is not None:
                    sheaf_penalty += self._edge_penalty_term(
                        edge_key, torch.matmul(m[0], V) - m[2]
                    )
            if node_i in frozen:
                m = self._match_edge(node_i, node_j, frozen[node_i], src_j)
                if m is not None:
                    sheaf_penalty += self._edge_penalty_term(
                        edge_key, torch.matmul(m[0], V) - m[2]
                    )

        return sheaf_penalty, after_comm

    # ── Freeze the cached full-pool pilots at the C→A boundary ────────────────

    @torch.no_grad()
    def _finalize_frozen_pilots(self) -> None:
        """Stack the per-sample-id cache into fixed (Z, keys, labels) per agent.

        The cache (filled over the last Phase-C epoch by ``_on_pilots_whitened``)
        spans the whole pilot pool, so every Phase-A batch finds its neighbours'
        reps by sample-id — no further communication needed.
        """
        frozen: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        for idx, per_id in self._phase_a_cache.items():
            if not per_id:
                continue
            ids = sorted(per_id)
            Z = torch.stack([per_id[k][0] for k in ids], dim=0)
            labels = torch.stack([per_id[k][1] for k in ids], dim=0)
            keys = torch.tensor(ids, device=Z.device, dtype=labels.dtype)
            frozen[idx] = (Z, keys, labels)
        self._frozen_pilots = frozen
