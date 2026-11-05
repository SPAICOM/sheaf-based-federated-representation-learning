"""DiSheafFRL: SheafFRL with a *directed*, confidence-gated sheaf penalty.

On every matched pilot row ``r`` of edge ``(i, j)`` each endpoint scores how
competent its own decoder is on its own latent (no labels):

    c_i,r = max softmax of agent i's decoder      (gate_score='softmax')
    c_i,r = max logit, z-scored over the batch     (gate_score='zlogit')

and the gate ``a_r = σ((c_i,r − c_j,r) / τ)`` (no gradient) decides who leads.
The symmetric penalty ``‖z_i V − z_j‖²`` is replaced by

    P = mean_r [ a_r ‖sg(z_i) V − z_j‖² + (1 − a_r) ‖z_i V − sg(z_j)‖² ]

so where agent i is the stronger one its latents are a frozen target and only
agent j's encoder moves toward them, and vice versa.  ``a_r = ½`` (equal
confidence, e.g. classes both agents train on) recovers the symmetric penalty
up to a factor ½.  Restriction maps ``V`` keep their gradient in both terms.
With ``penalty_norm='energy'`` each term is divided by the same energy, with the
leader side detached in its own term so the leader gets no gradient at all.
"""

from __future__ import annotations

import torch

from .sheaf_frl import SheafFRL


class DiSheafFRL(SheafFRL):
    def __init__(
        self,
        *,
        gate_score: str = 'softmax',
        gate_temperature: float = 0.1,
        **kwargs,
    ):
        if gate_score not in ('softmax', 'zlogit'):
            raise ValueError(f"gate_score must be 'softmax' or 'zlogit', got {gate_score!r}")
        if float(gate_temperature) <= 0.0:
            raise ValueError(f'gate_temperature must be > 0, got {gate_temperature}')
        super().__init__(**kwargs)
        if self._dual_lmb_enabled():
            raise ValueError('DiSheafFRL does not support learn_lmb (dual-ascent λ)')
        self._gate_score = gate_score
        self._gate_temperature = float(gate_temperature)
        self.hparams['gate_score'] = gate_score
        self.hparams['gate_temperature'] = self._gate_temperature
        # (node_i, node_j) -> unmapped matched rows (Z_i, Z_j) of the current step
        self._step_edge_rows: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def _edge_step_anchors(self, node_i: int, node_j: int):
        matched = super()._edge_step_anchors(node_i, node_j)
        if matched is not None:
            self._step_edge_rows[f'{node_i}_{node_j}'] = (matched[0], matched[2])
        return matched

    @torch.no_grad()
    def _competence(self, idx: int, Z: torch.Tensor) -> torch.Tensor:
        """Per-row confidence of agent ``idx``'s own decoder on its own latents."""
        decoder = self.agents[str(idx)].decoder
        was_training = decoder.training
        decoder.eval()  # do not touch the decoder's BN running stats with pilot rows
        logits = decoder(self._recolour_node(idx, Z.detach()))
        decoder.train(was_training)
        if self._gate_score == 'softmax':
            return logits.softmax(dim=1).max(dim=1).values
        m = logits.max(dim=1).values
        return (m - m.mean()) / m.std().clamp_min(1e-6)

    def _edge_penalty_term(self, edge_key, diff, sides=None):
        rows = self._step_edge_rows.pop(edge_key, None)
        if rows is None or edge_key not in self.stiefel_matrices:
            return super()._edge_penalty_term(edge_key, diff, sides)
        Z_i, Z_j = rows
        node_i, node_j = map(int, edge_key.split('_'))
        V = self.stiefel_matrices[edge_key]
        a = torch.sigmoid(
            (self._competence(node_i, Z_i) - self._competence(node_j, Z_j))
            / self._gate_temperature
        )
        A, B = Z_i @ V, Z_j                      # same rows as `sides`
        A_lead, B_lead = Z_i.detach() @ V, Z_j.detach()
        sq_i_leads = (A_lead - B).pow(2).sum(dim=1)   # only j's encoder (and V) move
        sq_j_leads = (A - B_lead).pow(2).sum(dim=1)   # only i's encoder (and V) move
        t_i = (a * sq_i_leads).mean()
        t_j = ((1.0 - a) * sq_j_leads).mean()
        if str(self.hparams.penalty_norm) == 'energy':
            e = lambda X: X.pow(2).sum(dim=1).mean()
            normed = t_i / (e(A_lead) + e(B)).clamp(min=1e-12) + t_j / (e(A) + e(B_lead)).clamp(min=1e-12)
        else:
            normed = (t_i + t_j) / (2.0 * diff.shape[1])
        if self.training:
            self.log(f'train/gate_mean_{edge_key}', a.mean(), on_step=False, on_epoch=True)
            self.log(f'train/gate_frac_i_leads_{edge_key}', (a > 0.5).float().mean(),
                     on_step=False, on_epoch=True)
        return normed
