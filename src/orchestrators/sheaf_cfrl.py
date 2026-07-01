"""Semantic-compression Sheaf-FRL orchestrator (Algorithm of Section 9).

This is the *compressed edge stalk* variant of :class:`SheafFRL`.  Instead of one
restriction map per edge that embeds the smaller stalk into the larger one, each
edge ``(i, j)`` owns **two fat semi-orthogonal maps** that project both node
stalks down into a shared compressed edge stalk of dimension ``c_ij < min(d_i,
d_j)``::

    V_ji ∈ R^{c × d_i},  V_ji V_ji^T = I_c        (projector for node i)
    V_ij ∈ R^{c × d_j},  V_ij V_ij^T = I_c        (projector for node j)

The coboundary / sheaf penalty driving the gradient phases is the compressed
total variation, measured in the edge stalk:

    TV_c(A_i) = Σ_j ‖ Ã_i V_ji^T − Ã_j V_ij^T ‖_F²        (row convention)

Phase B (the per-edge map refit) minimises a β-weighted blend of this *semantic*
misalignment and the *communication* loss — how well each node reconstructs the
other after the lossy round-trip ``V_·^T V_·`` through the edge stalk (Section
9.2).  For β > 0 the communication terms couple the two maps non-linearly (one
appears transposed inside a product with the other), so the naive alternating
Procrustes solution is wrong (Section 9.3); Phase B instead runs ``T_B`` SOC-ADMM
iterations per edge, each a closed-form pass of two generalised Sylvester solves
(updating ``V_ji`` then ``V_ij``), two polar projections onto the Stiefel
manifold (the auxiliaries ``Y_i``, ``Y_j``) and a scaled dual ascent (``U_i``,
``U_j``).  The decoupled three-phase schedule, whitening, and frozen-pilot
caching are inherited from :class:`CESheafFRL`; only the map structure, the
(both-live and Phase-A frozen) coboundary penalties, the Phase-B update and
``send_message`` differ.
"""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn

from src.communication.whitening import color, whiten
from src.orchestrators.ce_sheaf_frl import CESheafFRL


class SheafCFRL(CESheafFRL):
    """Compressed, communication-efficient Sheaf-FRL (Section 9 + decoupling).

    Compressed edge stalks (two fat semi-orthogonal maps per edge, refit by
    SOC-ADMM in Phase B) layered on the :class:`CESheafFRL` three-phase
    schedule.  It overrides only the *geometry* hooks — the both-live and
    Phase-A frozen coboundaries, the map update, and ``send_message`` — and
    inherits the phase schedule + frozen-pilot caching unchanged.
    """

    def __init__(
        self,
        *,
        compression_factor: float = 0.5,
        compression_inner_steps: int = 20,
        compression_init: str = 'pca',
        compression_beta: float = 0.5,
        compression_admm_alpha: float = 1.0,
        **kwargs,
    ):
        if str(compression_init) not in ('pca', 'truncated_identity'):
            raise ValueError(
                f"compression_init must be 'pca' or 'truncated_identity', "
                f'got {compression_init!r}.'
            )
        if not 0.0 < float(compression_factor) <= 1.0:
            raise ValueError(
                'compression_factor must be a fraction in (0, 1] of '
                'min(d_i, d_j) — the edge-stalk dimension to retain on each '
                f'edge (e.g. 0.5 keeps 50%); got {compression_factor!r}.'
            )
        if int(compression_inner_steps) < 1:
            raise ValueError('compression_inner_steps (T_B) must be >= 1.')
        if not 0.0 <= float(compression_beta) < 1.0:
            raise ValueError('compression_beta (β) must lie in [0, 1).')
        if float(compression_admm_alpha) <= 0.0:
            raise ValueError('compression_admm_alpha (α) must be > 0.')

        # SheafFRL.__init__ calls self._build_restriction_maps(...) (overridden
        # below) which only stashes the graph; the real maps are built here once
        # the compression hyperparameters are available.
        super().__init__(**kwargs)
        self._compression_factor = float(compression_factor)
        self._compression_inner_steps = int(compression_inner_steps)
        self._compression_init = str(compression_init)
        self._compression_beta = float(compression_beta)
        self._compression_admm_alpha = float(compression_admm_alpha)
        # Expose for logging / checkpointing (not captured by the parent's
        # save_hyperparameters, which only sees SheafFRL's signature).
        self.hparams['compression_factor'] = self._compression_factor
        self.hparams['compression_inner_steps'] = self._compression_inner_steps
        self.hparams['compression_init'] = self._compression_init
        self.hparams['compression_beta'] = self._compression_beta
        self.hparams['compression_admm_alpha'] = self._compression_admm_alpha
        self._build_compression_maps()

    # ── Map construction ──────────────────────────────────────────────────────

    @staticmethod
    def _proj_key(edge_key: str, node: int) -> str:
        """ParameterDict key for the projector of ``node`` on edge ``edge_key``."""
        return f'{edge_key}__{node}'

    def _build_restriction_maps(
        self, neighbors: dict, latent_dims_int: dict[int, int]
    ) -> None:
        """Invoked during ``super().__init__``: stash the graph, defer real maps.

        ``stiefel_matrices`` is kept empty (every base method that consults it is
        overridden here); the compressed maps are built by
        ``_build_compression_maps`` once the compression hyperparameters exist.
        """
        self.stiefel_matrices = nn.ParameterDict()
        self._latent_dims_int = {
            int(k): int(v) for k, v in latent_dims_int.items()
        }
        self._cfrl_neighbors = neighbors
        self._initialized_edges: set[str] = set()

    def _build_compression_maps(self) -> None:
        """Create two fat semi-orthogonal projectors per edge (truncated identity)."""
        self.compression_maps = nn.ParameterDict()
        # Each entry: (edge_key, node_a, node_b, c_ij).
        self._compression_edges: list[tuple[str, int, int, int]] = []
        self._initialized_edges = set()
        factor = self._compression_factor
        seen: set[str] = set()

        for i_raw, neighborset in self._cfrl_neighbors.items():
            for j_raw in neighborset:
                i, j = int(i_raw), int(j_raw)
                # Compression induces no natural orientation; pick a stable key
                # (larger dim first, ties broken by larger index) for determinism.
                if self._latent_dims_int[i] > self._latent_dims_int[j]:
                    a, b = i, j
                elif self._latent_dims_int[i] < self._latent_dims_int[j]:
                    a, b = j, i
                else:
                    a, b = max(i, j), min(i, j)
                edge_key = f'{a}_{b}'
                if edge_key in seen:
                    continue
                seen.add(edge_key)

                d_a, d_b = self._latent_dims_int[a], self._latent_dims_int[b]
                min_d = min(d_a, d_b)
                # Retain ``factor`` of the smaller node's dimension on the edge.
                c_ij = max(1, min(min_d, round(factor * min_d)))
                if c_ij >= min_d:
                    warnings.warn(
                        f'SheafCFRL: compression_factor={factor} keeps '
                        f'c_ij={c_ij} = min(d)={min_d} for edge {edge_key}; this '
                        'edge is not actually compressed.',
                        UserWarning,
                        stacklevel=2,
                    )
                # Truncated-identity init: first c_ij rows of I_d  → (c, d).
                self.compression_maps[self._proj_key(edge_key, a)] = (
                    nn.Parameter(
                        torch.eye(d_a)[:c_ij].clone(), requires_grad=False
                    )
                )
                self.compression_maps[self._proj_key(edge_key, b)] = (
                    nn.Parameter(
                        torch.eye(d_b)[:c_ij].clone(), requires_grad=False
                    )
                )
                self._compression_edges.append((edge_key, a, b, c_ij))

    def _find_edge(self, s: int, r: int) -> str | None:
        for edge_key, a, b, _c in self._compression_edges:
            if {a, b} == {int(s), int(r)}:
                return edge_key
        return None

    def _record_pilot_exchange(
        self,
        payloads_per_agent: dict,
        latents_per_agent: dict,
        prefix: str,
    ) -> None:
        """Record one pilot exchange using compressed c_ij-dimensional payloads.

        Each directed edge (i→j) carries latents_i projected into the c_ij-
        dimensional edge stalk (V_ji Z_i^T), not the full d_i-dimensional
        embedding.  Both directions of every undirected edge are recorded
        separately: a→b transmits K_a×c_ij scalars, b→a transmits K_b×c_ij.
        """
        self._record_communication_round(n_rounds=1, prefix=prefix)
        for _edge_key, a, b, c_ij in self._compression_edges:
            if a in payloads_per_agent:
                payload_a = payloads_per_agent[a]
                n_rows_a = (
                    payload_a.shape[0]
                    if isinstance(payload_a, torch.Tensor)
                    else latents_per_agent[a].shape[0]
                )
                self._record_communication(int(n_rows_a) * c_ij, prefix=prefix)
            if b in payloads_per_agent:
                payload_b = payloads_per_agent[b]
                n_rows_b = (
                    payload_b.shape[0]
                    if isinstance(payload_b, torch.Tensor)
                    else latents_per_agent[b].shape[0]
                )
                self._record_communication(int(n_rows_b) * c_ij, prefix=prefix)

    def on_train_start(self) -> None:
        super().on_train_start()
        # Reset PCA-warm-start tracking at the start of each fit.
        self._initialized_edges = set()

    # ── Linear-algebra helpers ────────────────────────────────────────────────

    @staticmethod
    def _polar_factor(M: torch.Tensor) -> torch.Tensor:
        """Orthonormal polar factor ``U W^T`` of ``M`` via thin SVD.

        Works for tall or wide ``M``: for tall ``M`` (d×c, d≥c) the result lies in
        ``St(d, c)`` (orthonormal columns); for wide ``M`` (c×d, c≤d) it has
        orthonormal rows.  In both cases it is the Euclidean projection of ``M``
        onto the (compact) Stiefel manifold — used for the cold-start Procrustes
        init and for the ADMM ``Y`` projection step (Section 9.3, Step 3).
        """
        try:
            U, _S, Wt = torch.linalg.svd(M, full_matrices=False)
        except RuntimeError:
            U, _S, Wt = torch.linalg.svd(M.cpu(), full_matrices=False)
            U, Wt = U.to(M.device), Wt.to(M.device)
        return U @ Wt

    @staticmethod
    def _sylvester_eig(
        P: torch.Tensor, lam: torch.Tensor, Q: torch.Tensor, R: torch.Tensor
    ) -> torch.Tensor:
        """Solve the generalised Sylvester equation ``P X Σ + X = R`` for ``X``.

        Caller supplies the eigendecomposition ``Σ = Q diag(λ) Qᵀ`` (``Σ`` is
        symmetric PSD).  ``P`` is ``(c, c)``; ``R`` and the returned ``X`` are
        ``(c, d)``; ``λ`` is ``(d,)`` and ``Q`` is ``(d, d)``.  Substituting
        ``X = Y Qᵀ`` decouples the ``d`` columns, leaving one ``c×c`` solve per
        eigenvalue: ``(λ_k P + I_c) y_k = (R Q)[:, k]``.  This is the symmetric
        specialisation of Bartels–Stewart used by the Phase-B ADMM (Section 9.3).
        """
        c = P.shape[0]
        Ic = torch.eye(c, device=P.device, dtype=P.dtype)
        # One c×c solve per eigenvalue, batched over the d columns.
        M = lam.view(-1, 1, 1) * P.unsqueeze(0) + Ic  # (d, c, c)
        rhs = (R @ Q).t().unsqueeze(-1)  # (d, c, 1)
        Y = torch.linalg.solve(M, rhs).squeeze(-1).t()  # (c, d)
        return Y @ Q.t()

    @staticmethod
    def _pca_init(Z: torch.Tensor, c: int) -> torch.Tensor:
        """Top-``c`` principal directions of pilots ``Z`` (K×d) as rows → (c, d)."""
        d = Z.shape[1]
        if Z.shape[0] - 1 < c:  # too few samples for c stable directions
            return torch.eye(d, device=Z.device, dtype=Z.dtype)[:c].clone()
        Zc = Z - Z.mean(0)
        try:
            _U, _S, Vh = torch.linalg.svd(Zc, full_matrices=False)
        except RuntimeError:
            _U, _S, Vh = torch.linalg.svd(Zc.cpu(), full_matrices=False)
            Vh = Vh.to(Z.device)
        if Vh.shape[0] < c:
            return torch.eye(d, device=Z.device, dtype=Z.dtype)[:c].clone()
        return Vh[:c].contiguous()

    # ── Compressed coboundary penalties (both-live + Phase-A frozen) ──────────

    def _both_live_alignment_losses(
        self,
        whitened_per_agent: dict[int, torch.Tensor],
        keys_per_agent: dict[int, torch.Tensor],
        labels_per_agent: dict[int, torch.Tensor],
        comm_weight: float,
        skip: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Phase-C / per-step compressed coboundary ‖Z_a V_aᵀ − Z_b V_bᵀ‖² + after-comm."""
        sheaf_penalty = torch.tensor(0.0, device=self.device)
        after_comm_task_loss = torch.tensor(0.0, device=self.device)
        if skip:
            return sheaf_penalty, after_comm_task_loss

        for edge_key, a, b, _c in self._compression_edges:
            if a not in whitened_per_agent or b not in whitened_per_agent:
                continue

            V_a = self.compression_maps[
                self._proj_key(edge_key, a)
            ]  # (c, d_a)
            V_b = self.compression_maps[
                self._proj_key(edge_key, b)
            ]  # (c, d_b)
            src_a = (
                whitened_per_agent[a],
                keys_per_agent[a],
                labels_per_agent[a],
            )
            src_b = (
                whitened_per_agent[b],
                keys_per_agent[b],
                labels_per_agent[b],
            )
            matched = self._match_edge(a, b, src_a, src_b)
            if matched is None:
                continue

            # Rows are already whitened (whitening was applied per node upstream).
            Z_a, y_a_shared, Z_b, y_b_shared = matched

            # Compressed coboundary: project both stalks into the c-dim edge space.
            P_a = torch.matmul(Z_a, V_a.t())  # (n, c)
            P_b = torch.matmul(Z_b, V_b.t())  # (n, c)
            diff = P_a - P_b
            sheaf_penalty += (diff**2).sum(dim=1).mean()

            # ── After-communication task loss (project → lift across the edge) ─
            if comm_weight > 0.0:
                agent_a = (
                    self.agents[str(a)] if str(a) in self.agents else None
                )
                agent_b = (
                    self.agents[str(b)] if str(b) in self.agents else None
                )
                _is_clf = lambda ag: (
                    getattr(ag, 'task_type', 'classification')
                    == 'classification'
                )

                # b → a: project b into edge (V_b), lift to a (V_a), recolour, decode.
                if agent_a is not None and _is_clf(agent_a):
                    Z_b_to_a = torch.matmul(
                        torch.matmul(Z_b.float(), V_b.t()), V_a
                    )
                    Z_b_to_a = self._recolour_node(a, Z_b_to_a)
                    logits_ba = agent_a.decoder(Z_b_to_a.to(dtype=Z_a.dtype))
                    after_comm_task_loss += agent_a.compute_loss(
                        logits_ba, y_b_shared.to(self.device)
                    )

                # a → b: project a into edge (V_a), lift to b (V_b), recolour, decode.
                if agent_b is not None and _is_clf(agent_b):
                    Z_a_to_b = torch.matmul(
                        torch.matmul(Z_a.float(), V_a.t()), V_b
                    )
                    Z_a_to_b = self._recolour_node(b, Z_a_to_b)
                    logits_ab = agent_b.decoder(Z_a_to_b.to(dtype=Z_b.dtype))
                    after_comm_task_loss += agent_b.compute_loss(
                        logits_ab, y_a_shared.to(self.device)
                    )

        return sheaf_penalty, after_comm_task_loss

    def _frozen_alignment_losses(
        self,
        whitened_per_agent: dict[int, torch.Tensor],
        keys_per_agent: dict[int, torch.Tensor],
        labels_per_agent: dict[int, torch.Tensor],
        skip: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Phase-A compressed coboundary against FROZEN neighbours.

        Two terms per edge, each projecting a live node and a detached frozen
        neighbour into the edge stalk; gradient reaches only the live node, and no
        pilot is communicated this step.
        """
        sheaf_penalty = torch.tensor(0.0, device=self.device)
        after_comm = torch.tensor(0.0, device=self.device)
        if skip:
            return sheaf_penalty, after_comm

        frozen = getattr(self, '_frozen_pilots', None)
        if not frozen:
            return self._both_live_alignment_losses(
                whitened_per_agent, keys_per_agent, labels_per_agent, 0.0, skip
            )

        for edge_key, a, b, _c in self._compression_edges:
            if a not in whitened_per_agent or b not in whitened_per_agent:
                continue
            V_a = self.compression_maps[
                self._proj_key(edge_key, a)
            ]  # (c, d_a)
            V_b = self.compression_maps[
                self._proj_key(edge_key, b)
            ]  # (c, d_b)
            src_a = (
                whitened_per_agent[a],
                keys_per_agent[a],
                labels_per_agent[a],
            )
            src_b = (
                whitened_per_agent[b],
                keys_per_agent[b],
                labels_per_agent[b],
            )
            # Pull node a (live) toward frozen b; then node b (live) toward frozen a.
            if b in frozen:
                m = self._match_edge(a, b, src_a, frozen[b])
                if m is not None:
                    d = torch.matmul(m[0], V_a.t()) - torch.matmul(
                        m[2], V_b.t()
                    )
                    sheaf_penalty += (d**2).sum(dim=1).mean()
            if a in frozen:
                m = self._match_edge(a, b, frozen[a], src_b)
                if m is not None:
                    d = torch.matmul(m[0], V_a.t()) - torch.matmul(
                        m[2], V_b.t()
                    )
                    sheaf_penalty += (d**2).sum(dim=1).mean()

        return sheaf_penalty, after_comm

    # ── Phase B: SOC-ADMM on the product Stiefel manifold ─────────────────────

    @torch.no_grad()
    def _admm_phase_b(
        self,
        V_a: torch.Tensor,
        V_b: torch.Tensor,
        Sigma_a: torch.Tensor,
        Sigma_b: torch.Tensor,
        Sigma_ab: torch.Tensor,
        c: int,
        t_b: int,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """SOC-ADMM for one edge (Section 9.3); returns Stiefel-feasible maps.

        Parameters mirror the spec with ``i ↔ a``, ``j ↔ b``::

            V_a = V_ji ∈ R^{c×d_a}  (projector for node a)
            V_b = V_ij ∈ R^{c×d_b}  (projector for node b)
            Σ_a, Σ_b                 node second-moment matrices
            Σ_ab = Ãa^T Ãb / n       cross-moment  (Σ_ji = Σ_ab^T)

        Each of the ``t_b`` iterations runs the four closed-form steps: two
        generalised Sylvester solves (update ``V_a`` then ``V_b``, the latter
        using the just-updated ``V_a``), two polar projections onto the Stiefel
        manifold (auxiliaries ``Y_a``, ``Y_b``) and a scaled dual ascent
        (``U_a``, ``U_b``).  The maps are read off from the converged auxiliaries
        (``V_a = Y_aᵀ``) so they are exactly row-orthonormal.  Returns
        ``(V_a, V_b, primal_residual)``.
        """
        beta = self._compression_beta
        alpha = self._compression_admm_alpha
        dev, dt = V_a.device, V_a.dtype
        Ic = torch.eye(c, device=dev, dtype=dt)
        Sigma_ba = Sigma_ab.t()  # (d_b, d_a) = Σ_ji

        # Σ_a, Σ_b are fixed across the loop → eigendecompose once for the
        # symmetric Sylvester solves (Σ = Q diag(λ) Qᵀ).
        lam_a, Q_a = torch.linalg.eigh(0.5 * (Sigma_a + Sigma_a.t()))
        lam_b, Q_b = torch.linalg.eigh(0.5 * (Sigma_b + Sigma_b.t()))

        # Section 9.4 init: the incoming maps already lie on the Stiefel manifold,
        # so Y⁰ = V⁰ᵀ (Stiefel-feasible) and the scaled duals start at zero.
        Y_a, Y_b = (
            V_a.t().contiguous(),
            V_b.t().contiguous(),
        )  # (d_a,c),(d_b,c)
        U_a, U_b = torch.zeros_like(Y_a), torch.zeros_like(Y_b)

        for _ in range(t_b):
            # Step 1 — update V_a = V_ji via the generalised Sylvester eq. (S1).
            A_b = (1.0 - beta) * Ic + beta * (V_b @ V_b.t())
            B_b = alpha * Ic + beta * (V_b @ Sigma_b @ V_b.t())
            C_b = (1.0 + beta) * (V_b @ Sigma_ba) + alpha * (Y_a - U_a).t()
            V_a = self._sylvester_eig(
                torch.linalg.solve(B_b, A_b),
                lam_a,
                Q_a,
                torch.linalg.solve(B_b, C_b),
            )

            # Step 2 — update V_b = V_ij via the generalised Sylvester eq. (S2),
            # using the V_a just computed above.
            A_a = (1.0 - beta) * Ic + beta * (V_a @ V_a.t())
            B_a = alpha * Ic + beta * (V_a @ Sigma_a @ V_a.t())
            C_a = (1.0 + beta) * (V_a @ Sigma_ab) + alpha * (Y_b - U_b).t()
            V_b = self._sylvester_eig(
                torch.linalg.solve(B_a, A_a),
                lam_b,
                Q_b,
                torch.linalg.solve(B_a, C_a),
            )

            # Step 3 — project the transposed maps onto the Stiefel manifold.
            Y_a = self._polar_factor(V_a.t() + U_a)  # (d_a, c) ∈ St(d_a, c)
            Y_b = self._polar_factor(V_b.t() + U_b)  # (d_b, c) ∈ St(d_b, c)

            # Step 4 — scaled dual ascent.
            U_a = U_a + V_a.t() - Y_a
            U_b = U_b + V_b.t() - Y_b

        primal = float(
            (V_a.t() - Y_a).norm().item() + (V_b.t() - Y_b).norm().item()
        )
        # Read off Stiefel-feasible maps from the converged auxiliaries.
        return Y_a.t().contiguous(), Y_b.t().contiguous(), primal

    @torch.no_grad()
    def _update_stiefel_matrices(
        self,
        latents_per_agent: dict[int, torch.Tensor],
        keys_per_agent: dict[int, torch.Tensor],
        whitening_ops=None,
        update_maps: bool = True,
    ) -> tuple[dict[str, float], dict]:
        if not self._compression_edges:
            return {}, {}

        edge_metrics: dict[str, float] = {}
        agent_normed, fitted_ops = self._whiten_epoch_latents(
            latents_per_agent, whitening_ops
        )
        t_b = self._compression_inner_steps

        for edge_key, a, b, c_ij in self._compression_edges:
            if a not in agent_normed or b not in agent_normed:
                continue

            # Epoch keys double as labels (they are class labels at epoch level).
            Z_a_f, keys_a_f, _la, Z_b_f, keys_b_f, _lb = (
                self._apply_edge_class_filter(
                    a,
                    b,
                    agent_normed[a],
                    keys_per_agent[a],
                    keys_per_agent[a],
                    agent_normed[b],
                    keys_per_agent[b],
                    keys_per_agent[b],
                )
            )
            shared = self._match_keys(
                A_i=Z_a_f, A_j=Z_b_f, keys_i=keys_a_f, keys_j=keys_b_f
            )
            if shared is None:
                continue
            Z_a, Z_b = shared

            param_a = self.compression_maps[self._proj_key(edge_key, a)]
            param_b = self.compression_maps[self._proj_key(edge_key, b)]
            dev = param_a.device
            Z_a, Z_b = Z_a.float().to(dev), Z_b.float().to(dev)

            V_a = param_a.data.clone()  # (c, d_a)
            V_b = param_b.data.clone()  # (c, d_b)
            # PCA warm start on the first Phase B for this edge (if requested);
            # later epochs warm-start from the maps already on the manifold.
            if (
                self._compression_init == 'pca'
                and edge_key not in self._initialized_edges
            ):
                V_a = self._pca_init(Z_a, c_ij)
                V_b = self._pca_init(Z_b, c_ij)

            if update_maps:
                # Normalised second-/cross-moment matrices (Σ ≈ I after
                # whitening).  The 1/n scaling — relative to the spec's raw ÃᵀÃ —
                # keeps the ADMM penalty α scale-free in the matched-pilot count.
                n = max(1, Z_a.shape[0])
                Sigma_a = (Z_a.t() @ Z_a) / n  # (d_a, d_a) = Σ_i
                Sigma_b = (Z_b.t() @ Z_b) / n  # (d_b, d_b) = Σ_j
                Sigma_ab = (Z_a.t() @ Z_b) / n  # (d_a, d_b) = Σ_ij
                V_a, V_b, primal = self._admm_phase_b(
                    V_a, V_b, Sigma_a, Sigma_b, Sigma_ab, c_ij, t_b
                )
                param_a.copy_(V_a.to(dtype=param_a.dtype, device=dev))
                param_b.copy_(V_b.to(dtype=param_b.dtype, device=dev))
                edge_metrics[
                    f'compressed_admm_primal_resid_edge_{edge_key}'
                ] = primal
                self._initialized_edges.add(edge_key)

            resid = ((Z_a @ V_a.t() - Z_b @ V_b.t()) ** 2).sum(dim=1).mean()
            edge_metrics[f'compressed_residual_edge_{edge_key}'] = float(
                resid.item()
            )

        return edge_metrics, fitted_ops

    # ── send_message: project sender into the edge stalk, lift into receiver ──

    @torch.no_grad()
    def send_message(
        self,
        sender_idx: int,
        receiver_idx: int,
        Z_sender: torch.Tensor,
    ) -> torch.Tensor:
        """Transport sender test latents to the receiver via the compressed edge.

        Pipeline: whiten with sender's g_φ → project to the edge stalk (V_sender)
        → lift into the receiver's stalk (V_receiver^T) → recolour with g*_φ.
        """
        use_learnable = self._use_learnable_whitening()
        dev = Z_sender.device
        work_dev = self.device if use_learnable else dev

        # Step 1 — whiten with the sender's statistics.
        if use_learnable:
            Z = self._whiten_pilots_frozen(sender_idx, Z_sender.to(work_dev))
        else:
            op_s = self._whitening_ops.get(sender_idx)
            Z = (
                whiten(Z_sender, op_s)
                if op_s is not None
                else Z_sender.float()
            )

        # Step 2 — project into the compressed edge stalk, then lift to receiver.
        edge_key = self._find_edge(sender_idx, receiver_idx)
        if edge_key is not None:
            V_s = (
                self.compression_maps[
                    self._proj_key(edge_key, int(sender_idx))
                ]
                .float()
                .to(work_dev)
            )
            V_r = (
                self.compression_maps[
                    self._proj_key(edge_key, int(receiver_idx))
                ]
                .float()
                .to(work_dev)
            )
            Z_lift = torch.matmul(torch.matmul(Z.to(work_dev), V_s.t()), V_r)
        else:
            Z_lift = Z.to(work_dev)

        # Step 3 — recolour with the receiver's statistics.
        if use_learnable:
            return self._colouring_layers[str(receiver_idx)](Z_lift).to(dev)
        op_r = self._whitening_ops.get(receiver_idx)
        return (color(Z_lift, op_r) if op_r is not None else Z_lift).to(dev)
