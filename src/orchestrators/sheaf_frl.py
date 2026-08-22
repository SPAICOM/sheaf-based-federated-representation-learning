"""
Sheaf-based Federated Representation Learning orchestrator.

This module implements the proposed federated learning framework with
Sheaf regularization that maintains aligned latent spaces across agents
through Stiefel manifold optimization of cross-covariance matrices.
Shared pilot batches provide the semantic correspondence needed to align
neighboring agents accurately.
"""

import warnings
from typing import Any

import torch
import torch.nn as nn

from src.communication.whitening import (
    SWBNColouringLayer,
    SWBNWhiteningLayer,
    WhiteningOp,
    color,
    fit_alignment,
    fit_whitening,
    whiten,
)
from src.mutualinfo._common import kmeans_cluster
from src.orchestrators.base_orchestrator import BaseOrchestrator
from src.utils.anchors import (
    AnchorConfig,
    communication_anchor_payload,
    parseval_normalize,
)


class SheafFRL(BaseOrchestrator):
    """Sheaf-based Federated Representation Learning orchestrator.

    Each edge's penalty is normalised by ``2·d_e`` (see
    :meth:`_edge_penalty_term`) so it is dimension-free and comparable
    across edges/configs regardless of latent size, before being weighted
    in one of three mutually exclusive ways:

    * **fixed** — one global ``max_lmb`` on every edge
      (``lambda_schedule=None``, ``learn_lmb=False``);
    * **scheduled** — ``max_lmb`` warmed up over training
      (``lambda_schedule='cosine' | 'exp'``, ``learn_lmb=False``);
    * **learned** — per-edge multipliers ``λ_e`` adapted online by projected
      dual ascent on the whitening-normalised residual constraint
      ``P_e / (2·d_e) ≤ dual_rho`` (``learn_lmb=True``; ``max_lmb`` and
      ``lambda_schedule`` are unused).  Whitened, unaligned latents give
      ``E[P_e] ≈ 2·d_e`` and perfect alignment gives 0, so ``dual_rho`` is a
      dimension-free "tolerated fraction of the unaligned residual energy".
      Each ``λ_e`` grows while its edge violates the tolerance and decays to
      exactly 0 once inside it, so warmup and schedules emerge automatically
      and hard (e.g. cross-group) edges receive more weight than easy ones.

    Communication is a *per-edge fixed anchor set*, re-selected every time the
    alignment map refreshes (:meth:`_rebuild_edge_anchor_caches`, on the same
    ``update_v_every_n_epochs`` / ``warmup_epochs`` cadence as
    ``_should_update_maps_at_epoch_end``) — **not** every step.  At each
    refresh, for every edge, matched pilot rows (already restricted to the
    union — or, with ``align_on_intersection=True``, the intersection — of the
    two endpoints' target classes via :meth:`_apply_edge_class_filter`) are
    reduced by one of four mutually exclusive ``anchor_selection`` strategies
    into a *fixed set of pilot identities* cached in ``self._edge_anchor_cache``:

    * **all** (default) — the full matched pool, unchanged;
    * **random** — exactly ``num_anchors`` rows, subsampled once at the
      refresh and reused verbatim (same physical samples) for every step
      until the next refresh;
    * **proto_class** — one (or ``protos_per_class``, via within-class
      k-means) prototype per class present on the edge, à la FedProto — the
      *membership* of each class group is fixed at the refresh, but its mean
      is recomputed every step from the members' current encodings;
    * **proto_kmeans** — unsupervised: k-means (``K=num_anchors``) is fit
      *once ever* (not every refresh) on the canonical higher-dimensional
      endpoint's pilots; the resulting sample-id → cluster assignment is
      cached and reused for the rest of training to induce matching
      prototypes on the other endpoint by averaging its rows under the same
      assignment — a one-shot, broadcast-once protocol.

    Between refreshes, every step re-encodes the *same* underlying pilot
    samples through each agent's current (still-training) parameters
    (:meth:`_edge_live_anchors`) — values change every step, identities don't
    — and this fresh re-encoding drives both the per-step sheaf penalty and,
    at the next refresh, the alignment-map refit itself
    (:meth:`_fit_edge_map`): the same anchors that were communicated build
    both.  ``num_anchors`` is therefore the *total* volume communicated per
    edge per refresh window, not a per-step recurring cost.

    With ``anchor_parseval_normalize=True`` the selected anchors are further
    prewhitened per side via :func:`~src.utils.anchors.parseval_normalize`
    before the penalty/after-comm terms are computed — most useful for the
    small anchor counts the ``random``/``proto_*`` strategies produce.
    """

    def __init__(
        self,
        agents: dict[int, nn.Module],
        neighbors: dict[int, set[int]],
        optimizer,
        max_lmb: float,
        latent_dims: dict,
        # local_steps: int = 1,
        anchor_strategy: str = 'pilots',
        num_anchors: int = 128,
        anchor_selection: str = 'all',
        protos_per_class: int = 1,
        anchor_parseval_normalize: bool = False,
        lambda_schedule: str | None = None,
        sparse_communication: bool = False,
        sparse_epsilon: float = 1e-2,
        update_v_every_n_epochs: int = 1,
        warmup_epochs: int = 0,
        log_latent_diagnostics: bool = False,
        use_general_maps: bool = False,
        soft_maps: bool = False,
        comm_task_coeff: float = 0.0,
        align_on_intersection: bool = False,
        learn_whitening: bool = True,
        learn_lmb: bool = False,
        dual_rho: float = 0.1,
        dual_lr: float = 0.01,
        dual_lmb_max: float = 100.0,
        **kwargs,
    ):
        super().__init__(
            agents=agents,
            neighbors=neighbors,
            optimizer=optimizer,
            log_latent_diagnostics=log_latent_diagnostics,
            **kwargs,
        )

        anchor_strategy = str(anchor_strategy)
        update_v_every_n_epochs = int(update_v_every_n_epochs)
        warmup_epochs = int(warmup_epochs)
        num_anchors = int(num_anchors)
        anchor_selection = str(anchor_selection)
        protos_per_class = int(protos_per_class)

        if anchor_strategy != 'pilots':
            raise ValueError(
                f'Unknown anchor_strategy: {anchor_strategy}. '
                "Valid options: ['pilots']"
            )
        if update_v_every_n_epochs < 1:
            raise ValueError('update_v_every_n_epochs must be at least 1')
        if warmup_epochs < 0:
            raise ValueError('warmup_epochs must be non-negative')
        if num_anchors < 1:
            raise ValueError('num_anchors must be at least 1')
        _valid_anchor_selections = (
            'all', 'random', 'proto_class', 'proto_kmeans'
        )
        if anchor_selection not in _valid_anchor_selections:
            raise ValueError(
                f'Unknown anchor_selection: {anchor_selection}. '
                f'Valid options: {list(_valid_anchor_selections)}'
            )
        if protos_per_class < 1:
            raise ValueError('protos_per_class must be at least 1')
        if learn_lmb:
            if lambda_schedule:
                raise ValueError(
                    'learn_lmb=True (per-edge dual ascent) and '
                    'lambda_schedule are mutually exclusive: the learned '
                    'multipliers replace the scheduled global coefficient.'
                )
            if not 0.0 < float(dual_rho) < 1.0:
                raise ValueError(
                    'dual_rho is the tolerated fraction of the unaligned '
                    f'residual energy and must lie in (0, 1); got {dual_rho}.'
                )
            if float(dual_lr) <= 0.0:
                raise ValueError('dual_lr must be > 0')
            if float(dual_lmb_max) <= 0.0:
                raise ValueError('dual_lmb_max must be > 0')

        if soft_maps and not use_general_maps:
            warnings.warn(
                'soft_maps=True requires use_general_maps=True. '
                'Forcing use_general_maps=True.',
                UserWarning,
                stacklevel=2,
            )
            use_general_maps = True

        self.save_hyperparameters()
        # Sparse-vs-dense byte comparison for the (rare) sparse_communication=True
        # path in _record_edge_exchange.  use_prototypes is always False here:
        # any prototype/random/k-means row reduction already happened upstream
        # via anchor_selection (_select_edge_anchors) before this is applied.
        self._sparse_payload_config = AnchorConfig(
            use_prototypes=False,
            sparse_communication=bool(sparse_communication),
            sparse_epsilon=float(sparse_epsilon),
        )
        self._latest_pilots: dict[
            int, tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]
        ] = {}
        self._whitening_ops: dict[int, WhiteningOp] = {}
        # anchor_selection='proto_kmeans' state: per-edge sample-id -> cluster-id
        # assignment, fit once (lazily, at the first refresh with enough pooled
        # pilots) and reused for the rest of a fit() call.
        self._edge_kmeans_assign: dict[tuple[int, int], dict[int, int]] = {}
        # Fixed per-edge anchor mechanism (selection cadence == map-refresh
        # cadence, see class docstring).  All three must exist from
        # construction, not just on_train_start: Lightning's sanity-check
        # validation pass runs _shared_eval (and therefore _edge_live_anchors)
        # *before* on_train_start fires.
        self._latest_train_pilots: dict[
            int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] = {}
        self._edge_anchor_cache: dict[tuple[int, int], dict[str, torch.Tensor]] = {}
        self._frozen_edge_anchors: dict[
            tuple[int, int],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        ] = {}
        latent_dims_int = {int(k): int(v) for k, v in latent_dims.items()}
        self._build_restriction_maps(neighbors, latent_dims_int)

        # ── Per-edge dual multipliers (learn_lmb mode) ─────────────────────────
        # Normalised residuals recorded by _edge_penalty_term during the current
        # step; consumed by _dual_ascent_step and the step metrics.
        self._step_edge_residuals: dict[str, torch.Tensor] = {}
        if learn_lmb:
            self._init_edge_duals(self.stiefel_matrices.keys())

        # ── Learnable (SWBN) whitening layers ─────────────────────────────────
        # Each agent owns a learnable whitening layer g_{phi_i} and a paired
        # colouring inverse g*_{phi_i}.  `_colouring_layers` is a plain dict
        # (not ModuleDict) because SWBNColouringLayer owns no tensors — it reads
        # phi_i from the paired whitening layer already registered in
        # `self.whitening_layers`.
        self.whitening_layers = nn.ModuleDict()
        self._colouring_layers: dict[str, SWBNColouringLayer] = {}
        if bool(learn_whitening):
            for idx, d in latent_dims_int.items():
                wl = SWBNWhiteningLayer(d)
                self.whitening_layers[str(idx)] = wl
                self._colouring_layers[str(idx)] = SWBNColouringLayer(wl)

    def _build_restriction_maps(
        self, neighbors: dict, latent_dims_int: dict[int, int]
    ) -> None:
        """Create one Stiefel restriction map per edge (truncated identity init).

        Edge key ``'{node_i}_{node_j}'`` is oriented so ``d_node_i >= d_node_j``;
        the map ``V`` has shape ``(d_i, d_j)`` and embeds node j's stalk into
        node i's.  Overridden by :class:`SheafCFRL`, which instead builds two
        fat semi-orthogonal maps per edge that project into a compressed edge
        stalk.
        """
        self.stiefel_matrices = nn.ParameterDict()
        learnable = bool(self.hparams.use_general_maps) and bool(
            self.hparams.soft_maps
        )
        for i_raw, neighborset in neighbors.items():
            for j_raw in neighborset:
                i, j = int(i_raw), int(j_raw)

                if latent_dims_int[i] > latent_dims_int[j]:
                    node_i, node_j = i, j
                elif latent_dims_int[i] < latent_dims_int[j]:
                    node_i, node_j = j, i
                else:
                    node_i, node_j = max(i, j), min(i, j)

                edge_key = f'{node_i}_{node_j}'
                if edge_key not in self.stiefel_matrices:
                    d_i = latent_dims_int[node_i]
                    d_j = latent_dims_int[node_j]
                    self.stiefel_matrices[edge_key] = nn.Parameter(
                        torch.eye(d_i, d_j), requires_grad=learnable
                    )

    # ── Per-edge λ: dual-ascent machinery ──────────────────────────────────────

    def _dual_lmb_enabled(self) -> bool:
        """True when per-edge dual-ascent multipliers replace the global λ."""
        return bool(getattr(self.hparams, 'learn_lmb', False))

    def _init_edge_duals(self, edge_keys) -> None:
        """(Re)create the per-edge dual multipliers ``λ_e`` (all zero).

        Called once the edge set is known: from ``__init__`` here (Stiefel
        edges) and again by :class:`SheafCFRL` once its compressed edges
        replace the (empty) Stiefel dict.  ``dual_lambdas`` is a buffer so the
        multipliers follow the module across devices and checkpoints.
        """
        self._dual_edge_index = {
            k: i for i, k in enumerate(sorted(edge_keys))
        }
        self.register_buffer(
            'dual_lambdas', torch.zeros(len(self._dual_edge_index))
        )

    def _edge_penalty_term(
        self, edge_key: str, diff: torch.Tensor
    ) -> torch.Tensor:
        """One edge's contribution to the sheaf penalty, per the active λ mode.

        ``diff`` holds the matched-row coboundary residuals in the edge's
        coboundary space, shape ``(n, d_e)`` — ``d_e = d_j`` for embedding
        maps, ``c_ij`` for compressed edge stalks (i.e. the alignment map's
        row dimension: each row of ``diff`` lives in this space).

        The raw penalty ``mean_rows ‖diff‖²`` is always normalised by
        ``2·d_e``, its whitened-unaligned baseline (whitened uncorrelated
        latents give ``E‖diff‖² ≈ 2·d_e``; perfect alignment 0), so the
        returned term ``P̂_e`` is dimension-free and always falls roughly
        within ``[0, 1]`` — comparable across edges of different latent
        sizes (and across configs sweeping the bottleneck dimension).

        Fixed/scheduled mode: returns ``P̂_e``; the global coefficient is
        applied once by ``_shared_eval``.

        Dual mode: returns ``λ_e · P̂_e`` and records ``P̂_e`` for the
        post-step dual update.  Phase-A frozen penalties contribute two
        terms per edge (one per live endpoint); both estimate the same edge
        residual, so their recordings are averaged.
        """
        penalty = (diff**2).sum(dim=1).mean()
        normed = penalty / (2.0 * diff.shape[1])
        if not self._dual_lmb_enabled():
            return normed
        resid = normed.detach()
        prev = self._step_edge_residuals.get(edge_key)
        self._step_edge_residuals[edge_key] = (
            resid if prev is None else 0.5 * (prev + resid)
        )
        # Read λ_e as a *copy*: indexing the buffer returns a view sharing its
        # version counter, and the multiply saves it for backward — the dual
        # update later this same step would then trip the in-place version
        # check at loss.backward() (same trap as SWBNWhiteningLayer's W).
        lam = self.dual_lambdas[self._dual_edge_index[edge_key]].clone()
        return lam * normed

    @torch.no_grad()
    def _dual_ascent_step(self) -> None:
        """Projected dual ascent on the residuals recorded this step.

        ``λ_e ← clip(λ_e + dual_lr·(P̂_e − dual_rho), [0, dual_lmb_max])`` —
        each multiplier integrates its edge's constraint violations, growing
        while the edge is misaligned beyond the tolerance and decaying (to
        exactly 0) once within it.  The integration itself averages out the
        per-step pilot-batch noise, so no extra smoothing is needed.
        """
        rho = float(self.hparams.dual_rho)
        lr = float(self.hparams.dual_lr)
        cap = float(self.hparams.dual_lmb_max)
        lams = self.dual_lambdas.clone()
        for edge_key, resid in self._step_edge_residuals.items():
            i = self._dual_edge_index[edge_key]
            lams[i] = lams[i] + lr * (resid.to(lams.device) - rho)
        # Update into a fresh tensor and *reassign* the buffer rather than
        # mutate it in place: this step's forward may hold saved reads of the
        # old tensor in the autograd graph (cf. SWBNWhiteningLayer._update_W).
        self.dual_lambdas = lams.clamp_(0.0, cap)

    def _use_learnable_whitening(self) -> bool:
        """True when SWBN learnable whitening is the active normalisation.

        ``self.whitening_layers`` is non-empty only when ``learn_whitening`` was
        requested *and* parseval/L2 normalisation are off (they are mutually
        exclusive with ZCA whitening), so this single check captures both.
        """
        return bool(self.whitening_layers)

    def _whiten_pilots_frozen(self, idx: int, Z: torch.Tensor) -> torch.Tensor:
        """Whiten ``Z`` through agent ``idx``'s SWBN layer without updating it.

        Used at epoch-end (Stiefel update) and at test (``send_message``), where
        the current ``phi_i`` must be *applied* rather than re-estimated, so the
        layer is forced into eval mode (frozen W / running stats) for the call.
        """
        layer = self.whitening_layers[str(idx)]
        was_training = layer.training
        layer.eval()
        try:
            return layer(Z)
        finally:
            layer.train(was_training)

    def _whiten_node_pilots(self, idx: int, A: torch.Tensor) -> torch.Tensor:
        """Whiten node ``idx``'s pilot latents **once** (independent of degree).

        Whitening is row-wise, so it commutes with the per-edge row matching:
        whitening up-front (one call per node) gives the same result as the old
        per-edge whitening while updating the SWBN layer's ``W`` / running stats
        only once per step regardless of how many neighbours node ``idx`` has.
        """
        if self._use_learnable_whitening():
            return self.whitening_layers[str(idx)](A)
        op = self._whitening_ops.get(idx)
        return whiten(A, op) if op is not None else A

    def _recolour_node(self, idx: int, Z: torch.Tensor) -> torch.Tensor:
        """Re-colour ``Z`` into node ``idx``'s native space (inverse of whitening)."""
        if self._use_learnable_whitening():
            return self._colouring_layers[str(idx)](Z)
        op = self._whitening_ops.get(idx)
        return color(Z, op) if op is not None else Z

    def _resolve_key(self, batch: dict, idx: int) -> int | str:
        str_key = str(idx)
        return str_key if str_key in batch else idx

    def _extract_pilot_batch(
        self, batch: dict, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Extract pilot data for an agent, handling global, private, or pairwise keys."""
        if f'pilot_{idx}' in batch:
            return (
                batch[f'pilot_{idx}'][0],
                batch[f'pilot_{idx}'][1],
                batch[f'pilot_{idx}'][2],
            )
        if f'global_pilot_{idx}' in batch:
            return (
                batch[f'global_pilot_{idx}'][0],
                batch[f'global_pilot_{idx}'][1],
                batch[f'global_pilot_{idx}'][2],
            )
        if 'global_pilot' in batch:
            return (
                batch['global_pilot'][0],
                batch['global_pilot'][1],
                batch['global_pilot'][2],
            )

        for key, value in batch.items():
            if isinstance(key, str) and key.startswith('pilot_'):
                parts = key.split('_')
                if len(parts) == 3:
                    i, j = int(parts[1]), int(parts[2])
                    if i == idx:
                        return value[0], value[1], value[2]
                    if j == idx:
                        return value[3], value[4], value[5]

        raise ValueError(f'Pilot batch missing for agent {idx}.')

    @torch.no_grad()
    def _fit_edge_map(
        self,
        edge_key: str,
        node_i: int,
        node_j: int,
        Z_i: torch.Tensor,
        Z_j: torch.Tensor,
        update_maps: bool,
    ) -> dict[str, float]:
        """Refit the Stiefel map for one edge from its fixed, already-whitened anchor pair.

        ``Z_i``/``Z_j`` are the fresh re-encoding of that edge's fixed anchor
        cache (:meth:`_edge_live_anchors`), called once per refresh from
        :meth:`_rebuild_edge_anchor_caches` — replaces the old epoch-pooled
        cross-covariance computation with the SVD/least-squares fit run
        directly on the anchors that also drive the per-step penalty.
        Overridden by :class:`SheafCFRL` to run its SOC-ADMM Phase-B update on
        the same pair instead.  ``update_maps=False`` (learnable general maps
        trained by gradient descent, ``use_general_maps and soft_maps``) still
        computes the diagnostic metrics but leaves the map parameter alone.
        """
        edge_metrics: dict[str, float] = {}
        V_param = self.stiefel_matrices.get(edge_key)
        if V_param is None or Z_i.shape[0] == 0:
            return edge_metrics
        param_device = V_param.device
        A_i, A_j = Z_i.float(), Z_j.float()

        C = torch.matmul(A_i.T, A_j)
        edge_metrics[f'crosscov_effective_rank_edge_{edge_key}'] = self._effective_rank(C)

        if self.hparams.use_general_maps:
            # Unconstrained least-squares: A s.t. A_i @ A.T ≈ A_j.
            # fit_alignment returns A of shape (d_j, d_i); V = A.T is (d_i, d_j).
            if update_maps:
                A = fit_alignment(A_i, A_j)
                V_param.copy_(A.T.to(dtype=V_param.dtype, device=param_device))
        else:
            C_svd = C + torch.randn_like(C) * 1e-6
            if not torch.isfinite(C_svd).all():
                warnings.warn(
                    f"_fit_edge_map: cross-covariance for edge {edge_key} "
                    f"contains non-finite values "
                    f"({(~torch.isfinite(C_svd)).sum().item()} entries). "
                    "Replacing with 0.0 — Stiefel update for this edge may be unreliable.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                C_svd = torch.nan_to_num(C_svd, nan=0.0, posinf=0.0, neginf=0.0)
            try:
                U, S, W_T = torch.linalg.svd(C_svd, full_matrices=False)
            except RuntimeError:
                C_svd = C_svd.cpu()
                U_cpu, S, W_T_cpu = torch.linalg.svd(C_svd, full_matrices=False)
                U, W_T = U_cpu.to(param_device), W_T_cpu.to(param_device)
            n_matched = A_i.shape[0]
            if n_matched > 1:
                edge_metrics[f'mean_canonical_correlation_edge_{edge_key}'] = (
                    float(S.float().mean().item()) / (n_matched - 1)
                )
            if update_maps:
                V_param.copy_(
                    torch.matmul(U, W_T).to(dtype=V_param.dtype, device=param_device)
                )

        return edge_metrics

    def _build_agent_target_classes(self) -> dict[int, set[int]] | None:
        """Build per-agent target-class sets from the datamodule, or return None."""
        dm = getattr(self.trainer, 'datamodule', None)
        if dm is None:
            return None
        groups: dict | None = getattr(dm, 'groups', None)
        group_tc: dict | None = getattr(dm, 'group_target_classes', None)
        if not groups or not group_tc:
            return None
        agent_tc: dict[int, set[int]] = {}
        for gid, agent_ids in groups.items():
            if gid in group_tc:
                tc_set = set(group_tc[gid])
                for aid in agent_ids:
                    agent_tc[int(aid)] = tc_set
        return agent_tc if agent_tc else None

    def _apply_edge_class_filter(
        self,
        node_i: int,
        node_j: int,
        A_i: torch.Tensor,
        keys_i: torch.Tensor,
        labels_i: torch.Tensor,
        A_j: torch.Tensor,
        keys_j: torch.Tensor,
        labels_j: torch.Tensor,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor, torch.Tensor,
    ]:
        """Filter pilot rows to the union of target classes for edge (node_i, node_j).

        labels_i / labels_j are the actual class labels aligned with A_i / A_j
        (may differ from keys when keys are sample identifiers).
        Returns filtered (A_i, keys_i, labels_i, A_j, keys_j, labels_j).
        No-op when _agent_target_classes is None.
        """
        if getattr(self, '_agent_target_classes', None) is None:
            return A_i, keys_i, labels_i, A_j, keys_j, labels_j
        tc_i = self._agent_target_classes.get(node_i)
        tc_j = self._agent_target_classes.get(node_j)
        if tc_i is None or tc_j is None:
            return A_i, keys_i, labels_i, A_j, keys_j, labels_j
        if getattr(self.hparams, 'align_on_intersection', False):
            target = tc_i & tc_j
        else:
            target = tc_i | tc_j
        if not target:
            return A_i, keys_i, labels_i, A_j, keys_j, labels_j
        union_t = torch.tensor(sorted(target), dtype=labels_i.dtype, device=labels_i.device)
        mask_i = torch.isin(labels_i, union_t)
        mask_j = torch.isin(labels_j, union_t.to(labels_j.device))
        return (
            A_i[mask_i], keys_i[mask_i], labels_i[mask_i],
            A_j[mask_j], keys_j[mask_j], labels_j[mask_j],
        )

    def _match_pilots_with_labels(
        self,
        A_i: torch.Tensor,
        keys_i: torch.Tensor,
        labels_i: torch.Tensor,
        A_j: torch.Tensor,
        keys_j: torch.Tensor,
        labels_j: torch.Tensor,
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        | None
    ):
        """Match pilot rows by shared key; return (A_i, y_i, A_j, y_j, keys) for matched rows.

        Geometric matcher: pairs up rows via shared sample id (or class label,
        depending on what ``keys`` carries), also collecting the per-row class
        labels (needed for the after-communication task loss) and the matched
        keys themselves (needed by the ``proto_kmeans`` anchor-selection
        strategy to look up its cached sample-id -> cluster-id assignment).
        """
        target_keys = set(keys_i.tolist()) & set(keys_j.tolist())
        if not target_keys:
            return None
        m_i, y_i_m, m_j, y_j_m, k_m = [], [], [], [], []
        for k in sorted(target_keys):
            idx_i = torch.where(keys_i == k)[0]
            idx_j = torch.where(keys_j == k)[0]
            mc = min(len(idx_i), len(idx_j))
            if mc > 0:
                m_i.append(A_i[idx_i[:mc]])
                y_i_m.append(labels_i[idx_i[:mc]])
                m_j.append(A_j[idx_j[:mc]])
                y_j_m.append(labels_j[idx_j[:mc]])
                k_m.append(
                    torch.full(
                        (mc,), k, dtype=keys_i.dtype, device=keys_i.device
                    )
                )
        if not m_i:
            return None
        return (
            torch.cat(m_i, dim=0),
            torch.cat(y_i_m, dim=0),
            torch.cat(m_j, dim=0),
            torch.cat(y_j_m, dim=0),
            torch.cat(k_m, dim=0),
        )

    # ── Edge-scoped anchor selection (anchor_selection) ────────────────────────

    def _edge_pairs(self) -> list[tuple[str, int, int]]:
        """Canonical ``(edge_key, node_i, node_j)`` triples for every alignment edge.

        Overridden by :class:`SheafCFRL`, whose edges live in
        ``_compression_edges`` rather than ``stiefel_matrices``.
        """
        return [(k, *map(int, k.split('_'))) for k in self.stiefel_matrices]

    def _select_edge_anchors(
        self,
        node_i: int,
        node_j: int,
        Z_i: torch.Tensor,
        y_i: torch.Tensor,
        Z_j: torch.Tensor,
        y_j: torch.Tensor,
        keys: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reduce matched, class-filtered edge rows to the actual alignment anchors.

        Dispatches on ``anchor_selection`` (see the class docstring for the four
        modes), then optionally Parseval-prewhitens each side independently via
        :func:`~src.utils.anchors.parseval_normalize`.
        """
        strategy = self.hparams.anchor_selection
        if strategy == 'all' or Z_i.shape[0] == 0:
            sel = (Z_i, y_i, Z_j, y_j)
        elif strategy == 'random':
            sel = self._random_edge_anchors(Z_i, y_i, Z_j, y_j)
        elif strategy == 'proto_class':
            sel = self._class_proto_edge_anchors(Z_i, y_i, Z_j, y_j)
        else:  # 'proto_kmeans'
            sel = self._kmeans_proto_edge_anchors(
                node_i, node_j, Z_i, y_i, Z_j, y_j, keys
            )
        if self.hparams.anchor_parseval_normalize:
            Zi_s, yi_s, Zj_s, yj_s = sel
            sel = (
                parseval_normalize(Zi_s),
                yi_s,
                parseval_normalize(Zj_s),
                yj_s,
            )
        return sel

    def _random_edge_anchors(
        self,
        Z_i: torch.Tensor,
        y_i: torch.Tensor,
        Z_j: torch.Tensor,
        y_j: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fresh random subset of ``num_anchors`` matched rows, resampled every step."""
        budget = int(self.hparams.num_anchors)
        n = Z_i.shape[0]
        if n <= budget:
            return Z_i, y_i, Z_j, y_j
        idx = torch.randperm(n, device=Z_i.device)[:budget]
        return Z_i[idx], y_i[idx], Z_j[idx], y_j[idx]

    def _class_proto_edge_anchors(
        self,
        Z_i: torch.Tensor,
        y_i: torch.Tensor,
        Z_j: torch.Tensor,
        y_j: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """One (or ``protos_per_class``) prototype pair per class present on the edge.

        Rows are already matched by shared sample id, so ``y_i`` and ``y_j``
        agree per row; grouping by ``y_i`` and averaging both sides over the
        *same* rows gives paired prototypes computed from the same underlying
        samples in each agent's own space.  With ``protos_per_class > 1`` each
        class is further split via k-means on its node_i rows
        (:func:`~src.mutualinfo._common.kmeans_cluster`), recomputed fresh every
        step — cheap since it only runs on the (typically small) class subset.
        """
        protos_per_class = int(self.hparams.protos_per_class)
        Zi_rows, yi_rows, Zj_rows, yj_rows = [], [], [], []
        for c in torch.unique(y_i, sorted=True).tolist():
            mask = y_i == c
            Zi_c, Zj_c = Z_i[mask], Z_j[mask]
            n_c = Zi_c.shape[0]
            if protos_per_class <= 1 or n_c <= protos_per_class:
                Zi_rows.append(Zi_c.mean(dim=0, keepdim=True))
                Zj_rows.append(Zj_c.mean(dim=0, keepdim=True))
                yi_rows.append(y_i.new_full((1,), c))
                yj_rows.append(y_j.new_full((1,), c))
                continue
            cluster_labels = kmeans_cluster(
                Zi_c.detach().float().cpu().numpy(), n_clusters=protos_per_class
            )
            cluster_ids = torch.as_tensor(
                cluster_labels, device=Z_i.device, dtype=torch.long
            )
            for cl in torch.unique(cluster_ids).tolist():
                cmask = cluster_ids == cl
                Zi_rows.append(Zi_c[cmask].mean(dim=0, keepdim=True))
                Zj_rows.append(Zj_c[cmask].mean(dim=0, keepdim=True))
                yi_rows.append(y_i.new_full((1,), c))
                yj_rows.append(y_j.new_full((1,), c))
        if not Zi_rows:
            return Z_i[:0], y_i[:0], Z_j[:0], y_j[:0]
        return (
            torch.cat(Zi_rows, dim=0),
            torch.cat(yi_rows, dim=0),
            torch.cat(Zj_rows, dim=0),
            torch.cat(yj_rows, dim=0),
        )

    def _kmeans_proto_edge_anchors(
        self,
        node_i: int,
        node_j: int,
        Z_i: torch.Tensor,
        y_i: torch.Tensor,
        Z_j: torch.Tensor,
        y_j: torch.Tensor,
        keys: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Induce node_j prototypes from node_i's cached, one-shot k-means clusters.

        Looks up the ``(node_i, node_j) -> {sample_id: cluster_id}`` assignment
        fit once, ever, by :meth:`_rebuild_edge_anchor_caches`; rows whose key
        was not part of the fitted pool are dropped.  Each surviving cluster is averaged
        independently on both sides (inducing node_j's prototype from whichever
        of *its own* matched rows fall in that node_i-defined cluster), with the
        majority label kept for the after-comm task loss.  Falls back to
        returning the unreduced rows when no assignment has been fit yet (e.g.
        still in warmup / not enough pooled pilots) — same graceful-fallback
        idiom as ``CESheafFRL._frozen_alignment_losses`` before the first
        frozen snapshot exists.
        """
        assign_map = self._edge_kmeans_assign.get((node_i, node_j))
        if not assign_map:
            return Z_i, y_i, Z_j, y_j
        cluster_ids = torch.tensor(
            [assign_map.get(int(k), -1) for k in keys.tolist()],
            device=Z_i.device,
            dtype=torch.long,
        )
        valid = cluster_ids >= 0
        if not bool(valid.any()):
            return Z_i, y_i, Z_j, y_j
        Z_i, y_i, Z_j, y_j = Z_i[valid], y_i[valid], Z_j[valid], y_j[valid]
        cluster_ids = cluster_ids[valid]
        Zi_rows, yi_rows, Zj_rows, yj_rows = [], [], [], []
        for cl in torch.unique(cluster_ids).tolist():
            mask = cluster_ids == cl
            Zi_rows.append(Z_i[mask].mean(dim=0, keepdim=True))
            Zj_rows.append(Z_j[mask].mean(dim=0, keepdim=True))
            vals, counts = torch.unique(y_i[mask], return_counts=True)
            majority = vals[counts.argmax()].reshape(1)
            yi_rows.append(majority)
            yj_rows.append(majority)
        return (
            torch.cat(Zi_rows, dim=0),
            torch.cat(yi_rows, dim=0),
            torch.cat(Zj_rows, dim=0),
            torch.cat(yj_rows, dim=0),
        )

    def _edge_live_anchors(
        self, node_i: int, node_j: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Re-encode edge ``(node_i, node_j)``'s fixed anchor cache fresh, then reduce.

        The one place both the per-step penalty (:meth:`_both_live_alignment_losses`)
        and the epoch-end map refit (:meth:`_rebuild_edge_anchor_caches`) get
        their anchors from: the underlying pilot *identities* are fixed for the
        whole refresh window (see :meth:`_rebuild_edge_anchor_caches`), but
        every call re-encodes them through the agents' *current* (still
        training) parameters — gradients/values are always fresh even though
        which samples they come from never changes mid-window.  Returns
        ``None`` before any cache exists for this edge (e.g. Lightning's
        sanity-check pass, which runs before ``on_train_start``).
        """
        cache = self._edge_anchor_cache.get((node_i, node_j))
        if cache is None:
            return None
        Z_i, y_i = self._edge_live_one_side(node_i, cache, 'i')
        Z_j, y_j = self._edge_live_one_side(node_j, cache, 'j')
        keys = cache['sample_ids'].to(self.device)
        return self._select_edge_anchors(node_i, node_j, Z_i, y_i, Z_j, y_j, keys)

    def _edge_live_one_side(
        self, node_idx: int, cache: dict[str, torch.Tensor], side: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Re-encode + whiten just one side (``'i'`` or ``'j'``) of an edge's anchor cache.

        Split out of :meth:`_edge_live_anchors` so :class:`CESheafFRL` can
        re-encode only the *live* node during Phase A, pairing it against the
        other side's frozen snapshot (``self._frozen_edge_anchors``) instead
        of a second live encoding.
        """
        agent = self.agents[str(node_idx)]
        x = cache[f'x_{side}'].to(self.device)
        Z = self._whiten_node_pilots(node_idx, agent.encode(x))
        y = cache[f'y_{side}'].to(self.device)
        return Z, y

    def _record_edge_exchange(
        self,
        edge_key: str,
        Z_i: torch.Tensor,
        Z_j: torch.Tensor,
        prefix: str = 'train',
    ) -> None:
        """Record the communication payload for one edge's freshly (re)selected anchors.

        Called once per edge per refresh from :meth:`_rebuild_edge_anchor_caches`
        — *not* once per step: under the fixed-anchor-cache design the same
        ``num_anchors`` rows drive every step's penalty until the next
        refresh, so the real "wire" cost is paid exactly once per window.
        ``edge_key`` is unused here but available to overrides (e.g.
        :class:`SheafCFRL`, which sends the shared compressed dimension
        ``c_ij`` on both sides instead of ``Z_i``/``Z_j``'s own raw dims).
        """
        n_rows = int(Z_i.shape[0])
        if n_rows <= 0:
            return
        if bool(self.hparams.sparse_communication):
            payload_i = communication_anchor_payload(
                anchor_matrix=Z_i.detach(), labels=None, config=self._sparse_payload_config
            )
            payload_j = communication_anchor_payload(
                anchor_matrix=Z_j.detach(), labels=None, config=self._sparse_payload_config
            )
            self._record_communication(payload_i, prefix=prefix)
            self._record_communication(payload_j, prefix=prefix)
        else:
            self._record_communication(n_rows * Z_i.shape[1], prefix=prefix)
            self._record_communication(n_rows * Z_j.shape[1], prefix=prefix)

    @torch.no_grad()
    def _rebuild_edge_anchor_caches(self) -> dict[str, float]:
        """(Re)select each edge's fixed anchor set and refit its alignment map from it.

        Called once per refresh window, from ``on_train_epoch_end``, gated by
        ``_should_update_maps_at_epoch_end`` — the same event that used to
        trigger the old epoch-pooled Stiefel refit.  For each edge: build a
        fresh, class-filtered, sample-id-matched candidate pool from the
        *last* train step's raw pilots (``self._latest_train_pilots``); reduce
        it to the fixed anchor identities that get reused every step until the
        next refresh (``random``: subsample once to ``num_anchors``;
        ``proto_kmeans``: fit its one-shot clustering the first time, then
        restrict to assigned rows; ``all``/``proto_class``: keep the full
        matched pool — their reduction is per-step averaging over a fixed
        membership, not identity selection, so it stays in
        :meth:`_edge_live_anchors`); cache it; immediately re-encode it once
        (the "just selected" pass) to record the communication cost, snapshot
        the frozen pair for :class:`CESheafFRL`'s Phase A, and fit the
        alignment map.
        """
        edge_metrics: dict[str, float] = {}
        update_maps = not (
            self.hparams.use_general_maps and self.hparams.soft_maps
        )
        recorded_round = False
        strategy = self.hparams.anchor_selection

        for edge_key, node_i, node_j in self._edge_pairs():
            if (
                node_i not in self._latest_train_pilots
                or node_j not in self._latest_train_pilots
            ):
                continue
            x_i, y_i, sid_i = (
                t.to(self.device) for t in self._latest_train_pilots[node_i]
            )
            x_j, y_j, sid_j = (
                t.to(self.device) for t in self._latest_train_pilots[node_j]
            )

            x_i_f, sid_i_f, y_i_f, x_j_f, sid_j_f, y_j_f = self._apply_edge_class_filter(
                node_i, node_j, x_i, sid_i, y_i, x_j, sid_j, y_j
            )
            matched = self._match_pilots_with_labels(
                x_i_f, sid_i_f, y_i_f, x_j_f, sid_j_f, y_j_f
            )
            if matched is None:
                continue
            x_i_m, y_i_m, x_j_m, y_j_m, sids_m = matched

            if strategy == 'random':
                budget = int(self.hparams.num_anchors)
                if x_i_m.shape[0] > budget:
                    idx = torch.randperm(x_i_m.shape[0], device=x_i_m.device)[:budget]
                    x_i_m, y_i_m, x_j_m, y_j_m, sids_m = (
                        x_i_m[idx], y_i_m[idx], x_j_m[idx], y_j_m[idx], sids_m[idx]
                    )
            elif strategy == 'proto_kmeans':
                pair = (node_i, node_j)
                K = int(self.hparams.num_anchors)
                if pair not in self._edge_kmeans_assign and x_i_m.shape[0] >= K:
                    z_ref = self.agents[str(node_i)].encode(x_i_m)
                    labels = kmeans_cluster(z_ref.float().cpu().numpy(), n_clusters=K)
                    self._edge_kmeans_assign[pair] = dict(
                        zip(sids_m.tolist(), labels.tolist())
                    )
                assign_map = self._edge_kmeans_assign.get(pair)
                if assign_map:
                    keep = torch.tensor(
                        [s in assign_map for s in sids_m.tolist()],
                        device=x_i_m.device,
                        dtype=torch.bool,
                    )
                    if bool(keep.any()):
                        x_i_m, y_i_m, x_j_m, y_j_m, sids_m = (
                            x_i_m[keep], y_i_m[keep], x_j_m[keep], y_j_m[keep], sids_m[keep]
                        )

            self._edge_anchor_cache[(node_i, node_j)] = {
                'x_i': x_i_m.cpu(), 'y_i': y_i_m.cpu(),
                'x_j': x_j_m.cpu(), 'y_j': y_j_m.cpu(),
                'sample_ids': sids_m.cpu(),
            }

            live = self._edge_live_anchors(node_i, node_j)
            if live is None:
                continue
            Z_i, y_i_l, Z_j, y_j_l = live
            if not recorded_round:
                self._record_communication_round(n_rounds=1, prefix='train')
                recorded_round = True
            self._record_edge_exchange(edge_key, Z_i, Z_j, prefix='train')
            self._frozen_edge_anchors[(node_i, node_j)] = (
                Z_i.detach(), y_i_l.detach(), Z_j.detach(), y_j_l.detach()
            )
            edge_metrics.update(
                self._fit_edge_map(edge_key, node_i, node_j, Z_i, Z_j, update_maps)
            )

        return edge_metrics

    def on_train_start(self) -> None:
        super().on_train_start()
        self._latest_pilots.clear()
        self._whitening_ops.clear()
        self._task_latent_buffer: dict[int, list[torch.Tensor]] = {}
        self._agent_target_classes: dict[int, set[int]] | None = (
            self._build_agent_target_classes()
        )
        # Fresh fit() call -> forget any proto_kmeans assignment and any fixed
        # anchor caches fit/selected previously.
        self._edge_kmeans_assign = {}
        self._latest_train_pilots = {}
        self._edge_anchor_cache = {}
        self._frozen_edge_anchors = {}

    def on_train_epoch_start(self) -> None:
        self._latest_pilots.clear()
        self._task_latent_buffer = {}

    @torch.no_grad()
    def on_train_epoch_end(self) -> None:
        if self._dual_lmb_enabled() and self._dual_edge_index:
            self.log_dict(
                {
                    f'train/dual_lmb_edge_{k}': float(self.dual_lambdas[i])
                    for k, i in self._dual_edge_index.items()
                },
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                add_dataloader_idx=False,
            )

        # Whether to refresh the fixed anchor caches + alignment maps this
        # epoch.  Hook: SheafFRL refreshes every `update_v_every_n_epochs`
        # after warmup; CESheafFRL refreshes only on Phase-C epochs.
        if self._should_update_maps_at_epoch_end():
            # With learnable SWBN whitening, phi_i lives in the persistent
            # layers (updated online every step); this closed-form ZCA fit is
            # only needed for learn_whitening=False.
            if not self._use_learnable_whitening():
                for idx, chunks in self._task_latent_buffer.items():
                    if chunks:
                        self._whitening_ops[idx] = fit_whitening(
                            torch.cat(chunks, dim=0).float()
                        )
            edge_metrics = self._rebuild_edge_anchor_caches()
            if edge_metrics:
                self.log_dict(
                    {f'train/{k}': v for k, v in edge_metrics.items()},
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    add_dataloader_idx=False,
                )

        self._latest_pilots.clear()
        self._task_latent_buffer = {}
        self._finalize_train_epoch_communication()
        self._log_train_comm_task_perf()

    # ── Epoch/communication hooks (overridden by CESheafFRL) ──────────────────

    def _should_update_maps_at_epoch_end(self) -> bool:
        """Refresh maps every ``update_v_every_n_epochs`` epochs, after warmup."""
        return (
            self.current_epoch >= self.hparams.warmup_epochs
            and self.current_epoch % self.hparams.update_v_every_n_epochs == 0
        )

    def on_validation_epoch_end(self) -> None:
        super().on_validation_epoch_end()

    def _compute_alignment_losses(
        self, comm_weight: float, skip: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-step entry point for the alignment losses (both-live).

        SheafFRL evaluates the both-live penalty every step; :class:`CESheafFRL`
        overrides this to phase-dispatch (Phase A → frozen coboundary, else
        both-live).  The geometry lives in ``_both_live_alignment_losses`` /
        ``_frozen_alignment_losses`` so subclasses can swap it (e.g. SheafCFRL's
        compressed coboundary) without re-implementing the schedule.
        """
        return self._both_live_alignment_losses(comm_weight, skip)

    def _both_live_alignment_losses(
        self, comm_weight: float, skip: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Both-live sheaf penalty + after-comm over all edges (embedding maps).

        One Stiefel map per edge, embedding coboundary ``z_i V − z_j``; both
        endpoints re-encoded fresh from their fixed anchor cache
        (:meth:`_edge_live_anchors`).  :class:`SheafCFRL` overrides this with
        the compressed two-map coboundary.  ``comm_weight > 0`` enables the
        after-comm term.
        """
        sheaf_penalty = torch.tensor(0.0, device=self.device)
        after_comm_task_loss = torch.tensor(0.0, device=self.device)
        if skip:
            return sheaf_penalty, after_comm_task_loss

        for edge_key, V in self.stiefel_matrices.items():
            node_i, node_j = map(int, edge_key.split('_'))
            matched = self._edge_live_anchors(node_i, node_j)
            if matched is None:
                continue

            # Rows are already whitened (inside _edge_live_anchors).
            Z_i, y_i_shared, Z_j, y_j_shared = matched
            diff = torch.matmul(Z_i, V) - Z_j
            sheaf_penalty += self._edge_penalty_term(edge_key, diff)

            # ── After-communication task loss ─────────────────────────────────
            # • j→i: align Z_j into node_i's space, decode with agent_i's decoder,
            #         compute task loss against node_j's pilot labels.
            # • i→j: align Z_i into node_j's space, decode with agent_j's decoder,
            #         compute task loss against node_i's pilot labels.
            if comm_weight > 0.0:
                agent_i = self.agents[str(node_i)] if str(node_i) in self.agents else None
                agent_j = self.agents[str(node_j)] if str(node_j) in self.agents else None
                _is_clf = lambda a: getattr(a, 'task_type', 'classification') == 'classification'

                # Inverse map: for Stiefel (semi-orthogonal cols) V.T; general → pinv(V).
                if self.hparams.use_general_maps:
                    V_inv = torch.linalg.pinv(V.float())
                else:
                    V_inv = V.float().T

                # j → i direction
                if agent_i is not None and _is_clf(agent_i):
                    Z_j_to_i = torch.matmul(Z_j.float(), V_inv)
                    Z_j_to_i = self._recolour_node(node_i, Z_j_to_i)
                    logits_ji = agent_i.decoder(Z_j_to_i.to(dtype=Z_i.dtype))
                    after_comm_task_loss += agent_i.compute_loss(
                        logits_ji, y_j_shared.to(self.device)
                    )

                # i → j direction
                if agent_j is not None and _is_clf(agent_j):
                    Z_i_to_j = torch.matmul(Z_i.float(), V.float())
                    Z_i_to_j = self._recolour_node(node_j, Z_i_to_j)
                    logits_ij = agent_j.decoder(Z_i_to_j.to(dtype=Z_j.dtype))
                    after_comm_task_loss += agent_j.compute_loss(
                        logits_ij, y_i_shared.to(self.device)
                    )

        return sheaf_penalty, after_comm_task_loss

    def _shared_eval(
        self,
        batch: dict[int, list[torch.Tensor]],
        batch_idx: int,
        prefix: str,
    ):
        if isinstance(batch, tuple):
            batch = batch[0]

        outputs = {}
        agent_losses = {}
        agent_performances = {}

        pilots_available = True

        for idx_str, agent in self.agents.items():
            idx = int(idx_str)

            # ── Task loss ────────────────────────────────────────────────────
            x_task, y_task = batch[self._resolve_key(batch, idx)]
            latent_task = agent.encode(x_task)
            y_hat = agent.decoder(latent_task)
            outputs[idx_str] = (y_hat.detach(), y_task)
            agent_losses[idx] = agent.compute_loss(y_hat, y_task)
            agent_performances[idx] = agent.task_performance(y_hat, y_task)

            if prefix == 'train':
                self._task_latent_buffer.setdefault(idx, []).append(
                    latent_task.detach().cpu()
                )

            # ── Pilot extraction ─────────────────────────────────────────────
            if pilots_available:
                try:
                    x_pilot, y_pilot, sample_ids = self._extract_pilot_batch(
                        batch, idx
                    )
                except ValueError:
                    pilots_available = False
                else:
                    self._latest_pilots[idx] = (x_pilot, y_pilot, sample_ids)
                    pilot_latents = agent.encode(x_pilot)
                    # Whiten every node's pilots ONCE per step for SWBN's online
                    # W / running-stat update and diagnostics; the alignment
                    # penalty no longer consumes this pass — it re-encodes each
                    # edge's fixed anchor cache separately (_edge_live_anchors).
                    self._whiten_node_pilots(idx, pilot_latents)

                    if prefix == 'train':
                        # Raw pilots (not yet encoded) from the most recent train
                        # step; used at the next map refresh to rebuild the fixed
                        # per-edge anchor caches (_rebuild_edge_anchor_caches).
                        self._latest_train_pilots[idx] = (
                            x_pilot.detach().cpu(),
                            y_pilot.detach().cpu(),
                            sample_ids.detach().cpu(),
                        )
                        if getattr(self.hparams, 'log_latent_diagnostics', False):
                            self.log(
                                f'train/global_pilot_effective_rank_agent_{idx}',
                                self._effective_rank(pilot_latents.detach().float()),
                                on_step=True,
                                on_epoch=False,
                                prog_bar=False,
                                add_dataloader_idx=False,
                            )

        total_task_loss = torch.stack(list(agent_losses.values())).sum()

        comm_weight = float(getattr(self.hparams, 'comm_task_coeff', 0.0))

        in_warmup = self.current_epoch < self.hparams.warmup_epochs
        self._step_edge_residuals = {}
        sheaf_penalty, after_comm_task_loss = self._compute_alignment_losses(
            comm_weight=comm_weight,
            skip=in_warmup,
        )

        # λ handling — fixed/scheduled: one global coefficient applied here;
        # dual (learn_lmb): every edge term already carries its own λ_e, and
        # the multipliers take a projected ascent step on the residuals just
        # recorded (training steps only).
        if self._dual_lmb_enabled():
            lmb_coeff = 1.0
            if prefix == 'train' and self._step_edge_residuals:
                self._dual_ascent_step()
        else:
            lmb_coeff = self._effective_lambda_reg()

        total_loss = (
            total_task_loss
            + lmb_coeff * sheaf_penalty
            + comm_weight * after_comm_task_loss
        )

        extra: dict[str, Any] = {
            # Online-learned whitened-coboundary residual at the current
            # Stiefel/compression maps.  The standardised cross-orchestrator
            # `misalignment_loss` (with per-edge breakdown) is instead logged
            # at train/test epoch end through the generic pilot-based
            # evaluator (see evaluate_communication_accuracy below), so it is
            # directly comparable with the post-training-alignment baselines.
            f'{prefix}/sheaf_penalty': sheaf_penalty,
        }
        if comm_weight > 0.0:
            extra[f'{prefix}/after_comm_task_loss'] = after_comm_task_loss
        if self._dual_lmb_enabled() and self._step_edge_residuals:
            extra[f'{prefix}/mean_edge_residual'] = torch.stack(
                list(self._step_edge_residuals.values())
            ).mean()
            extra[f'{prefix}/mean_dual_lmb'] = self.dual_lambdas.mean()

        self._log_shared_metrics(
            prefix=prefix,
            agent_losses=agent_losses,
            agent_performances=agent_performances,
            batch_size=self._resolve_batch_size(batch),
            agent_sample_counts=self._resolve_agent_sample_counts(batch),
            total_loss=total_loss,
            extra_metrics=extra,
            prog_bar=False,
            per_agent_loss_name='task_loss',
            skip_task_performance=(prefix == 'test'),
        )

        return outputs, total_loss

    # ── Communication accuracy evaluation ─────────────────────────────────────

    @torch.no_grad()
    def send_message(
        self,
        sender_idx: int,
        receiver_idx: int,
        Z_sender: torch.Tensor,
    ) -> torch.Tensor:
        """Transform sender's test latents into receiver's latent space.

        Pipeline:
        1. Whiten sender's test latents with the sender's whitening map g_{phi}.
        2. Apply the learned Stiefel alignment map for the (sender, receiver) edge.
        3. Re-colour with the receiver's colouring map g*_{phi}.

        When ``learn_whitening`` is on, g_{phi}/g*_{phi} are the per-agent SWBN
        layers (frozen here — applied, not re-estimated); otherwise they are the
        buffer-and-fit ``WhiteningOp``s in ``self._whitening_ops`` fitted on pilot
        latents at the end of the last training epoch.

        Parameters
        ----------
        sender_idx : int
            Index of the sending agent.
        receiver_idx : int
            Index of the receiving agent.
        Z_sender : torch.Tensor
            Raw test latent representations of the sender, shape ``(n, d_sender)``.

        Returns
        -------
        torch.Tensor
            Reconstructed representations in the receiver's latent space,
            shape ``(n, d_receiver)``, on the same device as ``Z_sender``.
        """
        op_sender = self._whitening_ops.get(sender_idx)
        op_receiver = self._whitening_ops.get(receiver_idx)
        use_learnable = self._use_learnable_whitening()

        dev = Z_sender.device
        # SWBN layer buffers live on the module device; run the pipeline there
        # and move the result back to the caller's device at the end.
        work_dev = self.device if use_learnable else dev

        # Step 1 — whiten with sender's training statistics (g_{phi_sender}).
        if use_learnable:
            Z = self._whiten_pilots_frozen(sender_idx, Z_sender.to(work_dev))
        elif op_sender is not None:
            Z = whiten(Z_sender, op_sender)
        else:
            Z = Z_sender.float()

        # Step 2 — apply the Stiefel alignment map.
        # Edge convention: key '{node_i}_{node_j}' with d_node_i >= d_node_j.
        # V maps  whitened node_i space → whitened node_j space  (shape d_i × d_j).
        # V.T maps whitened node_j space → whitened node_i space.
        edge_key_ij = f'{sender_idx}_{receiver_idx}'
        edge_key_ji = f'{receiver_idx}_{sender_idx}'

        if edge_key_ij in self.stiefel_matrices:
            V = self.stiefel_matrices[edge_key_ij].float().to(work_dev)
            Z_aligned = Z @ V
        elif edge_key_ji in self.stiefel_matrices:
            V = self.stiefel_matrices[edge_key_ji].float().to(work_dev)
            # For general (non-orthogonal) maps V.T is not the inverse; use pinv.
            V_inv = torch.linalg.pinv(V) if self.hparams.use_general_maps else V.T
            Z_aligned = Z @ V_inv
        else:
            Z_aligned = Z

        # Step 3 — re-colour with receiver's statistics (g*_{phi_receiver}).
        if use_learnable:
            return self._colouring_layers[str(receiver_idx)](Z_aligned).to(dev)
        if op_receiver is not None:
            return color(Z_aligned, op_receiver).to(dev)
        return Z_aligned.to(dev)

    @torch.no_grad()
    def _whiten_own_latents(self, idx: int, Z: torch.Tensor) -> torch.Tensor:
        """Whiten with agent ``idx``'s current whitening operator (frozen, applied not re-fit)."""
        if self._use_learnable_whitening():
            return self._whiten_pilots_frozen(idx, Z)
        op = self._whitening_ops.get(idx)
        if op is None:
            raise NotImplementedError(
                f'no whitening operator fitted for agent {idx}'
            )
        return whiten(Z, op)

    def _directed_alignment_map(
        self, sender_idx: int, receiver_idx: int
    ) -> torch.Tensor | None:
        """The Stiefel/general map stored for exactly sender_idx -> receiver_idx.

        Edges are stored once per pair under a single canonical key (see
        ``send_message``'s convention). Only the direction matching that
        stored key returns a map; the reverse direction returns ``None``
        rather than falling back to ``V.T``/``pinv(V)`` — those are a
        communication convenience, not a fitted map for that orientation, so
        they must not be used to score alignment quality.
        """
        edge_key = f'{sender_idx}_{receiver_idx}'
        V = self.stiefel_matrices.get(edge_key)
        if V is None:
            return None
        return V.float()

    def evaluate_communication_accuracy(
        self, dm, prefix: str = 'test'
    ) -> dict[str, float]:
        """Cross-agent accuracy plus the generic pilot-based misalignment loss.

        Extends the base evaluation with
        :meth:`BaseOrchestrator.evaluate_misalignment_loss`, which (via
        ``_whiten_own_latents`` / ``_directed_alignment_map`` above) measures
        the coboundary residual in the same whitened space and the same
        single stored direction per edge that ``{prefix}/sheaf_penalty``
        uses during training — no re-colouring, no decoder, no
        transpose/pseudo-inverse fallback for the reverse direction. This
        logs the same ``{prefix}/misalignment_loss`` and per-edge
        ``{prefix}/misalignment_loss_edge_i_j`` keys as the
        post-training-alignment baselines, computed by the identical
        evaluator, so the values are now *exactly* the same quantity as
        ``{prefix}/sheaf_penalty`` (up to which pilot samples/whitening
        operator snapshot are current at evaluation time) and directly
        comparable across orchestrators.
        """
        logs = dict(super().evaluate_communication_accuracy(dm, prefix=prefix))
        logs.update(self.evaluate_misalignment_loss(dm, prefix=prefix))
        return logs

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

