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
from src.orchestrators.base_orchestrator import BaseOrchestrator
from src.utils.anchors import (
    AnchorConfig,
    communication_anchor_payload,
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
        use_prototypes: bool = False,
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

        if anchor_strategy != 'pilots':
            raise ValueError(
                f'Unknown anchor_strategy: {anchor_strategy}. '
                "Valid options: ['pilots']"
            )
        if update_v_every_n_epochs < 1:
            raise ValueError('update_v_every_n_epochs must be at least 1')
        if warmup_epochs < 0:
            raise ValueError('warmup_epochs must be non-negative')
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
        self.anchor_config = AnchorConfig(
            use_prototypes=bool(use_prototypes),
            sparse_communication=bool(sparse_communication),
            sparse_epsilon=float(sparse_epsilon),
        )
        self._latest_pilots: dict[
            int, tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]
        ] = {}
        self._whitening_ops: dict[int, WhiteningOp] = {}
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

    def _compute_class_prototypes(
        self,
        anchor_matrix: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if anchor_matrix.numel() == 0 or labels.numel() == 0:
            return anchor_matrix[:0], labels[:0]

        prototypes, prototype_labels = [], []
        for class_label in torch.unique(labels, sorted=True).tolist():
            mask = labels == class_label
            if mask.any():
                prototypes.append(anchor_matrix[mask].mean(dim=0))
                prototype_labels.append(class_label)

        if not prototypes:
            return anchor_matrix[:0], labels[:0]

        return torch.stack(prototypes, dim=0), torch.tensor(
            prototype_labels,
            device=labels.device,
            dtype=labels.dtype,
        )

    def _apply_prototype_collapse(
        self,
        whitened_per_agent: dict[int, torch.Tensor],
        keys_per_agent: dict[int, torch.Tensor],
        labels_per_agent: dict[int, torch.Tensor],
    ) -> None:
        """Collapse each node's whitened pilot rows to one prototype per class.

        In prototype mode the per-step sheaf penalty aligns class prototypes
        rather than individual pilot rows.  The collapse runs *after* whitening:
        the SWBN online update keeps estimating its statistics on per-sample
        rows, and — whitening being affine — it commutes exactly with the class
        mean, so these prototypes equal whitened raw-latent prototypes.

        Keys and labels are both replaced by the prototype class labels
        (in-place in the dicts), so the downstream edge class filter
        (union/intersection of target classes via ``align_on_intersection``)
        and ``_match_keys`` operate class-wise: one matched row per shared
        class per edge.  Gradients flow through the class means.
        """
        for idx in list(whitened_per_agent):
            protos, proto_labels = self._compute_class_prototypes(
                whitened_per_agent[idx], labels_per_agent[idx]
            )
            whitened_per_agent[idx] = protos
            keys_per_agent[idx] = proto_labels
            labels_per_agent[idx] = proto_labels

    def _pilot_match_keys(
        self, y_pilot: torch.Tensor, sample_ids: torch.Tensor | None
    ) -> torch.Tensor:
        if sample_ids is not None:
            return sample_ids
        return y_pilot

    @torch.no_grad()
    def _encode_pilots_eval(
        self, agent: nn.Module, x_pilot: torch.Tensor
    ) -> torch.Tensor:
        was_training = agent.training
        agent.eval()
        try:
            return agent.encode(x_pilot)
        finally:
            agent.train(was_training)

    def _match_keys(
        self,
        A_i: torch.Tensor,
        A_j: torch.Tensor,
        keys_i: torch.Tensor,
        keys_j: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Universal geometric matcher: pairs up matrices via Sample IDs or Classes."""
        target_keys = set(keys_i.tolist()) & set(keys_j.tolist())

        if not target_keys:
            return None

        matched_i, matched_j = [], []
        for k in sorted(target_keys):
            idx_i = torch.where(keys_i == k)[0]
            idx_j = torch.where(keys_j == k)[0]
            matched_count = min(len(idx_i), len(idx_j))
            if matched_count > 0:
                matched_i.append(A_i[idx_i[:matched_count]])
                matched_j.append(A_j[idx_j[:matched_count]])

        if not matched_i:
            return None
        return torch.cat(matched_i, dim=0), torch.cat(matched_j, dim=0)

    @torch.no_grad()
    def _whiten_epoch_latents(
        self,
        latents_per_agent: dict[int, torch.Tensor],
        whitening_ops: dict[int, WhiteningOp] | None,
    ) -> tuple[dict[int, torch.Tensor], dict[int, WhiteningOp]]:
        """Whiten each agent's epoch-accumulated pilots once for the Phase-B update.

        SWBN layers are applied frozen (current phi_i, not re-estimated); closed-
        form ZCA uses the supplied ops or fits on the spot when not supplied.
        Returns whitened latents and any newly fitted ZCA ops (empty for SWBN).
        """
        use_learnable = self._use_learnable_whitening()
        agent_normed: dict[int, torch.Tensor] = {}
        fitted_ops: dict[int, WhiteningOp] = {}
        for idx, A in latents_per_agent.items():
            if use_learnable:
                normed = self._whiten_pilots_frozen(idx, A).to(
                    dtype=A.dtype, device=A.device
                )
            else:
                op = (
                    whitening_ops[idx]
                    if (whitening_ops and idx in whitening_ops)
                    else fit_whitening(A.float())
                )
                fitted_ops[idx] = op
                normed = whiten(A, op).to(dtype=A.dtype, device=A.device)
            if not torch.isfinite(normed).all():
                warnings.warn(
                    f"_whiten_epoch_latents: whitened latents for agent {idx} "
                    f"contain non-finite values "
                    f"({(~torch.isfinite(normed)).sum().item()} entries). "
                    "Replacing with 0.0 — check for training instability.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                normed = torch.nan_to_num(normed, nan=0.0, posinf=0.0, neginf=0.0)
            agent_normed[idx] = normed
        return agent_normed, fitted_ops

    @torch.no_grad()
    def _update_stiefel_matrices(
        self,
        latents_per_agent: dict[int, torch.Tensor],
        keys_per_agent: dict[int, torch.Tensor],
        whitening_ops: dict[int, WhiteningOp] | None = None,
        update_maps: bool = True,
    ) -> tuple[dict[str, float], dict[int, WhiteningOp]]:
        """Update Stiefel matrices via SVD of the whitened cross-covariance.

        Returns
        -------
        edge_metrics : dict mapping metric keys to float values.
            Contains both ``crosscov_effective_rank_edge_{k}`` (effective rank
            of the cross-covariance matrix C) and
            ``mean_canonical_correlation_edge_{k}`` (mean(S)/(n-1)) per edge.
        whitening_ops  : dict[agent_idx → WhiteningOp] fitted on the full-epoch
            latents passed in.  Callers should cache these for use in the next
            epoch's forward pass so whitening is never computed on a single
            mini-batch.
        """
        if not self.stiefel_matrices:
            return {}, {}

        edge_metrics: dict[str, float] = {}
        param_device = next(iter(self.stiefel_matrices.values())).device

        agent_normed, fitted_ops = self._whiten_epoch_latents(
            latents_per_agent, whitening_ops
        )

        for edge_key, V_param in self.stiefel_matrices.items():
            node_i, node_j = map(int, edge_key.split('_'))

            if node_i not in agent_normed or node_j not in agent_normed:
                continue

            # At epoch level keys_per_agent are class labels (not sample IDs),
            # so they double as both keys and labels for class-based filtering.
            A_i_f, keys_i_f, _, A_j_f, keys_j_f, _ = self._apply_edge_class_filter(
                node_i, node_j,
                agent_normed[node_i], keys_per_agent[node_i], keys_per_agent[node_i],
                agent_normed[node_j], keys_per_agent[node_j], keys_per_agent[node_j],
            )
            shared_rows = self._match_keys(
                A_i=A_i_f,
                A_j=A_j_f,
                keys_i=keys_i_f,
                keys_j=keys_j_f,
            )
            if shared_rows is None:
                continue

            A_i, A_j = shared_rows

            # Cross-covariance diagnostics are meaningful regardless of map type.
            C = torch.matmul(A_i.float().T, A_j.float())
            edge_metrics[f'crosscov_effective_rank_edge_{edge_key}'] = self._effective_rank(C)

            if self.hparams.use_general_maps:
                # Unconstrained least-squares: A s.t. A_i @ A.T ≈ A_j.
                # fit_alignment returns A of shape (d_j, d_i); V = A.T is (d_i, d_j).
                if update_maps:
                    A = fit_alignment(A_i.float(), A_j.float())
                    V_param.copy_(
                        A.T.to(dtype=V_param.dtype, device=param_device)
                    )
            else:
                C_svd = C + torch.randn_like(C) * 1e-6
                if not torch.isfinite(C_svd).all():
                    import warnings
                    warnings.warn(
                        f"_update_stiefel_matrices: cross-covariance for edge {edge_key} "
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

        return edge_metrics, fitted_ops

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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Match pilot rows by shared key; return (A_i, y_i, A_j, y_j) for matched rows.

        Extends _match_keys to also collect the per-row class labels, which are
        needed for the after-communication task loss.
        """
        target_keys = set(keys_i.tolist()) & set(keys_j.tolist())
        if not target_keys:
            return None
        m_i, y_i_m, m_j, y_j_m = [], [], [], []
        for k in sorted(target_keys):
            idx_i = torch.where(keys_i == k)[0]
            idx_j = torch.where(keys_j == k)[0]
            mc = min(len(idx_i), len(idx_j))
            if mc > 0:
                m_i.append(A_i[idx_i[:mc]])
                y_i_m.append(labels_i[idx_i[:mc]])
                m_j.append(A_j[idx_j[:mc]])
                y_j_m.append(labels_j[idx_j[:mc]])
        if not m_i:
            return None
        return (
            torch.cat(m_i, dim=0),
            torch.cat(y_i_m, dim=0),
            torch.cat(m_j, dim=0),
            torch.cat(y_j_m, dim=0),
        )

    def _match_edge(
        self,
        node_i: int,
        node_j: int,
        src_i: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        src_j: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Class-filter + key-match two pilot sources for edge (node_i, node_j).

        Each source is ``(A, keys, labels)``.  Returns ``(Z_i, y_i, Z_j, y_j)`` for
        the matched rows, or ``None``.  Used for both the live/live penalty and the
        live/frozen (Phase-A) penalty, so one source may be a frozen snapshot.
        """
        A_i, k_i, l_i = src_i
        A_j, k_j, l_j = src_j
        A_i_f, k_i_f, l_i_f, A_j_f, k_j_f, l_j_f = self._apply_edge_class_filter(
            node_i, node_j, A_i, k_i, l_i, A_j, k_j, l_j
        )
        return self._match_pilots_with_labels(
            A_i_f, k_i_f, l_i_f, A_j_f, k_j_f, l_j_f
        )

    def on_train_start(self) -> None:
        super().on_train_start()
        self._latest_pilots.clear()
        self._whitening_ops.clear()
        self._pilot_latent_buffer: dict[
            int, list[tuple[torch.Tensor, torch.Tensor]]
        ] = {}
        self._task_latent_buffer: dict[int, list[torch.Tensor]] = {}
        self._agent_target_classes: dict[int, set[int]] | None = (
            self._build_agent_target_classes()
        )

    def on_train_epoch_start(self) -> None:
        self._latest_pilots.clear()
        self._pilot_latent_buffer = {}
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

        # Whether to refresh the whitening ops + Stiefel maps at this epoch end.
        # Hook: SheafFRL refreshes every `update_v_every_n_epochs` after warmup;
        # CESheafFRL refreshes only on Phase-C epochs.
        do_update = self._should_update_maps_at_epoch_end()

        if not do_update:
            self._on_maps_refreshed(False)
            self._latest_pilots.clear()
            self._pilot_latent_buffer = {}
            self._task_latent_buffer = {}
            self._finalize_train_epoch_communication()
            self._log_train_comm_task_perf()
            return

        epoch_latents: dict[int, torch.Tensor] = {}
        epoch_keys: dict[int, torch.Tensor] = {}

        for idx, buf in self._pilot_latent_buffer.items():
            if not buf:
                continue

            raw_all = torch.cat([z for z, _ in buf], dim=0).to(self.device)
            label_all = torch.cat([y for _, y in buf], dim=0).to(self.device)

            # Prototype aggregation over the full epoch's pilots gives better
            # class estimates than a single last-batch.
            # Normalization (parseval/L2/whitening) is handled inside
            # _update_stiefel_matrices — do NOT apply it again here.
            if self.anchor_config.use_prototypes:
                final_A, final_keys = self._compute_class_prototypes(
                    raw_all, label_all
                )
            else:
                final_A = raw_all
                final_keys = self._pilot_match_keys(label_all, None)

            epoch_latents[idx] = final_A
            epoch_keys[idx] = final_keys

        if epoch_latents:
            train_whitening_ops: dict[int, WhiteningOp] = {}
            # With learnable SWBN whitening, phi_i lives in the persistent layers
            # (updated online during the epoch); the buffer-and-fit ops are unused.
            if not self._use_learnable_whitening():
                for idx, chunks in self._task_latent_buffer.items():
                    if chunks:
                        train_whitening_ops[idx] = fit_whitening(
                            torch.cat(chunks, dim=0).float()
                        )
            update_maps = not (
                self.hparams.use_general_maps and self.hparams.soft_maps
            )
            edge_metrics, new_ops = self._update_stiefel_matrices(
                epoch_latents,
                epoch_keys,
                whitening_ops=train_whitening_ops if train_whitening_ops else None,
                update_maps=update_maps,
            )
            # For soft maps the whitening ops come from training latents directly;
            # _update_stiefel_matrices returns them unchanged via fitted_ops.
            self._whitening_ops.update(new_ops)
            if edge_metrics:
                self.log_dict(
                    {f'train/{k}': v for k, v in edge_metrics.items()},
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    add_dataloader_idx=False,
                )

        # Hook (no-op in SheafFRL): CESheafFRL snapshots frozen neighbour pilots
        # here, after the maps + whitening ops have been refreshed and while
        # _latest_pilots is still available.
        self._on_maps_refreshed(True)

        self._latest_pilots.clear()
        self._pilot_latent_buffer = {}
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

    def _on_maps_refreshed(self, did_update: bool) -> None:
        """Called at every epoch end (no-op here; CESheafFRL snapshots pilots)."""

    def _exchanges_pilots(self, prefix: str) -> bool:
        """Whether pilots are exchanged (and the round recorded) this step.

        Always ``True`` for SheafFRL; CESheafFRL returns ``False`` during Phase A,
        where each node reuses frozen neighbour snapshots instead of communicating.
        """
        return True

    def _record_pilot_exchange(
        self,
        payloads_per_agent: dict[int, Any],
        latents_per_agent: dict[int, torch.Tensor],
        prefix: str,
    ) -> None:
        """Record one pilot exchange: 1 round + per-agent payload to all neighbours.

        Each agent broadcasts its full d_i-dimensional pilot payload to every
        neighbour.  SheafCFRL overrides this to record the compressed c_ij-dim
        payload instead.
        """
        self._record_communication_round(n_rounds=1, prefix=prefix)
        for idx, payload in payloads_per_agent.items():
            n_neighbors = len(
                self.hparams.neighbors.get(
                    idx,
                    self.hparams.neighbors.get(str(idx), set()),
                )
            )
            if n_neighbors > 0:
                self._record_communication(
                    payload,
                    n_transmissions=n_neighbors,
                    prefix=prefix,
                )

    def _on_pilots_whitened(
        self,
        whitened_per_agent: dict[int, torch.Tensor],
        keys_per_agent: dict[int, torch.Tensor],
        labels_per_agent: dict[int, torch.Tensor],
        prefix: str,
    ) -> None:
        """Called each step after the per-node whitening (no-op in SheafFRL).

        CESheafFRL uses it to cache the neighbours' whitened pilots received
        during the last Phase-C epoch, keyed by sample-id, so the whole pilot
        pool is available as a fixed target throughout the next Phase-A block.
        """

    def on_validation_epoch_end(self) -> None:
        super().on_validation_epoch_end()

    def _compute_alignment_losses(
        self,
        whitened_per_agent: dict[int, torch.Tensor],
        keys_per_agent: dict[int, torch.Tensor],
        labels_per_agent: dict[int, torch.Tensor],
        comm_weight: float,
        skip: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-step entry point for the alignment losses (both-live).

        SheafFRL evaluates the both-live penalty every step; :class:`CESheafFRL`
        overrides this to phase-dispatch (Phase A → frozen coboundary, else
        both-live).  The geometry lives in ``_both_live_alignment_losses`` /
        ``_frozen_alignment_losses`` so subclasses can swap it (e.g. SheafCFRL's
        compressed coboundary) without re-implementing the schedule.
        """
        return self._both_live_alignment_losses(
            whitened_per_agent, keys_per_agent, labels_per_agent, comm_weight, skip
        )

    def _both_live_alignment_losses(
        self,
        whitened_per_agent: dict[int, torch.Tensor],
        keys_per_agent: dict[int, torch.Tensor],
        labels_per_agent: dict[int, torch.Tensor],
        comm_weight: float,
        skip: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Both-live sheaf penalty + after-comm over all edges (embedding maps).

        One Stiefel map per edge, embedding coboundary ``z_i V − z_j``; both
        endpoints live.  :class:`SheafCFRL` overrides this with the compressed
        two-map coboundary.  ``comm_weight > 0`` enables the after-comm term.
        """
        sheaf_penalty = torch.tensor(0.0, device=self.device)
        after_comm_task_loss = torch.tensor(0.0, device=self.device)
        if skip:
            return sheaf_penalty, after_comm_task_loss

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
            matched = self._match_edge(node_i, node_j, src_i, src_j)
            if matched is None:
                continue

            # Rows are already whitened (whitening was applied per node upstream).
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
        latents_per_agent: dict[int, torch.Tensor] = {}
        payloads_per_agent: dict[int, Any] = {}
        keys_per_agent: dict[int, torch.Tensor] = {}
        # Class labels (y_pilot) aligned with latents_per_agent — always class
        # labels regardless of whether keys_per_agent carries sample IDs.
        labels_per_agent: dict[int, torch.Tensor] = {}

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

                    if prefix == 'train':
                        # Accumulate for epoch-level Stiefel update / whitening-op fitting.
                        self._pilot_latent_buffer.setdefault(idx, []).append(
                            (pilot_latents.detach().cpu(), y_pilot.cpu())
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

                    # Raw latents — whitening is applied below using ops from
                    # the previous epoch (self._whitening_ops).
                    latents_per_agent[idx] = pilot_latents
                    keys_per_agent[idx] = self._pilot_match_keys(y_pilot, sample_ids)
                    # Per-sample class labels aligned with the latent rows.
                    # Prototype mode replaces rows/keys/labels with per-class
                    # prototypes after whitening (_apply_prototype_collapse);
                    # keys must NOT be used here — with global pilots they are
                    # sample IDs, and grouping/filtering by them is meaningless.
                    labels_per_agent[idx] = y_pilot
                    payloads_per_agent[idx] = communication_anchor_payload(
                        anchor_matrix=latents_per_agent[idx],
                        labels=y_pilot,
                        config=self.anchor_config,
                    )

        total_task_loss = torch.stack(list(agent_losses.values())).sum()

        if (
            pilots_available
            and prefix in self._COMMUNICATION_SPLITS
            and prefix != 'test_monitor'
            and self._exchanges_pilots(prefix)
        ):
            self._record_pilot_exchange(payloads_per_agent, latents_per_agent, prefix)

        comm_weight = float(getattr(self.hparams, 'comm_task_coeff', 0.0))

        # Whiten every node's pilots ONCE, before the per-edge loop, so a node's
        # SWBN layer (and its W / running-stat online update) runs a single time
        # per step regardless of its degree.  Row-wise whitening commutes with
        # the per-edge row matching, so the matched rows are pre-whitened.
        whitened_per_agent: dict[int, torch.Tensor] = {
            idx: self._whiten_node_pilots(idx, A)
            for idx, A in latents_per_agent.items()
        }
        # Prototype mode: collapse to per-class prototypes AFTER whitening, so
        # the SWBN online update keeps seeing per-sample rows.  Placed before
        # _on_pilots_whitened so CESheafFRL's Phase-A snapshots cache prototypes
        # keyed by class, matching what the live side produces in Phase A.
        if self.anchor_config.use_prototypes:
            self._apply_prototype_collapse(
                whitened_per_agent, keys_per_agent, labels_per_agent
            )
        self._on_pilots_whitened(
            whitened_per_agent, keys_per_agent, labels_per_agent, prefix
        )

        in_warmup = self.current_epoch < self.hparams.warmup_epochs
        self._step_edge_residuals = {}
        sheaf_penalty, after_comm_task_loss = self._compute_alignment_losses(
            whitened_per_agent,
            keys_per_agent,
            labels_per_agent,
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

    def evaluate_communication_accuracy(
        self, dm, prefix: str = 'test'
    ) -> dict[str, float]:
        """Cross-agent accuracy plus the generic pilot-based misalignment loss.

        Extends the base evaluation with
        :meth:`BaseOrchestrator.evaluate_misalignment_loss`, routed through
        this orchestrator's learned ``send_message`` pipeline (whitening →
        Stiefel map → re-colouring).  This logs the same
        ``{prefix}/misalignment_loss`` and per-edge
        ``{prefix}/misalignment_loss_edge_i_j`` keys as the
        post-training-alignment baselines, computed by the identical
        evaluator, so the values are directly comparable across
        orchestrators.  The online whitened-coboundary residual remains
        available separately as ``{prefix}/sheaf_penalty``.
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

