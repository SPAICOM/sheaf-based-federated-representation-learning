"""Post-training alignment mixin for orchestrators.

Provides _fit_alignment_maps, send_message, _cleanup_alignment, and an
evaluate_communication_accuracy override that wraps the base class's generic
loop with conditional map fitting and cleanup.
"""

import torch

from src.communication.whitening import (
    WhiteningOp,
    color,
    common_pilot_indices,
    fit_alignment,
    fit_procrustes,
    fit_whitening,
    whiten,
)

VALID_ALIGNMENT_METHODS = ('general', 'procrustes')


class PostTrainingAlignmentMixin:
    """Mixin that adds post-hoc whitening + alignment to any BaseOrchestrator.

    Concrete classes must still implement on_train_epoch_end and _shared_eval.
    The alignment_method hparam (saved by the subclass) controls whether maps
    are general (least-squares) or procrustes (semi-orthogonal).  When
    alignment_method is None the send_message falls back to identity, so this
    mixin is safe to inherit in orchestrators with optional alignment.
    """

    @torch.no_grad()
    def _fit_alignment_maps(self, dm) -> bool:
        """Fit per-agent whitening operators and pairwise alignment maps.

        Steps:
          1. Encode each agent's full training set, fit a whitening operator.
          2. Encode each agent's pilot set, whiten with training operator.
          3. For every *undirected* edge, find common pilot samples and fit
             exactly one map M_{receiver←sender} using the configured method:
             - 'general'    : unconstrained least-squares (fit_alignment).
             - 'procrustes' : semi-orthogonal map (fit_procrustes).
             The (sender, receiver) orientation is fixed by latent dimension
             — sender = the endpoint with the larger latent dim, ties broken
             by the larger agent index — the same canonical convention
             ``SheafFRL._build_restriction_maps`` uses for its single shared
             restriction map per edge. This keeps the number of fitted maps,
             and their orientation, identical across baselines and SheafFRL
             so ``evaluate_misalignment_loss`` scores the exact same edges.

        Alignment maps are stored as right-multiply matrices M such that
        ``Z_white_sender @ M ≈ Z_white_receiver``. ``send_message`` covers
        the reverse direction by inverting this single map (transpose for
        'procrustes', pseudo-inverse for 'general') — see its docstring.

        Returns True if at least one alignment map was successfully fitted.
        """
        from PIL import Image as _PILImg
        from torchvision.transforms.functional import to_tensor as _pil_to_tensor

        train_datasets = getattr(dm, 'train_datasets', None)
        pilot_datasets = getattr(dm, 'pilot_datasets', None)
        if not train_datasets or not pilot_datasets:
            return False

        def _collate_x(batch):
            xs = []
            for item in batch:
                x = item[0]
                if isinstance(x, _PILImg.Image):
                    x = _pil_to_tensor(x)
                xs.append(x)
            return torch.stack(xs)

        # Step 1 — fit whitening on training latents.
        self._train_whitening_ops: dict[int, WhiteningOp] = {}
        for idx_str, agent in self.agents.items():
            if not hasattr(agent, 'encode'):
                continue
            idx = int(idx_str)
            train_ds = train_datasets.get(idx)
            if train_ds is None or len(train_ds) == 0:
                continue

            loader = torch.utils.data.DataLoader(
                train_ds,
                batch_size=256,
                shuffle=False,
                num_workers=0,
                collate_fn=_collate_x,
            )
            was_training = agent.training
            agent.eval()
            Zs = []
            for x_batch in loader:
                Zs.append(
                    agent.encode(x_batch.to(self.device)).detach().cpu().float()
                )
            agent.train(was_training)

            if Zs:
                self._train_whitening_ops[idx] = fit_whitening(
                    torch.cat(Zs, dim=0)
                )

        if not self._train_whitening_ops:
            return False

        # Step 2 — encode pilot latents, whiten with training operators.
        pilot_Z: dict[int, torch.Tensor] = {}
        for idx_str, agent in self.agents.items():
            if not hasattr(agent, 'encode'):
                continue
            idx = int(idx_str)
            if idx not in self._train_whitening_ops:
                continue

            pilot_ds = pilot_datasets.get(idx)
            if pilot_ds is None or len(pilot_ds) == 0:
                d = self._train_whitening_ops[idx].W.shape[0]
                pilot_Z[idx] = torch.empty(0, d)
                continue

            loader = torch.utils.data.DataLoader(
                pilot_ds,
                batch_size=256,
                shuffle=False,
                num_workers=0,
                collate_fn=_collate_x,
            )
            was_training = agent.training
            agent.eval()
            Zs = []
            for x_batch in loader:
                Zs.append(
                    agent.encode(x_batch.to(self.device)).detach().cpu().float()
                )
            agent.train(was_training)

            if Zs:
                Z_pilot = torch.cat(Zs, dim=0)
                pilot_Z[idx] = whiten(Z_pilot, self._train_whitening_ops[idx])
            else:
                d = self._train_whitening_ops[idx].W.shape[0]
                pilot_Z[idx] = torch.empty(0, d)

        # Step 3 — fit one alignment map per undirected edge.
        method = str(getattr(self.hparams, 'alignment_method', 'general'))
        neighbors_map: dict[int, set[int]] = {
            int(k): {int(n) for n in v}
            for k, v in self.hparams.neighbors.items()
        }
        latent_dims = {
            idx: op.W.shape[0] for idx, op in self._train_whitening_ops.items()
        }
        self._alignment_maps: dict[int, dict[int, torch.Tensor]] = {
            idx: {} for idx in pilot_Z
        }

        any_fitted = False
        seen_edges: set[frozenset[int]] = set()
        for i, receiver_set in neighbors_map.items():
            if i not in pilot_Z:
                continue
            for j in receiver_set:
                if j == i or j not in pilot_Z:
                    continue
                edge = frozenset((i, j))
                if edge in seen_edges:
                    continue
                seen_edges.add(edge)

                # Canonical orientation — mirrors SheafFRL's node_i/node_j
                # rule exactly (larger latent dim wins; ties go to the
                # larger agent index).
                if latent_dims[i] > latent_dims[j]:
                    sender_idx, receiver_idx = i, j
                elif latent_dims[i] < latent_dims[j]:
                    sender_idx, receiver_idx = j, i
                else:
                    sender_idx, receiver_idx = max(i, j), min(i, j)

                pi = pilot_datasets.get(sender_idx)
                pj = pilot_datasets.get(receiver_idx)
                if pi is None or pj is None:
                    continue

                idx_i, idx_j = common_pilot_indices(pi, pj)
                if len(idx_i) < 2:
                    continue

                X_i = pilot_Z[sender_idx][idx_i]
                X_j = pilot_Z[receiver_idx][idx_j]

                if method == 'procrustes':
                    M = fit_procrustes(X_i, X_j)
                else:
                    # fit_alignment returns A s.t. X_i @ A.T ≈ X_j;
                    # store as right-multiply M = A.T.
                    M = fit_alignment(X_i, X_j).T

                self._alignment_maps[sender_idx][receiver_idx] = M
                any_fitted = True

        return any_fitted

    @torch.no_grad()
    def send_message(
        self,
        sender_idx: int,
        receiver_idx: int,
        Z_sender: torch.Tensor,
    ) -> torch.Tensor:
        """Transform sender's test latents into the receiver's latent space.

        Pipeline: whiten (sender stats) → align (M_{receiver←sender}) →
        re-colour (receiver stats).  Only one map is fitted per undirected
        edge (see ``_fit_alignment_maps``); when queried in the opposite
        orientation, that single map is inverted instead — ``M.T`` for
        ``alignment_method == 'procrustes'`` (semi-orthogonal, so the
        transpose is exact) or ``pinv(M)`` otherwise — exactly mirroring
        ``SheafFRL.send_message``'s ``V.T``/``pinv(V)`` fallback for the
        edge's non-canonical direction. When alignment_method is None or no
        maps have been fitted, the method degrades to identity (returns
        Z_sender).

        Parameters
        ----------
        sender_idx : int
        receiver_idx : int
        Z_sender : torch.Tensor
            Raw test latent representations, shape ``(n, d_sender)``.

        Returns
        -------
        torch.Tensor
            Representations in the receiver's latent space.
        """
        train_ops = getattr(self, '_train_whitening_ops', {})
        alignment_maps = getattr(self, '_alignment_maps', {})

        op_sender = train_ops.get(sender_idx)
        op_receiver = train_ops.get(receiver_idx)

        M = alignment_maps.get(sender_idx, {}).get(receiver_idx)
        if M is None:
            M_rev = alignment_maps.get(receiver_idx, {}).get(sender_idx)
            if M_rev is not None:
                method = str(getattr(self.hparams, 'alignment_method', 'general'))
                M = (
                    M_rev.T
                    if method == 'procrustes'
                    else torch.linalg.pinv(M_rev)
                )

        dev = Z_sender.device

        Z_white = (
            whiten(Z_sender, op_sender)
            if op_sender is not None
            else Z_sender.float()
        )

        Z_aligned = Z_white @ M.to(dev) if M is not None else Z_white

        if op_receiver is not None:
            return color(Z_aligned, op_receiver).to(dev)
        return Z_aligned.to(dev)

    def _cleanup_alignment(self) -> None:
        self._train_whitening_ops = {}
        self._alignment_maps = {}

    def _whiten_own_latents(self, idx: int, Z: torch.Tensor) -> torch.Tensor:
        """Whiten with this agent's own post-hoc-fitted training-set operator."""
        op = getattr(self, '_train_whitening_ops', {}).get(idx)
        if op is None:
            raise NotImplementedError(
                f'no whitening operator fitted for agent {idx}'
            )
        return whiten(Z, op)

    def _directed_alignment_map(
        self, sender_idx: int, receiver_idx: int
    ) -> torch.Tensor | None:
        """The map fitted for exactly sender_idx -> receiver_idx.

        ``_fit_alignment_maps`` fits exactly one map per undirected edge, in
        the canonical orientation described there (mirroring SheafFRL's
        node_i/node_j rule). This returns that map only when queried in
        that exact orientation, and ``None`` for the reverse direction
        (never a transpose/pseudo-inverse of it — that inversion is a
        ``send_message`` communication convenience, not a valid alignment-
        quality measurement for the direction it wasn't fit for). So
        misalignment-loss scoring covers exactly one direction per edge,
        the same edges and orientation as ``SheafFRL._directed_alignment_map``.
        """
        return getattr(self, '_alignment_maps', {}).get(sender_idx, {}).get(
            receiver_idx
        )

    @torch.no_grad()
    def evaluate_communication_accuracy(
        self, dm, prefix: str = 'test'
    ) -> dict[str, float]:
        """Compute cross-agent accuracy, fitting alignment maps when configured.

        When ``hparams.alignment_method`` is set, per-agent whitening operators
        and pairwise alignment maps are fitted on pilot data, cross-agent
        accuracy is measured over the edges that could be fitted, and the maps
        are cleaned up afterward.  When it is ``None`` evaluation delegates to
        the base-class loop with an identity ``send_message`` (safe only when
        all agents share a latent space, e.g. FedAvg).

        Either way, the returned logs are merged with
        ``self.evaluate_misalignment_loss`` (computed while any fitted maps
        are still active, before cleanup) so the coboundary misalignment loss
        is always evaluated with the post-training alignment maps when those
        exist, matching the quantity SheafFRL calls ``sheaf_penalty``.

        ``prefix`` namespaces the returned metric keys (default ``'test'``).
        """
        alignment_method = getattr(self.hparams, 'alignment_method', None)
        if alignment_method is None:
            logs = dict(super().evaluate_communication_accuracy(dm, prefix=prefix))
            logs.update(self.evaluate_misalignment_loss(dm, prefix=prefix))
            return logs

        if not self._fit_alignment_maps(dm):
            return {}
        try:
            logs = dict(self._comm_accuracy_with_fitted_maps(dm, prefix=prefix))
            logs.update(self.evaluate_misalignment_loss(dm, prefix=prefix))
            return logs
        finally:
            self._cleanup_alignment()

    @torch.no_grad()
    def _comm_accuracy_with_fitted_maps(
        self, dm, prefix: str = 'test'
    ) -> dict[str, float]:
        """Measure cross-agent accuracy over edges with a fitted alignment map.

        Assumes ``_fit_alignment_maps`` has already populated
        ``self._train_whitening_ops`` and ``self._alignment_maps``.

        For each receiver agent j and each neighbour sender i:
          1. Send sender i's test latents through the pipeline:
             whiten → align (M_{j←i}) → re-colour.
          2. Receiver j's decoder classifies the reconstructed representations;
             top-1 accuracy is measured against the sender's ground-truth labels.

        Edges where no alignment map was fitted (e.g. no common pilots) are
        skipped so that only meaningful metrics are reported.

        Returns a dict of metric name → value; the caller is responsible for
        logging so that ``self.log_dict`` runs outside the no-grad context.
        """
        from PIL import Image as _PILImg
        from torchvision.transforms.functional import to_tensor as _pil_to_tensor

        test_datasets = getattr(dm, 'test_datasets', None)
        if not test_datasets:
            return {}

        def _collate_xy(batch):
            xs, ys = [], []
            for item in batch:
                x = item[0]
                if isinstance(x, _PILImg.Image):
                    x = _pil_to_tensor(x)
                xs.append(x)
                y = item[1]
                ys.append(
                    y if isinstance(y, torch.Tensor) else torch.tensor(y)
                )
            return torch.stack(xs), torch.stack(ys)

        test_Z: dict[int, torch.Tensor] = {}
        test_y: dict[int, torch.Tensor] = {}
        for idx_str, agent in self.agents.items():
            if not hasattr(agent, 'encode'):
                continue
            idx = int(idx_str)
            test_ds = test_datasets.get(idx)
            if test_ds is None or len(test_ds) == 0:
                continue

            loader = torch.utils.data.DataLoader(
                test_ds,
                batch_size=256,
                shuffle=False,
                num_workers=0,
                collate_fn=_collate_xy,
            )
            was_training = agent.training
            agent.eval()
            Zs, ys = [], []
            for x_batch, y_batch in loader:
                Zs.append(
                    agent.encode(x_batch.to(self.device)).detach().cpu().float()
                )
                ys.append(y_batch.cpu())
            agent.train(was_training)

            if Zs:
                test_Z[idx] = torch.cat(Zs, dim=0)
                test_y[idx] = torch.cat(ys, dim=0)

        if not test_Z:
            return {}

        neighbors_map: dict[int, set[int]] = {
            int(k): {int(n) for n in v}
            for k, v in self.hparams.neighbors.items()
        }

        # Communication accuracy is a classification-only metric: the
        # receiver's decoder is expected to return logits, so AE agents
        # (decoder returns an image) are skipped both as receivers and as
        # self-accuracy targets. They can still act as senders.
        def _is_classifier(agent) -> bool:
            return (
                getattr(agent, 'task_type', 'classification')
                == 'classification'
            )

        self_accs: dict[int, float] = {}
        logs: dict[str, float] = {}
        for idx_str, agent in self.agents.items():
            idx = int(idx_str)
            if not hasattr(agent, 'decoder') or idx not in test_Z:
                continue
            if not _is_classifier(agent):
                continue
            was_training = agent.training
            agent.eval()
            logits = agent.decoder(test_Z[idx].to(self.device))
            agent.train(was_training)
            preds = logits.argmax(dim=1).cpu()
            self_accs[idx] = float(
                (preds == test_y[idx]).float().mean().item()
            )
            logs[f'{prefix}/private_task_perf_agent_{idx}'] = self_accs[idx]

        if self_accs:
            logs[f'{prefix}/avg_private_task_perf'] = (
                sum(self_accs.values()) / len(self_accs)
            )

        receiver_comm_accs: dict[int, float] = {}
        task_fidelities: dict[int, float] = {}

        for idx_str, agent_receiver in self.agents.items():
            receiver_idx = int(idx_str)
            if not hasattr(agent_receiver, 'decoder'):
                continue
            if not _is_classifier(agent_receiver):
                continue
            if receiver_idx not in test_Z:
                continue

            neighbor_accs: list[float] = []
            for sender_idx in neighbors_map.get(receiver_idx, set()):
                if sender_idx not in test_Z:
                    continue
                # Only report metrics for edges where a map was fitted, in
                # either orientation — send_message inverts the map when
                # queried in the edge's non-canonical direction.
                if receiver_idx not in self._alignment_maps.get(
                    sender_idx, {}
                ) and sender_idx not in self._alignment_maps.get(
                    receiver_idx, {}
                ):
                    continue

                Z_colored = self.send_message(
                    sender_idx=sender_idx,
                    receiver_idx=receiver_idx,
                    Z_sender=test_Z[sender_idx],
                )

                was_training = agent_receiver.training
                agent_receiver.eval()
                logits = agent_receiver.decoder(Z_colored.to(self.device))
                agent_receiver.train(was_training)

                preds = logits.argmax(dim=1).cpu()
                acc = float(
                    (preds == test_y[sender_idx]).float().mean().item()
                )
                neighbor_accs.append(acc)

            if neighbor_accs:
                avg_acc = sum(neighbor_accs) / len(neighbor_accs)
                receiver_comm_accs[receiver_idx] = avg_acc
                logs[f'{prefix}/comm_task_perf_agent_{receiver_idx}'] = avg_acc

                self_acc = self_accs.get(receiver_idx, 0.0)
                fidelity = avg_acc / self_acc if self_acc > 0.0 else 0.0
                task_fidelities[receiver_idx] = fidelity
                logs[f'{prefix}/task_fidelity_agent_{receiver_idx}'] = fidelity

        if receiver_comm_accs:
            logs[f'{prefix}/avg_comm_task_perf'] = (
                sum(receiver_comm_accs.values()) / len(receiver_comm_accs)
            )
        if task_fidelities:
            logs[f'{prefix}/avg_task_fidelity'] = (
                sum(task_fidelities.values()) / len(task_fidelities)
            )

        return logs
