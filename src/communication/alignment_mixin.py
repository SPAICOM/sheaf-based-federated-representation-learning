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
          3. For every directed edge (sender→receiver), find common pilot
             samples and fit M_{receiver←sender} using the configured method:
             - 'general'    : unconstrained least-squares (fit_alignment).
             - 'procrustes' : semi-orthogonal map (fit_procrustes).

        Alignment maps are stored as right-multiply matrices M such that
        ``Z_white_sender @ M ≈ Z_white_receiver``.

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

        # Step 3 — fit directed alignment maps M_{receiver←sender} per edge.
        method = str(getattr(self.hparams, 'alignment_method', 'general'))
        neighbors_map: dict[int, set[int]] = {
            int(k): {int(n) for n in v}
            for k, v in self.hparams.neighbors.items()
        }
        self._alignment_maps: dict[int, dict[int, torch.Tensor]] = {
            idx: {} for idx in pilot_Z
        }

        any_fitted = False
        for sender_idx, receiver_set in neighbors_map.items():
            if sender_idx not in pilot_Z:
                continue
            for receiver_idx in receiver_set:
                if receiver_idx == sender_idx or receiver_idx not in pilot_Z:
                    continue

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
        re-colour (receiver stats).  When alignment_method is None or no maps
        have been fitted, the method degrades to identity (returns Z_sender).

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

    @torch.no_grad()
    def evaluate_communication_accuracy(self, dm) -> dict[str, float]:
        """Compute cross-agent accuracy, fitting alignment maps when configured.

        When hparams.alignment_method is not None, alignment maps are fitted
        on pilot data before evaluation and cleaned up afterward.  When it is
        None, evaluation delegates directly to the base class (send_message
        acts as identity since no maps are fitted).
        """
        alignment_method = getattr(self.hparams, 'alignment_method', None)
        if alignment_method is not None:
            success = self._fit_alignment_maps(dm)
            if not success:
                return {}
        try:
            logs = super().evaluate_communication_accuracy(dm)
        finally:
            if alignment_method is not None:
                self._cleanup_alignment()
        return logs
