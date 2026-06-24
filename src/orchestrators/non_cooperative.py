"""Non-cooperative multi-agent training baseline.

Each agent minimizes only its own task loss. No communication or model
alignment is performed during training, so cumulative communication remains
zero throughout the run.
"""

from typing import Any

import torch
import torch.nn as nn

from src.communication.alignment_mixin import (
    VALID_ALIGNMENT_METHODS,
    PostTrainingAlignmentMixin,
)
from src.orchestrators.base_orchestrator import BaseOrchestrator


class NonCooperativeLearning(PostTrainingAlignmentMixin, BaseOrchestrator):
    """Independent local-training baseline with shared evaluation logging."""

    def __init__(
        self,
        agents: dict[int, nn.Module],
        neighbors: dict[int, set[int]],
        optimizer: Any,
        alignment_method: str = 'general',
        **kwargs,
    ):
        super().__init__(
            agents=agents,
            neighbors=neighbors,
            optimizer=optimizer,
            **kwargs,
        )
        alignment_method = str(alignment_method)
        if alignment_method not in VALID_ALIGNMENT_METHODS:
            raise ValueError(
                f"Unknown alignment_method '{alignment_method}'. "
                f'Valid options: {VALID_ALIGNMENT_METHODS}'
            )
        self.save_hyperparameters(ignore=['agents'])

    def on_train_epoch_end(self) -> None:
        """No communication or aggregation is performed."""
        self._finalize_train_epoch_communication()
        return None

    def _shared_eval(
        self,
        batch: dict[int, list[torch.Tensor]],
        batch_idx: int,
        prefix: str,
    ) -> tuple[dict[str, Any], torch.Tensor]:
        outputs = self(batch)

        agent_losses = {}
        agent_performances = {}

        for idx, agent in self.agents.items():
            y_hat, y = outputs[idx]
            agent_losses[int(idx)] = agent.compute_loss(y_hat, y)
            agent_performances[int(idx)] = agent.task_performance(y_hat, y)

        total_loss, _avg_performance = self._log_shared_metrics(
            prefix=prefix,
            agent_losses=agent_losses,
            agent_performances=agent_performances,
            batch_size=self._resolve_batch_size(batch),
            agent_sample_counts=self._resolve_agent_sample_counts(batch),
            skip_task_performance=(prefix == 'test'),
        )

        return outputs, total_loss

    # ── Post-hoc alignment evaluation ─────────────────────────────────────────
    # _fit_alignment_maps, send_message, _cleanup_alignment come from the mixin.
    # evaluate_communication_accuracy is overridden here to skip edges where no
    # alignment map could be fitted (e.g. no common pilot samples).

    @torch.no_grad()
    def evaluate_communication_accuracy(self, dm) -> dict[str, float]:
        """Compute cross-agent accuracy via the alignment send_message pipeline.

        For each receiver agent j and each neighbour sender i:
          1. Fit per-agent whitening operators on training latents and pairwise
             alignment maps on the common global pilot set (post-hoc).
          2. Send sender i's test latents through the pipeline:
             whiten → align (M_{j←i}) → re-colour.
          3. Receiver j's decoder classifies the reconstructed representations;
             top-1 accuracy is measured against the sender's ground-truth labels.

        Edges where no alignment map was fitted (e.g. no common pilots) are
        skipped so that only meaningful metrics are reported.

        Returns a dict of metric name → value; the caller is responsible for
        logging so that ``self.log_dict`` runs outside the no-grad context.
        """
        from PIL import Image as _PILImg
        from torchvision.transforms.functional import to_tensor as _pil_to_tensor

        success = self._fit_alignment_maps(dm)
        if not success:
            return {}

        test_datasets = getattr(dm, 'test_datasets', None)
        if not test_datasets:
            self._cleanup_alignment()
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
            self._cleanup_alignment()
            return {}

        neighbors_map: dict[int, set[int]] = {
            int(k): {int(n) for n in v}
            for k, v in self.hparams.neighbors.items()
        }

        self_accs: dict[int, float] = {}
        logs: dict[str, float] = {}
        for idx_str, agent in self.agents.items():
            idx = int(idx_str)
            if not hasattr(agent, 'decoder') or idx not in test_Z:
                continue
            was_training = agent.training
            agent.eval()
            logits = agent.decoder(test_Z[idx].to(self.device))
            agent.train(was_training)
            preds = logits.argmax(dim=1).cpu()
            self_accs[idx] = float(
                (preds == test_y[idx]).float().mean().item()
            )
            logs[f'test/private_task_perf_agent_{idx}'] = self_accs[idx]

        if self_accs:
            logs['test/avg_private_task_perf'] = (
                sum(self_accs.values()) / len(self_accs)
            )

        receiver_comm_accs: dict[int, float] = {}
        task_fidelities: dict[int, float] = {}

        for idx_str, agent_receiver in self.agents.items():
            receiver_idx = int(idx_str)
            if not hasattr(agent_receiver, 'decoder'):
                continue
            if receiver_idx not in test_Z:
                continue

            neighbor_accs: list[float] = []
            for sender_idx in neighbors_map.get(receiver_idx, set()):
                if sender_idx not in test_Z:
                    continue
                # Only report metrics for edges where a map was fitted.
                if receiver_idx not in self._alignment_maps.get(sender_idx, {}):
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
                logs[f'test/comm_task_perf_agent_{receiver_idx}'] = avg_acc

                self_acc = self_accs.get(receiver_idx, 0.0)
                fidelity = avg_acc / self_acc if self_acc > 0.0 else 0.0
                task_fidelities[receiver_idx] = fidelity
                logs[f'test/task_fidelity_agent_{receiver_idx}'] = fidelity

        if receiver_comm_accs:
            logs['test/avg_comm_task_perf'] = (
                sum(receiver_comm_accs.values()) / len(receiver_comm_accs)
            )
        if task_fidelities:
            logs['test/avg_task_fidelity'] = (
                sum(task_fidelities.values()) / len(task_fidelities)
            )

        self._cleanup_alignment()
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
