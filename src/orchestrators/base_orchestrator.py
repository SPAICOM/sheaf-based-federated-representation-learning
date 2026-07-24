"""
Base orchestrator for federated learning with multiple agents.

This module defines the abstract interface for orchestrating training
across multiple agents in a federated learning setting. Subclasses must
implement epoch-end aggregation logic and evaluation procedures.
"""

import inspect
import math
import warnings
from abc import ABC, abstractmethod
from typing import Any

import lightning as l
import numpy as np
import torch
import torch.nn as nn
from hydra.utils import instantiate
from PIL import Image as _PILImage
from torchvision.transforms.functional import to_tensor as _pil_to_tensor

from src.communication.whitening import (
    color,
    common_pilot_indices,
    fit_alignment,
    fit_procrustes,
    fit_whitening,
    whiten,
)
from src.utils import calculate_communication_cost


class BaseOrchestrator(l.LightningModule, ABC):
    """Abstract base orchestrator for federated learning with multiple agents.

    Base class that defines the interface for orchestrating training across
    multiple agents in a federated learning setting. Subclasses must implement
    epoch-end aggregation logic and evaluation procedures.

    Parameters
    ----------
    agents : dict[int, nn.Module]
        Dictionary mapping agent indices to their model instances. Must not be
        empty.
    neighbors : dict[int, set[int]]
        Dictionary mapping each agent index to the set of its neighbor indices.
    optimizer : hydra config
        Optimizer configuration for training.

    Notes
    -----
    - Agents are stored in a ``ModuleDict`` so Lightning can track parameters.
    - Subclasses must implement ``on_train_epoch_end`` for epoch-level ops.
    - Subclasses must implement ``_shared_eval`` for evaluation logic.
    """

    _COMMUNICATION_SPLITS = (
        'train',
        'validation',
        'test_monitor',
        'test',
    )

    def __init__(
        self,
        agents: dict[int, nn.Module],
        neighbors: dict[int, set[int]],
        optimizer,
        log_latent_diagnostics: bool = True,
        pid_logging: bool = False,
        pid_methods: tuple[str, ...] = ('cvxpy',),
        pid_pairs: str = 'neighbors',
        pid_num_clusters: int = 10,
        pid_max_samples: int | None = 4000,
        pid_discrim_epochs: int = 40,
        pid_align_epochs: int = 10,
        pid_batch_size: int = 256,
        pid_hidden_dim: int = 32,
        pid_embed_dim: int = 10,
        **kwargs: Any,
    ):
        super().__init__()
        if kwargs:
            unexpected = sorted(kwargs)
            warnings.warn(
                f'BaseOrchestrator received unexpected kwargs: {unexpected}',
                RuntimeWarning,
                stacklevel=2,
            )
        self.save_hyperparameters(ignore=['agents', 'cfg'])

        assert len(agents) > 0, 'The "agents" dictionary must be not empty'

        self.agents = nn.ModuleDict(
            {str(idx): agent for idx, agent in agents.items()}
        )
        self._validation_prefixes_seen: set[str] = set()
        self._latent_buffer: dict[int, list[torch.Tensor]] = {}
        self._global_pilot_buffer: dict[int, list[torch.Tensor]] = {}
        self._reset_communication_state()

    def _empty_communication_state(self) -> dict[str, float]:
        """Return a zero-initialized communication state."""
        return {
            'kilobytes': 0.0,
            'rounds': 0.0,
        }

    def _reset_communication_state(self) -> None:
        """Reset cumulative communication accounting for all stages."""
        self._communication_by_split = {
            split: self._empty_communication_state()
            for split in self._COMMUNICATION_SPLITS
        }

    def on_train_start(self) -> None:
        """Reset communication accounting at the start of training."""
        self._reset_communication_state()

    def on_validation_start(self) -> None:
        """Reset validation communication accounting at validation start."""
        self._validation_prefixes_seen = set()
        self._reset_split_communication('validation')
        self._reset_split_communication('test_monitor')
        self._reset_latent_buffer()
        self._reset_global_pilot_buffer()

    def on_test_start(self) -> None:
        """Reset test communication accounting at test start."""
        self._reset_split_communication('test')

    def _reset_split_communication(self, prefix: str) -> None:
        """Reset communication accounting for a single stage."""
        self._communication_by_split[prefix] = (
            self._empty_communication_state()
        )

    # ── Latent diagnostics ────────────────────────────────────────────────────

    def _reset_latent_buffer(self) -> None:
        self._latent_buffer = {}

    def _reset_global_pilot_buffer(self) -> None:
        self._global_pilot_buffer = {}

    @staticmethod
    def _latent_sparsity(Z: torch.Tensor, epsilon: float) -> float:
        """Mean number of entries with |z| > epsilon per sample."""
        return float((Z.abs() > epsilon).float().sum(dim=1).mean().item())

    @staticmethod
    def _effective_rank(Z: torch.Tensor) -> float:
        """Roy-Vetterli effective rank: exp(H(p)) where p are normalised singular values."""
        if Z.ndim != 2 or Z.shape[0] < 2 or Z.shape[1] < 1:
            return 1.0
        try:
            s = torch.linalg.svdvals(Z.float())
            s = s[s > 1e-10]
            if s.numel() < 1:
                return 1.0
            p = s / s.sum()
            return float(torch.exp(-(p * p.log()).sum()).item())
        except RuntimeError:
            return float('nan')

    @staticmethod
    def _isotropy(Z: torch.Tensor) -> float:
        """Ratio of min to max per-dim variance. 1.0 = perfectly isotropic, ~0 = strongly anisotropic."""
        if Z.ndim != 2 or Z.shape[0] < 2 or Z.shape[1] < 1:
            return 0.0
        v = torch.var(Z.float(), dim=0)
        v_max = float(v.max().item())
        return float(v.min().item() / v_max) if v_max > 0 else 0.0

    def _collect_all_val_latents(self) -> None:
        """Encode the full validation dataset through every agent.

        Called once at validation epoch end. Accesses each agent's validation
        dataset directly from the datamodule so sparsity and effective rank are
        computed on the complete representation space, not on per-batch chunks.
        """
        dm = getattr(self.trainer, 'datamodule', None)
        if dm is None:
            return
        val_datasets = getattr(dm, 'val_datasets', None)
        if not val_datasets:
            return

        def _collate(batch):
            xs = []
            for item in batch:
                x = item[0]
                if isinstance(x, _PILImage.Image):
                    x = _pil_to_tensor(x)
                xs.append(x)
            return torch.stack(xs)

        self._reset_latent_buffer()

        for idx_str, agent in self.agents.items():
            if not hasattr(agent, 'encode'):
                continue
            idx = int(idx_str)
            val_ds = val_datasets.get(idx)
            if val_ds is None or len(val_ds) == 0:
                continue
            loader = torch.utils.data.DataLoader(
                val_ds,
                batch_size=256,
                shuffle=False,
                num_workers=0,
                collate_fn=_collate,
            )
            chunks = []
            with torch.no_grad():
                for x in loader:
                    x = x.to(self.device)
                    latent = agent.encode(x)
                    chunks.append(latent.detach().cpu())
            self._latent_buffer[idx] = chunks

    def _log_latent_diagnostics(self) -> None:
        """Compute and log sparsity and effective rank on the full validation latent space."""
        if not self._latent_buffer:
            return
        epsilon = float(getattr(self.hparams, 'sparse_epsilon', 1e-2))
        agent_matrices = {
            idx: torch.cat(chunks, dim=0)
            for idx, chunks in self._latent_buffer.items()
        }
        logs = {
            f'validation/latent_sparsity_agent_{idx}': self._latent_sparsity(
                Z, epsilon
            )
            for idx, Z in agent_matrices.items()
        }
        logs.update(
            {
                f'validation/latent_effective_rank_agent_{idx}': self._effective_rank(
                    Z
                )
                for idx, Z in agent_matrices.items()
            }
        )
        logs.update(
            {
                f'validation/latent_isotropy_agent_{idx}': self._isotropy(Z)
                for idx, Z in agent_matrices.items()
            }
        )
        self.log_dict(
            logs,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            add_dataloader_idx=False,
        )
        self._reset_latent_buffer()

    def _collect_all_global_pilot_latents(self) -> None:
        """Encode the full unique pilot dataset through every agent.

        Called once at validation epoch end.  Accesses the datamodule directly
        so each pilot sample is encoded exactly once, regardless of how many
        times it appears in the per-batch combined loader.
        """
        dm = getattr(self.trainer, 'datamodule', None)
        if dm is None:
            return
        pilot_datasets = getattr(dm, 'pilot_datasets', None)
        if not pilot_datasets:
            return
        pilot_ds = pilot_datasets.get(0)
        if pilot_ds is None or len(pilot_ds) == 0:
            return

        from PIL import Image as _PILImage
        from torchvision.transforms.functional import (
            to_tensor as _pil_to_tensor,
        )

        def _collate(batch):
            xs = []
            for item in batch:
                x = item[0]
                if isinstance(x, _PILImage.Image):
                    x = _pil_to_tensor(x)
                xs.append(x)
            return torch.stack(xs)

        loader = torch.utils.data.DataLoader(
            pilot_ds,
            batch_size=256,
            shuffle=False,
            num_workers=0,
            collate_fn=_collate,
        )

        self._reset_global_pilot_buffer()

        for idx_str, agent in self.agents.items():
            if not hasattr(agent, 'encode'):
                continue
            idx = int(idx_str)
            chunks = []
            with torch.no_grad():
                for x in loader:
                    x = x.to(self.device)
                    latent = agent.encode(x)
                    chunks.append(latent.detach().cpu())
            self._global_pilot_buffer[idx] = chunks

    def _log_global_pilot_diagnostics(self) -> None:
        """Compute and log sparsity, effective rank, and isotropy on the full global-pilot latent space."""
        if not self._global_pilot_buffer:
            return
        epsilon = float(getattr(self.hparams, 'sparse_epsilon', 1e-2))
        agent_matrices = {
            idx: torch.cat(chunks, dim=0)
            for idx, chunks in self._global_pilot_buffer.items()
        }
        logs = {
            f'validation/global_pilot_sparsity_agent_{idx}': self._latent_sparsity(Z, epsilon)
            for idx, Z in agent_matrices.items()
        }
        logs.update({
            f'validation/global_pilot_effective_rank_agent_{idx}': self._effective_rank(Z)
            for idx, Z in agent_matrices.items()
        })
        logs.update({
            f'validation/global_pilot_isotropy_agent_{idx}': self._isotropy(Z)
            for idx, Z in agent_matrices.items()
        })
        self.log_dict(
            logs,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            add_dataloader_idx=False,
        )
        self._reset_global_pilot_buffer()

    # ── Lambda schedule ───────────────────────────────────────────────────────

    def _effective_lambda_reg(self) -> float:
        """Return the current regularization coefficient, applying schedule if configured.

        Reads ``max_lmb`` and ``lambda_schedule`` from hparams. When
        ``lambda_schedule`` is ``null``/``None`` the coefficient is constant at
        ``max_lmb``. Otherwise it is warmed up from 0 to ``max_lmb`` over the
        full training run:

        - ``cosine``: λ(t) = max_lmb · (1 − cos(π·t)) / 2
        - ``exp``:    λ(t) = max_lmb · (1 − exp(−5·t))

        where t = current_epoch / (max_epochs − 1) ∈ [0, 1].
        """
        max_lmb = float(self.hparams.max_lmb)
        schedule = getattr(self.hparams, 'lambda_schedule', None)

        if not schedule:
            return max_lmb

        trainer = getattr(self, '_trainer', None)
        max_epochs = (
            int(trainer.max_epochs)
            if trainer is not None and trainer.max_epochs
            else 1
        )
        t = float(self.current_epoch) / max(max_epochs - 1, 1)
        t = min(max(t, 0.0), 1.0)

        if schedule == 'cosine':
            return max_lmb * (1.0 - math.cos(math.pi * t)) / 2.0
        elif schedule == 'exp':
            return max_lmb * (1.0 - math.exp(-5.0 * t))
        else:
            raise ValueError(
                f"Unknown lambda_schedule '{schedule}'. "
                "Valid options: [null, 'cosine', 'exp']"
            )

    def _metric_tensor(self, value: Any) -> torch.Tensor:
        """Ensure a metric value is a scalar tensor on the module device.

        Accepts either a ``torch.Tensor`` (returned as-is) or any numeric
        type that can be cast to ``float``. Used to normalise heterogeneous
        outputs from ``compute_loss`` and ``task_performance`` before
        stacking or logging.
        """
        if isinstance(value, torch.Tensor):
            return value
        return torch.tensor(float(value), device=self.device)

    def _record_communication(
        self,
        payload: Any,
        *,
        n_transmissions: int = 1,
        prefix: str = 'train',
    ) -> dict[str, float]:
        """Accumulate communication volume in kilobytes for a payload.

        Parameters
        ----------
        payload : Any
            The tensor or structure being transmitted.
        n_transmissions : int, optional
            Number of neighbours the payload is sent to. The total cost
            is ``payload_size × n_transmissions``. A value < 1 records
            zero cost and is silently ignored (default: 1).
        """
        if int(n_transmissions) < 1:
            return {'kilobytes': 0.0}

        cost = calculate_communication_cost(
            payload,
            n_transmissions=n_transmissions,
        )
        stage_state = self._communication_by_split[prefix]
        stage_state['kilobytes'] += float(cost['kilobytes'])
        return {'kilobytes': float(cost['kilobytes'])}

    def _record_communication_round(
        self,
        n_rounds: int = 1,
        *,
        prefix: str = 'train',
    ) -> None:
        """Accumulate protocol-level communication rounds."""
        if int(n_rounds) < 1:
            return None
        self._communication_by_split[prefix]['rounds'] += float(int(n_rounds))
        return None

    def _communication_metrics(self, prefix: str) -> dict[str, float]:
        """Return cumulative communication metrics ready for ``log_dict``.

        Parameters
        ----------
        prefix : str
            Logging stage prefix (e.g. ``'train'``, ``'validation'``).
            Keys are formatted as ``'{prefix}/communication_*'``.
        """
        stage_state = self._communication_by_split[prefix]
        return {
            f'{prefix}/communication_kilobytes': stage_state['kilobytes'],
            f'{prefix}/communication_rounds': stage_state['rounds'],
        }

    def _paired_communication_metrics(
        self,
        prefix: str,
        *,
        source_prefix: str = 'train',
    ) -> dict[str, float]:
        """Return one stage's cumulative communication under another prefix.

        This pairs evaluation metrics with the cumulative communication budget
        spent to obtain the evaluated model, making W&B accuracy-vs-budget
        plots possible even when the evaluation stage itself performs no
        communication.
        """
        source_state = self._communication_by_split[source_prefix]
        return {
            (
                f'{prefix}/{source_prefix}_communication_kilobytes_cumulative'
            ): source_state['kilobytes'],
            (
                f'{prefix}/{source_prefix}_communication_rounds_cumulative'
            ): source_state['rounds'],
        }

    def _finalize_train_epoch_communication(self) -> None:
        """Log cumulative train communication metrics once per epoch."""
        communication_logs = self._communication_metrics('train')
        if 'max_lmb' in inspect.signature(type(self).__init__).parameters:
            communication_logs['train/lambda_reg'] = (
                self._effective_lambda_reg()
            )
        self.log_dict(
            communication_logs,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            add_dataloader_idx=False,
        )

    def _finalize_stage_communication(self, prefix: str) -> None:
        """Log cumulative stage communication metrics once per stage epoch."""
        communication_logs = self._communication_metrics(prefix)
        self.log_dict(
            communication_logs,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            add_dataloader_idx=False,
        )

    def _finalize_paired_communication(
        self,
        prefix: str,
        *,
        source_prefix: str = 'train',
    ) -> None:
        """Log cumulative communication from one stage under another prefix."""
        communication_logs = self._paired_communication_metrics(
            prefix,
            source_prefix=source_prefix,
        )
        self.log_dict(
            communication_logs,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            add_dataloader_idx=False,
        )

    def on_validation_epoch_end(self) -> None:
        """Log cumulative validation communication metrics."""
        for prefix in sorted(self._validation_prefixes_seen):
            if prefix == 'test_monitor':
                continue
            self._finalize_stage_communication(prefix)
        if getattr(self.hparams, 'log_latent_diagnostics', True):
            self._collect_all_val_latents()
            self._log_latent_diagnostics()
            self._collect_all_global_pilot_latents()
            self._log_global_pilot_diagnostics()

    def on_test_epoch_end(self) -> None:
        """Log cumulative test communication metrics."""
        self._finalize_stage_communication('test')
        self._finalize_paired_communication('test', source_prefix='train')

    @torch.no_grad()
    def evaluate_communication_accuracy(
        self, dm, prefix: str = 'test'
    ) -> dict[str, float]:
        """Compute cross-agent classification accuracy via the send_message pipeline.

        Generic across orchestrators: relies only on ``self.send_message`` to
        translate the sender's test latents into the receiver's latent space.
        Subclasses opt in by overriding ``on_test_epoch_end`` to call this
        method and log the returned dictionary.

        For each receiver agent j and each neighbour sender i:
          1. ``self.send_message(i, j, Z_i)`` converts the sender's test
             latents into the receiver's local space.
          2. The receiver's decoder classifies the reconstructed
             representations and accuracy is measured against the sender's
             ground-truth labels.

        ``prefix`` namespaces the returned metric keys (default ``'test'``);
        pass ``'train'`` to track this against the held-out split at other
        points during training without colliding with the test-time metrics.

        Returns a dict of metric name → value; the caller is responsible for
        logging so that ``self.log_dict`` runs outside the no-grad context.
        """
        test_datasets = getattr(dm, 'test_datasets', None)
        if not test_datasets:
            return {}

        def _collate_xy(batch):
            xs, ys = [], []
            for item in batch:
                x = item[0]
                if isinstance(x, _PILImage.Image):
                    x = _pil_to_tensor(x)
                xs.append(x)
                y = item[1]
                ys.append(
                    y if isinstance(y, torch.Tensor) else torch.tensor(y)
                )
            return torch.stack(xs), torch.stack(ys)

        # Encode test latents and collect labels for every agent.
        test_Z: dict[int, torch.Tensor] = {}
        test_y: dict[int, torch.Tensor] = {}

        for idx_str, agent in self.agents.items():
            if not hasattr(agent, 'encode') or not hasattr(agent, 'decoder'):
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
                z = (
                    agent.encode(x_batch.to(self.device))
                    .detach()
                    .cpu()
                    .float()
                )
                Zs.append(z)
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
            return getattr(agent, 'task_type', 'classification') == 'classification'

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
            logs[f'{prefix}/avg_private_task_perf'] = sum(self_accs.values()) / len(self_accs)

        # Communication accuracy and task fidelity per receiver.
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

                Z_received = self.send_message(
                    sender_idx=sender_idx,
                    receiver_idx=receiver_idx,
                    Z_sender=test_Z[sender_idx],
                )

                was_training = agent_receiver.training
                agent_receiver.eval()
                logits = agent_receiver.decoder(Z_received.to(self.device))
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
            logs[f'{prefix}/avg_comm_task_perf'] = sum(
                receiver_comm_accs.values()
            ) / len(receiver_comm_accs)
        if task_fidelities:
            logs[f'{prefix}/avg_task_fidelity'] = sum(
                task_fidelities.values()
            ) / len(task_fidelities)

        # Non-neighbour cross-agent accuracy is a test-only metric: it always
        # requires post-hoc map fitting (no orchestrator trains maps for
        # non-edges), so it is evaluated once rather than every train epoch.
        if prefix == 'test':
            logs.update(
                self.evaluate_heterophil_communication_accuracy(
                    dm, prefix=prefix
                )
            )

        return logs

    @torch.no_grad()
    def evaluate_heterophil_communication_accuracy(
        self, dm, prefix: str = 'test'
    ) -> dict[str, float]:
        """Cross-agent accuracy over *non*-edges via post-hoc alignment maps.

        Counterpart to ``evaluate_communication_accuracy``: measures how well
        each agent's decoder classifies test latents received from the agents
        it is NOT connected to in the communication graph. Since no
        orchestrator fits or learns maps for non-edges during training, the
        maps here are always fitted post hoc, with the identical procedure for
        every orchestrator:

          1. Fit a whitening operator per agent on its training latents.
          2. Whiten pilot latents; for every unordered non-neighbour pair fit
             exactly one alignment map on the common pilot samples, in the
             same canonical orientation used for graph edges (sender = the
             endpoint with the larger latent dim, ties broken by the larger
             agent index — the convention shared by
             ``SheafFRL._build_restriction_maps`` and
             ``PostTrainingAlignmentMixin._fit_alignment_maps``).
          3. For each receiver and each non-neighbour sender, transform the
             sender's test latents (whiten → align → re-colour) and measure
             the receiver's decoder accuracy against the sender's labels.
             The pair's non-canonical direction reuses the single fitted map
             inverted — transpose for 'procrustes', pseudo-inverse otherwise
             — mirroring ``send_message``.

        The map type follows ``hparams.alignment_method`` when set and
        defaults to 'procrustes' otherwise. All fitted state is local to this
        call — no orchestrator attributes are read or written — so it is safe
        regardless of any alignment machinery the orchestrator itself carries.

        Returns ``{prefix}/heterophil_comm_task_perf_agent_{j}`` per receiver
        and their mean ``{prefix}/avg_heterophil_comm_task_perf``; the caller
        is responsible for logging.
        """
        train_datasets = getattr(dm, 'train_datasets', None)
        pilot_datasets = getattr(dm, 'pilot_datasets', None)
        test_datasets = getattr(dm, 'test_datasets', None)
        if not train_datasets or not pilot_datasets or not test_datasets:
            return {}

        def _collate_xy(batch):
            xs, ys = [], []
            for item in batch:
                x = item[0]
                if isinstance(x, _PILImage.Image):
                    x = _pil_to_tensor(x)
                xs.append(x)
                y = item[1]
                ys.append(
                    y if isinstance(y, torch.Tensor) else torch.tensor(y)
                )
            return torch.stack(xs), torch.stack(ys)

        def _encode(agent, dataset):
            loader = torch.utils.data.DataLoader(
                dataset,
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
                    agent.encode(x_batch.to(self.device))
                    .detach()
                    .cpu()
                    .float()
                )
                ys.append(y_batch.cpu())
            agent.train(was_training)
            if not Zs:
                return None, None
            return torch.cat(Zs, dim=0), torch.cat(ys, dim=0)

        # Step 1 — post-hoc whitening operators from training latents.
        whitening_ops = {}
        for idx_str, agent in self.agents.items():
            if not hasattr(agent, 'encode'):
                continue
            idx = int(idx_str)
            train_ds = train_datasets.get(idx)
            if train_ds is None or len(train_ds) == 0:
                continue
            Z, _ = _encode(agent, train_ds)
            if Z is not None:
                whitening_ops[idx] = fit_whitening(Z)

        if not whitening_ops:
            return {}

        # Step 2 — whitened pilot latents.
        pilot_Z: dict[int, torch.Tensor] = {}
        for idx_str, agent in self.agents.items():
            if not hasattr(agent, 'encode'):
                continue
            idx = int(idx_str)
            if idx not in whitening_ops:
                continue
            pilot_ds = pilot_datasets.get(idx)
            if pilot_ds is None or len(pilot_ds) == 0:
                continue
            Z, _ = _encode(agent, pilot_ds)
            if Z is not None:
                pilot_Z[idx] = whiten(Z, whitening_ops[idx])

        neighbors_map: dict[int, set[int]] = {
            int(k): {int(n) for n in v}
            for k, v in self.hparams.neighbors.items()
        }
        latent_dims = {
            idx: op.W.shape[0] for idx, op in whitening_ops.items()
        }

        method = getattr(self.hparams, 'alignment_method', None)
        method = str(method) if method is not None else 'procrustes'

        # Step 3 — one map per unordered non-neighbour pair, canonical
        # orientation (larger latent dim wins; ties go to the larger index).
        hetero_maps: dict[tuple[int, int], torch.Tensor] = {}
        agent_ids = sorted(pilot_Z)
        for pos, i in enumerate(agent_ids):
            for j in agent_ids[pos + 1:]:
                if j in neighbors_map.get(i, set()) or i in neighbors_map.get(
                    j, set()
                ):
                    continue

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
                    M = fit_alignment(X_i, X_j).T
                hetero_maps[(sender_idx, receiver_idx)] = M

        if not hetero_maps:
            return {}

        # Step 4 — encode test latents and labels.
        test_Z: dict[int, torch.Tensor] = {}
        test_y: dict[int, torch.Tensor] = {}
        for idx_str, agent in self.agents.items():
            if not hasattr(agent, 'encode'):
                continue
            idx = int(idx_str)
            if idx not in whitening_ops:
                continue
            test_ds = test_datasets.get(idx)
            if test_ds is None or len(test_ds) == 0:
                continue
            Z, y = _encode(agent, test_ds)
            if Z is not None:
                test_Z[idx] = Z
                test_y[idx] = y

        if not test_Z:
            return {}

        def _is_classifier(agent) -> bool:
            return (
                getattr(agent, 'task_type', 'classification')
                == 'classification'
            )

        # Step 5 — receiver decodes each non-neighbour sender's test latents.
        receiver_accs: dict[int, float] = {}
        logs: dict[str, float] = {}
        for idx_str, agent_receiver in self.agents.items():
            receiver_idx = int(idx_str)
            if not hasattr(agent_receiver, 'decoder'):
                continue
            if not _is_classifier(agent_receiver):
                continue
            if receiver_idx not in test_Z:
                continue

            accs: list[float] = []
            for sender_idx in test_Z:
                if sender_idx == receiver_idx:
                    continue
                if sender_idx in neighbors_map.get(receiver_idx, set()):
                    continue

                M = hetero_maps.get((sender_idx, receiver_idx))
                if M is None:
                    M_rev = hetero_maps.get((receiver_idx, sender_idx))
                    if M_rev is None:
                        continue
                    M = (
                        M_rev.T
                        if method == 'procrustes'
                        else torch.linalg.pinv(M_rev)
                    )

                Z_white = whiten(
                    test_Z[sender_idx], whitening_ops[sender_idx]
                )
                Z_recv = color(Z_white @ M, whitening_ops[receiver_idx])

                was_training = agent_receiver.training
                agent_receiver.eval()
                logits = agent_receiver.decoder(Z_recv.to(self.device))
                agent_receiver.train(was_training)

                preds = logits.argmax(dim=1).cpu()
                accs.append(
                    float((preds == test_y[sender_idx]).float().mean().item())
                )

            if accs:
                avg_acc = sum(accs) / len(accs)
                receiver_accs[receiver_idx] = avg_acc
                logs[
                    f'{prefix}/heterophil_comm_task_perf_agent_{receiver_idx}'
                ] = avg_acc

        if receiver_accs:
            logs[f'{prefix}/avg_heterophil_comm_task_perf'] = sum(
                receiver_accs.values()
            ) / len(receiver_accs)

        return logs

    def _log_train_comm_task_perf(self) -> None:
        """Log ``train/avg_comm_task_perf`` against the held-out test split.

        Mirrors the test-time cross-agent accuracy logged in each
        orchestrator's ``on_test_epoch_end`` (via
        ``evaluate_communication_accuracy``), but runs every train epoch under
        the ``'train'`` prefix so alignment quality can be tracked live
        instead of only once at the very end.  Concrete orchestrators call
        this from their own ``on_train_epoch_end``.

        When the orchestrator's ``evaluate_communication_accuracy`` also
        reports the coboundary misalignment loss (post-training-alignment
        methods refit their maps for every call, so each epoch's value uses
        maps fitted on the current representations; SheafFRL-family methods
        evaluate their learned maps), ``train/misalignment_loss`` and its
        per-edge breakdown are logged too.
        """
        dm = getattr(self.trainer, 'datamodule', None)
        if dm is None:
            return
        logs = self.evaluate_communication_accuracy(dm, prefix='train')
        keep = {
            key: value
            for key, value in logs.items()
            if key == 'train/avg_comm_task_perf' or 'misalignment_loss' in key
        }
        if keep:
            self.log_dict(
                keep,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                add_dataloader_idx=False,
            )

    def _whiten_own_latents(self, idx: int, Z: torch.Tensor) -> torch.Tensor:
        """Whiten agent ``idx``'s own raw latents with its own whitening operator.

        Used by :meth:`evaluate_misalignment_loss` to stay in the *whitened*
        coboundary space — the same space SheafFRL's ``sheaf_penalty`` is
        computed in — rather than the raw, re-coloured space ``send_message``
        produces. Concrete orchestrators with a fitted/learned whitening
        operator per agent must override this; the default raises
        ``NotImplementedError`` so callers can skip cleanly.
        """
        raise NotImplementedError(
            f'{self.__class__.__name__} does not implement _whiten_own_latents'
        )

    def _directed_alignment_map(
        self, sender_idx: int, receiver_idx: int
    ) -> torch.Tensor | None:
        """The map fitted/learned for exactly the sender → receiver direction.

        Returns ``None`` when no map exists in that exact orientation — in
        particular, this must *never* fall back to a transpose or
        pseudo-inverse of the reverse edge's map (that is a communication
        convenience ``send_message`` uses, not a valid alignment-quality
        measurement for the direction it wasn't fit for). Concrete
        orchestrators must override; the default raises
        ``NotImplementedError`` so callers can skip cleanly.
        """
        raise NotImplementedError(
            f'{self.__class__.__name__} does not implement _directed_alignment_map'
        )

    @torch.no_grad()
    def evaluate_misalignment_loss(self, dm, prefix: str = 'test') -> dict[str, float]:
        """Cross-agent coboundary misalignment loss on matched pilot pairs.

        For every directed edge (sender → receiver) that has a map fitted
        exactly in that orientation (via ``self._directed_alignment_map`` —
        never a transpose/pseudo-inverse of the reverse edge), whitens both
        sender and receiver pilot latents with their own whitening operators
        (via ``self._whiten_own_latents``), applies the map, and diffs
        against the receiver's own whitened latents on the pilot samples
        common to both (matched by sample id via ``common_pilot_indices``).

        This is computed entirely in whitened space — no re-colouring back
        into raw scale, no decoder involved — so it is *exactly* the
        quantity SheafFRL calls ``sheaf_penalty`` (there evaluated online via
        its own learned Stiefel maps every step); here it is evaluated
        generically so post-training-alignment baselines are directly
        comparable to it, not merely analogous.

        Normalised by ``2·d_e`` — for whitened (unit-variance, uncorrelated)
        latents this is the expected residual under *no* alignment, so the
        result is a dimension-free "fraction of total variation left
        unaligned" (see :meth:`SheafFRL._edge_penalty_term`).

        Orchestrators without both hooks implemented (raising
        ``NotImplementedError``), or edges with no map fitted in the queried
        direction, are silently skipped.

        Returns a dict of metric name → value; the caller logs it.
        """
        pilot_datasets = getattr(dm, 'pilot_datasets', None)
        if not pilot_datasets:
            return {}

        def _collate_x(batch):
            xs = []
            for item in batch:
                x = item[0]
                if isinstance(x, _PILImage.Image):
                    x = _pil_to_tensor(x)
                xs.append(x)
            return torch.stack(xs)

        pilot_Z: dict[int, torch.Tensor] = {}
        for idx_str, agent in self.agents.items():
            if not hasattr(agent, 'encode'):
                continue
            idx = int(idx_str)
            pilot_ds = pilot_datasets.get(idx)
            if pilot_ds is None or len(pilot_ds) == 0:
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
                pilot_Z[idx] = torch.cat(Zs, dim=0)

        if not pilot_Z:
            return {}

        neighbors_map: dict[int, set[int]] = {
            int(k): {int(n) for n in v}
            for k, v in self.hparams.neighbors.items()
        }

        per_receiver: dict[int, list[float]] = {}
        logs: dict[str, float] = {}
        for sender_idx, receiver_set in neighbors_map.items():
            if sender_idx not in pilot_Z:
                continue
            pi = pilot_datasets.get(sender_idx)
            for receiver_idx in receiver_set:
                if receiver_idx == sender_idx or receiver_idx not in pilot_Z:
                    continue
                pj = pilot_datasets.get(receiver_idx)
                if pi is None or pj is None:
                    continue

                try:
                    M = self._directed_alignment_map(sender_idx, receiver_idx)
                except NotImplementedError:
                    continue
                if M is None:
                    continue

                idx_i, idx_j = common_pilot_indices(pi, pj)
                if len(idx_i) < 2:
                    continue

                Z_i = pilot_Z[sender_idx][idx_i]
                Z_j = pilot_Z[receiver_idx][idx_j]
                try:
                    Z_i_w = (
                        self._whiten_own_latents(sender_idx, Z_i.to(self.device))
                        .detach()
                        .cpu()
                        .float()
                    )
                    Z_j_w = (
                        self._whiten_own_latents(receiver_idx, Z_j.to(self.device))
                        .detach()
                        .cpu()
                        .float()
                    )
                except NotImplementedError:
                    continue

                diff = Z_i_w @ M.detach().cpu().float() - Z_j_w
                d_e = diff.shape[1]
                if d_e == 0:
                    continue
                normed = float(
                    ((diff**2).sum(dim=1).mean() / (2.0 * d_e)).item()
                )
                logs[
                    f'{prefix}/misalignment_loss_edge_{sender_idx}_{receiver_idx}'
                ] = normed
                per_receiver.setdefault(receiver_idx, []).append(normed)

        all_vals = [v for vs in per_receiver.values() for v in vs]
        if all_vals:
            logs[f'{prefix}/misalignment_loss'] = sum(all_vals) / len(all_vals)

        return logs

    def _validation_prefix(self, dataloader_idx: int) -> str:
        """Map validation dataloader indices to stable logging prefixes."""
        return 'validation' if int(dataloader_idx) == 0 else 'test_monitor'

    def _log_shared_metrics(
        self,
        *,
        prefix: str,
        agent_losses: dict[int, Any],
        agent_performances: dict[int, Any],
        batch_size: int,
        agent_sample_counts: dict[int, int] | None = None,
        total_loss: torch.Tensor | None = None,
        extra_metrics: dict[str, Any] | None = None,
        prog_bar: bool = True,
        per_agent_loss_name: str = 'task_loss',
        skip_task_performance: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Log per-agent and aggregate metrics for val/test reporting."""
        if prefix == 'test_monitor':
            return torch.tensor(0.0, device=self.device), torch.tensor(
                0.0, device=self.device
            )
        normalized_losses = {
            idx: self._metric_tensor(loss)
            for idx, loss in agent_losses.items()
        }
        normalized_performances = {
            idx: self._metric_tensor(performance)
            for idx, performance in agent_performances.items()
        }

        per_agent_logs = {}
        for idx in sorted(normalized_losses):
            loss = normalized_losses[idx]
            per_agent_logs[f'{prefix}/{per_agent_loss_name}_agent_{idx}'] = loss
            if not skip_task_performance:
                per_agent_logs[f'{prefix}/task_performance_agent_{idx}'] = (
                    normalized_performances[idx]
                )

        if per_agent_logs:
            self.log_dict(
                per_agent_logs,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
                add_dataloader_idx=False,
            )

        if normalized_losses:
            losses_tensor = torch.stack(list(normalized_losses.values()))
            total_task_loss = losses_tensor.sum()
        else:
            losses_tensor = torch.tensor([0.0], device=self.device)
            total_task_loss = torch.tensor(0.0, device=self.device)

        if normalized_performances:
            performances_tensor = torch.stack(
                list(normalized_performances.values())
            )
            avg_performance = performances_tensor.mean()
        else:
            performances_tensor = torch.tensor([0.0], device=self.device)
            avg_performance = torch.tensor(0.0, device=self.device)

        if agent_sample_counts:
            sample_weights = torch.tensor(
                [
                    float(max(agent_sample_counts.get(idx, 0), 0))
                    for idx in sorted(normalized_losses)
                ],
                device=self.device,
            )
        else:
            sample_weights = torch.ones(
                len(normalized_losses),
                device=self.device,
            )

        if len(normalized_losses) > 0 and float(sample_weights.sum()) > 0.0:
            global_task_loss = (
                losses_tensor * sample_weights
            ).sum() / sample_weights.sum()
            global_task_performance = (
                performances_tensor * sample_weights
            ).sum() / sample_weights.sum()
        else:
            global_task_loss = torch.tensor(0.0, device=self.device)
            global_task_performance = torch.tensor(0.0, device=self.device)

        resolved_total_loss = (
            total_task_loss
            if total_loss is None
            else self._metric_tensor(total_loss)
        )

        aggregate_logs: dict[str, Any] = {
            f'{prefix}/task_loss_total_epoch': total_task_loss,
            f'{prefix}/global_task_loss_epoch': global_task_loss,
            f'{prefix}/total_loss_epoch': resolved_total_loss,
        }
        if not skip_task_performance:
            aggregate_logs[f'{prefix}/avg_task_performance_epoch'] = avg_performance
            aggregate_logs[f'{prefix}/global_task_performance_epoch'] = global_task_performance
        if extra_metrics is not None:
            aggregate_logs.update(extra_metrics)

        self.log_dict(
            aggregate_logs,
            on_step=False,
            on_epoch=True,
            prog_bar=prog_bar,
            batch_size=batch_size,
            add_dataloader_idx=False,
        )
        return total_task_loss, avg_performance

    def _resolve_batch_size(
        self,
        batch: dict[int, list[torch.Tensor]]
        | tuple[dict[int, list[torch.Tensor]]],
    ) -> int:
        """Infer an explicit batch size from a multi-agent combined batch."""
        if isinstance(batch, tuple):
            batch = batch[0]

        batch_sizes = []
        for key, values in batch.items():
            if isinstance(key, str) and 'pilot' in key:
                continue
            if not isinstance(values, (list, tuple)) or not values:
                continue
            x = values[0]
            if isinstance(x, torch.Tensor) and x.ndim > 0:
                batch_sizes.append(int(x.shape[0]))

        if batch_sizes:
            return max(batch_sizes)
        return 1

    def _resolve_agent_sample_counts(
        self,
        batch: dict[int, list[torch.Tensor]]
        | tuple[dict[int, list[torch.Tensor]]],
    ) -> dict[int, int]:
        """Infer per-agent sample counts from a combined multi-agent batch."""
        if isinstance(batch, tuple):
            batch = batch[0]

        sample_counts: dict[int, int] = {}
        for idx_str in self.agents:
            idx = int(idx_str)
            key = idx if idx in batch else idx_str
            if key not in batch:
                continue

            values = batch[key]
            if not isinstance(values, (list, tuple)) or len(values) == 0:
                continue

            x = values[0]
            if isinstance(x, torch.Tensor) and x.ndim > 0:
                sample_counts[idx] = int(x.shape[0])

        return sample_counts

    @abstractmethod
    def on_train_epoch_end(self):
        """Perform epoch-level aggregation or updates.

        Called at the end of each training epoch. Subclasses should implement
        this method to perform operations such as federated averaging,
        parameter synchronization, or model aggregation across agents.
        """
        pass

    @abstractmethod
    def send_message(
        self,
        sender_idx: int,
        receiver_idx: int,
        Z_sender: torch.Tensor,
    ) -> torch.Tensor:
        """Transform sender's latent representations into receiver's latent space.

        Mandatory in every subclass. Used at evaluation time to measure how well
        a receiver agent can classify data from a neighbour after the
        communication pipeline (whitening, alignment, re-colouring, etc.).

        Parameters
        ----------
        sender_idx : int
            Index of the agent sending the message.
        receiver_idx : int
            Index of the agent receiving the message.
        Z_sender : torch.Tensor
            Test latent representations from the sender, shape ``(n, d_sender)``.

        Returns
        -------
        torch.Tensor
            Transformed representations in the receiver's latent space,
            shape ``(n, d_receiver)``.
        """
        pass

    def forward(
        self,
        batch: dict[int, list[torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        """Forward pass for multiple agents.

        Parameters
        ----------
        batch : dict[int, list[torch.Tensor]]
            Dictionary mapping agent indices to (input, label) pairs.

        Returns
        -------
        dict[str, tuple[torch.Tensor, torch.Tensor]]
            String-keyed dict (agent index as str) mapping each agent to
            a ``(y_hat, y)`` pair where ``y_hat`` are raw logits and
            ``y`` are the ground-truth labels.
        """
        # Lightning's CombinedLoader wraps batches in a tuple; unwrap it
        # to get the underlying dict before dispatching to agents.
        if isinstance(batch, tuple):
            batch = batch[0]

        outputs = {}

        # Run forward pass for each agent on their respective data
        for idx, agent in self.agents.items():
            # Handle both string and int keys in batch dictionary
            key = idx if idx in batch else int(idx)
            x, y = batch[key]

            # Get predictions (logits) from agent
            y_hat = agent(x)

            # Store predictions with labels for loss computation
            outputs[idx] = (y_hat, y)

        return outputs

    def _shared_eval(
        self,
        batch: dict[int, list[torch.Tensor]],
        batch_idx: int,
        prefix: str,
    ) -> tuple[dict[str, Any], torch.Tensor]:
        """Compute losses and metrics for validation/test steps.

        Parameters
        ----------
        batch : dict[int, list[torch.Tensor]]
            Dictionary mapping agent indices to (input, label) pairs.
        batch_idx : int
            Index of the current batch.
        prefix : str
            Prefix for logging (e.g., 'train', 'validation', 'test').

        Returns
        -------
        tuple[dict, torch.Tensor]
            Tuple of (outputs, total_loss) where outputs maps agent indices to
            (prediction, label) pairs and total_loss is the summed loss.

        Raises
        ------
        NotImplementedError
            Subclasses must override this method.
        """
        raise NotImplementedError(
            f'{self.__class__.__name__} must implement _shared_eval'
        )

    def training_step(
        self,
        batch: dict[int, list[torch.Tensor]],
        batch_idx: int,
    ) -> torch.Tensor:
        """Execute a single training step.

        Parameters
        ----------
        batch : dict[int, list[torch.Tensor]]
            Dictionary mapping agent indices to (input, label) pairs.
        batch_idx : int
            Index of the current batch.

        Returns
        -------
        torch.Tensor
            Training loss for the step.
        """
        _, loss = self._shared_eval(
            batch=batch,
            batch_idx=batch_idx,
            prefix='train',
        )
        self.log(
            'train/total_loss_step',
            loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            batch_size=self._resolve_batch_size(batch),
        )
        return loss

    def test_step(
        self,
        batch: dict[int, list[torch.Tensor]],
        batch_idx: int,
    ) -> None:
        """Execute a single test step.

        Parameters
        ----------
        batch : dict[int, list[torch.Tensor]]
            Dictionary mapping agent indices to (input, label) pairs.
        batch_idx : int
            Index of the current batch.
        """
        self._shared_eval(
            batch=batch,
            batch_idx=batch_idx,
            prefix='test',
        )

    def validation_step(
        self,
        batch: dict[int, list[torch.Tensor]],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> dict[int, torch.Tensor]:
        """Execute a single validation step.

        Parameters
        ----------
        batch : dict[int, list[torch.Tensor]]
            Dictionary mapping agent indices to (input, label) pairs.
        batch_idx : int
            Index of the current batch.

        Returns
        -------
        dict[int, tuple[torch.Tensor, torch.Tensor]]
            Dictionary mapping agent indices to (prediction, label) pairs.
        """
        prefix = self._validation_prefix(dataloader_idx)
        if prefix == 'test_monitor':
            return {}
        self._validation_prefixes_seen.add(prefix)
        output, _ = self._shared_eval(
            batch=batch,
            batch_idx=batch_idx,
            prefix=prefix,
        )
        return output

    def predict_step(
        self,
        batch: dict[int, list[torch.Tensor]],
        batch_idx: int,
    ) -> dict[int, torch.Tensor]:
        """Execute a single prediction step.

        Parameters
        ----------
        batch : dict[int, list[torch.Tensor]]
            Dictionary mapping agent indices to (input, label) pairs.
        batch_idx : int
            Index of the current batch.

        Returns
        -------
        dict[int, tuple[torch.Tensor, torch.Tensor]]
            Dictionary mapping agent indices to (prediction, label) pairs.
        """
        return self(batch)

    def configure_optimizers(self) -> dict[str, object]:
        """Configure the optimizer for training.

        Returns
        -------
        dict[str, object]
            Dictionary containing the configured optimizer.
        """
        optimizer = instantiate(
            self.hparams.optimizer,
            params=self.parameters(),
        )
        return {
            'optimizer': optimizer,
        }

    # ── PID logging on training data ─────────────────────────────────────────
    #
    # Information-theoretic decomposition of ``I(Z_i, Z_j; Y)`` between every
    # pair of agents at the end of training (Liang et al., 2023,
    # arXiv:2302.12247).  Each pair gets four non-negative values logged as
    # ``train/pid_{method}_pair_{i}_{j}/{redundancy|unique_i|unique_j|synergy}``
    # plus the underlying mutual informations for sanity-checking.

    def _flush_pid_logs(self, logs: dict[str, float]) -> None:
        """Push PID metrics to every attached experiment logger.

        ``self.log_dict`` is not reliable inside ``on_train_end`` because
        the per-epoch aggregator has already been torn down — Lightning
        either drops the values or emits a deprecation warning depending
        on the version.  Writing through ``logger.log_metrics`` (or
        ``logger.experiment.log`` for W&B) bypasses that aggregator and
        flushes the metrics immediately under the final ``global_step``.
        """
        if not logs:
            return
        step = int(getattr(self, 'global_step', 0) or 0)
        loggers = getattr(self.trainer, 'loggers', None)
        if not loggers:
            single = getattr(self.trainer, 'logger', None)
            loggers = [single] if single is not None else []
        for lg in loggers:
            log_metrics = getattr(lg, 'log_metrics', None)
            if callable(log_metrics):
                try:
                    log_metrics(metrics=logs, step=step)
                    continue
                except TypeError:
                    log_metrics(logs, step)
                    continue
            experiment = getattr(lg, 'experiment', None)
            log_fn = getattr(experiment, 'log', None)
            if callable(log_fn):
                log_fn({**logs, 'global_step': step})

    @torch.no_grad()
    def _encode_full_training_set(
        self, dm
    ) -> dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]]:
        """Encode each agent's training set once.

        Returns
        -------
        per_agent : dict[int, (Z, Y, sample_ids | None)]
            Latent matrix, label vector, and optional sample identifiers
            (used to pair samples across agents).  Tensors live on CPU
            to keep VRAM available for the neural PID estimator.
        """
        train_datasets = getattr(dm, 'train_datasets', None)
        if not train_datasets:
            return {}

        def _collate(batch):
            xs, ys, sids = [], [], []
            for item in batch:
                x = item[0]
                if isinstance(x, _PILImage.Image):
                    x = _pil_to_tensor(x)
                xs.append(x)
                y = item[1]
                ys.append(
                    y if isinstance(y, torch.Tensor) else torch.tensor(y)
                )
                if len(item) >= 3:
                    sid = item[2]
                    sids.append(
                        sid
                        if isinstance(sid, torch.Tensor)
                        else torch.tensor(sid)
                    )
            xs_t = torch.stack(xs)
            ys_t = torch.stack(ys)
            sids_t = torch.stack(sids) if sids else None
            return xs_t, ys_t, sids_t

        out: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]] = {}
        for idx_str, agent in self.agents.items():
            if not hasattr(agent, 'encode'):
                continue
            idx = int(idx_str)
            ds = train_datasets.get(idx)
            if ds is None or len(ds) == 0:
                continue
            loader = torch.utils.data.DataLoader(
                ds,
                batch_size=256,
                shuffle=False,
                num_workers=0,
                collate_fn=_collate,
            )
            was_training = agent.training
            agent.eval()
            zs, ys, sids = [], [], []
            for x, y, sid in loader:
                x = x.to(self.device)
                z = agent.encode(x).detach().cpu().float()
                zs.append(z)
                ys.append(y.cpu().long())
                if sid is not None:
                    sids.append(sid.cpu().long())
            agent.train(was_training)
            if zs:
                out[idx] = (
                    torch.cat(zs, dim=0),
                    torch.cat(ys, dim=0),
                    torch.cat(sids, dim=0) if sids else None,
                )
        return out

    @staticmethod
    def _pair_by_sample_ids(
        z_i: torch.Tensor,
        y_i: torch.Tensor,
        sid_i: torch.Tensor | None,
        z_j: torch.Tensor,
        y_j: torch.Tensor,
        sid_j: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Match samples between two agents.

        Preference order: (1) shared ``sample_ids``; (2) positional
        pairing on the shorter of the two when both lack ids and have
        identical labels in order; (3) failure → ``None``.
        """
        if sid_i is not None and sid_j is not None:
            ids_i = sid_i.numpy()
            ids_j = sid_j.numpy()
            common = np.intersect1d(ids_i, ids_j, assume_unique=False)
            if common.size == 0:
                return None
            pos_i = {int(v): k for k, v in enumerate(ids_i.tolist())}
            pos_j = {int(v): k for k, v in enumerate(ids_j.tolist())}
            sel_i = [pos_i[int(c)] for c in common]
            sel_j = [pos_j[int(c)] for c in common]
            z_i_p = z_i[sel_i]
            z_j_p = z_j[sel_j]
            y_p = y_i[sel_i]
            if not torch.equal(y_p, y_j[sel_j]):
                return None
            return z_i_p, z_j_p, y_p

        n = min(len(y_i), len(y_j))
        if n == 0 or not torch.equal(y_i[:n], y_j[:n]):
            return None
        return z_i[:n], z_j[:n], y_i[:n]

    def _resolve_pid_num_classes(self, y: torch.Tensor) -> int | None:
        """Best-effort lookup of the global label cardinality."""
        dm = getattr(self.trainer, 'datamodule', None)
        if dm is not None:
            n = getattr(dm, 'num_classes', None)
            if isinstance(n, dict):
                for v in n.values():
                    if isinstance(v, int):
                        return int(v)
            elif isinstance(n, int):
                return int(n)
        return int(y.max().item()) + 1 if y.numel() else None

    def _iter_pid_pairs(self) -> list[tuple[int, int]]:
        """Return the unordered agent pairs to score PID on.

        ``pid_pairs='neighbors'`` (default) → graph edges from
        ``self.hparams.neighbors``; ``pid_pairs='all'`` → every pair
        of agents (full clique).  Each pair appears once, with the
        smaller index first.
        """
        mode = str(getattr(self.hparams, 'pid_pairs', 'neighbors')).lower()
        if mode == 'neighbors':
            neighbors_map = getattr(self.hparams, 'neighbors', {}) or {}
            seen: set[tuple[int, int]] = set()
            for i, neigh in neighbors_map.items():
                for j in neigh:
                    a, b = int(i), int(j)
                    if a == b:
                        continue
                    edge = (min(a, b), max(a, b))
                    seen.add(edge)
            return sorted(seen)
        if mode == 'all':
            idxs = sorted(int(k) for k in self.agents)
            return [
                (idxs[a], idxs[b])
                for a in range(len(idxs))
                for b in range(a + 1, len(idxs))
            ]
        raise ValueError(
            f"Unknown pid_pairs={mode!r}. Valid: ['neighbors', 'all']."
        )

    def _compute_and_log_pair_pid(
        self,
        encoded: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]],
    ) -> dict[str, float]:
        """Run PID estimators on every selected pair; return a log dict."""
        from src.mutualinfo import batch_pid, cvxpy_pid

        methods = tuple(self.hparams.pid_methods)
        logs: dict[str, float] = {}

        for i, j in self._iter_pid_pairs():
            if i not in encoded or j not in encoded:
                continue

            z_i, y_i, sid_i = encoded[i]
            z_j, y_j, sid_j = encoded[j]
            paired = self._pair_by_sample_ids(z_i, y_i, sid_i, z_j, y_j, sid_j)
            if paired is None:
                continue
            z_a, z_b, y = paired
            if y.numel() < 32:
                continue

            num_classes = self._resolve_pid_num_classes(y)
            pair_tag = f'{i}_{j}'

            if 'cvxpy' in methods or 'cvx' in methods:
                try:
                    res = cvxpy_pid(
                        z_a,
                        z_b,
                        y,
                        n_clusters=int(self.hparams.pid_num_clusters),
                        max_samples=self.hparams.pid_max_samples,
                        seed=int(getattr(self, 'global_step', 0)),
                    )
                    prefix = f'train/pid_cvxpy_pair_{pair_tag}'
                    logs[f'{prefix}/redundancy'] = res.redundancy
                    logs[f'{prefix}/unique_{i}'] = res.unique1
                    logs[f'{prefix}/unique_{j}'] = res.unique2
                    logs[f'{prefix}/synergy'] = res.synergy
                    logs[f'{prefix}/mi_y_z1'] = res.mi_y_z1
                    logs[f'{prefix}/mi_y_z2'] = res.mi_y_z2
                    logs[f'{prefix}/mi_y_z1z2'] = res.mi_y_z1z2
                    logs[f'{prefix}/total_information'] = res.total_information()
                except Exception as exc:
                    warnings.warn(
                        f'CVXPY PID failed on pair {pair_tag}: {exc}',
                        RuntimeWarning,
                        stacklevel=2,
                    )

            if 'batch' in methods or 'neural' in methods:
                try:
                    res = batch_pid(
                        z_a,
                        z_b,
                        y,
                        num_classes=num_classes,
                        hidden_dim=int(self.hparams.pid_hidden_dim),
                        embed_dim=int(self.hparams.pid_embed_dim),
                        discrim_epochs=int(self.hparams.pid_discrim_epochs),
                        align_epochs=int(self.hparams.pid_align_epochs),
                        batch_size=int(self.hparams.pid_batch_size),
                        device=self.device,
                        seed=int(getattr(self, 'global_step', 0)),
                    )
                    prefix = f'train/pid_batch_pair_{pair_tag}'
                    logs[f'{prefix}/redundancy'] = res.redundancy
                    logs[f'{prefix}/unique_{i}'] = res.unique1
                    logs[f'{prefix}/unique_{j}'] = res.unique2
                    logs[f'{prefix}/synergy'] = res.synergy
                    logs[f'{prefix}/mi_y_z1'] = res.mi_y_z1
                    logs[f'{prefix}/mi_y_z2'] = res.mi_y_z2
                    logs[f'{prefix}/mi_y_z1z2'] = res.mi_y_z1z2
                    logs[f'{prefix}/total_information'] = res.total_information()
                except Exception as exc:
                    warnings.warn(
                        f'Batch PID failed on pair {pair_tag}: {exc}',
                        RuntimeWarning,
                        stacklevel=2,
                    )
        return logs

    def _encode_pilot_set(
        self, dm
    ) -> dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]]:
        """Encode the shared pilot split through every agent.

        Pilot datasets have the same ``sample_ids`` across all agents, so
        ``_pair_by_sample_ids`` will always find a non-empty intersection.
        Falls back to an empty dict if ``dm.pilot_datasets`` does not exist.
        """
        pilot_datasets = getattr(dm, 'pilot_datasets', None)
        if not pilot_datasets:
            return {}

        def _collate(batch):
            xs, ys, sids = [], [], []
            for item in batch:
                x = item[0]
                if isinstance(x, _PILImage.Image):
                    x = _pil_to_tensor(x)
                xs.append(x)
                y = item[1]
                ys.append(
                    y if isinstance(y, torch.Tensor) else torch.tensor(y)
                )
                if len(item) >= 3:
                    sid = item[2]
                    sids.append(
                        sid
                        if isinstance(sid, torch.Tensor)
                        else torch.tensor(sid)
                    )
            xs_t = torch.stack(xs)
            ys_t = torch.stack(ys)
            sids_t = torch.stack(sids) if sids else None
            return xs_t, ys_t, sids_t

        out: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]] = {}
        for idx_str, agent in self.agents.items():
            if not hasattr(agent, 'encode'):
                continue
            idx = int(idx_str)
            ds = pilot_datasets.get(idx)
            if ds is None or len(ds) == 0:
                continue
            loader = torch.utils.data.DataLoader(
                ds,
                batch_size=256,
                shuffle=False,
                num_workers=0,
                collate_fn=_collate,
            )
            was_training = agent.training
            agent.eval()
            zs, ys, sids = [], [], []
            for x, y, sid in loader:
                x = x.to(self.device)
                z = agent.encode(x).detach().cpu().float()
                zs.append(z)
                ys.append(y.cpu().long())
                if sid is not None:
                    sids.append(sid.cpu().long())
            agent.train(was_training)
            if zs:
                out[idx] = (
                    torch.cat(zs, dim=0),
                    torch.cat(ys, dim=0),
                    torch.cat(sids, dim=0) if sids else None,
                )
        return out

    def on_train_end(self) -> None:
        """Run the PID estimators on the final encoder outputs (one shot).

        No-op unless ``pid_logging=True`` in the hparams.  All metrics
        are flushed under the final ``global_step`` to whatever logger
        the trainer is using.

        Encoding priority: pilot_datasets (shared across agents, guaranteed
        to have overlapping sample IDs) → train_datasets (fallback for
        configurations without a pilot split, e.g. uniform partitioning).
        """
        if not bool(getattr(self.hparams, 'pid_logging', False)):
            return
        dm = getattr(self.trainer, 'datamodule', None)
        if dm is None:
            return
        encoded = self._encode_pilot_set(dm)
        if not encoded:
            encoded = self._encode_full_training_set(dm)
        if not encoded:
            return
        logs = self._compute_and_log_pair_pid(encoded)
        self._flush_pid_logs(logs)


if __name__ == '__main__':
    pass
