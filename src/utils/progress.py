"""Run-progress callbacks: a JSON heartbeat file and a per-epoch printed line.

``StatusFileCallback`` writes one JSON status file per run for the tmux
dashboard; ``EpochProgressBar`` prints one line per finished epoch, which is
what survives a stdout redirected into a log file.

Files land in ``logs/status/<sweep>/<job>.json`` and are rewritten atomically
at most every ``every_s`` seconds. ``scripts/runs_dashboard.py`` renders them;
Claude can read them directly instead of tailing logs.

Sweep name comes from ``$SWEEP`` (set by ``just launch``), else the Hydra job
name. ``$SWEEP_TOTAL`` (optional) lets the dashboard estimate the sweep ETA.
"""

from __future__ import annotations

import json
import os
import socket
import time
from pathlib import Path
from typing import Any

from lightning.pytorch.callbacks import Callback

_METRIC_KEYS = ('acc', 'loss', 'comm', 'rank', 'misalign', 'mrr', 'lmb')


def _hydra_job() -> tuple[str, int, list[str]]:
    try:
        from hydra.core.hydra_config import HydraConfig

        hc = HydraConfig.get()
        num = hc.job.num if hc.job.num is not None else 0
        return hc.job.name, int(num), list(hc.overrides.task)
    except Exception:
        return 'run', 0, []


class StatusFileCallback(Callback):
    def __init__(
        self,
        root: str | Path = 'logs/status',
        every_s: float = 10.0,
        max_metrics: int = 8,
    ) -> None:
        job_name, self.job_num, self.overrides = _hydra_job()
        self.sweep = os.environ.get('SWEEP') or job_name
        self.path = (
            Path(root)
            / self.sweep
            / f'job_{self.job_num:03d}_{os.getpid()}.json'
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.every_s = every_s
        self.max_metrics = max_metrics
        self.t0 = time.time()
        self.t_epoch = self.t0
        self.epoch_times: list[float] = []
        self._last_write = 0.0

    # ── helpers ──────────────────────────────────────────────────────────────
    def _metrics(self, trainer) -> dict[str, float]:
        out = {}
        for k, v in trainer.callback_metrics.items():
            if any(s in k.lower() for s in _METRIC_KEYS):
                try:
                    out[k] = round(float(v), 4)
                except (TypeError, ValueError):
                    continue
        return dict(sorted(out.items())[: self.max_metrics])

    def _write(
        self, trainer, status: str, force: bool = False, **extra: Any
    ) -> None:
        now = time.time()
        if not force and now - self._last_write < self.every_s:
            return
        self._last_write = now
        max_ep = trainer.max_epochs or 0
        ep = trainer.current_epoch
        n_batches = trainer.num_training_batches
        batch = getattr(
            trainer.fit_loop.epoch_loop.batch_progress.current, 'ready', 0
        )
        frac_epoch = (
            (batch / n_batches)
            if isinstance(n_batches, int) and n_batches
            else 0.0
        )
        done = ep + frac_epoch
        elapsed = now - self.t0
        eta = elapsed / done * (max_ep - done) if done > 0 and max_ep else None
        payload = {
            'sweep': self.sweep,
            'job': self.job_num,
            'overrides': self.overrides,
            'status': status,
            'epoch': ep,
            'max_epochs': max_ep,
            'batch': batch,
            'n_batches': n_batches if isinstance(n_batches, int) else None,
            'elapsed_s': round(elapsed),
            'eta_s': round(eta) if eta is not None else None,
            'last_epoch_s': round(self.epoch_times[-1], 1)
            if self.epoch_times
            else None,
            'metrics': self._metrics(trainer),
            'pid': os.getpid(),
            'host': socket.gethostname(),
            'updated': now,
            **extra,
        }
        tmp = self.path.with_suffix('.tmp')
        tmp.write_text(json.dumps(payload, indent=1))
        os.replace(tmp, self.path)

    # ── hooks ────────────────────────────────────────────────────────────────
    def on_fit_start(self, trainer, pl_module) -> None:
        self.t0 = self.t_epoch = time.time()
        self._write(trainer, 'running', force=True)

    def on_train_batch_end(self, trainer, *args, **kwargs) -> None:
        self._write(trainer, 'running')

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        now = time.time()
        self.epoch_times.append(now - self.t_epoch)
        self.t_epoch = now
        # Orchestrator epoch-end work (alignment fits, comm eval) runs after this.
        self._write(trainer, 'epoch-end', force=True)

    def on_validation_start(self, trainer, pl_module) -> None:
        if not trainer.sanity_checking:
            self._write(trainer, 'validating', force=True)

    def on_fit_end(self, trainer, pl_module) -> None:
        self._write(trainer, 'testing', force=True)

    def on_test_end(self, trainer, pl_module) -> None:
        self._write(trainer, 'done', force=True)

    def on_exception(self, trainer, pl_module, exception) -> None:
        self._write(trainer, 'failed', force=True, error=repr(exception)[:300])


class EpochProgressBar(Callback):
    """One printed line per finished epoch, for logs you read with ``tail -f``.

    Lightning's own bar is carriage-return driven, so a redirected stdout gets
    either nothing or thousands of fragments.  This prints exactly one line per
    epoch with plain ``print(..., flush=True)``::

        Epochs [██████░░░░] 12/40 | 52s/epoch | ETA 24.3m

    ``s/epoch`` and the ETA come from the mean wall-clock time per completed
    epoch since ``on_fit_start``, so they include validation and the
    orchestrator's epoch-end work (alignment refits, comm eval) rather than the
    training loop alone.
    """

    def __init__(self, width: int = 10, prefix: str = 'Epochs') -> None:
        self.width = max(int(width), 1)
        self.prefix = prefix
        self.t0 = time.time()

    @staticmethod
    def _fmt_duration(seconds: float) -> str:
        if seconds < 90:
            return f'{seconds:.0f}s'
        if seconds < 90 * 60:
            return f'{seconds / 60:.1f}m'
        return f'{seconds / 3600:.1f}h'

    def _bar(self, done: int, total: int) -> str:
        filled = (
            min(self.width, round(self.width * done / total)) if total else 0
        )
        return '█' * filled + '░' * (self.width - filled)

    def on_fit_start(self, trainer, pl_module) -> None:
        self.t0 = time.time()

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        done = trainer.current_epoch + 1
        total = trainer.max_epochs or 0
        elapsed = time.time() - self.t0
        per_epoch = elapsed / done
        parts = [
            f'{self.prefix} [{self._bar(done, total)}] {done}/{total or "?"}',
            f'{self._fmt_duration(per_epoch)}/epoch',
        ]
        if total:
            parts.append(
                f'ETA {self._fmt_duration(per_epoch * (total - done))}'
            )
        print(' | '.join(parts), flush=True)
