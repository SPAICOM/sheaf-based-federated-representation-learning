"""Visualize multi-agent experiment metrics from WandB run logs.

By default, fetches runs from the wandb cloud API (``--entity``/``--project``)
— local ``logs/wandb`` run directories are routinely cleaned up once a run has
synced, so the cloud is the reliable source of truth. Pass ``--local`` to
instead read straight from the on-disk WandB transaction logs written by
``scripts/multi_agent_experiment.py`` (under ``logs/wandb/run-*``, no cloud
access) — this still auto-falls back to the cloud API if nothing local is
found. Every orchestrator in ``EXCLUDED_ORCHS`` (currently ``CESheafFRL`` and
``SheafCFRL``) is dropped from the run set entirely, before anything below is
built — edit that set directly to bring one back. Either way it produces:

  1. ``test/avg_comm_task_perf`` vs ``shift_strength``, one curve per
     orchestrator, and a companion figure of ``test/avg_private_task_perf``
     vs ``shift_strength`` the same way. ComFed is never plotted here — it
     sits far below the others and squashes the y-axis; its numbers instead
     go in the dedicated ComFed table (part 4 below). Pass ``--together`` to
     draw the two metrics side by side in one figure (same per-panel
     proportions as the standalone plots) with a single shared legend sitting
     just above both panels, in three columns, instead of two separate
     figures.
     By default (``adjusted=True``) each orchestrator's points are dodged
     slightly off the shared shift value and the spread across agents is
     drawn as small std error-bar caps (bar-plot style) rather than
     seaborn's translucent band, so overlapping curves stay legible; pass
     ``adjusted=False`` for the original undodged line + std-band rendering.
     All shift-strength plots use xticks restricted to the shifts actually
     run.

  2. For each ``shift_strength``, a *separate* figure of the training task
     performance (mean across agents ± std band) over epochs, one curve per
     orchestrator, each drawn with a distinct line style.

  3. For each ``shift_strength``, a table (printed + CSV + markdown) with one
     row per orchestrator (including ComFed):
       - avg. comm task perf (± std across agents)
       - avg. private task perf (± std across agents)
       - communication rounds (training-cumulative)
       - communication kilobytes (training-cumulative)
       - number of parameters: agent weights (+ sheaf/coupling maps)
       - estimated training FLOPs: 3 x per-sample forward FLOPs x
         batch_size x total steps (agent-only; see note below)
       - runtime proxy: total run wall-clock (``_runtime``); no eval-only
         timer is logged, so this is fit+test combined

  4. A dedicated ComFed table (printed + CSV + markdown), one row per
     ``shift_strength``, with its avg. comm + private task perf (± std
     across agents) — since ComFed is excluded from the plots above, this is
     the only place its accuracy across shifts is visible.

Parameter counting
------------------
"Learnable parameters" is reported as ``agent_params (+ map_params)`` where:
  * ``agent_params`` — the SGD-trained weights inside each agent (comparable
    across all orchestrators);
  * ``map_params``   — *everything the orchestrator holds on top of the agents*:
    the restriction maps (``stiefel_matrices``, frozen ``requires_grad=False``),
    the SWBN whitening matrices ``W`` (buffers, updated by the custom SWBN rule),
    the whitening affine ``gamma``/``beta``, and — for Sheaf-FMTL — the
    ``projection_matrices`` (updated manually in ``on_train_epoch_end``).
    This is exactly the set of "maps not updated by autodiff" the counting is
    meant to include. Test-time alignment maps ``A_{j<-i}`` are excluded: they
    are transient and their footprint is already captured by the comm-kb column.

Counts are obtained by reconstructing agents + orchestrator from each run's own
stored config (the config evolved across sweeps, so counts are read per run).

Runs are scoped by their wandb *project*, defaulting to
``multi_hetero_agents_true``. This matters: ``study_name`` is reused across
projects, so filtering on it leaks runs from unrelated sweeps.

Usage:
    python scripts/plot_multiagent_metrics.py   # remote, your default entity
    python scripts/plot_multiagent_metrics.py \\
        --entity my-team \\
        --project multi_hetero_agents_true \\
        --out_dir results/multi_agent/plots
    python scripts/plot_multiagent_metrics.py --no-params   # skip param counts

    # Read from logs/wandb instead (falls back to remote if empty).
    python scripts/plot_multiagent_metrics.py --local --wandb_dir logs/wandb

When several runs share the same (orchestrator, shift) — e.g. a re-run ComFed —
``--select`` chooses which to keep:
    python scripts/plot_multiagent_metrics.py --select best      # best comm perf
    python scripts/plot_multiagent_metrics.py --select latest    # most recent (default)
    python scripts/plot_multiagent_metrics.py --select best \\
        --select-metric test/avg_private_task_perf               # best by another metric
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from matplotlib.container import ErrorbarContainer

# ── Orchestrator display order / labels / styles ──────────────────────────────
# This canonical order is shared (imported) by every plotting script in this
# directory (plot_bottleneck_metrics.py, plot_hetero_bottleneck_metrics.py)
# so that a given orchestrator always gets the same color/marker/linestyle
# everywhere, regardless of which (or how many) other orchestrators happen to
# be present in that script's own run set — see ``_style_maps`` below.
#
# The first six keep their original tab10 slots (blue / orange / green / red /
# purple / brown) exactly as they were in this script before bottleneck-plot
# consistency was introduced — SheafFRL=red and CESheafFRL=violet in
# particular. The rest are appended afterwards rather than interleaved, so
# adding them can't shift anyone else's color/marker.
ORCH_ORDER = [
    'NonCooperativeLearning',
    'ComFed',
    'SheafFMTL',
    'SheafFRL',
    'CESheafFRL',
    'SheafCFRL',
    'FedProto',
    'FedMuscle',
    'FederatedLearning',
]
ORCH_LABELS = {
    'NonCooperativeLearning': 'Non-cooperative',
    'ComFed': 'ComFed',
    'SheafFMTL': 'Sheaf-FMTL',
    'SheafFRL': 'Sheaf-FRL',
    'CESheafFRL': 'CE-Sheaf-FRL',
    'SheafCFRL': 'Sheaf-CFRL',
    'FedProto': 'FedProto',
    'FedMuscle': 'FedMuscle',
    'FederatedLearning': 'Federated',
}
_LINESTYLES = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 1))]
_PALETTE = sns.color_palette('tab10', n_colors=max(len(ORCH_ORDER), 10))
_MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', 'h', '<']

# Orchestrators dropped from every run set across all three plotting scripts
# here (plot_multiagent_metrics.py, plot_bottleneck_metrics.py,
# plot_hetero_bottleneck_metrics.py) — e.g. a model whose results aren't
# trustworthy/complete yet. Not derived from the data: edit this set directly
# to bring an orchestrator back.
EXCLUDED_ORCHS: set[str] = {'CESheafFRL', 'SheafCFRL'}


def drop_excluded_orchs(
    runs: dict[tuple[str, float], dict[str, Any]],
) -> dict[tuple[str, float], dict[str, Any]]:
    """Drop every run whose orchestrator is in :data:`EXCLUDED_ORCHS`."""
    return {k: v for k, v in runs.items() if k[0] not in EXCLUDED_ORCHS}

# Shared matplotlib style (figure size, font sizes, LaTeX text) applied to
# every figure this script draws.
_MPLSTYLE = (
    Path(__file__).resolve().parent.parent
    / 'config'
    / 'plotting'
    / 'plt.mplstyle'
)


def _savefig(fig: plt.Figure, out_path: Path) -> Path:
    """Save ``fig`` as both ``out_path`` (a ``.png``) and a sibling ``.pdf``."""
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    fig.savefig(out_path.with_suffix('.pdf'), bbox_inches='tight')
    return out_path

_AGENT_TRAIN_RE = re.compile(r'^train/task_performance_agent_(\d+)$')
_AGENT_COMM_RE = re.compile(r'^test/comm_task_perf_agent_(\d+)$')
_AGENT_PRIV_RE = re.compile(r'^test/private_task_perf_agent_(\d+)$')

# MNIST-style geometry for reconstructing agents (this experiment is MNIST).
_DATASET_GEOMETRY = {
    'mnist': (1, 28),
    'fmnist': (1, 28),
    'cifar10': (3, 32),
    'cifar100': (3, 32),
}


# ── WandB local-log parsing ───────────────────────────────────────────────────


def _unwrap(cfg: dict[str, Any], key: str) -> Any:
    """Read ``key`` from a WandB ``config.yaml`` (values are ``{value: ...}``)."""
    node = cfg.get(key)
    if isinstance(node, dict) and 'value' in node:
        return node['value']
    return node


def _load_raw_config(run_dir: Path) -> dict[str, Any] | None:
    cfg_path = run_dir / 'files' / 'config.yaml'
    if not cfg_path.exists():
        return None
    with cfg_path.open() as fh:
        return yaml.safe_load(fh) or {}


def read_run_project(run_dir: Path) -> str | None:
    """Return the wandb project a run belongs to (from its RunRecord).

    The project is a run property, not a config value, so it is read from the
    RunRecord at the top of the ``.wandb`` datastore (scanning only the first
    few records keeps this cheap even across hundreds of runs).
    """
    from wandb.proto import wandb_internal_pb2 as pb
    from wandb.sdk.internal.datastore import DataStore

    wf = next(run_dir.glob('*.wandb'), None)
    if wf is None:
        return None
    ds = DataStore()
    try:
        ds.open_for_scan(str(wf))
    except Exception:
        return None
    for _ in range(50):  # RunRecord sits near the start of the log
        try:
            data = ds.scan_data()
        except Exception:
            break
        if data is None:
            break
        rec = pb.Record()
        rec.ParseFromString(data)
        if rec.WhichOneof('record_type') == 'run':
            return rec.run.project or None
    return None


def read_run_meta(run_dir: Path) -> dict[str, Any] | None:
    """Return metadata + summary for a run, or None if unusable."""
    raw = _load_raw_config(run_dir)
    if raw is None:
        return None

    orch = _unwrap(raw, 'orchestrator')
    orch_name = (
        str(orch['_target_']).split('.')[-1]
        if isinstance(orch, dict) and '_target_' in orch
        else None
    )
    dataset = _unwrap(raw, 'dataset')
    shift = (
        dataset.get('shift_strength') if isinstance(dataset, dict) else None
    )
    if orch_name is None or shift is None:
        return None

    summ_path = run_dir / 'files' / 'wandb-summary.json'
    summary: dict[str, Any] = {}
    if summ_path.exists():
        with summ_path.open() as fh:
            summary = json.load(fh)

    return {
        'dir': run_dir,
        'orch': orch_name,
        'shift': float(shift),
        'study': _unwrap(raw, 'study_name'),
        'raw_config': raw,
        'summary': summary,
        'mtime': run_dir.stat().st_mtime,
    }


def _iter_history_rows(run_dir: Path):
    """Yield each logged history row of a run as a ``{key: value}`` dict."""
    from wandb.proto import wandb_internal_pb2 as pb
    from wandb.sdk.internal.datastore import DataStore

    wf = next(run_dir.glob('*.wandb'), None)
    if wf is None:
        return
    ds = DataStore()
    ds.open_for_scan(str(wf))
    while True:
        try:
            data = ds.scan_data()
        except Exception:
            break
        if data is None:
            break
        rec = pb.Record()
        rec.ParseFromString(data)
        if rec.WhichOneof('record_type') != 'history':
            continue
        row: dict[str, Any] = {}
        for it in rec.history.item:
            key = it.key or '.'.join(it.nested_key)
            try:
                row[key] = json.loads(it.value_json)
            except (json.JSONDecodeError, TypeError):
                row[key] = it.value_json
        yield row


def _history_rows(meta: dict[str, Any]):
    """Yield history rows for a run's meta, regardless of local vs. remote."""
    if 'dir' in meta:
        yield from _iter_history_rows(meta['dir'])
    else:
        yield from meta['run'].scan_history()


def read_train_task_matrix(meta: dict[str, Any]) -> np.ndarray | None:
    """Return a (n_agents, n_epochs) matrix of ``train/task_performance_agent_i``."""
    per_agent: dict[int, list[float]] = {}
    for row in _history_rows(meta):
        for key, val in row.items():
            m = _AGENT_TRAIN_RE.match(key)
            if m is not None and isinstance(val, (int, float)):
                per_agent.setdefault(int(m.group(1)), []).append(float(val))
    if not per_agent:
        return None
    n_ep = min(len(v) for v in per_agent.values())
    if n_ep == 0:
        return None
    return np.array(
        [per_agent[i][:n_ep] for i in sorted(per_agent)], dtype=float
    )


def _per_agent_values(
    summary: dict[str, Any], pattern: re.Pattern
) -> list[float]:
    return [
        float(v)
        for k, v in summary.items()
        if pattern.match(k) and isinstance(v, (int, float))
    ]


# ── Discovery / dedup ─────────────────────────────────────────────────────────


def _selection_score(
    meta: dict[str, Any], select: str, metric: str
) -> tuple[float, ...]:
    """Sort key for picking one run among duplicates of the same (orch, shift).

    ``latest`` ranks by mtime; ``best`` ranks by the chosen summary ``metric``
    (higher = better), falling back to mtime as a tie-breaker. Runs missing the
    metric score -inf so a run that has it always wins.
    """
    if select == 'best':
        val = meta['summary'].get(metric)
        val = float(val) if isinstance(val, (int, float)) else float('-inf')
        return (val, meta['mtime'])
    return (meta['mtime'],)


def _dedup_runs(
    candidates: dict[tuple[str, float], list[dict[str, Any]]],
    select: str,
    select_metric: str,
    label: Callable[[dict[str, Any]], str],
) -> dict[tuple[str, float], dict[str, Any]]:
    """Pick one run per ``(orch, shift)`` cell and report discarded duplicates.

    Shared by the local and remote discovery paths — both produce the same
    ``candidates`` shape (meta dicts with ``summary``/``mtime``), so the
    selection logic (and its printout) only needs to live once.
    """
    best: dict[tuple[str, float], dict[str, Any]] = {}
    for key, metas in candidates.items():
        chosen = max(
            metas,
            key=lambda m: _selection_score(m, select, metric=select_metric),
        )
        best[key] = chosen
        if len(metas) > 1:
            orch, shift = key
            score = chosen['summary'].get(select_metric)
            print(
                f'  {orch} @ shift {shift:g}: {len(metas)} runs → kept '
                f'{label(chosen)} ({select}'
                + (
                    f', {select_metric}={score:.3f}'
                    if select == 'best' and isinstance(score, (int, float))
                    else ''
                )
                + ')'
            )
    return best


def discover_runs(
    wandb_dir: Path,
    project: str | None,
    study_name: str | None = None,
    select: str = 'latest',
    select_metric: str = 'test/avg_comm_task_perf',
) -> dict[tuple[str, float], dict[str, Any]]:
    """Return {(orch, shift): meta}, one chosen run per cell.

    Runs are scoped by their wandb ``project`` (a run property, read from the
    RunRecord) — not by ``study_name``, which is reused across projects and so
    leaks runs from unrelated sweeps.

    When several completed runs share the same ``(orch, shift)`` (e.g. repeated
    ComFed runs), ``select`` decides which one is kept:
      * ``latest`` — most recent by mtime (default, reproduces old behaviour);
      * ``best``   — highest ``select_metric`` (default the plotted comm perf).
    """
    from collections import defaultdict

    candidates: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(
        list
    )
    for run_dir in sorted(wandb_dir.glob('run-*')):
        if project is not None and read_run_project(run_dir) != project:
            continue
        meta = read_run_meta(run_dir)
        if meta is None:
            continue
        if study_name is not None and str(meta['study']) != study_name:
            continue
        if 'test/avg_comm_task_perf' not in meta['summary']:
            continue  # "completed" = test phase finished
        candidates[(meta['orch'], meta['shift'])].append(meta)

    return _dedup_runs(
        candidates, select, select_metric, label=lambda m: m['dir'].name
    )


# ── Remote discovery (wandb cloud API) ─────────────────────────────────────────
# Used as an automatic fallback when the local ``logs/wandb`` scan above turns
# up nothing (e.g. the local run directories were cleaned up after syncing) —
# same shape of {(orch, shift): meta} so every downstream consumer (plots,
# tables, param counting) works unmodified regardless of where a run came from.


def _remote_mtime(run: Any, summary: dict[str, Any]) -> float:
    """Best-effort recency timestamp for a remote run (mirrors local mtime)."""
    ts = summary.get('_timestamp')
    if isinstance(ts, (int, float)):
        return float(ts)
    try:
        from datetime import datetime

        return datetime.fromisoformat(
            str(run.created_at).replace('Z', '+00:00')
        ).timestamp()
    except (ValueError, TypeError, AttributeError):
        return 0.0


def read_run_meta_remote(run: Any) -> dict[str, Any] | None:
    """Return metadata + summary for a remote wandb ``Run``, or None if unusable.

    Mirrors :func:`read_run_meta` field-for-field, except ``dir`` (a local
    path) is replaced by ``run`` (the API handle, needed later to fetch
    per-step history via ``run.scan_history()``).
    """
    # api.runs() hands back lazily-loaded runs (LIGHTWEIGHT_RUN_FRAGMENT):
    # .config/.summary are empty until a full attribute load is forced.
    run.load(force=True)
    raw = dict(run.config)
    orch = _unwrap(raw, 'orchestrator')
    orch_name = (
        str(orch['_target_']).split('.')[-1]
        if isinstance(orch, dict) and '_target_' in orch
        else None
    )
    dataset = _unwrap(raw, 'dataset')
    shift = (
        dataset.get('shift_strength') if isinstance(dataset, dict) else None
    )
    if orch_name is None or shift is None:
        return None

    summary = dict(run.summary)
    return {
        'run': run,
        'orch': orch_name,
        'shift': float(shift),
        'study': _unwrap(raw, 'study_name'),
        'raw_config': raw,
        'summary': summary,
        'mtime': _remote_mtime(run, summary),
    }


def discover_runs_remote(
    api: Any,
    entity: str,
    project: str,
    study_name: str | None = None,
    select: str = 'latest',
    select_metric: str = 'test/avg_comm_task_perf',
    max_workers: int = 8,
) -> dict[tuple[str, float], dict[str, Any]]:
    """Same contract as :func:`discover_runs`, scanning a wandb cloud project.

    Filters server-side on ``state == 'finished'`` (cheap), then fetches each
    run's config/summary in a small thread pool since those are independent
    HTTP round-trips.
    """
    from collections import defaultdict
    from concurrent.futures import ThreadPoolExecutor

    remote_runs = list(
        api.runs(f'{entity}/{project}', filters={'state': 'finished'})
    )
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        metas = list(pool.map(read_run_meta_remote, remote_runs))

    candidates: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(
        list
    )
    for meta in metas:
        if meta is None:
            continue
        if study_name is not None and str(meta['study']) != study_name:
            continue
        if 'test/avg_comm_task_perf' not in meta['summary']:
            continue
        candidates[(meta['orch'], meta['shift'])].append(meta)

    return _dedup_runs(
        candidates, select, select_metric, label=lambda m: m['run'].id
    )


def _ordered_orchs(orchs: set[str]) -> list[str]:
    known = [o for o in ORCH_ORDER if o in orchs]
    extra = sorted(o for o in orchs if o not in ORCH_ORDER)
    return known + extra


def _style_maps(orchs: list[str]) -> tuple[dict, dict, dict]:
    """Return per-orchestrator ``{color}``, ``{marker}``, ``{linestyle}`` dicts.

    Each is keyed off the orchestrator's fixed position in ``ORCH_ORDER``
    (any name not in it is appended, sorted, after the known ones) rather
    than its position within ``orchs`` — so a given orchestrator keeps the
    same look whether it is plotted alongside all 9 known orchestrators or
    just one other.
    """
    extra = sorted(o for o in orchs if o not in ORCH_ORDER)
    canonical = ORCH_ORDER + extra
    colors = {o: _PALETTE[canonical.index(o) % len(_PALETTE)] for o in orchs}
    markers = {o: _MARKERS[canonical.index(o) % len(_MARKERS)] for o in orchs}
    styles = {
        o: _LINESTYLES[canonical.index(o) % len(_LINESTYLES)] for o in orchs
    }
    return colors, markers, styles


# ── Parameter counting (reconstruct agents + orchestrator from config) ─────────

_MAE = None
_PARAM_CACHE: dict[str, dict | None] = {}


def _load_mae():
    global _MAE
    if _MAE is None:
        import importlib.util

        here = Path(__file__).resolve().parent
        spec = importlib.util.spec_from_file_location(
            'multi_agent_experiment', here / 'multi_agent_experiment.py'
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _MAE = mod
    return _MAE


class _StubDatamodule:
    """Minimal datamodule stand-in so _build_agents can size the encoders."""

    def __init__(self, n_agents: int, channels: int, img_size: int) -> None:
        self.models = list(range(n_agents))
        self.num_classes = {'label': 10}
        self.input_dims = {str(i): channels for i in range(n_agents)}
        self.input_shape = (channels, img_size, img_size)


def _dataset_geometry(cfg) -> tuple[int, int]:
    name = str(getattr(cfg.dataset, 'name', 'mnist')).lower()
    for key, geom in _DATASET_GEOMETRY.items():
        if key in name:
            return geom
    return (1, 28)


def _param_signature(raw_config: dict[str, Any]) -> str:
    """Config subset that determines parameter counts (shift-independent)."""
    keep = ('orchestrator', 'model', 'graph', 'seed')
    sig = {k: _unwrap(raw_config, k) for k in keep}
    ds = _unwrap(raw_config, 'dataset') or {}
    if isinstance(ds, dict):
        sig['dataset'] = {k: ds.get(k) for k in ('name', 'n_agents', 'groups')}
    return json.dumps(sig, sort_keys=True, default=str)


def count_model_stats(meta: dict[str, Any]) -> dict | None:
    """Return {agent_params, map_params, total, fwd_flops_total} or None.

    ``fwd_flops_total`` is the forward-pass FLOP count (torch ``FlopCounterMode``
    convention) for a *single* input, summed across all agents. It depends only
    on the agent architectures, so it is identical across orchestrators that
    share the same agents — the training-FLOP estimate scales it by
    ``3 x batch_size x steps`` in the table builder.
    """
    sig = _param_signature(meta['raw_config'])
    if sig in _PARAM_CACHE:
        return _PARAM_CACHE[sig]

    result: dict | None = None
    try:
        import sys

        import torch

        repo_root = str(Path(__file__).resolve().parent.parent)
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

        from omegaconf import OmegaConf

        from src.utils.graph_generator import generate_neighbors

        mae = _load_mae()
        raw = meta['raw_config']
        cfg = OmegaConf.create(
            {
                k: _unwrap(raw, k)
                for k in raw
                if k not in ('_wandb', 'wandb_version')
            }
        )
        n = int(cfg.dataset.n_agents)
        channels, img = _dataset_geometry(cfg)

        atc = mae._parse_agent_target_classes(cfg)
        if atc is not None:
            for i in range(n):
                atc.setdefault(i, set())
        neighbors = generate_neighbors(
            mode=cfg.graph.neighbors_mode,
            n_agents=n,
            seed=cfg.graph.get('seed', 42),
            p=cfg.graph.get('p', 0.3),
            m=cfg.graph.get('m', 3),
            manual=cfg.graph.get('neighbors', {}),
            target_classes=atc,
            max_edge_frac=cfg.graph.get('max_edge_frac', 0.4),
            similarity=cfg.graph.get('similarity', 'intersection'),
        )
        per_agents_cfg = mae._parse_per_agent_cfg(cfg)
        agents, latent_dims = mae._build_agents(
            cfg, _StubDatamodule(n, channels, img), per_agents_cfg
        )
        orch = mae._build_orchestrator(
            cfg, agents, neighbors, latent_dims, per_agents_cfg
        )

        agent_params = sum(p.numel() for p in orch.agents.parameters())
        registered = sum(p.numel() for p in orch.parameters())
        whitening_W = 0
        if hasattr(orch, 'whitening_layers'):
            whitening_W = sum(
                m.W.numel() for m in orch.whitening_layers.values()
            )
        map_params = (registered - agent_params) + whitening_W

        # Forward FLOPs for one input, summed across agents.
        from torch.utils.flop_counter import FlopCounterMode

        example = torch.zeros(1, channels, img, img)
        fwd_flops_total = 0
        for agent in agents.values():
            agent.eval()
            counter = FlopCounterMode(display=False)
            with counter, torch.no_grad():
                agent(example)
            fwd_flops_total += int(counter.get_total_flops())

        result = {
            'agent_params': int(agent_params),
            'map_params': int(map_params),
            'total': int(agent_params + map_params),
            'fwd_flops_total': int(fwd_flops_total),
        }
    except Exception as exc:  # noqa: BLE001 — table still useful without counts
        print(
            f'  [warn] model-stats reconstruction failed for '
            f'{meta["orch"]}: {exc}'
        )
        result = None

    _PARAM_CACHE[sig] = result
    return result


def _fmt_count(n: int) -> str:
    if n >= 1e9:
        return f'{n / 1e9:.2f} B'
    if n >= 1e6:
        return f'{n / 1e6:.2f} M'
    if n >= 1e3:
        return f'{n / 1e3:.1f} K'
    return str(n)


def _fmt_flops(n: float) -> str:
    for suffix, div in (('E', 1e18), ('P', 1e15), ('T', 1e12), ('G', 1e9)):
        if n >= div:
            return f'{n / div:.2f} {suffix}FLOPs'
    return f'{n / 1e6:.1f} MFLOPs'


def _fmt_seconds(s: float) -> str:
    return f'{int(round(s))} s ({s / 60:.1f} min)'


# ── Plot 1: metric vs x (shift, bottleneck dim, …) ─────────────────────────────
# These helpers are x-axis-agnostic (``x_col``/``xlabel``/``xscale``) and are
# reused as-is by plot_bottleneck_metrics.py and plot_hetero_bottleneck_metrics.py
# so every "metric vs sweep parameter" plot in this family looks and behaves
# identically (dodge, error-bar-cap styling, legend handling, xticks).


def _agent_metric_long_df(
    runs: dict[tuple[str, float], dict[str, Any]],
    pattern: re.Pattern,
    value_col: str,
    x_col: str = 'shift',
) -> pd.DataFrame:
    """One row per (orchestrator, x, agent) with a per-agent metric value."""
    rows = []
    for (orch, x), meta in runs.items():
        for val in _per_agent_values(meta['summary'], pattern):
            rows.append(
                {
                    'orchestrator': ORCH_LABELS.get(orch, orch),
                    '_orch': orch,
                    x_col: x,
                    value_col: val,
                }
            )
    return pd.DataFrame(rows)


def metric_summary_df(
    df: pd.DataFrame,
    orchs: list[str],
    value_col: str,
    x_col: str = 'shift',
) -> pd.DataFrame:
    """Per-(orchestrator, x) mean/std table, e.g. for saving alongside a plot."""
    rows = []
    for orch in orchs:
        sub = df[df['_orch'] == orch]
        if sub.empty:
            continue
        grouped = sub.groupby(x_col)[value_col]
        means = grouped.mean()
        stds = grouped.std(ddof=0)
        for x in sorted(means.index):
            rows.append(
                {
                    'orchestrator': ORCH_LABELS.get(orch, orch),
                    x_col: x,
                    value_col: means[x],
                    f'{value_col}_std': stds[x],
                }
            )
    return pd.DataFrame(rows)


def _dodge_offsets(n: int, xvals: list[float]) -> np.ndarray:
    """Small per-curve x-offsets so same-x markers don't overlap.

    Sized as a fraction of the smallest gap between consecutive x values (or,
    with a single x value, a fraction of its magnitude) so the dodge stays
    visually subordinate to the real x spacing.
    """
    if n <= 1:
        return np.zeros(n)
    if len(xvals) > 1:
        gap = min(np.diff(sorted(xvals)))
    else:
        gap = abs(xvals[0]) if xvals and xvals[0] != 0 else 1.0
    width = gap * 0.4
    return np.linspace(-width / 2, width / 2, n)


def _plot_metric_on_ax(
    ax: plt.Axes,
    df: pd.DataFrame,
    orchs: list[str],
    colors: dict,
    markers: dict,
    value_col: str,
    ylabel: str,
    adjusted: bool,
    x_col: str = 'shift',
    xlabel: str = 'Distribution shift strength',
    xscale: str = 'linear',
    xticklabel_fmt: Callable[[float], str] | None = None,
    xticklabel_rotation: float = 0,
) -> None:
    """Draw one metric-vs-x panel on ``ax``.

    Shared by the standalone (:func:`plot_metric_vs_x`) and combined
    (:func:`plot_two_metrics_vs_x`) figures. ``adjusted=True`` dodges each
    orchestrator's points a little off the shared x value — so markers
    landing on the same x tick don't stack on top of each other — and draws
    the per-x agent std as small error-bar caps (as in a bar plot) instead of
    seaborn's translucent band, which is hard to read once curves are
    dodged. ``adjusted=False`` keeps the original undodged line + std-band
    rendering.
    """
    order = [o for o in orchs if o in set(df['_orch'])]
    df = df[df['_orch'].isin(order)]
    xvals = sorted(df[x_col].unique())

    if adjusted:
        offsets = _dodge_offsets(len(order), xvals)
        for off, orch in zip(offsets, order):
            sub = df[df['_orch'] == orch]
            grouped = sub.groupby(x_col)[value_col]
            means = grouped.mean().reindex(xvals)
            stds = grouped.std(ddof=0).reindex(xvals).fillna(0.0)
            mask = means.notna().to_numpy()
            xs = np.asarray(xvals, dtype=float) + off
            ax.errorbar(
                xs[mask],
                means.to_numpy()[mask],
                yerr=stds.to_numpy()[mask],
                color=colors[orch],
                marker=markers[orch],
                linestyle='-',
                capsize=3,
                elinewidth=1.0,
                label=ORCH_LABELS.get(orch, orch),
            )
    else:
        palette = {ORCH_LABELS.get(o, o): colors[o] for o in order}
        marker_map = {ORCH_LABELS.get(o, o): markers[o] for o in order}
        hue_order = [ORCH_LABELS.get(o, o) for o in order]
        sns.lineplot(
            data=df,
            x=x_col,
            y=value_col,
            hue='orchestrator',
            style='orchestrator',
            hue_order=hue_order,
            style_order=hue_order,
            palette=palette,
            markers=marker_map,
            dashes=False,
            errorbar='sd',
            estimator='mean',
            ax=ax,
        )
    if xscale == 'log':
        ax.set_xscale('log')
    ax.set_xticks(xvals)
    if xticklabel_fmt is not None:
        ax.set_xticklabels(
            [xticklabel_fmt(x) for x in xvals],
            rotation=xticklabel_rotation,
            ha='right' if xticklabel_rotation else 'center',
        )
    if xscale == 'log':
        ax.minorticks_off()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def _strip_errorbar_caps(handles: list) -> list:
    """Replace ``ErrorbarContainer`` legend handles with just their line.

    ``ax.errorbar`` legend entries otherwise draw the little error-bar caps
    next to the marker/line, which clutters the legend.
    """
    return [h[0] if isinstance(h, ErrorbarContainer) else h for h in handles]


def plot_metric_vs_x(
    df: pd.DataFrame,
    orchs: list[str],
    colors: dict,
    markers: dict,
    out_dir: Path,
    fname: str,
    value_col: str,
    ylabel: str,
    adjusted: bool = True,
    x_col: str = 'shift',
    xlabel: str = 'Distribution shift strength',
    xscale: str = 'linear',
    xticklabel_fmt: Callable[[float], str] | None = None,
    xticklabel_rotation: float = 0,
) -> None:
    """Plot ``value_col`` (mean across agents) vs ``x_col``, one curve per orch."""
    with plt.style.context(str(_MPLSTYLE)):
        fig, ax = plt.subplots()
        _plot_metric_on_ax(
            ax,
            df,
            orchs,
            colors,
            markers,
            value_col,
            ylabel,
            adjusted,
            x_col=x_col,
            xlabel=xlabel,
            xscale=xscale,
            xticklabel_fmt=xticklabel_fmt,
            xticklabel_rotation=xticklabel_rotation,
        )
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            _strip_errorbar_caps(handles),
            labels,
            title='Method',
            frameon=True,
        )
        sns.despine()
        out = out_dir / fname
        _savefig(fig, out)
        plt.close(fig)
    print(f'  saved → {out}')


def plot_two_metrics_vs_x(
    comm_df: pd.DataFrame,
    priv_df: pd.DataFrame,
    orchs: list[str],
    colors: dict,
    markers: dict,
    out_dir: Path,
    fname: str,
    adjusted: bool = True,
    x_col: str = 'shift',
    xlabel: str = 'Distribution shift strength',
    xscale: str = 'linear',
    xticklabel_fmt: Callable[[float], str] | None = None,
    xticklabel_rotation: float = 0,
    comm_ylabel: str = 'Avg. communication accuracy',
    priv_ylabel: str = 'Avg. private accuracy',
) -> None:
    """Comm + private accuracy vs x, side by side in one figure.

    Each panel keeps the same size/proportions as the corresponding
    standalone plot (the figure is just twice as wide, one row of two axes),
    and the two per-panel legends are merged into a single legend, in three
    columns, sitting just above both axes.
    """
    specs = [
        (comm_df, 'comm_task_perf', comm_ylabel),
        (priv_df, 'private_task_perf', priv_ylabel),
    ]
    with plt.style.context(str(_MPLSTYLE)):
        base_w, base_h = plt.rcParams['figure.figsize']
        fig, axes = plt.subplots(1, 2, figsize=(2 * base_w, base_h))
        for ax, (df, value_col, ylabel) in zip(axes, specs):
            _plot_metric_on_ax(
                ax,
                df,
                orchs,
                colors,
                markers,
                value_col,
                ylabel,
                adjusted,
                x_col=x_col,
                xlabel=xlabel,
                xscale=xscale,
                xticklabel_fmt=xticklabel_fmt,
                xticklabel_rotation=xticklabel_rotation,
            )
        handles, labels = axes[0].get_legend_handles_labels()
        top = max(ax.get_position().y1 for ax in axes)
        fig.legend(
            _strip_errorbar_caps(handles),
            labels,
            title='Method',
            loc='lower center',
            bbox_to_anchor=(0.5, top + 0.03),
            ncol=3,
            frameon=True,
            borderaxespad=0.0,
        )
        out = out_dir / fname
        _savefig(fig, out)
        plt.close(fig)
    print(f'  saved → {out}')


# ── Plot 2: training curves per shift ─────────────────────────────────────────


def plot_train_curves_per_shift(
    runs: dict[tuple[str, float], dict[str, Any]],
    orchs: list[str],
    colors: dict,
    styles: dict,
    out_dir: Path,
) -> None:
    shifts = sorted({shift for _, shift in runs})
    for shift in shifts:
        with plt.style.context(str(_MPLSTYLE)):
            fig, ax = plt.subplots()
            any_curve = False
            for orch in orchs:
                meta = runs.get((orch, shift))
                if meta is None:
                    continue
                mat = read_train_task_matrix(meta)
                if mat is None:
                    continue
                mean = mat.mean(axis=0)
                std = mat.std(axis=0)
                epochs = np.arange(1, mat.shape[1] + 1)
                ax.plot(
                    epochs,
                    mean,
                    color=colors[orch],
                    linestyle=styles[orch],
                    label=ORCH_LABELS.get(orch, orch),
                )
                ax.fill_between(
                    epochs,
                    mean - std,
                    mean + std,
                    color=colors[orch],
                    alpha=0.15,
                    linewidth=0,
                )
                any_curve = True
            if not any_curve:
                plt.close(fig)
                continue
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Train task performance')
            # ax.set_title(f'Training task performance — shift strength {shift:g}')
            ax.legend(title='Method', frameon=True)
            sns.despine()
            out = out_dir / f'train_task_perf_shift_{shift:g}.png'
            _savefig(fig, out)
            plt.close(fig)
        print(f'  saved → {out}')


# ── Part 3: per-shift tables ──────────────────────────────────────────────────


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float('nan'), float('nan')
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std())


def build_shift_table(
    runs: dict[tuple[str, float], dict[str, Any]],
    orchs: list[str],
    shift: float,
    with_params: bool,
) -> pd.DataFrame:
    rows = []
    for orch in orchs:
        meta = runs.get((orch, shift))
        if meta is None:
            continue
        summ = meta['summary']
        comm_mean, comm_std = _mean_std(
            _per_agent_values(summ, _AGENT_COMM_RE)
        )
        priv_mean, priv_std = _mean_std(
            _per_agent_values(summ, _AGENT_PRIV_RE)
        )
        row: dict[str, Any] = {
            'orchestrator': ORCH_LABELS.get(orch, orch),
            'comm_task_perf_mean': comm_mean,
            'comm_task_perf_std': comm_std,
            'private_task_perf_mean': priv_mean,
            'private_task_perf_std': priv_std,
            'comm_rounds': summ.get('train/communication_rounds'),
            'comm_kb': summ.get('train/communication_kilobytes'),
            # Proxy for evaluation time: total run wall-clock (fit + test).
            'runtime_s': summ.get('_runtime'),
        }
        if with_params:
            stats = count_model_stats(meta)
            if stats is None:
                row['agent_params'] = None
                row['map_params'] = None
                row['total_params'] = None
                row['train_flops_est'] = None
            else:
                row['agent_params'] = stats['agent_params']
                row['map_params'] = stats['map_params']
                row['total_params'] = stats['total']
                # Estimated training FLOPs: 3 (fwd+bwd) x per-sample forward
                # FLOPs x samples processed (batch_size x total steps).
                ds = _unwrap(meta['raw_config'], 'dataset') or {}
                batch_size = (
                    ds.get('batch_size') if isinstance(ds, dict) else None
                )
                steps = summ.get('trainer/global_step')
                if batch_size is not None and steps is not None:
                    row['train_flops_est'] = (
                        3
                        * stats['fwd_flops_total']
                        * int(batch_size)
                        * int(steps)
                    )
                else:
                    row['train_flops_est'] = None
        rows.append(row)
    return pd.DataFrame(rows)


def _pretty_table(df: pd.DataFrame, with_params: bool) -> pd.DataFrame:
    """Human-readable string columns for printing / markdown."""
    out = pd.DataFrame()
    out['Orchestrator'] = df['orchestrator']
    out['Avg comm task perf'] = [
        f'{m:.3f} ± {s:.3f}'
        for m, s in zip(df['comm_task_perf_mean'], df['comm_task_perf_std'])
    ]
    out['Avg private task perf'] = [
        f'{m:.3f} ± {s:.3f}'
        for m, s in zip(
            df['private_task_perf_mean'], df['private_task_perf_std']
        )
    ]
    out['Comm rounds'] = [
        '—' if r is None else f'{int(r):,}' for r in df['comm_rounds']
    ]
    out['Comm kB'] = [
        '—' if k is None else f'{float(k):,.0f}' for k in df['comm_kb']
    ]
    if with_params:
        params = []
        for _, r in df.iterrows():
            if r.get('agent_params') is None:
                params.append('n/a')
            elif r['map_params']:
                params.append(
                    f'{_fmt_count(int(r["agent_params"]))} '
                    f'(+{_fmt_count(int(r["map_params"]))} maps)'
                )
            else:
                params.append(_fmt_count(int(r['agent_params'])))
        out['Params (agents + maps)'] = params
        out['Est. train FLOPs'] = [
            'n/a' if f is None else _fmt_flops(float(f))
            for f in df['train_flops_est']
        ]
    out['Runtime (proxy)'] = [
        '—' if s is None else _fmt_seconds(float(s)) for s in df['runtime_s']
    ]
    return out


def _to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ['| ' + ' | '.join(cols) + ' |']
    lines.append('| ' + ' | '.join(['---'] * len(cols)) + ' |')
    for _, r in df.iterrows():
        lines.append('| ' + ' | '.join(str(r[c]) for c in cols) + ' |')
    return '\n'.join(lines)


def build_tables(
    runs: dict[tuple[str, float], dict[str, Any]],
    orchs: list[str],
    out_dir: Path,
    with_params: bool,
) -> None:
    tables_dir = out_dir / 'tables'
    tables_dir.mkdir(parents=True, exist_ok=True)
    shifts = sorted({shift for _, shift in runs})
    md_blocks = []
    for shift in shifts:
        raw = build_shift_table(runs, orchs, shift, with_params)
        if raw.empty:
            continue
        raw.to_csv(tables_dir / f'table_shift_{shift:g}.csv', index=False)
        pretty = _pretty_table(raw, with_params)
        title = f'### Distribution shift strength = {shift:g}'
        block = f'{title}\n\n{_to_markdown(pretty)}\n'
        md_blocks.append(block)
        print(f'\n{title}')
        print(pretty.to_string(index=False))
    if md_blocks:
        md_path = tables_dir / 'summary_tables.md'
        md_path.write_text('\n'.join(md_blocks))
        print(f'\n  saved tables → {tables_dir}')


def build_comfed_table(
    runs: dict[tuple[str, float], dict[str, Any]], out_dir: Path
) -> None:
    """ComFed's avg comm + private accuracy across shifts, one row per shift.

    ComFed is excluded from every shift plot (it sits far below the other
    methods and squashes the y-axis), so this table is the only place its
    accuracy across shifts is visible.
    """
    rows = []
    for (orch, shift), meta in runs.items():
        if orch != 'ComFed':
            continue
        summ = meta['summary']
        comm_mean, comm_std = _mean_std(_per_agent_values(summ, _AGENT_COMM_RE))
        priv_mean, priv_std = _mean_std(_per_agent_values(summ, _AGENT_PRIV_RE))
        rows.append(
            {
                'shift': shift,
                'comm_task_perf_mean': comm_mean,
                'comm_task_perf_std': comm_std,
                'private_task_perf_mean': priv_mean,
                'private_task_perf_std': priv_std,
            }
        )
    if not rows:
        return
    raw = pd.DataFrame(rows).sort_values('shift')

    tables_dir = out_dir / 'tables'
    tables_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(tables_dir / 'comfed_vs_shift.csv', index=False)

    pretty = pd.DataFrame()
    pretty['Shift'] = [f'{s:g}' for s in raw['shift']]
    pretty['Avg comm task perf'] = [
        f'{m:.3f} ± {s:.3f}'
        for m, s in zip(raw['comm_task_perf_mean'], raw['comm_task_perf_std'])
    ]
    pretty['Avg private task perf'] = [
        f'{m:.3f} ± {s:.3f}'
        for m, s in zip(
            raw['private_task_perf_mean'], raw['private_task_perf_std']
        )
    ]

    title = '### ComFed — accuracy vs distribution shift'
    md_path = tables_dir / 'comfed_vs_shift.md'
    md_path.write_text(f'{title}\n\n{_to_markdown(pretty)}\n')
    print(f'\n{title}')
    print(pretty.to_string(index=False))
    print(f'\n  saved → {md_path}')


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--wandb_dir', type=Path, default=Path('logs/wandb'))
    parser.add_argument(
        '--project',
        type=str,
        default='multi_hetero_agents_true',
        help='Only include runs from this wandb project (none = all projects).',
    )
    parser.add_argument(
        '--study_name',
        type=str,
        default='none',
        help='Optional extra filter on config study_name (none = no filter).',
    )
    parser.add_argument(
        '--out_dir', type=Path, default=Path('results/multi_agent/plots')
    )
    parser.add_argument(
        '--select',
        choices=['latest', 'best'],
        default='latest',
        help='When several runs share an (orchestrator, shift): keep the most '
        'recent (latest, default) or the best-scoring (best).',
    )
    parser.add_argument(
        '--select-metric',
        type=str,
        default='test/avg_comm_task_perf',
        help='Summary metric maximised when --select best (default: the '
        'plotted comm task perf).',
    )
    parser.add_argument(
        '--no-params',
        action='store_true',
        help='Skip parameter counting (avoids reconstructing the models).',
    )
    parser.add_argument(
        '--entity',
        type=str,
        default=None,
        help='wandb entity to use for the (default) remote fetch (default: '
        'your wandb default entity). Ignored with --local.',
    )
    parser.add_argument(
        '--local',
        action='store_true',
        help='Scan local logs/wandb run directories instead of the wandb '
        'cloud API. Still auto-falls back to the API if nothing local is '
        'found.',
    )
    parser.add_argument(
        '--together',
        action='store_true',
        help='Draw comm + private accuracy vs shift side by side in one '
        'figure (shared legend on top, 2 columns) instead of two separate '
        'figures.',
    )
    args = parser.parse_args()

    project = None if args.project.lower() == 'none' else args.project
    study = None if args.study_name.lower() == 'none' else args.study_name

    def _fetch_remote() -> dict[tuple[str, float], dict[str, Any]]:
        if project is None:
            raise SystemExit(
                'Remote fetching needs an explicit --project (use --local '
                'for an all-projects local scan).'
            )
        import wandb

        api = wandb.Api()
        entity = args.entity or api.default_entity
        if entity is None:
            raise SystemExit(
                'No wandb entity available for the remote fetch (pass '
                '--entity, or run `wandb login`).'
            )
        print(
            f'Scanning wandb cloud project {entity}/{project!r} '
            f'(study={study!r}, select={args.select!r}) …'
        )
        return discover_runs_remote(
            api,
            entity,
            project,
            study,
            select=args.select,
            select_metric=args.select_metric,
        )

    if args.local:
        print(
            f'Scanning {args.wandb_dir} (project={project!r}, study={study!r}, '
            f'select={args.select!r}) …'
        )
        runs = discover_runs(
            args.wandb_dir,
            project,
            study,
            select=args.select,
            select_metric=args.select_metric,
        )
        if not runs and project is not None:
            print(
                f'No completed local runs found in {args.wandb_dir} — '
                f'falling back to the wandb cloud API …'
            )
            runs = _fetch_remote()
    else:
        runs = _fetch_remote()

    if not runs:
        raise SystemExit(
            f'No completed runs found for project {project!r} '
            f'({"local" if args.local else "remote"}).'
        )

    runs = drop_excluded_orchs(runs)
    if not runs:
        raise SystemExit(
            f'No completed runs left for project {project!r} after '
            f'excluding {sorted(EXCLUDED_ORCHS)}.'
        )

    orchs = _ordered_orchs({o for o, _ in runs})
    colors, markers, styles = _style_maps(orchs)
    shifts = sorted({shift for _, shift in runs})
    print(f'Found {len(runs)} runs — orchestrators: {orchs}')
    print(f'  shift strengths: {shifts}')

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style='whitegrid', context='paper', font_scale=1.4)

    # ComFed is never drawn in the shift plots (it sits far below the other
    # methods and squashes the y-axis) — its own accuracy across shifts goes
    # in the dedicated table (part 4) instead.
    plot_orchs = [o for o in orchs if o != 'ComFed']

    # ── Plot 1: comm + private task performance vs shift ──────────────────────
    print('\nPlot 1: communication + private task performance vs shift …')
    comm_df = _agent_metric_long_df(runs, _AGENT_COMM_RE, 'comm_task_perf')
    priv_df = _agent_metric_long_df(runs, _AGENT_PRIV_RE, 'private_task_perf')
    if args.together:
        plot_two_metrics_vs_x(
            comm_df,
            priv_df,
            plot_orchs,
            colors,
            markers,
            args.out_dir,
            'comm_and_priv_task_perf_vs_shift.png',
        )
    else:
        plot_metric_vs_x(
            comm_df,
            plot_orchs,
            colors,
            markers,
            args.out_dir,
            'comm_task_perf_vs_shift.png',
            'comm_task_perf',
            'Avg. communication accuracy',
        )
        plot_metric_vs_x(
            priv_df,
            plot_orchs,
            colors,
            markers,
            args.out_dir,
            'private_task_perf_vs_shift.png',
            'private_task_perf',
            'Avg. private accuracy',
        )

    # ── Plot 2 (per-shift training curves) ────────────────────────────────────
    print('\nPlot 2: training task performance per shift …')
    plot_train_curves_per_shift(runs, orchs, colors, styles, args.out_dir)

    # ── Part 3 (per-shift tables) ─────────────────────────────────────────────
    print('\nPart 3: per-shift summary tables …')
    build_tables(runs, orchs, args.out_dir, with_params=not args.no_params)

    # ── Part 4 (ComFed accuracy vs shift table) ───────────────────────────────
    print('\nPart 4: ComFed accuracy vs shift table …')
    build_comfed_table(runs, args.out_dir)

    print('\nDone.')


if __name__ == '__main__':
    main()
