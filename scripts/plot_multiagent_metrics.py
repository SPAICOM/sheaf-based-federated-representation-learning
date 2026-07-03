"""Visualize multi-agent experiment metrics from local WandB run logs.

Reads the WandB transaction logs written by ``scripts/multi_agent_experiment.py``
(under ``logs/wandb/run-*``) — no cloud access required, everything is parsed
straight from the on-disk ``.wandb`` datastore and ``files/`` of each run — and
produces:

  1. ``test/avg_comm_task_perf`` vs ``shift_strength``, one curve per
     orchestrator, with seaborn's translucent std band (spread across agents).
     Two versions are written: one including ComFed, one without it (ComFed
     sits far below the others and squashes the y-axis).

  2. For each ``shift_strength``, a *separate* figure of the training task
     performance (mean across agents ± std band) over epochs, one curve per
     orchestrator, each drawn with a distinct line style.

  3. For each ``shift_strength``, a table (printed + CSV + markdown) with one
     row per orchestrator:
       - avg. comm task perf (± std across agents)
       - avg. private task perf (± std across agents)
       - communication rounds (training-cumulative)
       - communication kilobytes (training-cumulative)
       - number of parameters: agent weights (+ sheaf/coupling maps)
       - estimated training FLOPs: 3 x per-sample forward FLOPs x
         batch_size x total steps (agent-only; see note below)
       - runtime proxy: total run wall-clock (``_runtime``); no eval-only
         timer is logged, so this is fit+test combined

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

Runs are scoped by their wandb *project* (read from the RunRecord), defaulting
to ``multi_hetero_agents_true``. This matters: ``study_name`` is reused across
projects, so filtering on it leaks runs from unrelated sweeps.

Usage:
    python scripts/plot_multiagent_metrics.py
    python scripts/plot_multiagent_metrics.py \\
        --wandb_dir logs/wandb \\
        --project multi_hetero_agents_true \\
        --out_dir results/multi_agent/plots
    python scripts/plot_multiagent_metrics.py --no-params   # skip param counts

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
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

# ── Orchestrator display order / labels / styles ──────────────────────────────
ORCH_ORDER = [
    'NonCooperativeLearning',
    'ComFed',
    'FedProto',
    'FedMuscle',
    'SheafFMTL',
    'SheafFRL',
    'CESheafFRL',
    'SheafCFRL',
]
ORCH_LABELS = {
    'NonCooperativeLearning': 'Non-cooperative',
    'ComFed': 'ComFed',
    'FedProto': 'FedProto',
    'FedMuscle': 'FedMuscle',
    'SheafFMTL': 'Sheaf-FMTL',
    'SheafFRL': 'Sheaf-FRL',
    'CESheafFRL': 'CE-Sheaf-FRL',
    'SheafCFRL': 'Sheaf-CFRL',
}
_LINESTYLES = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 1))]

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
    shift = dataset.get('shift_strength') if isinstance(dataset, dict) else None
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


def read_train_task_matrix(run_dir: Path) -> np.ndarray | None:
    """Return a (n_agents, n_epochs) matrix of ``train/task_performance_agent_i``."""
    per_agent: dict[int, list[float]] = {}
    for row in _iter_history_rows(run_dir):
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


def _per_agent_values(summary: dict[str, Any], pattern: re.Pattern) -> list[float]:
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

    best: dict[tuple[str, float], dict[str, Any]] = {}
    for key, metas in candidates.items():
        chosen = max(metas, key=lambda m: _selection_score(m, select, metric=select_metric))
        best[key] = chosen
        if len(metas) > 1:
            orch, shift = key
            score = chosen['summary'].get(select_metric)
            print(
                f'  {orch} @ shift {shift:g}: {len(metas)} runs → kept '
                f'{chosen["dir"].name} ({select}'
                + (f", {select_metric}={score:.3f}" if select == 'best'
                   and isinstance(score, (int, float)) else '')
                + ')'
            )
    return best


def _ordered_orchs(orchs: set[str]) -> list[str]:
    known = [o for o in ORCH_ORDER if o in orchs]
    extra = sorted(o for o in orchs if o not in ORCH_ORDER)
    return known + extra


def _style_maps(orchs: list[str]) -> tuple[dict, dict]:
    palette = sns.color_palette('tab10', n_colors=max(len(orchs), 3))
    colors = {o: palette[i] for i, o in enumerate(orchs)}
    styles = {o: _LINESTYLES[i % len(_LINESTYLES)] for i, o in enumerate(orchs)}
    return colors, styles


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
        sig['dataset'] = {
            k: ds.get(k) for k in ('name', 'n_agents', 'groups')
        }
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
        print(f'  [warn] model-stats reconstruction failed for '
              f'{meta["orch"]}: {exc}')
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


# ── Plot 1: comm task perf vs shift ───────────────────────────────────────────


def _comm_long_df(
    runs: dict[tuple[str, float], dict[str, Any]],
) -> pd.DataFrame:
    """One row per (orchestrator, shift, agent) with per-agent comm task perf."""
    rows = []
    for (orch, shift), meta in runs.items():
        for val in _per_agent_values(meta['summary'], _AGENT_COMM_RE):
            rows.append(
                {
                    'orchestrator': ORCH_LABELS.get(orch, orch),
                    '_orch': orch,
                    'shift': shift,
                    'comm_task_perf': val,
                }
            )
    return pd.DataFrame(rows)


def plot_comm_vs_shift(
    df: pd.DataFrame,
    orchs: list[str],
    colors: dict,
    out_dir: Path,
    fname: str,
) -> None:
    order = [o for o in orchs if o in set(df['_orch'])]
    palette = {ORCH_LABELS.get(o, o): colors[o] for o in order}
    hue_order = [ORCH_LABELS.get(o, o) for o in order]

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.lineplot(
        data=df,
        x='shift',
        y='comm_task_perf',
        hue='orchestrator',
        style='orchestrator',
        hue_order=hue_order,
        style_order=hue_order,
        palette=palette,
        markers=True,
        dashes=False,
        errorbar='sd',
        estimator='mean',
        ax=ax,
    )
    ax.set_xlabel('Distribution shift strength')
    ax.set_ylabel('Avg communication task performance')
    ax.set_title('Communication task performance vs distribution shift')
    ax.legend(title='Orchestrator', frameon=True)
    sns.despine()
    out = out_dir / fname
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'  saved → {out}')
    plt.close(fig)


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
        fig, ax = plt.subplots(figsize=(9, 6))
        any_curve = False
        for orch in orchs:
            meta = runs.get((orch, shift))
            if meta is None:
                continue
            mat = read_train_task_matrix(meta['dir'])
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
                linewidth=2.5,
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
        ax.set_ylabel('Train task performance (mean across agents)')
        ax.set_title(f'Training task performance — shift strength {shift:g}')
        ax.legend(title='Orchestrator', frameon=True)
        sns.despine()
        out = out_dir / f'train_task_perf_shift_{shift:g}.png'
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f'  saved → {out}')
        plt.close(fig)


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
                batch_size = ds.get('batch_size') if isinstance(ds, dict) else None
                steps = summ.get('trainer/global_step')
                if batch_size is not None and steps is not None:
                    row['train_flops_est'] = (
                        3 * stats['fwd_flops_total'] * int(batch_size)
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
    args = parser.parse_args()

    project = None if args.project.lower() == 'none' else args.project
    study = None if args.study_name.lower() == 'none' else args.study_name
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
    if not runs:
        raise SystemExit(
            f'No completed runs found in {args.wandb_dir} '
            f'for project {project!r}.'
        )

    orchs = _ordered_orchs({o for o, _ in runs})
    colors, styles = _style_maps(orchs)
    shifts = sorted({shift for _, shift in runs})
    print(f'Found {len(runs)} runs — orchestrators: {orchs}')
    print(f'  shift strengths: {shifts}')

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style='whitegrid', context='paper', font_scale=1.4)

    # ── Plot 1 (with + without ComFed) ────────────────────────────────────────
    print('\nPlot 1: communication task performance vs shift …')
    comm_df = _comm_long_df(runs)
    plot_comm_vs_shift(
        comm_df, orchs, colors, args.out_dir,
        'comm_task_perf_vs_shift.png',
    )
    no_comfed = comm_df[comm_df['_orch'] != 'ComFed']
    if not no_comfed.empty:
        plot_comm_vs_shift(
            no_comfed, [o for o in orchs if o != 'ComFed'], colors,
            args.out_dir, 'comm_task_perf_vs_shift_no_comfed.png',
        )

    # ── Plot 2 (per-shift training curves) ────────────────────────────────────
    print('\nPlot 2: training task performance per shift …')
    plot_train_curves_per_shift(runs, orchs, colors, styles, args.out_dir)

    # ── Part 3 (per-shift tables) ─────────────────────────────────────────────
    print('\nPart 3: per-shift summary tables …')
    build_tables(runs, orchs, args.out_dir, with_params=not args.no_params)

    print('\nDone.')


if __name__ == '__main__':
    main()
