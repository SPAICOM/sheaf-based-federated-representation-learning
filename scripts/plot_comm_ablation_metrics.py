"""Visualize the ``comm_ablation`` sweep: test comm accuracy vs comm fraction.

Companion to ``plot_bottleneck_metrics.py`` / ``plot_anchor_selection_metrics.py``,
but for the *communication-fraction* ablation
(``config/hydra/multiagent_mnist_comm_ablation.yaml``): 15 heterogeneous MNIST
agents at a fixed shift, with the fraction of the 100 training epochs that
communicate swept by the ``comm_frac`` group (``f1`` … ``f90``, i.e. an exact
collaborative-epoch count out of 100), crossed with ``orchestrator.local_reg``
(false / true). Everything else is held constant, so the only thing varying is
how often agents communicate (and whether they regularize toward frozen
neighbours during the communication-free local epochs).

The curves are grouped by **``orchestrator.local_reg``** (the only non-comm
swept variable). It produces one figure, one curve per ``local_reg`` value, of

    ``test/avg_comm_task_perf``  (avg. communication accuracy)  vs comm fraction

The x-axis is the number of communication epochs, and since every run runs for
``total_epochs = 100``, the raw absolute epochs are reported (``10``, ``30``,
``60``, …) rather than a percentage. The per-agent spread around the average is
drawn as a translucent band (identical to the other scripts).

By default it additionally produces a single figure with **two y-axes** on the
same shared x-axis: communication accuracy on the left and cumulative training
communication volume (``train/communication_kilobytes``) on the right, both as
one curve per ``local_reg`` — pass ``--no-dual-axis`` to skip it. A companion
CSV (one row per (local_reg, comm fraction)) is always written.

By default, fetches runs from the wandb cloud API (``--entity``/``--project``)
— local ``logs/wandb`` run directories are routinely cleaned up once a run has
synced, so the cloud is the reliable source of truth. Pass ``--local`` to
instead read straight from the on-disk logs under ``logs/wandb/run-*`` (no
cloud access) — this still auto-falls back to the cloud API if nothing local
is found.

Runs are scoped by their wandb *project*, so runs from other projects that
reuse the same study name are never mixed in.

Usage:
    python scripts/plot_comm_ablation_metrics.py   # remote, your default entity
    python scripts/plot_comm_ablation_metrics.py \\
        --entity my-team --project comm_ablation \\
        --out_dir results/comm_ablation/plots

    # Read from logs/wandb instead (falls back to remote if empty).
    python scripts/plot_comm_ablation_metrics.py --local

    # Skip the twin-axis (accuracy + KB) figure.
    python scripts/plot_comm_ablation_metrics.py --no-dual-axis
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Reuse the WandB log-parsing / figure-saving helpers, the shared mplstyle and
# the canonincal orchestrator labels/order from the multi-agent plotting script,
# so this figure looks like every other one in this directory. Each *series*
# here is an ``orchestrator.local_reg`` value (false / true).
from plot_multiagent_metrics import (  # noqa: E402
    _AGENT_COMM_RE,
    _MPLSTYLE,
    _load_raw_config,
    _remote_mtime,
    _savefig,
    _unwrap,
    read_run_project,
)

# Summary metric keys (per-run scalars stored on the wandb run summary).
COMM_PERF_KEY = 'test/avg_comm_task_perf'
COMM_KB_KEY = 'train/communication_kilobytes'
COMM_MODEL_SIZE_KEY = 'model_size_kb'

# Total epoch horizon every comm_ablation run uses (see
# multiagent_mnist_comm_ablation.yaml: ``total_epochs: ${trainer.max_epochs}``
# with ``trainer.max_epochs: 100``). The comm fraction thus directly equals the
# number of communication epochs, which we report as-is on the x-axis.
TOTAL_EPOCHS = 100

# ── local_reg series display order / labels ───────────────────────────────────
# Two curves: regularization during the communication-free local epochs is off /
# on. Fixed order so a given value always sorts / styles the same.
LOCAL_REG_ORDER = [False, True]
LOCAL_REG_LABELS = {
    False: 'Local reg: off',
    True: 'Local reg: on',
}
_PALETTE = sns.color_palette('tab10', n_colors=10)
_MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', 'h', '<']


def _local_reg_label(value: bool) -> str:
    return LOCAL_REG_LABELS.get(value, str(value))


def _style_maps(values: list[bool]) -> tuple[dict, dict]:
    """Return per-``local_reg`` ``{color}``/``{marker}`` dicts.

    Assigned by each value's position in ``LOCAL_REG_ORDER``, so a given
    ``local_reg`` value always gets the same look.
    """
    canonical = LOCAL_REG_ORDER + [
        v for v in values if v not in LOCAL_REG_ORDER
    ]
    colors = {v: _PALETTE[canonical.index(v) % len(_PALETTE)] for v in values}
    markers = {v: _MARKERS[canonical.index(v) % len(_MARKERS)] for v in values}
    return colors, markers


# ── Config field extraction ───────────────────────────────────────────────────


def _orch_name(raw_config: dict[str, Any]) -> str | None:
    """Return the orchestrator class name from ``orchestrator._target_``."""
    orch = _unwrap(raw_config, 'orchestrator')
    if isinstance(orch, dict) and '_target_' in orch:
        return str(orch['_target_']).split('.')[-1]
    return None


def _orch_field(raw_config: dict[str, Any], field: str) -> Any:
    """Return ``orchestrator.<field>`` from a run's stored wandb config."""
    orch = _unwrap(raw_config, 'orchestrator')
    if isinstance(orch, dict):
        return orch.get(field)
    return None


def _build_meta(
    raw: dict[str, Any], summary: dict[str, Any], mtime: float
) -> dict[str, Any] | None:
    """Shared local/remote meta builder — returns ``None`` for unusable runs."""
    orch = _orch_name(raw)
    local_reg = _orch_field(raw, 'local_reg')
    comm_percentage = _orch_field(raw, 'comm_percentage')
    if orch is None or local_reg is None or comm_percentage is None:
        return None
    if COMM_PERF_KEY not in summary:
        return None  # "completed" = test phase finished
    comm_epochs = int(round(float(comm_percentage) / 100.0 * TOTAL_EPOCHS))
    return {
        'orch': orch,
        'local_reg': bool(local_reg),
        'comm_epochs': comm_epochs,
        'summary': summary,
        'mtime': mtime,
    }


# ── Discovery / dedup ─────────────────────────────────────────────────────────
# Keyed on (orchestrator, local_reg, comm_epochs): one chosen run per
# (local_reg, x) cell, keeping the most recent (by mtime).

RunKey = tuple[str, bool, int]


def _run_key(meta: dict[str, Any]) -> RunKey:
    return (meta['orch'], meta['local_reg'], meta['comm_epochs'])


def discover_runs(
    wandb_dir: Path, project: str | None
) -> dict[RunKey, dict[str, Any]]:
    """Return ``{(orch, local_reg, comm_epochs): meta}`` from local logs."""
    best: dict[RunKey, dict[str, Any]] = {}
    for run_dir in sorted(wandb_dir.glob('run-*')):
        if project is not None and read_run_project(run_dir) != project:
            continue
        raw = _load_raw_config(run_dir)
        if raw is None:
            continue
        summ_path = run_dir / 'files' / 'wandb-summary.json'
        if not summ_path.exists():
            continue
        with summ_path.open() as fh:
            summary = json.load(fh)
        meta = _build_meta(raw, summary, run_dir.stat().st_mtime)
        if meta is None:
            continue
        key = _run_key(meta)
        if key not in best or meta['mtime'] > best[key]['mtime']:
            best[key] = meta
    return best


def discover_runs_remote(
    api: Any, entity: str, project: str, max_workers: int = 8
) -> dict[RunKey, dict[str, Any]]:
    """Same contract as :func:`discover_runs`, scanning a wandb cloud project.

    Used as an automatic fallback when the local ``logs/wandb`` scan finds
    nothing (e.g. the local run directories were cleaned up after syncing).
    """
    from concurrent.futures import ThreadPoolExecutor

    remote_runs = list(
        api.runs(f'{entity}/{project}', filters={'state': 'finished'})
    )

    def _meta(run: Any) -> dict[str, Any] | None:
        # api.runs() hands back lazily-loaded runs: .config/.summary are empty
        # until a full attribute load is forced.
        run.load(force=True)
        summary = dict(run.summary)
        return _build_meta(
            dict(run.config), summary, _remote_mtime(run, summary)
        )

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        metas = list(pool.map(_meta, remote_runs))

    best: dict[RunKey, dict[str, Any]] = {}
    for meta in metas:
        if meta is None:
            continue
        key = _run_key(meta)
        if key not in best or meta['mtime'] > best[key]['mtime']:
            best[key] = meta
    return best


# ── Data shaping ──────────────────────────────────────────────────────────────


def _per_agent_values(summary: dict[str, Any], pattern) -> list[float]:
    """Per-agent values matching ``pattern`` from a run summary, sorted by idx."""
    indexed = [
        (int(m.group(1)), float(summary[k]))
        for k in summary
        if (m := pattern.match(k)) is not None
    ]
    indexed.sort(key=lambda t: t[0])
    return [v for _, v in indexed]


def _comm_df(runs: dict[RunKey, dict[str, Any]]) -> pd.DataFrame:
    """Long-format DataFrame: one row per (local_reg, comm_epochs, agent)."""
    rows = []
    for (_, local_reg, comm_epochs), meta in runs.items():
        rows.extend(
            {
                '_local_reg': bool(local_reg),
                'local_reg_label': _local_reg_label(local_reg),
                'comm_epochs': comm_epochs,
                'comm_task_perf': val,
            }
            for val in _per_agent_values(meta['summary'], _AGENT_COMM_RE)
        )
    return pd.DataFrame(rows)


def _kb_series(runs: dict[RunKey, dict[str, Any]]) -> pd.DataFrame:
    """Per-cell summary of cumulative training KB vs comm fraction."""
    rows = []
    for (_, local_reg, comm_epochs), meta in runs.items():
        kb = meta['summary'].get(COMM_KB_KEY)
        rows.append(
            {
                '_local_reg': bool(local_reg),
                'local_reg_label': _local_reg_label(local_reg),
                'comm_epochs': comm_epochs,
                'comm_kb': kb,
            }
        )
    return pd.DataFrame(rows)


# ── Plotting ──────────────────────────────────────────────────────────────────


def _ordered_local_regs(values: set[bool]) -> list[bool]:
    known = [v for v in LOCAL_REG_ORDER if v in values]
    return known + sorted(v for v in values if v not in LOCAL_REG_ORDER)


def _plot_comm_vs_fraction(
    comm_df: pd.DataFrame,
    local_regs: list[bool],
    colors: dict,
    markers: dict,
    out_dir: Path,
    fname: str,
    xlabel: str,
) -> None:
    """Comm accuracy vs comm fraction, one curve per local_reg (translucent band)."""
    with plt.style.context(str(_MPLSTYLE)):
        fig, ax = plt.subplots()
        xvals = sorted(comm_df['comm_epochs'].unique())
        for lr in local_regs:
            sub = comm_df[comm_df['_local_reg'] == lr]
            centers, lowers, uppers = _grouped_center_bounds(
                sub, 'comm_task_perf', 'comm_epochs', xvals
            )
            mask = ~np.isnan(centers)
            xs = np.asarray(xvals, dtype=float)[mask]
            m = centers[mask]
            ax.plot(
                xs,
                m,
                color=colors[lr],
                marker=markers[lr],
                linestyle='-',
                label=_local_reg_label(lr),
            )
            ax.fill_between(
                xs,
                lowers[mask],
                uppers[mask],
                color=colors[lr],
                alpha=0.12,
                linewidth=0,
            )
        ax.set_xticks(xvals)
        ax.set_xticklabels([str(int(x)) for x in xvals])
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Avg. communication accuracy')
        ax.legend(title='Local reg', frameon=True)
        sns.despine()
        out = out_dir / fname
        _savefig(fig, out)
        plt.close(fig)
    print(f'  saved → {out}')


def _plot_dual_axis(
    comm_df: pd.DataFrame,
    kb_df: pd.DataFrame,
    local_regs: list[bool],
    colors: dict,
    markers: dict,
    out_dir: Path,
    fname: str,
    xlabel: str,
) -> None:
    """Comm accuracy (left y-axis) + training KB (right y-axis) vs comm fraction.

    Both share the same x-axis (communication epochs) in a single figure, with
    twin y-axes so the two metrics' scales don't interfere. One curve per
    ``local_reg`` per y-axis.
    """
    with plt.style.context(str(_MPLSTYLE)):
        fig, ax = plt.subplots()
        ax2 = ax.twinx()
        xvals = sorted(comm_df['comm_epochs'].unique())
        for lr in local_regs:
            sub = comm_df[comm_df['_local_reg'] == lr]
            centers, lowers, uppers = _grouped_center_bounds(
                sub, 'comm_task_perf', 'comm_epochs', xvals
            )
            mask = ~np.isnan(centers)
            xs = np.asarray(xvals, dtype=float)[mask]
            m = centers[mask]
            ax.plot(
                xs,
                m,
                color=colors[lr],
                marker=markers[lr],
                linestyle='-',
                label=_local_reg_label(lr),
            )
            ax.fill_between(
                xs,
                lowers[mask],
                uppers[mask],
                color=colors[lr],
                alpha=0.12,
                linewidth=0,
            )
        for lr in local_regs:
            sub = kb_df[kb_df['_local_reg'] == lr]
            ksub = sub.dropna(subset=['comm_kb']).sort_values('comm_epochs')
            if ksub.empty:
                continue
            ax2.plot(
                ksub['comm_epochs'],
                ksub['comm_kb'],
                color=colors[lr],
                marker=markers[lr],
                linestyle='--',
                label=_local_reg_label(lr),
            )
        ax.set_xticks(xvals)
        ax.set_xticklabels([str(int(x)) for x in xvals])
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Avg. communication accuracy')
        ax2.set_ylabel('Training communication (KB)')
        ax.xaxis.set_ticks_position('bottom')
        ax2.xaxis.set_ticks_position('bottom')
        ax.yaxis.label.set_color('tab:blue')
        ax2.yaxis.label.set_color('tab:blue')
        ax.tick_params(axis='y', colors='tab:blue')
        ax2.tick_params(axis='y', colors='tab:blue')
        ax.legend(title='Local reg', frameon=True)
        sns.despine(ax=ax)
        sns.despine(ax=ax2)
        out = out_dir / fname
        _savefig(fig, out)
        plt.close(fig)
    print(f'  saved → {out}')


def _grouped_center_bounds(
    df: pd.DataFrame, value_col: str, x_col: str, xvals: list
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-x (mean, min, max) across agents for one local_reg, aligned to xvals."""
    grouped = df.groupby(x_col)[value_col]
    means = grouped.mean().reindex(xvals)
    lowers = grouped.min().reindex(xvals)
    uppers = grouped.max().reindex(xvals)
    return means.values, lowers.values, uppers.values


def _summary_csv(
    comm_df: pd.DataFrame,
    kb_df: pd.DataFrame,
    local_regs: list[bool],
    out_csv: Path,
) -> None:
    """Per-(local_reg, comm fraction) mean/std comm perf + training KB table."""
    rows = []
    xvals = sorted(comm_df['comm_epochs'].unique())
    for lr in local_regs:
        sub = comm_df[comm_df['_local_reg'] == lr]
        for x in xvals:
            vals = sub.loc[sub['comm_epochs'] == x, 'comm_task_perf']
            if vals.empty:
                continue
            kb = kb_df.loc[
                (kb_df['_local_reg'] == lr) & (kb_df['comm_epochs'] == x),
                'comm_kb',
            ]
            rows.append(
                {
                    'local_reg': _local_reg_label(lr),
                    'comm_epochs': x,
                    'comm_task_perf_mean': vals.mean(),
                    'comm_task_perf_std': vals.std(ddof=0),
                    'comm_kb': (
                        float(kb.iloc[0]) if not kb.dropna().empty else None
                    ),
                }
            )
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f'  saved → {out_csv}')


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--wandb_dir', type=Path, default=Path('logs/wandb'))
    parser.add_argument('--project', type=str, default='comm_ablation')
    parser.add_argument(
        '--out_dir', type=Path, default=Path('results/comm_ablation/plots')
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
        '--no-dual-axis',
        action='store_true',
        help='Skip the twin-axis figure (communication accuracy + training KB '
        'on one shared x-axis).',
    )
    args = parser.parse_args()

    project = None if args.project.lower() == 'none' else args.project

    def _fetch_remote() -> dict[RunKey, dict[str, Any]]:
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
        print(f'Scanning wandb cloud project {entity}/{project!r} …')
        return discover_runs_remote(api, entity, project)

    if args.local:
        print(f'Scanning {args.wandb_dir} (project={project!r}) …')
        runs = discover_runs(args.wandb_dir, project)
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

    local_regs = _ordered_local_regs({lr for _, lr, _ in runs})
    colors, markers = _style_maps(local_regs)
    comm_epochs = sorted({x for _, _, x in runs})
    print(
        f'Found {len(runs)} runs — local_reg: {local_regs}, '
        f'comm epochs: {comm_epochs}'
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style='whitegrid', context='paper', font_scale=1.4)

    comm_df = _comm_df(runs)
    xlabel = 'Communication epochs (of 100)'

    _plot_comm_vs_fraction(
        comm_df,
        local_regs,
        colors,
        markers,
        args.out_dir,
        'comm_vs_comm_fraction.png',
        xlabel,
    )
    if not args.no_dual_axis:
        kb_df = _kb_series(runs)
        _plot_dual_axis(
            comm_df,
            kb_df,
            local_regs,
            colors,
            markers,
            args.out_dir,
            'comm_and_kb_vs_comm_fraction.png',
            xlabel,
        )
    else:
        kb_df = _kb_series(runs)

    _summary_csv(
        comm_df,
        kb_df,
        local_regs,
        args.out_dir / 'comm_vs_comm_fraction.csv',
    )
    print('\nDone.')


if __name__ == '__main__':
    main()
