"""Visualize the ``anchor_selection_sfrl`` sweep: metrics vs number of anchors.

Companion to ``plot_multiagent_metrics.py`` / ``plot_hetero_bottleneck_metrics.py``
/ ``plot_network_density_metrics.py``, but for the *anchor-selection* sweep
(``config/hydra/multiagent_mnist_anchors.yaml``): 15 heterogeneous MNIST agents
at a fixed distribution shift, with ``orchestrator.num_anchors`` swept (8 … 512)
under each ``orchestrator.anchor_selection`` strategy (``all`` / ``random`` /
``proto_class`` / ``proto_kmeans``) — i.e. *how* the per-edge anchor budget is
chosen and *how big* it is, with everything else held constant.

The curves are grouped by **(orchestrator, alignment method, anchor-selection)
method** — e.g. "Sheaf-FRL + Procrustes + Random subset" — since the project
mixes more than one orchestrator, more than one ``orchestrator.alignment_method``
(``general`` / ``procrustes`` / ``relative``), and more than one anchor strategy.
It produces one figure, one curve per method, of

    ``test/avg_comm_task_perf``  (avg. communication accuracy)  vs num anchors

The y-value is a per-run scalar logged to the run summary, so there is no
per-agent spread band (this sweep is single-seed). ``anchor_selection=all``
ignores ``num_anchors`` (it aligns *every* matched pilot row regardless), so any
method using it is drawn as a horizontal reference line spanning the x-range
rather than a per-``num_anchors`` curve — controlled by ``--baseline`` (default
``all``). The companion CSV additionally carries ``train/communication_kilobytes``
per method for reference.

By default, fetches runs from the wandb cloud API (``--entity``/``--project``)
— local ``logs/wandb`` run directories are routinely cleaned up once a run has
synced, so the cloud is the reliable source of truth. Pass ``--local`` to
instead read straight from the on-disk logs under ``logs/wandb/run-*`` (no
cloud access) — this still auto-falls back to the cloud API if nothing local
is found. Either way, the (method, num_anchors) → metrics table is also
written to CSV.

Runs are scoped by their wandb *project* (a run property, not a config value),
so runs from other projects that reuse the same study name are never mixed in.

Usage:
    python scripts/plot_anchor_selection_metrics.py   # remote, your default entity
    python scripts/plot_anchor_selection_metrics.py \\
        --entity my-team --project anchor_selection_sfrl \\
        --out_dir results/anchor_selection/plots

    # Read from logs/wandb instead (falls back to remote if empty).
    python scripts/plot_anchor_selection_metrics.py --local
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
# the canonical orchestrator labels/order from the multi-agent plotting script,
# so this figure looks like every other one in this directory. Each *series*
# here is an (orchestrator, alignment_method, anchor_selection) triple.
from plot_multiagent_metrics import (  # noqa: E402
    _MPLSTYLE,
    ORCH_ORDER,
    ORCH_LABELS,
    _load_raw_config,
    _remote_mtime,
    _savefig,
    _unwrap,
    read_run_project,
)

# Summary metric keys (per-run scalars stored on the wandb run summary).
COMM_PERF_KEY = 'test/avg_comm_task_perf'
COMM_KB_KEY = 'train/communication_kilobytes'

# ── Alignment-method display order / labels ──────────────────────────────────
# ``VALID_ALIGNMENT_METHODS`` in src/communication/alignment_mixin.py; a run
# with the field unset is grouped under 'general' (the mixin's runtime default).
ALIGN_ORDER = ['general', 'procrustes', 'relative']
ALIGN_LABELS = {
    'general': 'General',
    'procrustes': 'Procrustes',
    'relative': 'Relative',
}

# ── Anchor-selection display order / labels ──────────────────────────────────
# Fixed order so a given strategy always sorts / styles the same regardless of
# which subset of strategies a project happens to contain.
ANCHOR_ORDER = ['all', 'random', 'proto_class', 'proto_kmeans']
ANCHOR_LABELS = {
    'all': 'All anchors',
    'random': 'Random subset',
    'proto_class': 'Class prototypes',
    'proto_kmeans': 'K-means prototypes',
}
_PALETTE = sns.color_palette('tab10', n_colors=10)
_MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', 'h', '<']

# A "combo" (series) is an ``(orchestrator, alignment_method, anchor_selection)`` triple.
Combo = tuple[str, str, str]


def _combo_label(combo: Combo) -> str:
    """Legend label: ``<Orchestrator> + <alignment> + <anchor strategy>``."""
    orch, align, sel = combo
    return (
        f'{ORCH_LABELS.get(orch, orch)} + '
        f'{ALIGN_LABELS.get(align, align)} + '
        f'{ANCHOR_LABELS.get(sel, sel)}'
    )


def _combo_sort_key(combo: Combo) -> tuple[int, str, int, str, int, str]:
    """Sort methods by orchestrator, then alignment method, then strategy."""
    orch, align, sel = combo
    oi = ORCH_ORDER.index(orch) if orch in ORCH_ORDER else len(ORCH_ORDER)
    ai = ALIGN_ORDER.index(align) if align in ALIGN_ORDER else len(ALIGN_ORDER)
    si = ANCHOR_ORDER.index(sel) if sel in ANCHOR_ORDER else len(ANCHOR_ORDER)
    return (oi, orch, ai, align, si, sel)


def _ordered_combos(combos: set[Combo]) -> list[Combo]:
    return sorted(combos, key=_combo_sort_key)


def _style_maps(combos: list[Combo]) -> tuple[dict, dict]:
    """Return per-method ``{color}``/``{marker}`` dicts.

    Assigned by each method's position in the canonically-ordered list, so a
    given (orchestrator, alignment, strategy) keeps a stable look for a fixed
    set of methods.
    """
    colors = {c: _PALETTE[i % len(_PALETTE)] for i, c in enumerate(combos)}
    markers = {c: _MARKERS[i % len(_MARKERS)] for i, c in enumerate(combos)}
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
    align = _orch_field(raw, 'alignment_method')
    selection = _orch_field(raw, 'anchor_selection')
    num_anchors = _orch_field(raw, 'num_anchors')
    if orch is None or selection is None or num_anchors is None:
        return None
    if COMM_PERF_KEY not in summary:
        return None  # "completed" = test phase finished
    return {
        'orch': orch,
        # Unset alignment_method → 'general' (the runtime default in the
        # PostTrainingAlignmentMixin), so those runs group cleanly.
        'alignment_method': str(align) if align is not None else 'general',
        'anchor_selection': str(selection),
        'num_anchors': int(num_anchors),
        'summary': summary,
        'mtime': mtime,
    }


# ── Discovery / dedup ─────────────────────────────────────────────────────────
# Keyed on (orchestrator, alignment_method, anchor_selection, num_anchors): one
# chosen run per (method, x) cell, keeping the most recent (by mtime).

RunKey = tuple[str, str, str, int]


def _run_key(meta: dict[str, Any]) -> RunKey:
    return (
        meta['orch'],
        meta['alignment_method'],
        meta['anchor_selection'],
        meta['num_anchors'],
    )


def discover_runs(
    wandb_dir: Path, project: str | None
) -> dict[RunKey, dict[str, Any]]:
    """Return ``{(orch, align, anchor_selection, num_anchors): meta}`` from local logs."""
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
        return _build_meta(dict(run.config), summary, _remote_mtime(run, summary))

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


def _series_xy(
    runs: dict[RunKey, dict[str, Any]], combo: Combo, value_key: str
) -> tuple[np.ndarray, np.ndarray]:
    """Sorted ``(num_anchors, value)`` arrays for one (orch, align, strategy) method."""
    orch, align, sel = combo
    pts = []
    for (o, a, s, na), meta in runs.items():
        if o != orch or a != align or s != sel:
            continue
        val = meta['summary'].get(value_key)
        if isinstance(val, (int, float)):
            pts.append((na, float(val)))
    pts.sort()
    xs = np.array([p[0] for p in pts], dtype=float)
    ys = np.array([p[1] for p in pts], dtype=float)
    return xs, ys


def summary_df(
    runs: dict[RunKey, dict[str, Any]], combos: list[Combo]
) -> pd.DataFrame:
    """One row per (method, num_anchors) with both plotted metrics."""
    rows = []
    for combo in combos:
        orch, align, sel = combo
        for (o, a, s, na), meta in sorted(runs.items()):
            if o != orch or a != align or s != sel:
                continue
            rows.append(
                {
                    'orchestrator': ORCH_LABELS.get(orch, orch),
                    'alignment_method': ALIGN_LABELS.get(align, align),
                    'anchor_selection': ANCHOR_LABELS.get(sel, sel),
                    'num_anchors': na,
                    'avg_comm_task_perf': meta['summary'].get(COMM_PERF_KEY),
                    'train_communication_kb': meta['summary'].get(COMM_KB_KEY),
                }
            )
    return pd.DataFrame(rows)


# ── Plotting ──────────────────────────────────────────────────────────────────


def _plot_metric_on_ax(
    ax: plt.Axes,
    runs: dict[RunKey, dict[str, Any]],
    combos: list[Combo],
    colors: dict,
    markers: dict,
    value_key: str,
    ylabel: str,
    xvals: list[float],
    baseline: set[str],
) -> None:
    """Draw one metric-vs-num_anchors panel, one curve per method.

    A method whose anchor strategy is in ``baseline`` (e.g. ``all``, which
    ignores ``num_anchors``) is drawn as a horizontal reference line spanning
    the x-range instead of a per-``num_anchors`` curve.
    """
    for combo in combos:
        xs, ys = _series_xy(runs, combo, value_key)
        if len(xs) == 0:
            continue
        _orch, _align, sel = combo
        label = _combo_label(combo)
        if sel in baseline:
            ax.axhline(
                float(np.mean(ys)),
                color=colors[combo],
                linestyle='--',
                linewidth=1.5,
                label=label,
            )
        else:
            ax.plot(
                xs,
                ys,
                color=colors[combo],
                marker=markers[combo],
                linestyle='-',
                label=label,
            )
    if xvals:
        ax.set_xscale('log', base=2)
        ax.set_xticks(xvals)
        ax.set_xticklabels([str(int(x)) for x in xvals])
        ax.minorticks_off()
    ax.set_xlabel('Number of anchors')
    ax.set_ylabel(ylabel)


def plot_anchor_metrics(
    runs: dict[RunKey, dict[str, Any]],
    combos: list[Combo],
    colors: dict,
    markers: dict,
    out_dir: Path,
    fname: str,
    baseline: set[str],
) -> None:
    """Avg. communication accuracy vs num anchors, one curve per method."""
    # The x-axis is defined by the *swept* (non-baseline) methods; a baseline
    # like ``all`` carries an arbitrary num_anchors that shouldn't stretch it.
    xvals = sorted({na for (_o, _a, sel, na) in runs if sel not in baseline})
    if not xvals:
        xvals = sorted({na for (*_, na) in runs})

    with plt.style.context(str(_MPLSTYLE)):
        fig, ax = plt.subplots()
        _plot_metric_on_ax(
            ax,
            runs,
            combos,
            colors,
            markers,
            COMM_PERF_KEY,
            'Avg. communication accuracy',
            xvals,
            baseline,
        )
        # One legend. With the longer combined labels it's drawn above the
        # panel, spanning its width; two columns unless there are very few
        # methods.
        handles, labels = ax.get_legend_handles_labels()
        top = ax.get_position().y1
        fig.legend(
            handles,
            labels,
            title='Method',
            loc='lower center',
            bbox_to_anchor=(0.5, top + 0.03),
            ncol=1 if len(labels) <= 3 else 2,
            frameon=True,
            borderaxespad=0.0,
        )
        sns.despine(fig=fig)
        out = out_dir / fname
        _savefig(fig, out)
        plt.close(fig)
    print(f'  saved → {out}')


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--wandb_dir', type=Path, default=Path('logs/wandb'))
    parser.add_argument('--project', type=str, default='anchor_selection_sfrl')
    parser.add_argument(
        '--out_dir', type=Path, default=Path('results/anchor_selection/plots')
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
        '--baseline',
        nargs='*',
        default=['all'],
        help='Anchor-selection strategies drawn as a horizontal reference '
        'line (num_anchors-independent) instead of a curve. Default: all. '
        'Pass with no values to disable.',
    )
    args = parser.parse_args()
    baseline = set(args.baseline)

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

    combos = _ordered_combos({(o, a, s) for (o, a, s, _na) in runs})
    colors, markers = _style_maps(combos)
    nas = sorted({na for (*_, na) in runs})
    print(f'Found {len(runs)} runs — methods:')
    for combo in combos:
        print(f'  {_combo_label(combo)}')
    print(f'  num_anchors: {nas}')

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style='whitegrid', context='paper', font_scale=1.4)

    print('\nPlot: avg. communication accuracy vs num anchors …')
    plot_anchor_metrics(
        runs,
        combos,
        colors,
        markers,
        args.out_dir,
        'comm_perf_vs_num_anchors.png',
        baseline,
    )

    table = summary_df(runs, combos)
    out = args.out_dir / 'anchor_selection_metrics.csv'
    table.to_csv(out, index=False)
    print(f'  saved → {out}')
    print()
    print(table.to_string(index=False))

    print('\nDone.')


if __name__ == '__main__':
    main()
