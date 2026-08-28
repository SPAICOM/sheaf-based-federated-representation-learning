"""Visualize the ``netowrk_analysis_sfrl`` sweep: accuracy vs graph density.

Companion to ``plot_multiagent_metrics.py``, but for the *network-analysis*
sweep (``config/hydra/multiagent_mnist_network_analyisi.yaml``): 15
heterogeneous MNIST agents at a fixed distribution shift, with
``graph.max_edge_frac`` swept (0.2 … 0.9) so the communication graph gets
denser while everything else is held constant.

X-axis: the **actual edge density** of the communication graph,
``|E| / (n(n-1)/2)`` — *not* ``graph.max_edge_frac`` itself, which is only the
edge *budget* handed to ``class_overlap_neighbors`` and differs from the
realized density in two ways (see ``src/utils/graph_generator.py``):

  * the budget is floored — ``int(max_edge_frac * n(n-1)/2)`` — so e.g. with
    ``n = 15`` (105 possible edges) ``max_edge_frac = 0.3`` buys 31 edges, a
    density of 0.295, not 0.300;
  * connectivity wins over sparsity — the budget is raised to at least
    ``n - 1`` edges so the maximum-weight spanning tree always fits, which
    matters for small ``max_edge_frac`` on large graphs.

The density is therefore recomputed per run by rebuilding that run's graph
from its own stored config (:func:`plot_multiagent_metrics.reconstruct_neighbors`,
the same reconstruction behind the degree-weighted comm estimator). Pass
``--x-source max_edge_frac`` to plot the raw config knob instead, and
``--x-units fraction`` to label the axis in [0, 1] instead of percent.

Y-axis: the same two metrics as every other plotting script here, one curve
per orchestrator:

    1. ``test/avg_comm_task_perf``    (communication task performance)
    2. ``test/avg_private_task_perf`` (private task performance)

Both reuse the plotting machinery of ``plot_multiagent_metrics.py``
(:func:`plot_metric_vs_x` / :func:`plot_two_metrics_vs_x`), so colors,
markers and legend are identical to the shift-strength and bottleneck
figures: the comm curve is the per-agent **degree-weighted mean**
(``--comm-estimator``, weighting each agent by its share of total graph
degree — which itself changes as the graph densifies) and the private curve
the plain **mean** (``--priv-estimator``). The per-agent spread is hidden by
default (``--errorbar-pi 0``); pass e.g. ``--errorbar-pi 100`` for a
translucent band spanning the full min-max range, or ``--errorbar-pi 50``
for the interquartile range. Pass ``--together`` to draw both panels in one
figure.

Alongside the figures it writes, per metric, the mean/std summary CSV, plus
``graph_density.csv`` — the ``max_edge_frac`` → realized ``density`` mapping
(edge counts and mean degree included), which is also printed.

Every orchestrator in ``EXCLUDED_ORCHS`` (from ``plot_multiagent_metrics``) is
dropped from the run set before anything is plotted.

Runs are scoped by their wandb *project* (a run property, not a config value),
so runs from other projects that reuse the same study name are never mixed in.

Usage:
    python scripts/plot_network_density_metrics.py   # remote, your default entity
    python scripts/plot_network_density_metrics.py --together
    python scripts/plot_network_density_metrics.py \\
        --entity my-team --project netowrk_analysis_sfrl \\
        --out_dir results/network_analysis/plots
    python scripts/plot_network_density_metrics.py --x-source max_edge_frac

    # Read from logs/wandb instead (falls back to remote if empty).
    python scripts/plot_network_density_metrics.py --local
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import seaborn as sns

# Reuse the WandB log-parsing helpers, orchestrator labels/order/exclusions and
# the shared metric-vs-x plotting machinery from the multi-agent plotting
# script, so this figure looks like every other one in this directory.
from plot_multiagent_metrics import (  # noqa: E402
    _AGENT_COMM_RE,
    _AGENT_PRIV_RE,
    EXCLUDED_ORCHS,
    ORCH_ORDER,
    _agent_metric_long_df,
    _dedup_runs,
    _load_raw_config,
    _remote_mtime,
    _style_maps,
    _unwrap,
    drop_excluded_orchs,
    metric_summary_df,
    plot_metric_vs_x,
    plot_two_metrics_vs_x,
    read_run_project,
    reconstruct_neighbors,
)

X_COL = 'density'


def graph_stats(raw_config: dict[str, Any]) -> dict[str, float] | None:
    """Return the realized graph size/density for a run's stored config.

    ``{n_agents, n_edges, max_edges, density, mean_degree}`` where ``density``
    is ``|E| / (n(n-1)/2)`` of the graph actually built for that run — the
    quantity ``graph.max_edge_frac`` only approximates (it is a floored edge
    budget, raised to ``n - 1`` when connectivity demands it). ``None`` when
    the graph can't be reconstructed.
    """
    neighbors = reconstruct_neighbors(raw_config)
    if neighbors is None:
        return None
    n = len(neighbors)
    if n < 2:
        return None
    degrees = sum(len(nb) for nb in neighbors.values())
    n_edges = degrees / 2
    max_edges = n * (n - 1) / 2
    return {
        'n_agents': float(n),
        'n_edges': n_edges,
        'max_edges': max_edges,
        'density': n_edges / max_edges,
        'mean_degree': degrees / n,
    }


def _max_edge_frac(raw_config: dict[str, Any]) -> float | None:
    """Return ``graph.max_edge_frac`` from a run's stored config."""
    graph = _unwrap(raw_config, 'graph')
    if not isinstance(graph, dict):
        return None
    frac = graph.get('max_edge_frac')
    return float(frac) if isinstance(frac, (int, float)) else None


def _shift_strength(raw_config: dict[str, Any]) -> float | None:
    """Return ``dataset.shift_strength`` from a run's stored config."""
    dataset = _unwrap(raw_config, 'dataset')
    if not isinstance(dataset, dict):
        return None
    shift = dataset.get('shift_strength')
    return float(shift) if isinstance(shift, (int, float)) else None


def _build_meta(
    raw: dict[str, Any],
    summary: dict[str, Any],
    mtime: float,
    x_source: str,
    shift_strength: float | None,
) -> dict[str, Any] | None:
    """Shared local/remote meta builder — returns ``None`` for unusable runs.

    ``raw_config`` is kept on the meta because the degree-weighted comm
    estimator rebuilds the graph from it (see
    :func:`plot_multiagent_metrics._agent_metric_long_df`); here the graph
    varies *with* the x-axis, so those weights genuinely change per point.
    """
    orch = _unwrap(raw, 'orchestrator')
    orch_name = (
        str(orch['_target_']).split('.')[-1]
        if isinstance(orch, dict) and '_target_' in orch
        else None
    )
    if orch_name is None:
        return None
    if 'test/avg_comm_task_perf' not in summary:
        return None  # "completed" = test phase finished
    if shift_strength is not None:
        shift = _shift_strength(raw)
        if shift is None or not math.isclose(
            shift, shift_strength, abs_tol=1e-6
        ):
            return None

    stats = graph_stats(raw)
    frac = _max_edge_frac(raw)
    x = frac if x_source == 'max_edge_frac' else (stats or {}).get('density')
    if x is None:
        return None

    return {
        'orch': orch_name,
        X_COL: float(x),
        'max_edge_frac': frac,
        'graph_stats': stats,
        'shift': _shift_strength(raw),
        'raw_config': raw,
        'summary': summary,
        'mtime': mtime,
    }


def discover_runs(
    wandb_dir: Path,
    project: str | None,
    x_source: str = 'density',
    shift_strength: float | None = None,
    select: str = 'latest',
    select_metric: str = 'test/avg_comm_task_perf',
) -> dict[tuple[str, float], dict[str, Any]]:
    """Return ``{(orch, x): meta}``, one chosen run per cell, from local logs.

    Same dedup contract as :func:`plot_multiagent_metrics.discover_runs`
    (``select`` = ``latest`` by mtime, or ``best`` by ``select_metric``), only
    keyed on graph density instead of shift strength.
    """
    candidates: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(
        list
    )
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
        meta = _build_meta(
            raw,
            summary,
            run_dir.stat().st_mtime,
            x_source,
            shift_strength,
        )
        if meta is None:
            continue
        meta['dir'] = run_dir
        candidates[(meta['orch'], meta[X_COL])].append(meta)

    return _dedup_runs(
        candidates,
        select,
        select_metric,
        label=lambda m: m['dir'].name,
        x_label=x_source,
    )


def discover_runs_remote(
    api: Any,
    entity: str,
    project: str,
    x_source: str = 'density',
    shift_strength: float | None = None,
    select: str = 'latest',
    select_metric: str = 'test/avg_comm_task_perf',
    max_workers: int = 8,
) -> dict[tuple[str, float], dict[str, Any]]:
    """Same contract as :func:`discover_runs`, scanning a wandb cloud project."""
    from concurrent.futures import ThreadPoolExecutor

    remote_runs = list(
        api.runs(f'{entity}/{project}', filters={'state': 'finished'})
    )

    def _meta(run: Any) -> dict[str, Any] | None:
        # api.runs() hands back lazily-loaded runs: .config/.summary are empty
        # until a full attribute load is forced.
        run.load(force=True)
        summary = dict(run.summary)
        meta = _build_meta(
            dict(run.config),
            summary,
            _remote_mtime(run, summary),
            x_source,
            shift_strength,
        )
        if meta is not None:
            meta['run'] = run
        return meta

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        metas = list(pool.map(_meta, remote_runs))

    candidates: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(
        list
    )
    for meta in metas:
        if meta is not None:
            candidates[(meta['orch'], meta[X_COL])].append(meta)

    return _dedup_runs(
        candidates,
        select,
        select_metric,
        label=lambda m: m['run'].id,
        x_label=x_source,
    )


def _ordered_orchs(orchs: set[str]) -> list[str]:
    known = [o for o in ORCH_ORDER if o in orchs]
    return known + sorted(o for o in orchs if o not in ORCH_ORDER)


def build_density_table(
    runs: dict[tuple[str, float], dict[str, Any]], out_dir: Path
) -> pd.DataFrame:
    """Save + return the ``max_edge_frac`` → realized-density mapping.

    One row per distinct communication graph in the sweep. This is the table
    that shows *how far* the config knob sits from the density actually
    realized (floored edge budget, connectivity floor) — the reason the x-axis
    is recomputed rather than read off ``graph.max_edge_frac``.
    """
    rows = {}
    for meta in runs.values():
        stats = meta.get('graph_stats')
        if stats is None:
            continue
        rows[(meta.get('max_edge_frac'), stats['density'])] = {
            'max_edge_frac': meta.get('max_edge_frac'),
            'n_agents': int(stats['n_agents']),
            'n_edges': int(stats['n_edges']),
            'max_edges': int(stats['max_edges']),
            'density': stats['density'],
            'mean_degree': stats['mean_degree'],
        }
    df = pd.DataFrame(rows.values())
    if df.empty:
        return df
    df = df.sort_values('density').reset_index(drop=True)
    out = out_dir / 'graph_density.csv'
    df.to_csv(out, index=False)
    print(f'  saved → {out}')
    print()
    print(df.to_string(index=False))
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--wandb_dir', type=Path, default=Path('logs/wandb'))
    parser.add_argument(
        '--project', type=str, default='netowrk_analysis_sfrl'
    )
    parser.add_argument(
        '--out_dir', type=Path, default=Path('results/network_analysis/plots')
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
        help='Draw comm + private accuracy vs graph density side by side in '
        'one figure (shared legend) instead of two separate figures.',
    )
    parser.add_argument(
        '--x-source',
        choices=['density', 'max_edge_frac'],
        default='density',
        help='What goes on the x-axis: the realized edge density of each '
        "run's reconstructed communication graph (default), or the raw "
        'graph.max_edge_frac config knob (only an upper edge budget — see '
        'the module docstring).',
    )
    parser.add_argument(
        '--x-units',
        choices=['percent', 'fraction'],
        default='percent',
        help='Label the x-axis as a percentage of the maximum possible edges '
        '(default) or as a fraction in [0, 1].',
    )
    parser.add_argument(
        '--shift_strength',
        type=float,
        default=None,
        help='Only keep runs whose dataset.shift_strength matches this value. '
        'Off by default (the sweep varies only the graph); pass it if the '
        'project ever mixes shifts, so incomparable runs cannot share an '
        '(orchestrator, density) cell.',
    )
    parser.add_argument(
        '--select',
        choices=['latest', 'best'],
        default='latest',
        help='When several runs share an (orchestrator, density): keep the '
        'most recent (latest, default) or the best-scoring (best).',
    )
    parser.add_argument(
        '--select-metric',
        type=str,
        default='test/avg_comm_task_perf',
        help='Summary metric maximised when --select best (default: the '
        'plotted comm task perf).',
    )
    parser.add_argument(
        '--comm-estimator',
        choices=['mean', 'median', 'weighted_mean'],
        default='weighted_mean',
        help='Point/line statistic drawn across agents for the comm task '
        'perf curve (default: weighted_mean — weights each agent by its '
        'share of total communication-graph degree).',
    )
    parser.add_argument(
        '--priv-estimator',
        choices=['mean', 'median', 'weighted_mean'],
        default='mean',
        help='Point/line statistic drawn across agents for the private task '
        'perf curve (default: mean, unweighted).',
    )
    parser.add_argument(
        '--errorbar-pi',
        type=float,
        default=0.0,
        help='Percentile-interval width (0-100) drawn around each line. 0 '
        '(default) hides the interval entirely; e.g. 100 spans the full '
        'min-max range across agents, 50 the interquartile range.',
    )
    args = parser.parse_args()
    errorbar = ('pi', args.errorbar_pi) if args.errorbar_pi > 0 else None

    project = None if args.project.lower() == 'none' else args.project

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
        print(f'Scanning wandb cloud project {entity}/{project!r} …')
        return discover_runs_remote(
            api,
            entity,
            project,
            x_source=args.x_source,
            shift_strength=args.shift_strength,
            select=args.select,
            select_metric=args.select_metric,
        )

    if args.local:
        print(f'Scanning {args.wandb_dir} (project={project!r}) …')
        runs = discover_runs(
            args.wandb_dir,
            project,
            x_source=args.x_source,
            shift_strength=args.shift_strength,
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
    colors, markers, _ = _style_maps(orchs)
    xvals = sorted({x for _, x in runs})
    shifts = sorted({m['shift'] for m in runs.values() if m['shift'] is not None})
    print(f'Found {len(runs)} runs — orchestrators: {orchs}')
    print(f'  {args.x_source}: {[round(x, 4) for x in xvals]}')
    print(f'  shift strengths: {[round(s, 4) for s in shifts]}')
    if len(shifts) > 1 and args.shift_strength is None:
        print(
            '  [warn] runs span several shift strengths — pass '
            '--shift_strength to keep only one (curves otherwise mix '
            'incomparable runs).'
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style='whitegrid', context='paper', font_scale=1.4)

    # Percent is only a relabelling of the same [0, 1] quantity, so scale the
    # values themselves and keep the tick formatter trivial.
    scale = 100.0 if args.x_units == 'percent' else 1.0
    if args.x_source == 'density':
        base_label = 'Communication graph density'
    else:
        base_label = 'Max edge fraction'
    # ``%`` has to be escaped: the shared mplstyle renders text through LaTeX,
    # where a bare % starts a comment and would swallow the rest of the label.
    xlabel = rf'{base_label} (\%)' if args.x_units == 'percent' else base_label
    ndigits = 1 if args.x_units == 'percent' else 3

    scaled = {(o, x * scale): meta for (o, x), meta in runs.items()}
    x_kwargs = {
        'x_col': X_COL,
        'xlabel': xlabel,
        # Densities land on non-round values (a floored edge budget, e.g. 31 of
        # 105 edges = 29.5%), so ticks are labelled from the real values with
        # trailing zeros trimmed rather than snapped to the config knob.
        'xticklabel_fmt': lambda x: f'{round(x, ndigits):g}',
        'adjusted': False,
        'show_error': errorbar is not None,
        'errorbar': errorbar,
    }
    comm_df = _agent_metric_long_df(
        scaled, _AGENT_COMM_RE, 'comm_task_perf', x_col=X_COL
    )
    priv_df = _agent_metric_long_df(
        scaled, _AGENT_PRIV_RE, 'private_task_perf', x_col=X_COL
    )

    if args.together:
        print('\nPlot: communication + private task performance vs density …')
        plot_two_metrics_vs_x(
            comm_df,
            priv_df,
            orchs,
            colors,
            markers,
            args.out_dir,
            f'comm_and_priv_task_perf_vs_{args.x_source}.png',
            legend_loc='right',
            legend_anchor='lower left',
            comm_estimator=args.comm_estimator,
            priv_estimator=args.priv_estimator,
            **x_kwargs,
        )
    else:
        print('\nPlot 1: communication task performance vs density …')
        plot_metric_vs_x(
            comm_df,
            orchs,
            colors,
            markers,
            args.out_dir,
            f'comm_task_perf_vs_{args.x_source}.png',
            'comm_task_perf',
            'Avg. communication accuracy',
            estimator=args.comm_estimator,
            **x_kwargs,
        )
        print('\nPlot 2: private task performance vs density …')
        plot_metric_vs_x(
            priv_df,
            orchs,
            colors,
            markers,
            args.out_dir,
            f'private_task_perf_vs_{args.x_source}.png',
            'private_task_perf',
            'Avg. private accuracy',
            estimator=args.priv_estimator,
            **x_kwargs,
        )

    for df, value_col in (
        (comm_df, 'comm_task_perf'),
        (priv_df, 'private_task_perf'),
    ):
        summary = metric_summary_df(df, orchs, value_col, x_col=X_COL)
        out = args.out_dir / f'{value_col}_vs_{args.x_source}.csv'
        summary.to_csv(out, index=False)
        print(f'\n  saved → {out}')
        print()
        print(summary.to_string(index=False))

    print('\nGraph density realized per max_edge_frac …')
    build_density_table(runs, args.out_dir)

    print('\nDone.')


if __name__ == '__main__':
    main()
