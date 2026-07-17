"""Visualize the ``sfrl_hetero_bottleneck`` sweep from WandB run logs.

Companion to ``plot_bottleneck_metrics.py``, but for the *heterogeneous*
2-agent bottleneck sweep (``config/hydra/hetero_rate_2agents_mnist_hetero_bottleneck.yaml``):
each agent has a differently-shaped encoder (different depth / hidden dims),
but both are hand-matched to share the same *last* encoder dim — the
bottleneck — across the pair (see ``config/hydra/model/hetero_bottleneck_*.yaml``).
The bottleneck dimension is read from agent 0's ``encoder_hidden_dims[-1]``
in each run's stored config (agent 1's last dim is the same by construction).

By default, fetches runs from the wandb cloud API (``--entity``/``--project``)
— local ``logs/wandb`` run directories are routinely cleaned up once a run has
synced, so the cloud is the reliable source of truth. Pass ``--local`` to
instead read straight from the on-disk logs under ``logs/wandb/run-*`` (no
cloud access) — this still auto-falls back to the cloud API if nothing local
is found.

Every orchestrator in ``EXCLUDED_ORCHS`` (from ``plot_multiagent_metrics``,
currently ``CESheafFRL`` and ``SheafCFRL``) is dropped from the run set
entirely before anything is plotted or tabulated — edit that set directly to
bring one back.

Either way it produces two metrics against the bottleneck dimension on the
x-axis, one curve per orchestrator:

    1. ``test/avg_comm_task_perf``    (communication task performance)
    2. ``test/avg_private_task_perf`` (private task performance)

These reuse the exact plotting machinery from ``plot_multiagent_metrics.py``
(:func:`plot_metric_vs_x` / :func:`plot_two_metrics_vs_x`), so the styling is
identical: by default (``adjusted=True``) each orchestrator's points are
dodged slightly off the shared bottleneck-dim value and the per-agent spread
(± std of ``test/comm_task_perf_agent_{i}`` / ``test/private_task_perf_agent_{i}``)
is drawn as small error-bar caps (bar-plot style) rather than a translucent
band. Pass ``--together`` to draw both metrics side by side in one figure
(same per-panel proportions as the standalone plots) with a single shared
legend, in three columns, sitting just above both panels, instead of two
separate figures. Either way, the mean/std summary is also written to CSV.

Runs are scoped by their wandb *project* (a run property, not a config
value), so runs from other projects that reuse the same study name are never
mixed in.

Usage:
    python scripts/plot_hetero_bottleneck_metrics.py   # remote, your default entity
    python scripts/plot_hetero_bottleneck_metrics.py \\
        --entity my-team \\
        --project sfrl_hetero_bottleneck --out_dir results/hetero_bottleneck/plots
    python scripts/plot_hetero_bottleneck_metrics.py --together

    # Read from logs/wandb instead (falls back to remote if empty).
    python scripts/plot_hetero_bottleneck_metrics.py --local
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import seaborn as sns

# Reuse the WandB log-parsing helpers, orchestrator labels/exclusions, and the
# shared metric-vs-x plotting machinery from the multi-agent plotting script —
# the hetero-bottleneck project mixes runs from the current sweep config
# (FedProto/FedMuscle/SheafFMTL) with earlier runs (e.g.
# NonCooperativeLearning/SheafFRL), so the full label set is reused rather
# than a config-specific subset, and colors/markers/plot styling stay
# consistent with every other plotting script here.
from plot_multiagent_metrics import (  # noqa: E402
    _AGENT_COMM_RE,
    _AGENT_PRIV_RE,
    EXCLUDED_ORCHS,
    ORCH_ORDER,
    _agent_metric_long_df,
    _load_raw_config,
    _remote_mtime,
    _style_maps,
    _unwrap,
    drop_excluded_orchs,
    metric_summary_df,
    plot_metric_vs_x,
    plot_two_metrics_vs_x,
    read_run_project,
)


def _bottleneck_dim(raw_config: dict[str, Any]) -> float | None:
    """Return the shared bottleneck dim: agent 0's ``encoder_hidden_dims[-1]``.

    Both agents in a ``hetero_bottleneck_*`` model config are hand-matched to
    share this value (see ``config/hydra/model/hetero_bottleneck_*.yaml``), so
    reading it off either agent is equivalent — agent 0 is picked
    arbitrarily.
    """
    model = _unwrap(raw_config, 'model')
    if not isinstance(model, dict):
        return None
    agents = model.get('agents')
    if not isinstance(agents, dict):
        return None
    for key in sorted(agents, key=str):
        agent = agents[key]
        if isinstance(agent, dict):
            enc = agent.get('encoder_hidden_dims')
            if isinstance(enc, (list, tuple)) and enc:
                return float(enc[-1])
    return None


def discover_runs(
    wandb_dir: Path,
    project: str | None,
) -> dict[tuple[str, float], dict[str, Any]]:
    """Return {(orch, bottleneck_dim): meta} keeping the latest completed run."""
    best: dict[tuple[str, float], dict[str, Any]] = {}
    for run_dir in sorted(wandb_dir.glob('run-*')):
        if project is not None and read_run_project(run_dir) != project:
            continue
        raw = _load_raw_config(run_dir)
        if raw is None:
            continue
        orch = _unwrap(raw, 'orchestrator')
        orch_name = (
            str(orch['_target_']).split('.')[-1]
            if isinstance(orch, dict) and '_target_' in orch
            else None
        )
        dim = _bottleneck_dim(raw)
        if orch_name is None or dim is None:
            continue

        summ_path = run_dir / 'files' / 'wandb-summary.json'
        if not summ_path.exists():
            continue
        with summ_path.open() as fh:
            summary = json.load(fh)
        if 'test/avg_comm_task_perf' not in summary:
            continue  # only completed runs

        key = (orch_name, dim)
        mtime = run_dir.stat().st_mtime
        if key not in best or mtime > best[key]['mtime']:
            best[key] = {
                'orch': orch_name,
                'bottleneck_dim': dim,
                'summary': summary,
                'mtime': mtime,
            }
    return best


def discover_runs_remote(
    api: Any,
    entity: str,
    project: str,
) -> dict[tuple[str, float], dict[str, Any]]:
    """Same contract as :func:`discover_runs`, scanning a wandb cloud project.

    Used as an automatic fallback when the local ``logs/wandb`` scan finds
    nothing (e.g. the local run directories were cleaned up after syncing).
    """
    from concurrent.futures import ThreadPoolExecutor

    remote_runs = list(
        api.runs(f'{entity}/{project}', filters={'state': 'finished'})
    )

    def _meta(run: Any) -> dict[str, Any] | None:
        # api.runs() hands back lazily-loaded runs: .config/.summary are
        # empty until a full attribute load is forced.
        run.load(force=True)
        raw = dict(run.config)
        orch = _unwrap(raw, 'orchestrator')
        orch_name = (
            str(orch['_target_']).split('.')[-1]
            if isinstance(orch, dict) and '_target_' in orch
            else None
        )
        dim = _bottleneck_dim(raw)
        if orch_name is None or dim is None:
            return None
        summary = dict(run.summary)
        if 'test/avg_comm_task_perf' not in summary:
            return None
        return {
            'orch': orch_name,
            'bottleneck_dim': dim,
            'summary': summary,
            'mtime': _remote_mtime(run, summary),
        }

    with ThreadPoolExecutor(max_workers=8) as pool:
        metas = list(pool.map(_meta, remote_runs))

    best: dict[tuple[str, float], dict[str, Any]] = {}
    for meta in metas:
        if meta is None:
            continue
        key = (meta['orch'], meta['bottleneck_dim'])
        if key not in best or meta['mtime'] > best[key]['mtime']:
            best[key] = meta
    return best


def _ordered_orchs(orchs: set[str]) -> list[str]:
    known = [o for o in ORCH_ORDER if o in orchs]
    return known + sorted(o for o in orchs if o not in ORCH_ORDER)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--wandb_dir', type=Path, default=Path('logs/wandb'))
    parser.add_argument(
        '--project', type=str, default='sfrl_hetero_bottleneck'
    )
    parser.add_argument(
        '--out_dir', type=Path, default=Path('results/hetero_bottleneck/plots')
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
        help='Draw comm + private accuracy vs bottleneck dim side by side in '
        'one figure (shared legend on top, 3 columns) instead of two '
        'separate figures.',
    )
    args = parser.parse_args()

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

    runs = {(o, dim): meta for (o, dim), meta in runs.items() if dim != 4}
    if not runs:
        raise SystemExit(
            f'No completed runs left for project {project!r} after '
            f'excluding bottleneck dim 4.'
        )

    runs = drop_excluded_orchs(runs)
    if not runs:
        raise SystemExit(
            f'No completed runs left for project {project!r} after '
            f'excluding {sorted(EXCLUDED_ORCHS)}.'
        )

    orchs = _ordered_orchs({o for o, _ in runs})
    colors, markers, _ = _style_maps(orchs)
    dims = sorted({dim for _, dim in runs})
    print(f'Found {len(runs)} runs — orchestrators: {orchs}')
    print(f'  bottleneck dims: {[int(x) for x in dims]}')

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style='whitegrid', context='paper', font_scale=1.4)

    x_kwargs = {
        'x_col': 'bottleneck_dim',
        'xlabel': 'Bottleneck dimension',
        'xscale': 'log',
        'xticklabel_fmt': lambda x: str(int(x)),
        'xticklabel_rotation': 45,
    }
    comm_df = _agent_metric_long_df(
        runs, _AGENT_COMM_RE, 'comm_task_perf', x_col='bottleneck_dim'
    )
    priv_df = _agent_metric_long_df(
        runs, _AGENT_PRIV_RE, 'private_task_perf', x_col='bottleneck_dim'
    )

    if args.together:
        print(
            '\nPlot: communication + private task performance vs '
            'bottleneck dim …'
        )
        plot_two_metrics_vs_x(
            comm_df,
            priv_df,
            orchs,
            colors,
            markers,
            args.out_dir,
            'comm_and_priv_task_perf_vs_bottleneck_dim.png',
            **x_kwargs,
        )
    else:
        print('\nPlot 1: communication task performance vs bottleneck dim …')
        plot_metric_vs_x(
            comm_df,
            orchs,
            colors,
            markers,
            args.out_dir,
            'comm_task_perf_vs_bottleneck_dim.png',
            'comm_task_perf',
            'Avg. communication accuracy',
            **x_kwargs,
        )
        print('\nPlot 2: private task performance vs bottleneck dim …')
        plot_metric_vs_x(
            priv_df,
            orchs,
            colors,
            markers,
            args.out_dir,
            'private_task_perf_vs_bottleneck_dim.png',
            'private_task_perf',
            'Avg. private accuracy',
            **x_kwargs,
        )

    comm_summary = metric_summary_df(
        comm_df, orchs, 'comm_task_perf', x_col='bottleneck_dim'
    )
    comm_summary.to_csv(
        args.out_dir / 'comm_task_perf_vs_bottleneck_dim.csv', index=False
    )
    print(f'  saved → {args.out_dir / "comm_task_perf_vs_bottleneck_dim.csv"}')
    print()
    print(comm_summary.to_string(index=False))

    priv_summary = metric_summary_df(
        priv_df, orchs, 'private_task_perf', x_col='bottleneck_dim'
    )
    priv_summary.to_csv(
        args.out_dir / 'private_task_perf_vs_bottleneck_dim.csv', index=False
    )
    print(
        f'  saved → {args.out_dir / "private_task_perf_vs_bottleneck_dim.csv"}'
    )
    print()
    print(priv_summary.to_string(index=False))

    print('\nDone.')


if __name__ == '__main__':
    main()
