"""Visualize the ``sfrl_bottleneck`` sweep from WandB run logs.

By default, fetches runs from the wandb cloud API (``--entity``/``--project``)
— local ``logs/wandb`` run directories are routinely cleaned up once a run has
synced, so the cloud is the reliable source of truth. Pass ``--local`` to
instead read straight from the on-disk logs under ``logs/wandb/run-*`` (no
cloud access) — this still auto-falls back to the cloud API if nothing local
is found. Either way it plots communication task performance as a function of
the latent (bottleneck) dimension:

    x-axis : ``latent_dim``               (logged in each run's config)
    y-axis : ``test/avg_comm_task_perf``  (final test-time scalar)
    curves : one per orchestrator

This reuses the exact plotting machinery from ``plot_multiagent_metrics.py``
(:func:`plot_metric_vs_x`), so the styling is identical: by default
(``adjusted=True``) each orchestrator's points are dodged slightly off the
shared latent-dim value and the spread (± std) of the per-agent
``test/comm_task_perf_agent_{i}`` around the average is drawn as small
error-bar caps (bar-plot style) rather than a translucent band. Every
orchestrator in ``EXCLUDED_ORCHS`` (from ``plot_multiagent_metrics``,
currently ``CESheafFRL`` and ``SheafCFRL``) is dropped from the run set
entirely before plotting — edit that set directly to bring one back.

Runs are scoped by their wandb *project*, so runs from other projects that
reuse the same study name are never mixed in.

Usage:
    python scripts/plot_bottleneck_metrics.py   # remote, your default entity
    python scripts/plot_bottleneck_metrics.py \\
        --entity my-team \\
        --project sfrl_bottleneck --out_dir results/bottleneck/plots

    # Read from logs/wandb instead (falls back to remote if empty).
    python scripts/plot_bottleneck_metrics.py --local
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import seaborn as sns

# Reuse the WandB log-parsing helpers, orchestrator labels/exclusions, and the
# shared metric-vs-x plotting machinery from the multi-agent plotting script —
# so the same orchestrator always looks the same, and every plot behaves
# identically, across every plotting script here.
from plot_multiagent_metrics import (  # noqa: E402
    _AGENT_COMM_RE,
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
    read_run_project,
)


def discover_runs(
    wandb_dir: Path,
    project: str | None,
    exclude_latent: set[float] | None = None,
) -> dict[tuple[str, float], dict[str, Any]]:
    """Return {(orch, latent_dim): meta} keeping the latest completed run.

    ``exclude_latent`` drops runs at those latent dimensions (non-destructive —
    the run logs are left untouched, they are just skipped here).
    """
    exclude_latent = exclude_latent or set()
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
        latent_dim = _unwrap(raw, 'latent_dim')
        if orch_name is None or latent_dim is None:
            continue
        if float(latent_dim) in exclude_latent:
            continue

        summ_path = run_dir / 'files' / 'wandb-summary.json'
        if not summ_path.exists():
            continue
        import json

        with summ_path.open() as fh:
            summary = json.load(fh)
        if 'test/avg_comm_task_perf' not in summary:
            continue  # only completed runs

        key = (orch_name, float(latent_dim))
        mtime = run_dir.stat().st_mtime
        if key not in best or mtime > best[key]['mtime']:
            best[key] = {
                'orch': orch_name,
                'latent_dim': float(latent_dim),
                'summary': summary,
                'mtime': mtime,
            }
    return best


def discover_runs_remote(
    api: Any,
    entity: str,
    project: str,
    exclude_latent: set[float] | None = None,
) -> dict[tuple[str, float], dict[str, Any]]:
    """Same contract as :func:`discover_runs`, scanning a wandb cloud project.

    Used as an automatic fallback when the local ``logs/wandb`` scan finds
    nothing (e.g. the local run directories were cleaned up after syncing).
    """
    from concurrent.futures import ThreadPoolExecutor

    exclude_latent = exclude_latent or set()
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
        latent_dim = _unwrap(raw, 'latent_dim')
        if orch_name is None or latent_dim is None:
            return None
        if float(latent_dim) in exclude_latent:
            return None
        summary = dict(run.summary)
        if 'test/avg_comm_task_perf' not in summary:
            return None
        return {
            'orch': orch_name,
            'latent_dim': float(latent_dim),
            'summary': summary,
            'mtime': _remote_mtime(run, summary),
        }

    with ThreadPoolExecutor(max_workers=8) as pool:
        metas = list(pool.map(_meta, remote_runs))

    best: dict[tuple[str, float], dict[str, Any]] = {}
    for meta in metas:
        if meta is None:
            continue
        key = (meta['orch'], meta['latent_dim'])
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
    parser.add_argument('--project', type=str, default='sfrl_bottleneck')
    parser.add_argument(
        '--out_dir', type=Path, default=Path('results/bottleneck/plots')
    )
    parser.add_argument(
        '--exclude_latent',
        type=float,
        nargs='*',
        default=[128.0],
        help='Latent dims to drop (default: 128). Pass nothing to keep all.',
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
    args = parser.parse_args()

    project = None if args.project.lower() == 'none' else args.project
    exclude = set(args.exclude_latent)
    if exclude:
        print(f'Excluding latent dims: {sorted(int(x) for x in exclude)}')

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
        return discover_runs_remote(api, entity, project, exclude_latent=exclude)

    if args.local:
        print(f'Scanning {args.wandb_dir} (project={project!r}) …')
        runs = discover_runs(args.wandb_dir, project, exclude_latent=exclude)
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
    latents = sorted({latent for _, latent in runs})
    print(f'Found {len(runs)} runs — orchestrators: {orchs}')
    print(f'  latent dims: {[int(x) for x in latents]}')

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style='whitegrid', context='paper', font_scale=1.4)

    comm_df = _agent_metric_long_df(
        runs, _AGENT_COMM_RE, 'comm_task_perf', x_col='latent_dim'
    )
    plot_metric_vs_x(
        comm_df,
        orchs,
        colors,
        markers,
        args.out_dir,
        'comm_task_perf_vs_latent_dim.png',
        'comm_task_perf',
        'Avg. communication accuracy',
        x_col='latent_dim',
        xlabel='Latent (bottleneck) dimension',
        xticklabel_fmt=lambda x: str(int(x)),
        xticklabel_rotation=45,
    )

    summary = metric_summary_df(
        comm_df, orchs, 'comm_task_perf', x_col='latent_dim'
    )
    summary.to_csv(
        args.out_dir / 'comm_task_perf_vs_latent_dim.csv', index=False
    )
    print(f'  saved → {args.out_dir / "comm_task_perf_vs_latent_dim.csv"}')
    print()
    print(summary.to_string(index=False))
    print('\nDone.')


if __name__ == '__main__':
    main()
