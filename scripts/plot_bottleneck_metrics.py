"""Visualize the ``sfrl_bottleneck`` sweep from local WandB run logs.

Reads the WandB logs under ``logs/wandb/run-*`` (no cloud access) for the
``sfrl_bottleneck`` project and plots communication task performance as a
function of the latent (bottleneck) dimension:

    x-axis : ``latent_dim``               (logged in each run's config)
    y-axis : ``test/avg_comm_task_perf``  (final test-time scalar)
    curves : one per orchestrator         (Federated / Non-cooperative / Sheaf-FRL)

A translucent band shows the spread (± std) of the per-agent
``test/comm_task_perf_agent_{i}`` around the average.

Runs are scoped by their wandb *project* (read from the RunRecord), so runs
from other projects that reuse the same study name are never mixed in.

Usage:
    python scripts/plot_bottleneck_metrics.py
    python scripts/plot_bottleneck_metrics.py \\
        --project sfrl_bottleneck --out_dir results/bottleneck/plots
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Reuse the WandB local-log helpers from the multi-agent plotting script.
from plot_multiagent_metrics import (  # noqa: E402
    _AGENT_COMM_RE,
    _load_raw_config,
    _per_agent_values,
    _unwrap,
    read_run_project,
)

# Orchestrator display order / labels for the bottleneck sweep.
ORCH_ORDER = ['NonCooperativeLearning', 'FederatedLearning', 'SheafFRL']
ORCH_LABELS = {
    'NonCooperativeLearning': 'Non-cooperative',
    'FederatedLearning': 'Federated',
    'SheafFRL': 'Sheaf-FRL',
}


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


def _ordered_orchs(orchs: set[str]) -> list[str]:
    known = [o for o in ORCH_ORDER if o in orchs]
    return known + sorted(o for o in orchs if o not in ORCH_ORDER)


def plot_comm_vs_latent(
    runs: dict[tuple[str, float], dict[str, Any]],
    orchs: list[str],
    out_dir: Path,
) -> pd.DataFrame:
    palette = sns.color_palette('tab10', n_colors=max(len(orchs), 3))
    colors = {o: palette[i] for i, o in enumerate(orchs)}
    markers = ['o', 's', '^', 'D', 'v', 'P']

    fig, ax = plt.subplots(figsize=(9, 6))
    table_rows = []
    for idx, orch in enumerate(orchs):
        pts = []
        for (o, latent), meta in runs.items():
            if o != orch:
                continue
            avg = float(meta['summary']['test/avg_comm_task_perf'])
            per_agent = _per_agent_values(meta['summary'], _AGENT_COMM_RE)
            std = float(np.std(per_agent)) if per_agent else 0.0
            pts.append((latent, avg, std))
            table_rows.append(
                {
                    'orchestrator': ORCH_LABELS.get(orch, orch),
                    'latent_dim': latent,
                    'avg_comm_task_perf': avg,
                    'comm_task_perf_std': std,
                }
            )
        if not pts:
            continue
        pts.sort()
        xs = np.array([p[0] for p in pts])
        ys = np.array([p[1] for p in pts])
        ax.plot(
            xs,
            ys,
            marker=markers[idx % len(markers)],
            markersize=9,
            linewidth=2.5,
            color=colors[orch],
            label=ORCH_LABELS.get(orch, orch),
        )

    latents = sorted({latent for _, latent in runs})
    ax.set_xticks(latents)
    ax.set_xticklabels([str(int(x)) for x in latents], rotation=45, ha='right')
    ax.set_xlabel('Latent (bottleneck) dimension')
    ax.set_ylabel('Avg communication task performance')
    ax.set_title('Communication task performance vs latent dimension')
    ax.legend(title='Orchestrator', frameon=True)
    sns.despine()

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / 'comm_task_perf_vs_latent_dim.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'  saved → {out}')
    plt.close(fig)
    return pd.DataFrame(table_rows).sort_values(['orchestrator', 'latent_dim'])


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
    args = parser.parse_args()

    project = None if args.project.lower() == 'none' else args.project
    exclude = set(args.exclude_latent)
    print(f'Scanning {args.wandb_dir} (project={project!r}) …')
    if exclude:
        print(f'  excluding latent dims: {sorted(int(x) for x in exclude)}')
    runs = discover_runs(args.wandb_dir, project, exclude_latent=exclude)
    if not runs:
        raise SystemExit(
            f'No completed runs found in {args.wandb_dir} '
            f'for project {project!r}.'
        )

    orchs = _ordered_orchs({o for o, _ in runs})
    latents = sorted({latent for _, latent in runs})
    print(f'Found {len(runs)} runs — orchestrators: {orchs}')
    print(f'  latent dims: {[int(x) for x in latents]}')

    df = plot_comm_vs_latent(runs, orchs, args.out_dir)
    df.to_csv(args.out_dir / 'comm_task_perf_vs_latent_dim.csv', index=False)
    print(f'  saved → {args.out_dir / "comm_task_perf_vs_latent_dim.csv"}')
    print()
    print(df.to_string(index=False))
    print('\nDone.')


if __name__ == '__main__':
    main()
