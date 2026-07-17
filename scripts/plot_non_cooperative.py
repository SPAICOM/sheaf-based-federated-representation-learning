"""Plot non-cooperative evaluation results.

Reads all parquets from results/non_cooperative/ and produces four figures:
  1. Network avg accuracy vs shift_strength
  2. Homophilous vs heterophilous accuracy vs shift_strength (line + mean)
  3. Network avg task fidelity vs shift_strength
  4. Homophilous vs heterophilous task fidelity vs shift_strength (line + mean)

Task fidelity = cross_accuracy / self_accuracy (per source agent, per shift).

Usage:
    python scripts/plot_non_cooperative.py
    python scripts/plot_non_cooperative.py --results_dir path/to/non_cooperative/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ── Load data ─────────────────────────────────────────────────────────────────


def load_results(results_dir: Path) -> pd.DataFrame:
    files = sorted(results_dir.glob('*.parquet'))
    if not files:
        raise FileNotFoundError(f'No parquet files found in {results_dir}')
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    # Deduplicate: keep latest run per (source, target, shift, seed).
    df = (
        df.sort_values('shift_strength')
        .drop_duplicates(
            subset=['source_agent', 'target_agent', 'shift_strength', 'seed'],
            keep='last',
        )
        .reset_index(drop=True)
    )
    return df


def add_task_fidelity(df: pd.DataFrame) -> pd.DataFrame:
    self_acc = (
        df[df['is_self']]
        .set_index(['source_agent', 'shift_strength', 'seed'])['accuracy']
        .rename('self_accuracy')
    )
    cross = df[~df['is_self']].copy()
    cross = cross.join(self_acc, on=['source_agent', 'shift_strength', 'seed'])
    cross['task_fidelity'] = cross['accuracy'] / cross['self_accuracy'].clip(
        lower=1e-6
    )
    return cross


# ── Plotting helpers ──────────────────────────────────────────────────────────


def _save(fig: plt.Figure, path: Path, name: str) -> None:
    out = path / name
    fig.savefig(out, dpi=150, bbox_inches='tight')
    fig.savefig(out.with_suffix('.pdf'), bbox_inches='tight')
    print(f'  saved → {out}')
    plt.close(fig)


def plot_network_avg(
    cross: pd.DataFrame,
    y: str,
    ylabel: str,
    title: str,
    out_dir: Path,
    fname: str,
) -> None:
    agg = cross.groupby(['shift_strength', 'seed'])[y].mean().reset_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.lineplot(
        data=agg,
        x='shift_strength',
        y=y,
        ax=ax,
        marker='o',
        errorbar='sd',
    )
    ax.set_xlabel('Shift strength')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(-0.02, 1.02)
    sns.despine()
    _save(fig, out_dir, fname)


def plot_homo_hetero(
    cross: pd.DataFrame,
    y: str,
    ylabel: str,
    title: str,
    out_dir: Path,
    fname: str,
) -> None:
    cross = cross.copy()
    cross['pair_type'] = cross['homophilous'].map(
        {True: 'homophilous', False: 'heterophilous'}
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.lineplot(
        data=cross,
        x='shift_strength',
        y=y,
        hue='pair_type',
        style='pair_type',
        ax=ax,
        marker='o',
        errorbar='sd',
        estimator='mean',
    )
    ax.set_xlabel('Shift strength')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(-0.02, 1.02)
    ax.legend(title='Pair type')
    sns.despine()
    _save(fig, out_dir, fname)


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--results_dir',
        type=Path,
        default=Path('results/non_cooperative'),
    )
    args = parser.parse_args()

    results_dir: Path = args.results_dir
    out_dir = results_dir / 'plots'
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'Loading parquets from {results_dir} …')
    df = load_results(results_dir)
    cross = add_task_fidelity(df)

    sns.set_theme(style='whitegrid', context='paper', font_scale=1.2)

    print('Generating plots …')

    # 1. Network avg accuracy
    plot_network_avg(
        cross,
        y='accuracy',
        ylabel='Avg cross-accuracy',
        title='Network average cross-accuracy vs distribution shift',
        out_dir=out_dir,
        fname='net_avg_accuracy.png',
    )

    # 2. Homophilous vs heterophilous accuracy
    plot_homo_hetero(
        cross,
        y='accuracy',
        ylabel='Cross-accuracy',
        title='Homophilous vs heterophilous accuracy vs distribution shift',
        out_dir=out_dir,
        fname='homo_hetero_accuracy.png',
    )

    # 3. Network avg task fidelity
    plot_network_avg(
        cross,
        y='task_fidelity',
        ylabel='Avg task fidelity',
        title='Network average task fidelity vs distribution shift',
        out_dir=out_dir,
        fname='net_avg_task_fidelity.png',
    )

    # 4. Homophilous vs heterophilous task fidelity
    plot_homo_hetero(
        cross,
        y='task_fidelity',
        ylabel='Task fidelity',
        title='Homophilous vs heterophilous task fidelity vs distribution shift',
        out_dir=out_dir,
        fname='homo_hetero_task_fidelity.png',
    )

    print('Done.')


if __name__ == '__main__':
    main()
