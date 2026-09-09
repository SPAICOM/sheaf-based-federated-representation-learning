"""Plot masked-reconstruction MSE metrics against mask overlap/sharedness.

Reads the parquet summaries written by ``scripts/reconstruction_experiment.py``
and produces two curves, one for private visible-region MSE and one for
cross-agent communication MSE.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ORCH_ORDER = ['NonCooperativeLearning', 'SheafFRL']
ORCH_LABELS = {
    'NonCooperativeLearning': 'Non-cooperative',
    'SheafFRL': 'Sheaf-FRL',
}
_PALETTE = sns.color_palette('tab10', n_colors=10)
ORCH_COLORS = {
    'NonCooperativeLearning': _PALETTE[0],
    'SheafFRL': _PALETTE[3],
}
ORCH_MARKERS = {
    'NonCooperativeLearning': 'o',
    'SheafFRL': 'D',
}
ORCH_LINESTYLES = {
    'NonCooperativeLearning': '-',
    'SheafFRL': '--',
}

_MPLSTYLE = (
    Path(__file__).resolve().parent.parent
    / 'config'
    / 'plotting'
    / 'plt.mplstyle'
)


def _savefig(fig: plt.Figure, out_path: Path) -> Path:
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    fig.savefig(out_path.with_suffix('.pdf'), bbox_inches='tight')
    return out_path


def _load_results(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for path in sorted(args.results_dir.glob('reconstruction__*.parquet')):
        try:
            df = pd.read_parquet(path)
        except Exception as exc:
            print(f'skip unreadable parquet {path}: {exc}')
            continue
        df = df.copy()
        df['source_path'] = str(path)
        df['source_mtime'] = path.stat().st_mtime
        rows.append(df)

    if not rows:
        raise SystemExit(f'No reconstruction parquet files in {args.results_dir}')

    data = pd.concat(rows, ignore_index=True)
    data = data[data['orchestrator'].isin(ORCH_ORDER)]
    data = data[data['mask_mode'] == args.mask_mode]
    if args.seed is not None and 'seed' in data.columns:
        data = data[data['seed'] == args.seed]
    x_col = _x_column(args.mask_mode)
    if (
        args.mask_mode in {'agent_random_spatial', 'constant_visible_shared'}
        and (
            x_col not in data.columns
            or data[x_col].isna().all()
        )
    ):
        inactive_x_col = f'inactive_{x_col}'
        if inactive_x_col in data.columns:
            x_col = inactive_x_col
    data = data.dropna(subset=[x_col])
    data['x_value'] = data[x_col].astype(float)
    data = data[
        (data['x_value'] >= args.min_value)
        & (data['x_value'] <= args.max_value)
    ]
    if data.empty:
        raise SystemExit('No rows match the requested filters.')

    # Keep the latest run for each orchestrator/overlap/agent. This lets a
    # corrected rerun replace an older parquet without deleting local results.
    data = data.sort_values('source_mtime')
    return data.drop_duplicates(
        subset=['orchestrator', 'x_value', 'agent'],
        keep='last',
    )


def _x_column(mask_mode: str) -> str:
    if mask_mode == 'constant_visible_shared':
        return 'constant_shared_visible_probability'
    if mask_mode == 'agent_random_spatial':
        return 'random_shared_visible_probability'
    return 'region_overlap_fraction'


def _x_label(mask_mode: str) -> str:
    if mask_mode == 'constant_visible_shared':
        return 'Shared fraction of visible blocks'
    if mask_mode == 'agent_random_spatial':
        return 'Shared visible probability'
    return 'Overlap'


def _file_prefix(mask_mode: str) -> str:
    if mask_mode == 'constant_visible_shared':
        return 'constant_visible_shared'
    if mask_mode == 'agent_random_spatial':
        return 'random_shared'
    return 'overlap'


def _communication_mse_column(data: pd.DataFrame) -> str:
    if 'comm_mse' in data.columns and data['comm_mse'].notna().any():
        return 'comm_mse'
    legacy_col = 'comm_mse_tx_missing_rx_visible'
    if legacy_col in data.columns and data[legacy_col].notna().any():
        return legacy_col
    raise SystemExit('No communication MSE column found in matching rows.')


def _aggregate(data: pd.DataFrame) -> pd.DataFrame:
    comm_col = _communication_mse_column(data)
    metrics = ['private_mse_visible', comm_col]
    grouped = (
        data.groupby(['orchestrator', 'x_value'])[metrics]
        .agg(['mean', 'std'])
        .reset_index()
    )
    grouped.columns = [
        '_'.join(col).strip('_') if isinstance(col, tuple) else col
        for col in grouped.columns
    ]
    if comm_col != 'comm_mse':
        grouped = grouped.rename(
            columns={
                f'{comm_col}_mean': 'comm_mse_mean',
                f'{comm_col}_std': 'comm_mse_std',
            }
        )
    return grouped


def _plot_metric(
    summary: pd.DataFrame,
    metric: str,
    ylabel: str,
    xlabel: str,
    out_path: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for orch in ORCH_ORDER:
        sub = summary[summary['orchestrator'] == orch].sort_values('x_value')
        if sub.empty:
            continue
        y_col = f'{metric}_mean'
        err_col = f'{metric}_std'
        ax.plot(
            sub['x_value'],
            sub[y_col],
            label=ORCH_LABELS.get(orch, orch),
            color=ORCH_COLORS[orch],
            marker=ORCH_MARKERS[orch],
            linestyle=ORCH_LINESTYLES[orch],
            linewidth=2.0,
            markersize=5.0,
        )
        if err_col in sub and sub[err_col].notna().any():
            lower = sub[y_col] - sub[err_col].fillna(0.0)
            upper = sub[y_col] + sub[err_col].fillna(0.0)
            ax.fill_between(
                sub['x_value'].to_numpy(),
                lower.to_numpy(),
                upper.to_numpy(),
                color=ORCH_COLORS[orch],
                alpha=0.16,
                linewidth=0,
            )

    x_values = sorted(summary['x_value'].dropna().unique())
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x_values)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    return _savefig(fig, out_path)


def _write_markdown_table(df: pd.DataFrame, out_path: Path) -> None:
    headers = list(df.columns)
    with out_path.open('w', encoding='utf-8') as f:
        f.write('| ' + ' | '.join(headers) + ' |\n')
        f.write('| ' + ' | '.join(['---'] * len(headers)) + ' |\n')
        for _, row in df.iterrows():
            values = []
            for value in row:
                if isinstance(value, float):
                    values.append(f'{value:.6f}')
                else:
                    values.append(str(value))
            f.write('| ' + ' | '.join(values) + ' |\n')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--results-dir',
        type=Path,
        default=Path('results/reconstruction'),
    )
    parser.add_argument(
        '--out-dir',
        type=Path,
        default=Path('results/reconstruction/plots'),
    )
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mask-mode', default='region_overlap')
    parser.add_argument('--min-value', type=float, default=0.1)
    parser.add_argument('--max-value', type=float, default=0.9)
    parser.add_argument(
        '--min-overlap',
        type=float,
        default=None,
        help='Deprecated alias for --min-value.',
    )
    parser.add_argument(
        '--max-overlap',
        type=float,
        default=None,
        help='Deprecated alias for --max-value.',
    )
    args = parser.parse_args()
    if args.min_overlap is not None:
        args.min_value = args.min_overlap
    if args.max_overlap is not None:
        args.max_value = args.max_overlap

    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = _load_results(args)
    summary = _aggregate(data)
    prefix = _file_prefix(args.mask_mode)
    xlabel = _x_label(args.mask_mode)

    table_path = args.out_dir / f'{prefix}_mse_summary.csv'
    summary.to_csv(table_path, index=False)
    _write_markdown_table(
        summary,
        args.out_dir / f'{prefix}_mse_summary.md',
    )

    with plt.style.context(_MPLSTYLE if _MPLSTYLE.exists() else 'default'):
        plt.rcParams['text.usetex'] = False
        private_path = _plot_metric(
            summary,
            'private_mse_visible',
            'Private visible MSE',
            xlabel,
            args.out_dir / f'private_mse_vs_{prefix}.png',
        )
        comm_path = _plot_metric(
            summary,
            'comm_mse',
            'Communication MSE',
            xlabel,
            args.out_dir / f'comm_mse_vs_{prefix}.png',
        )

    print(f'Summary saved -> {table_path}')
    print(f'Private MSE plot saved -> {private_path}')
    print(f'Communication MSE plot saved -> {comm_path}')


if __name__ == '__main__':
    main()
