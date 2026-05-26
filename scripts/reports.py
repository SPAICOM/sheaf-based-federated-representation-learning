"""Generate publication-quality comparison plots from W&B runs."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Keep Matplotlib quiet in sandboxed/cluster jobs where $HOME may be read-only.
os.environ.setdefault(
    'MPLCONFIGDIR',
    str(Path('/tmp/sheaf_reports_matplotlib').resolve()),
)

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import (
    FuncFormatter,
    LogLocator,
    MaxNLocator,
    NullFormatter,
)

import wandb

# =====================================================================
# 1. CONFIGURATION
# =====================================================================
PROJECT_PATH = 'avino-1905974-sapienza-universit-di-roma/rotated_mnist'
OUTPUT_DIR = Path('reports')
OUTPUT_FORMATS = ('pdf', 'png')

COOPERATIVE_RUNS = {
    'itozic1c': 'Sheaf-FRL (Prototype Anchors)',
    '0ug3sabn': 'Sheaf-FRL (Global Pilots)',
    'beql9d40': 'Sheaf-FMTL',
    'h480pcxo': 'd-PSGD',
    'xrtc17kk': 'd-FedU',
}

BASELINE_RUN = 'lu0anc3d'
BASELINE_LABEL = 'Local Baseline'

ACC_METRIC = 'test_monitor/avg_task_performance_epoch'
KB_METRIC = 'test_monitor/train_communication_kilobytes_cumulative'
ROUNDS_METRIC = 'test_monitor/train_communication_rounds_cumulative'

METHOD_ORDER = list(dict.fromkeys(COOPERATIVE_RUNS.values()))


@dataclass(frozen=True)
class MethodStyle:
    color: str
    marker: str
    linestyle: str = '-'


METHOD_STYLES = {
    # Okabe-Ito inspired, colorblind-safe palette.
    'Sheaf-FRL (Prototype Anchors)': MethodStyle('#0072B2', 'o'),
    'Sheaf-FRL (Global Pilots)': MethodStyle('#009E73', 's'),
    'Sheaf-FMTL': MethodStyle('#D55E00', '^'),
    'd-PSGD': MethodStyle('#CC79A7', 'D'),
    'd-FedU': MethodStyle('#E69F00', 'P'),
}
# =====================================================================


def configure_plotting() -> None:
    """Apply a compact, vector-friendly style for paper figures."""
    sns.set_theme(context='paper', style='whitegrid')
    plt.rcParams.update(
        {
            'figure.dpi': 150,
            'savefig.dpi': 600,
            'savefig.bbox': 'tight',
            'font.family': 'DejaVu Sans',
            'font.size': 9,
            'axes.labelsize': 10,
            'axes.titlesize': 10,
            'axes.linewidth': 0.8,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'legend.fontsize': 8,
            'legend.title_fontsize': 8,
            'lines.linewidth': 1.9,
            'lines.markersize': 4.5,
            'grid.linewidth': 0.45,
            'grid.alpha': 0.28,
            'pdf.fonttype': 42,
            'ps.fonttype': 42,
            'svg.fonttype': 'none',
        }
    )


def _last_per_step(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or '_step' not in frame.columns:
        return frame
    return (
        frame.sort_values('_step')
        .groupby('_step', as_index=False, dropna=False)
        .last()
    )


def _coerce_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors='coerce')
    return frame


def fetch_data() -> tuple[pd.DataFrame, float | None]:
    api = wandb.Api()
    runs = api.runs(PROJECT_PATH)

    coop_data = []
    baseline_max_acc: float | None = None

    print('Fetching and aligning asynchronous data from W&B...\n')
    for run in runs:
        if run.id in COOPERATIVE_RUNS or run.name in COOPERATIVE_RUNS:
            label = COOPERATIVE_RUNS.get(
                run.id, COOPERATIVE_RUNS.get(run.name)
            )
            print(f'[{label}] Downloading and aligning timelines...')

            val_history = list(
                run.scan_history(keys=['_step', 'epoch', ACC_METRIC])
            )
            comm_history = list(
                run.scan_history(keys=['_step', KB_METRIC, ROUNDS_METRIC])
            )

            val_df = _last_per_step(pd.DataFrame(val_history))
            comm_df = _last_per_step(pd.DataFrame(comm_history))

            if val_df.empty or comm_df.empty:
                print('  Skipped: missing validation or communication data.')
                continue

            merged_df = pd.merge(
                val_df,
                comm_df,
                on='_step',
                how='outer',
                suffixes=('', '_comm'),
            ).sort_values('_step')
            merged_df = _coerce_numeric(
                merged_df,
                ['_step', 'epoch', ACC_METRIC, KB_METRIC, ROUNDS_METRIC],
            )
            merged_df[KB_METRIC] = merged_df[KB_METRIC].ffill().fillna(0)
            merged_df[ROUNDS_METRIC] = (
                merged_df[ROUNDS_METRIC].ffill().fillna(0)
            )

            aligned_df = merged_df.dropna(subset=[ACC_METRIC]).copy()
            if aligned_df.empty:
                print('  Skipped: no aligned accuracy rows.')
                continue
            aligned_df['Algorithm'] = label
            coop_data.append(aligned_df)

        elif run.id == BASELINE_RUN or run.name == BASELINE_RUN:
            print(f'[{BASELINE_LABEL}] Extracting max accuracy...')
            df = _coerce_numeric(
                pd.DataFrame(list(run.scan_history(keys=[ACC_METRIC]))),
                [ACC_METRIC],
            )
            if not df.empty and ACC_METRIC in df.columns:
                candidate = df[ACC_METRIC].max()
                if pd.notna(candidate):
                    baseline_max_acc = float(candidate)

    if not coop_data:
        raise ValueError('\nCRITICAL: No cooperative run data found.')

    plot_df = pd.concat(coop_data, ignore_index=True)
    plot_df['Algorithm'] = pd.Categorical(
        plot_df['Algorithm'],
        categories=METHOD_ORDER,
        ordered=True,
    )
    return plot_df, baseline_max_acc


# =====================================================================
# 2. PUBLICATION-READY PLOTTING
# =====================================================================
def _accuracy_formatter(max_value: float) -> tuple[FuncFormatter, str]:
    if max_value <= 1.5:
        return FuncFormatter(
            lambda y, _pos: f'{100 * y:.0f}'
        ), 'Test accuracy (%)'
    return FuncFormatter(lambda y, _pos: f'{y:.2f}'), 'Test accuracy'


def _format_axis(
    ax: plt.Axes,
    *,
    use_log_x: bool,
    x_metric: str,
    x_label: str,
    y_label: str,
) -> None:
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

    if x_metric == 'epoch':
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    elif not use_log_x:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))

    ax.grid(True, which='major', axis='both')
    if use_log_x:
        ax.set_xscale('log')
        ax.xaxis.set_major_locator(LogLocator(base=10))
        ax.xaxis.set_minor_locator(
            LogLocator(base=10, subs=(2, 3, 4, 5, 6, 7, 8, 9))
        )
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.grid(True, which='minor', axis='x', alpha=0.12, linewidth=0.35)


def _save_figure(fig: plt.Figure, filename: str | Path) -> None:
    output_base = OUTPUT_DIR / Path(filename).stem
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for extension in OUTPUT_FORMATS:
        output_path = output_base.with_suffix(f'.{extension}')
        fig.savefig(output_path)
        print(f'Saved {output_path}')


def create_plot(
    df: pd.DataFrame,
    baseline_acc: float | None,
    x_metric: str,
    x_label: str,
    filename: str,
    use_log_x: bool = False,
) -> None:
    if x_metric not in df.columns:
        print(f'Skipping {filename}: missing metric {x_metric!r}.')
        return

    plot_df = df.dropna(subset=[x_metric, ACC_METRIC]).copy()
    if use_log_x:
        plot_df = plot_df[plot_df[x_metric] > 0]
    if plot_df.empty:
        print(f'Skipping {filename}: no plottable rows.')
        return

    baseline_value = (
        float(baseline_acc)
        if baseline_acc is not None and pd.notna(baseline_acc)
        else 0.0
    )
    max_acc = max(float(plot_df[ACC_METRIC].max()), baseline_value)
    y_formatter, y_label = _accuracy_formatter(max_acc)

    fig, ax = plt.subplots(figsize=(3.45, 2.45), constrained_layout=True)

    for algorithm in METHOD_ORDER:
        group = (
            plot_df[plot_df['Algorithm'] == algorithm]
            .sort_values(x_metric)
            .drop_duplicates(subset=[x_metric], keep='last')
        )
        if group.empty:
            continue

        style = METHOD_STYLES.get(algorithm, MethodStyle('#4D4D4D', 'o'))
        markevery = max(1, len(group) // 8)
        ax.plot(
            group[x_metric],
            group[ACC_METRIC],
            label=algorithm,
            color=style.color,
            linestyle=style.linestyle,
            marker=style.marker,
            markevery=markevery,
            markerfacecolor='white',
            markeredgewidth=0.9,
        )

    if baseline_value > 0:
        baseline_label = f'{BASELINE_LABEL} ({100 * baseline_value:.1f}%)'
        if max_acc > 1.5:
            baseline_label = f'{BASELINE_LABEL} ({baseline_value:.2f})'
        ax.axhline(
            y=baseline_value,
            color='#222222',
            linestyle=(0, (4, 2)),
            linewidth=1.25,
            label=baseline_label,
            zorder=0,
        )

    _format_axis(
        ax,
        use_log_x=use_log_x,
        x_metric=x_metric,
        x_label=x_label,
        y_label=y_label,
    )
    ax.yaxis.set_major_formatter(y_formatter)

    y_min = max(0.0, float(plot_df[ACC_METRIC].min()) - 0.04)
    y_max = min(1.02, max_acc + 0.035) if max_acc <= 1.5 else max_acc * 1.04
    ax.set_ylim(y_min, y_max)

    if use_log_x:
        x_min = float(plot_df[x_metric].min())
        x_max = float(plot_df[x_metric].max())
        ax.set_xlim(left=max(1.0, x_min * 0.8), right=x_max * 1.15)
    else:
        ax.margins(x=0.03)

    ax.legend(
        loc='lower right',
        frameon=True,
        framealpha=0.94,
        borderpad=0.45,
        handlelength=1.8,
        labelspacing=0.35,
    )

    _save_figure(fig, filename)
    plt.close(fig)


if __name__ == '__main__':
    configure_plotting()
    df, baseline_acc = fetch_data()

    create_plot(
        df,
        baseline_acc,
        'epoch',
        'Epoch',
        'acc_vs_epoch',
        use_log_x=False,
    )
    create_plot(
        df,
        baseline_acc,
        KB_METRIC,
        'Communication volume (kB)',
        'acc_vs_kb',
        use_log_x=True,
    )
    create_plot(
        df,
        baseline_acc,
        ROUNDS_METRIC,
        'Communication rounds',
        'acc_vs_rounds',
        use_log_x=False,
    )
