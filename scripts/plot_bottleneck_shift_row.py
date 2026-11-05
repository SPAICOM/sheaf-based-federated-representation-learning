"""Shift-strength + bottleneck-dim sweeps in one row of four panels.

Combines the two side-by-side figures produced with ``--together`` by
``plot_multiagent_metrics.py`` (15-agent shift-strength sweep) and
``plot_hetero_bottleneck_metrics.py`` (2-agent heterogeneous bottleneck sweep)
into a single figure, left to right:

    (a) avg. private accuracy       vs distribution shift strength
    (b) avg. communication accuracy vs distribution shift strength
    (c) avg. private accuracy       vs bottleneck dimension
    (d) avg. communication accuracy vs bottleneck dimension

Each panel is tagged with its letter below the x-axis label, so the paper
can refer to it directly.

Runs are fetched from the wandb cloud API with the same discovery, run
selection, orchestrator exclusions (``EXCLUDED_ORCHS``) and estimators as the
two source scripts' defaults, so each pair of panels shows exactly the
numbers of the corresponding standalone figure. The shared legend lists every
orchestrator present in either sweep (the bottleneck sweep's set is a
superset of the shift sweep's), and colors/markers come from the canonical
``ORCH_ORDER`` so a method looks the same in all four panels. Fonts are
enlarged on top of ``config/plotting/plt.mplstyle`` so the figure stays
legible when scaled down to the full text width.

The data is fetched once and one figure is saved per ``--legend`` variant
(``<fname>_legend_<variant>.{png,pdf}``):

    top    one-row legend above all four panels, outside the axes
    b      legend inside panel (b), two columns in the empty band between
           the collaborative curves and ComFed
    c      legend inside panel (c), lower right (the axis is extended
           downward to make room below the saturated curves)

Panel (d) has no free area large enough for the legend.

Usage:
    uv run python scripts/plot_bottleneck_shift_row.py
    uv run python scripts/plot_bottleneck_shift_row.py --legend top
    uv run python scripts/plot_bottleneck_shift_row.py --entity my-team \\
        --out_dir results/combined/plots
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from plot_hetero_bottleneck_metrics import (
    discover_runs_remote as discover_bottleneck_runs_remote,
)
from plot_multiagent_metrics import (
    _AGENT_COMM_RE,
    _AGENT_PRIV_RE,
    _MPLSTYLE,
    ORCH_LABELS,
    _agent_metric_long_df,
    _center_and_bounds,
    _ordered_orchs,
    _plot_metric_on_ax,
    _savefig,
    _style_maps,
    drop_excluded_orchs,
    read_run_meta_remote,
)

# Font/marker overrides on top of plt.mplstyle: four panels side by side are
# shrunk ~4x more than a standalone plot once placed in the paper, so the
# text is scaled up and the markers down relative to the per-panel size.
_RC = {
    'font.size': 30,
    'axes.labelsize': 30,
    'xtick.labelsize': 28,
    'ytick.labelsize': 28,
    'legend.fontsize': 25,
    'legend.title_fontsize': 27,
    'lines.markersize': 13,
    'lines.linewidth': 3,
}
# Taller than it would otherwise need to be to leave room for the (a)-(d)
# tags below the x labels without shrinking the axes under the y labels.
_PANEL_SIZE = (7.8, 7.7)
# Extra horizontal space (inches) between the shift pair (a, b) and the
# bottleneck pair (c, d), on top of the uniform tight_layout padding.
_GROUP_GAP = 1.6
_LEGEND_VARIANTS = ('top', 'b', 'c')


def _add_group_gap(fig: plt.Figure, axes, split: int, gap: float) -> None:
    """Widen ``fig`` by ``gap`` inches and open that space before ``axes[split]``.

    Axes keep their size in inches: every axis at or after ``split`` is
    shifted right by ``gap``, the rest stay put.
    """
    old_w, h = fig.get_size_inches()
    new_w = old_w + gap
    positions = [ax.get_position() for ax in axes]
    fig.set_size_inches(new_w, h)
    for i, (ax, pos) in enumerate(zip(axes, positions)):
        x0 = pos.x0 * old_w + (gap if i >= split else 0.0)
        ax.set_position(
            [x0 / new_w, pos.y0, pos.width * old_w / new_w, pos.height]
        )


def discover_shift_runs_by_seed(
    api: Any,
    entity: str,
    project: str,
    seeds: list[int] | None = None,
    orchs: set[str] | None = None,
) -> dict[tuple[str, float, int], dict[str, Any]]:
    """Return ``{(orch, shift, seed): meta}`` for one wandb project.

    Unlike ``plot_multiagent_metrics.discover_runs_remote``, which keeps one
    run per ``(orch, shift)``, the seed is part of the key — that is what the
    error bars are computed over. Within a key (a re-run of the same cell) the
    most recent run wins. Runs with no ``test/avg_comm_task_perf`` in their
    summary are dropped: the project also holds aborted one-minute runs that
    would otherwise shadow a real one.
    """
    from concurrent.futures import ThreadPoolExecutor

    remote = list(api.runs(f'{entity}/{project}', filters={'state': 'finished'}))
    with ThreadPoolExecutor(max_workers=8) as pool:
        metas = list(pool.map(read_run_meta_remote, remote))

    best: dict[tuple[str, float, int], dict[str, Any]] = {}
    for meta in metas:
        if meta is None:
            continue
        if 'test/avg_comm_task_perf' not in meta['summary']:
            continue
        orch = meta['orch']
        if orchs is not None and orch not in orchs:
            continue
        seed = meta['raw_config'].get('seed')
        if seed is None or (seeds is not None and int(seed) not in seeds):
            continue
        key = (orch, float(meta['shift']), int(seed))
        if key not in best or meta['mtime'] > best[key]['mtime']:
            best[key] = meta
    return best


def seed_metric_df(
    runs: dict[tuple[str, float, int], dict[str, Any]],
    pattern: Any,
    value_col: str,
    estimator: str,
) -> pd.DataFrame:
    """One row per ``(orchestrator, shift, seed)``, agents already aggregated.

    Each row is that run's across-agent statistic — the plain mean, or the
    degree-weighted mean for the communication metric, exactly as the
    single-seed panels compute it — so the only spread left for the plot to
    show is the spread across seeds.
    """
    rows = []
    for seed in sorted({sd for _, _, sd in runs}):
        per_seed = {
            (orch, shift): meta
            for (orch, shift, sd), meta in runs.items()
            if sd == seed
        }
        long_df = _agent_metric_long_df(per_seed, pattern, value_col)
        if long_df.empty:
            continue
        for (orch, shift), group in long_df.groupby(['_orch', 'shift']):
            center, _, _ = _center_and_bounds(
                group[value_col].to_numpy(),
                estimator,
                None,
                weights=group['weight'].to_numpy(),
            )
            rows.append(
                {
                    'orchestrator': ORCH_LABELS.get(orch, orch),
                    '_orch': orch,
                    'shift': shift,
                    'seed': seed,
                    value_col: center,
                }
            )
    return pd.DataFrame(rows)


def _plot_seed_metric_on_ax(
    ax: plt.Axes,
    df: pd.DataFrame,
    orchs: list[str],
    colors: dict,
    markers: dict,
    value_col: str,
    ylabel: str,
    xlabel: str,
) -> None:
    """Mean across seeds with seaborn's default error bars, one line per orch.

    ``errorbar`` is deliberately not passed, so the band is seaborn's default
    (a 95% bootstrap confidence interval of the mean over the seeds present
    for that orchestrator).
    """
    present = [o for o in orchs if o in set(df['_orch'])]
    labels = [ORCH_LABELS.get(o, o) for o in present]
    sns.lineplot(
        data=df[df['_orch'].isin(present)],
        x='shift',
        y=value_col,
        hue='orchestrator',
        style='orchestrator',
        hue_order=labels,
        style_order=labels,
        palette={ORCH_LABELS.get(o, o): colors[o] for o in present},
        markers={ORCH_LABELS.get(o, o): markers[o] for o in present},
        dashes=False,
        estimator='mean',
        legend=False,
        ax=ax,
        linewidth=plt.rcParams['lines.linewidth'],
        markersize=plt.rcParams['lines.markersize'],
    )
    ax.set_xticks(sorted(df['shift'].unique()))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def plot_row(
    panels: list[tuple],
    orchs: list[str],
    colors: dict,
    markers: dict,
    legend: str,
    out_path: Path,
) -> None:
    """Draw ``panels`` in one row, tag them (a), (b), …, and place the legend.

    ``panels`` holds one dict per panel: ``seeded`` panels are drawn as a
    mean over seeds with seaborn's default error band, the others as a single
    curve. ``legend`` is one of :data:`_LEGEND_VARIANTS`; it always lists
    every orchestrator in ``orchs``, since no single panel carries them all.
    """
    with plt.style.context(str(_MPLSTYLE)), plt.rc_context(_RC):
        w, h = _PANEL_SIZE
        fig, axes = plt.subplots(1, len(panels), figsize=(len(panels) * w, h))
        for i, (ax, panel) in enumerate(zip(axes, panels)):
            if panel['seeded']:
                _plot_seed_metric_on_ax(
                    ax,
                    panel['df'],
                    orchs,
                    colors,
                    markers,
                    panel['value_col'],
                    panel['ylabel'],
                    panel['x_kwargs']['xlabel'],
                )
            else:
                _plot_metric_on_ax(
                    ax,
                    panel['df'],
                    orchs,
                    colors,
                    markers,
                    panel['value_col'],
                    panel['ylabel'],
                    adjusted=False,
                    show_error=False,
                    estimator=panel['estimator'],
                    **panel['x_kwargs'],
                )
            ax.annotate(
                f'({chr(ord("a") + i)})',
                xy=(0.5, 0.0),
                xycoords=ax.xaxis.label,
                xytext=(0, -12),
                textcoords='offset points',
                ha='center',
                va='top',
                fontsize=plt.rcParams['axes.labelsize'],
            )
        sns.despine(fig=fig)
        # Built from the style maps rather than scraped off an axis: the
        # seeded panels draw through seaborn with legend=False, and no single
        # panel carries every orchestrator anyway.
        labels = [ORCH_LABELS.get(o, o) for o in orchs]
        handles = [
            Line2D(
                [],
                [],
                color=colors[o],
                marker=markers[o],
                linestyle='-',
                linewidth=plt.rcParams['lines.linewidth'],
                markersize=plt.rcParams['lines.markersize'],
            )
            for o in orchs
        ]

        if legend == 'c':
            # Private accuracy saturates near 1 from d=64 on: extend the axis
            # downward so the legend sits in empty space below the curves.
            axes[2].set_ylim(bottom=0.3)
            axes[2].legend(
                handles, labels, title='Method', loc='lower right', frameon=True
            )
        elif legend == 'b':
            # Anchor the y range at [0, 1] so the empty band between the
            # collaborative curves (>= 0.70) and ComFed (<= 0.16) is tall
            # enough for a compact two-column legend.
            axes[1].set_ylim(0.0, 1.0)
            axes[1].legend(
                handles,
                labels,
                title='Method',
                loc='center',
                bbox_to_anchor=(0.5, 0.43),
                ncol=2,
                fontsize=23,
                title_fontsize=25,
                handlelength=1.0,
                handletextpad=0.4,
                columnspacing=0.8,
                borderpad=0.3,
                labelspacing=0.3,
                frameon=True,
            )
        fig.tight_layout(w_pad=0.0)
        _add_group_gap(fig, axes, split=2, gap=_GROUP_GAP)
        if legend == 'top':
            fig.legend(
                handles,
                labels,
                loc='lower center',
                bbox_to_anchor=(0.5, 1.0),
                ncol=len(labels),
                fontsize=plt.rcParams['axes.labelsize'],
                columnspacing=1.8,
                frameon=True,
                borderaxespad=0.0,
            )
        _savefig(fig, out_path)
        plt.close(fig)
    print(f'  saved → {out_path} (+ .pdf)')


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--entity',
        type=str,
        default=None,
        help='wandb entity (default: your wandb default entity).',
    )
    parser.add_argument(
        '--bottleneck_project', type=str, default='sfrl_hetero_bottleneck'
    )
    parser.add_argument(
        '--bottleneck_shift_strength',
        type=float,
        default=0.7,
        help='dataset.shift_strength the bottleneck-sweep runs are filtered on.',
    )
    # The 4-method shift sweep behind results/multi_agent/plots/*_vs_shift.png
    # (plot_multiagent_metrics.py itself defaults to multi_hetero_agents_true).
    parser.add_argument('--shift_project', type=str, default='shift_distr')
    # FedProto/FedMuscle were run in the Sapienza team's own copy of the
    # sweep; its SheafFRL/NonCooperativeLearning runs are a different (much
    # weaker, private acc ~0.5) configuration and are deliberately not taken.
    parser.add_argument(
        '--shift_extra_project',
        type=str,
        default='avino-1905974-sapienza-universit-di-roma/shift_distr',
        help='"[entity/]project" holding extra shift runs to merge in; '
        '"none" to use only --shift_project.',
    )
    parser.add_argument(
        '--shift_extra_orchs',
        nargs='*',
        default=['FedProto', 'FedMuscle'],
        help='Orchestrators taken from --shift_extra_project (empty: all). '
        'Runs already present in --shift_project are never overridden.',
    )
    parser.add_argument(
        '--out_dir', type=Path, default=Path('results/combined/plots')
    )
    parser.add_argument(
        '--fname',
        type=str,
        default='comm_and_priv_task_perf_vs_shift_and_bottleneck',
        help='Output file stem; "_legend_<variant>.png/.pdf" is appended.',
    )
    parser.add_argument(
        '--seeds',
        nargs='+',
        type=int,
        default=[1, 2, 3],
        help='Seeds averaged in the shift panels (error bars are computed '
        'over them). Seed 42 is excluded by default: for every method but '
        'SheafFRL its only runs are the July 2026 ones, made with the code '
        'as it was before the SheafFRL/alignment rewrite, so pooling it with '
        'the current seeds would mix two code vintages.',
    )
    parser.add_argument(
        '--legend',
        nargs='+',
        choices=_LEGEND_VARIANTS,
        default=list(_LEGEND_VARIANTS),
        help='Legend placement variant(s) to render (default: all).',
    )
    args = parser.parse_args()

    import wandb

    api = wandb.Api()
    entity = args.entity or api.default_entity
    if entity is None:
        raise SystemExit(
            'No wandb entity available (pass --entity, or run `wandb login`).'
        )

    print(f'Scanning {entity}/{args.bottleneck_project!r} …')
    b_runs = discover_bottleneck_runs_remote(
        api,
        entity,
        args.bottleneck_project,
        shift_strength=args.bottleneck_shift_strength,
    )
    # Same filtering as plot_hetero_bottleneck_metrics.py.
    b_runs = drop_excluded_orchs(
        {(o, d): m for (o, d), m in b_runs.items() if d >= 16}
    )
    seeds = sorted(set(args.seeds))
    print(f'Scanning {entity}/{args.shift_project!r} (seeds {seeds}) …')
    s_runs = drop_excluded_orchs(
        discover_shift_runs_by_seed(api, entity, args.shift_project, seeds)
    )
    if args.shift_extra_project.lower() != 'none':
        extra_entity, _, extra_project = args.shift_extra_project.rpartition('/')
        extra_entity = extra_entity or entity
        keep = set(args.shift_extra_orchs) or None
        print(f'Scanning {extra_entity}/{extra_project!r} (extra) …')
        extra = drop_excluded_orchs(
            discover_shift_runs_by_seed(
                api, extra_entity, extra_project, seeds, orchs=keep
            )
        )
        added = 0
        for key, meta in extra.items():
            if key in s_runs:  # the main project wins on overlap
                continue
            s_runs[key] = meta
            added += 1
        print(f'  merged {added} extra runs ({sorted(keep) if keep else "all orchs"})')
    per_cell = {}
    for orch, shift, seed in s_runs:
        per_cell.setdefault(orch, set()).add(seed)
    for orch in sorted(per_cell):
        print(f'  {orch}: seeds {sorted(per_cell[orch])}')
    if not b_runs or not s_runs:
        raise SystemExit('No completed runs found for one of the two sweeps.')

    orchs = _ordered_orchs({o for o, _ in b_runs} | {o for o, _, _ in s_runs})
    colors, markers, _ = _style_maps(orchs)
    print(f'Orchestrators: {orchs}')

    b_comm = _agent_metric_long_df(
        b_runs, _AGENT_COMM_RE, 'comm_task_perf', x_col='bottleneck_dim'
    )
    b_priv = _agent_metric_long_df(
        b_runs, _AGENT_PRIV_RE, 'private_task_perf', x_col='bottleneck_dim'
    )
    s_comm = seed_metric_df(
        s_runs, _AGENT_COMM_RE, 'comm_task_perf', 'weighted_mean'
    )
    s_priv = seed_metric_df(
        s_runs, _AGENT_PRIV_RE, 'private_task_perf', 'mean'
    )

    bottleneck_x = {
        'x_col': 'bottleneck_dim',
        'xlabel': 'Bottleneck dimension',
        'xscale': 'log',
        'xticklabel_fmt': lambda x: str(int(x)),
    }
    shift_x = {'x_col': 'shift', 'xlabel': 'Distribution shift strength'}
    # (df, value_col, ylabel, estimator, x kwargs) — estimators match the
    # source scripts' defaults (degree-weighted comm mean on the graph sweep).
    def panel(df, value_col, ylabel, estimator, x_kwargs, seeded):
        return {
            'df': df,
            'value_col': value_col,
            'ylabel': ylabel,
            'estimator': estimator,
            'x_kwargs': x_kwargs,
            'seeded': seeded,
        }

    # Only the shift sweep has been repeated over seeds, so only (a)/(b) get
    # error bars; the bottleneck panels stay single-curve.
    panels = [
        panel(s_priv, 'private_task_perf', 'Avg. private accuracy', 'mean', shift_x, True),
        panel(s_comm, 'comm_task_perf', 'Avg. communication accuracy', 'weighted_mean', shift_x, True),
        panel(b_priv, 'private_task_perf', 'Avg. private accuracy', 'mean', bottleneck_x, False),
        panel(b_comm, 'comm_task_perf', 'Avg. communication accuracy', 'mean', bottleneck_x, False),
    ]

    sns.set_theme(style='whitegrid', context='paper', font_scale=1.4)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for legend in args.legend:
        plot_row(
            panels,
            orchs,
            colors,
            markers,
            legend,
            args.out_dir / f'{args.fname}_legend_{legend}.png',
        )


if __name__ == '__main__':
    main()
