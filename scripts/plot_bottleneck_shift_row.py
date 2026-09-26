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

import matplotlib.pyplot as plt
import seaborn as sns
from plot_hetero_bottleneck_metrics import (
    discover_runs_remote as discover_bottleneck_runs_remote,
)
from plot_multiagent_metrics import (
    _AGENT_COMM_RE,
    _AGENT_PRIV_RE,
    _MPLSTYLE,
    ORCH_LABELS,
    _agent_metric_long_df,
    _ordered_orchs,
    _plot_metric_on_ax,
    _savefig,
    _style_maps,
    discover_runs_remote,
    drop_excluded_orchs,
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


def plot_row(
    panels: list[tuple],
    orchs: list[str],
    colors: dict,
    markers: dict,
    legend: str,
    out_path: Path,
) -> None:
    """Draw ``panels`` in one row, tag them (a), (b), …, and place the legend.

    ``panels`` holds ``(df, value_col, ylabel, estimator, x_kwargs)`` tuples;
    ``legend`` is one of :data:`_LEGEND_VARIANTS`. The legend lists the
    union of the orchestrators drawn in any panel, in ``orchs`` order.
    """
    with plt.style.context(str(_MPLSTYLE)), plt.rc_context(_RC):
        w, h = _PANEL_SIZE
        fig, axes = plt.subplots(1, len(panels), figsize=(len(panels) * w, h))
        for i, (ax, (df, value_col, ylabel, estimator, x_kwargs)) in enumerate(
            zip(axes, panels)
        ):
            _plot_metric_on_ax(
                ax,
                df,
                orchs,
                colors,
                markers,
                value_col,
                ylabel,
                adjusted=False,
                show_error=False,
                estimator=estimator,
                **x_kwargs,
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
        # The shift sweep lacks FedProto/FedMuscle, so no single panel
        # carries every orchestrator.
        by_label = {}
        for ax in axes:
            for handle, label in zip(*ax.get_legend_handles_labels()):
                by_label.setdefault(label, handle)
        labels = [
            ORCH_LABELS.get(o, o) for o in orchs if ORCH_LABELS.get(o, o) in by_label
        ]
        handles = [by_label[label] for label in labels]

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
    print(f'Scanning {entity}/{args.shift_project!r} …')
    s_runs = drop_excluded_orchs(
        discover_runs_remote(api, entity, args.shift_project, None)
    )
    if args.shift_extra_project.lower() != 'none':
        extra_entity, _, extra_project = args.shift_extra_project.rpartition('/')
        extra_entity = extra_entity or entity
        print(f'Scanning {extra_entity}/{extra_project!r} (extra) …')
        extra = drop_excluded_orchs(
            discover_runs_remote(api, extra_entity, extra_project, None)
        )
        keep = set(args.shift_extra_orchs)
        added = 0
        for key, meta in extra.items():
            if keep and key[0] not in keep:
                continue
            if key in s_runs:  # the main project wins on overlap
                continue
            s_runs[key] = meta
            added += 1
        print(f'  merged {added} extra runs ({sorted(keep) or "all orchs"})')
    if not b_runs or not s_runs:
        raise SystemExit('No completed runs found for one of the two sweeps.')

    orchs = _ordered_orchs({o for o, _ in b_runs} | {o for o, _ in s_runs})
    colors, markers, _ = _style_maps(orchs)
    print(f'Orchestrators: {orchs}')

    b_comm = _agent_metric_long_df(
        b_runs, _AGENT_COMM_RE, 'comm_task_perf', x_col='bottleneck_dim'
    )
    b_priv = _agent_metric_long_df(
        b_runs, _AGENT_PRIV_RE, 'private_task_perf', x_col='bottleneck_dim'
    )
    s_comm = _agent_metric_long_df(s_runs, _AGENT_COMM_RE, 'comm_task_perf')
    s_priv = _agent_metric_long_df(s_runs, _AGENT_PRIV_RE, 'private_task_perf')

    bottleneck_x = {
        'x_col': 'bottleneck_dim',
        'xlabel': 'Bottleneck dimension',
        'xscale': 'log',
        'xticklabel_fmt': lambda x: str(int(x)),
    }
    shift_x = {'x_col': 'shift', 'xlabel': 'Distribution shift strength'}
    # (df, value_col, ylabel, estimator, x kwargs) — estimators match the
    # source scripts' defaults (degree-weighted comm mean on the graph sweep).
    panels = [
        (s_priv, 'private_task_perf', 'Avg. private accuracy', 'mean', shift_x),
        (s_comm, 'comm_task_perf', 'Avg. communication accuracy', 'weighted_mean', shift_x),
        (b_priv, 'private_task_perf', 'Avg. private accuracy', 'mean', bottleneck_x),
        (b_comm, 'comm_task_perf', 'Avg. communication accuracy', 'mean', bottleneck_x),
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
