r"""MNIST 4 agents anchor-selection ablation: original vs proto_kmeans vs stratified anchors.

DiSheafFRL (lambda .01, CenterNorm) and non-coop, K = 128/384/768/1536 pilots/anchors, 3 seeds per point.
Reads the offline scope scores (no W&B, no checkpoints) and writes into results/mnist_4agents_shift/anchors/:
  per_run.csv       every scored run (fractions): scripts/score_mnist4_anchors.py cache + the original K=384/768/1536
                    rows of results/mnist_4agents_shift/swbn_scopes/per_run.csv
  summary_wide.csv  mean / sample SD / n per (method, arm, K) and scope (%)
  summary.md        tables (+ DiSheafFRL minus non-coop gap)
  anchors_study.{pdf,png}         (a) Comm all classes, (b) Comm on C_i \ C_j, (c) Comm on C_i
  anchors_study_shared.{pdf,png}  (a) Comm on C_i \ C_j, (b) Comm on C_i ∩ C_j, (c) Comm on C_i
                                  (layout of results/mnist_4agents_shift/maps/pilot_study_shared, panel (c) = Comm on C_i instead of Private)
i = receiver, j = sender. Solid = DiSheafFRL, dotted = non-coop, one colour per strategy.
Usage: uv run python scripts/plot_mnist4_anchors.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.plotting import use_style  # noqa: E402

OUT = ROOT / 'results' / 'mnist_4agents_shift' / 'anchors'
OLD = ROOT / 'results' / 'mnist_4agents_shift' / 'swbn_scopes' / 'per_run.csv'
KS = [128, 384, 768, 1536]
# summary name -> per_run.csv column
SCOPES = {
    'Private': 'private',
    'Private seen': 'Private target classes',
    'Comm': 'Comm',
    'Sender unseen / receiver seen': 'Comm sender unseen / receiver seen',
    'Receiver seen': 'Comm receiver seen',
    'Both seen': 'Comm sender seen / receiver seen',
}
STRATEGIES = {
    'original': ('Original (K pilots)', 'tab:blue'),
    'kmeans': ('K-means prototypes', 'tab:purple'),
    'strat': ('Stratified', 'tab:green'),
}
FIGURES = {
    'anchors_study': [
        ('Comm', r'(a) Comm. accuracy, all classes'),
        (
            'Sender unseen / receiver seen',
            r'(b) Comm. accuracy on $\mathcal{C}_i\setminus\mathcal{C}_j$',
        ),
        ('Receiver seen', r'(c) Comm. accuracy on $\mathcal{C}_i$'),
    ],
    'anchors_study_shared': [
        (
            'Sender unseen / receiver seen',
            r'(a) Comm. accuracy on $\mathcal{C}_i\setminus\mathcal{C}_j$',
        ),
        (
            'Both seen',
            r'(b) Comm. accuracy on $\mathcal{C}_i\cap\mathcal{C}_j$',
        ),
        ('Receiver seen', r'(c) Comm. accuracy on $\mathcal{C}_i$'),
    ],
}
COLS = [
    'method',
    'arm',
    'group',
    'K',
    'seed',
    'source_run',
    'private',
    'Private target classes',
    'Comm',
    'Comm sender unseen / receiver seen',
    'Comm receiver seen',
    'Comm sender seen / receiver seen',
    'remote_comm',
    'note',
]


def load_runs() -> pd.DataFrame:
    """Scored runs of the ablation: new scores plus the original K>=384 rows of the pilot study."""
    new = pd.read_csv(OUT / 'per_run_scored.csv')
    new['note'] = new.group.map(
        lambda g: 'warm-start k-means' if 'warm' in g else ''
    )
    old = pd.read_csv(OLD)
    old = old[
        old.method.isin(['DiSheafFRL + CenterNorm', 'Non-coop'])
        & old.pilots.isin([384, 768, 1536])
    ].rename(
        columns={'run': 'source_run', 'pilots': 'K', 'Private': 'private'}
    )
    old['method'] = old.method.str.replace(' + CenterNorm', '', regex=False)
    old['arm'], old['group'], old['remote_comm'] = (
        'original',
        'pilot study',
        np.nan,
    )
    old['note'] = 'from results/mnist_4agents_shift/swbn_scopes'
    old[['K', 'seed']] = old[['K', 'seed']].astype(int)
    runs = pd.concat([new[COLS], old[COLS]])
    return runs.sort_values(
        ['method', 'arm', 'K', 'group', 'seed']
    ).reset_index(drop=True)


def summarise(runs: pd.DataFrame) -> pd.DataFrame:
    """Mean / sample SD / n (%) per (method, arm, K, k-means fit variant)."""
    d = runs.copy()
    d[[*SCOPES.values()]] *= 100
    d['kmeans_fit'] = np.where(
        d.arm == 'kmeans',
        np.where(d.group.str.contains('warm'), 'warm', 'cold'),
        '',
    )
    rows = []
    for (m, a, k, fit), t in d.groupby(['method', 'arm', 'K', 'kmeans_fit']):
        r = {
            'method': m,
            'arm': a,
            'K': k,
            'kmeans_fit': fit,
            'n_seeds': len(t),
            'seeds': ','.join(map(str, sorted(t.seed))),
        }
        for name, c in SCOPES.items():
            r[f'{name} mean'] = t[c].mean()
            r[f'{name} sd'] = t[c].std(ddof=1) if len(t) > 1 else np.nan
        rows.append(r)
    w = pd.DataFrame(rows)
    w['o'] = w.arm.map({'original': 0, 'kmeans': 1, 'strat': 2})
    return w.sort_values(['method', 'o', 'K', 'kmeans_fit']).drop(columns='o')


def cell(r, name):
    """Table cell: mean ± SD (SD omitted for a single seed)."""
    sd = '' if pd.isna(r[f'{name} sd']) else f' ± {r[f"{name} sd"]:.2f}'
    return f'{r[f"{name} mean"]:.2f}{sd}'


def write_markdown(w: pd.DataFrame) -> None:
    """summary.md: one table per method plus the DiSheafFRL - non-coop Comm gap."""
    shown = [
        'Private',
        'Private seen',
        'Comm',
        'Sender unseen / receiver seen',
        'Receiver seen',
        'Both seen',
    ]
    md = [
        '# Anchor-selection ablation: original vs proto_kmeans vs stratified',
        '',
        '4-agent MNIST (7 classes/agent, fully connected), lambda .01, CenterNorm, bs256, 100 epochs. Accuracy %,',
        'mean ± sample SD over seeds, pooled over the 12 directed edges (i = receiver, j = sender), rescored offline',
        'from final checkpoints (overall Comm asserted vs remote W&B, 2e-3). `n` = number of seeds. Per-run numbers:',
        '`per_run.csv`; wide stats: `summary_wide.csv`; run ids: `provenance.md`.',
        '',
    ]
    head = '| arm | K | n | Private | Private on C_i | Comm | Comm on C_i\\C_j | Comm on C_i | Comm on C_i∩C_j |'
    for m in ['Non-coop', 'DiSheafFRL']:
        md += [f'## {m}', '', head, '|' + '---|' * 9]
        for _, r in w[w.method == m].iterrows():
            arm = r.arm + (f' ({r.kmeans_fit})' if r.arm == 'kmeans' else '')
            md.append(
                f'| {arm} | {r.K} | {r.n_seeds} | '
                + ' | '.join(cell(r, n) for n in shown)
                + ' |'
            )
        md.append('')
    # cold k-means only for K=128 (the warm seed-42 row is a separate fit variant)
    c = w[~((w.arm == 'kmeans') & (w.kmeans_fit == 'warm') & (w.K == 128))]
    c = c.set_index(['method', 'arm', 'K'])['Comm mean'].unstack('method')
    md += [
        '## DiSheafFRL − non-coop gap (Comm, points)',
        '',
        '| arm | K=128 | K=384 | K=768 | K=1536 |',
        '|---|---|---|---|---|',
    ]
    for a in STRATEGIES:
        gaps = [
            f'{c.loc[(a, k), "DiSheafFRL"] - c.loc[(a, k), "Non-coop"]:+.2f}'
            if (a, k) in c.index
            else '—'
            for k in KS
        ]
        md.append(f'| {a} | ' + ' | '.join(gaps) + ' |')
    md += [
        '',
        '## Notes',
        '',
        '- The k-means DiSheafFRL arm mixes two fit variants: cold start at K=128/384, warm start (kmeans_warm_start=true) at',
        '  K=768/1536; a warm K=128 seed-42 row is listed separately.',
        '- Original K=384/768/1536 come from results/mnist_4agents_shift/swbn_scopes/per_run.csv; original K=128 is scored by',
        '  scripts/score_mnist4_anchors.py.',
        '- Original non-coop = checkpoints trained with K pilots; k-means/stratified non-coop = 4096-pilot checkpoints refit',
        '  with the K-anchor selection (private accuracy is therefore identical across those two arms).',
        '- Single-seed noise floor ~±1.5 pt (3-seed SDs here are far below it).',
    ]
    (OUT / 'summary.md').write_text('\n'.join(md) + '\n')


def write_provenance(runs: pd.DataFrame) -> None:
    """provenance.md: the W&B run behind every scored row."""
    lines = [
        'Projects: mnist_4agents_shift_anchors_{kmeans,strat} (DiSheafFRL; groups K<K>, K<K>_warm = warm-start k-means),',
        'mnist_4agents_shift (original K=128..1536: cnorm_p128_l.01 / pilot-study groups; non-coop 4096-pilot sources',
        'mnist4_di_clean_bs256_check_non_cooperative_p4096 rescored under each (arm, K)). Rows: method, arm, K, seed -> source run.',
        '',
    ]
    lines += [
        f'- {r.method}, {r.arm}, K={r.K}, seed {r.seed} ({r.group}): {r.source_run}'
        for r in runs.itertuples()
    ]
    (OUT / 'provenance.md').write_text('\n'.join(lines) + '\n')


def plot(w: pd.DataFrame) -> None:
    """Both figures, house style with the compact overrides of the pilot-study figures."""
    # k-means K=128: cold start only (the warm seed-42 point is a separate fit variant)
    w = w[~((w.arm == 'kmeans') & (w.kmeans_fit == 'warm') & (w.K == 128))]
    use_style()
    plt.rcParams.update(
        {
            'text.usetex': False,
            'font.size': 9,
            'axes.titlesize': 9,
            'axes.labelsize': 9,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'legend.fontsize': 8,
            'lines.linewidth': 1.2,
            'lines.markersize': 3,
        }
    )
    for stem, panels in FIGURES.items():
        fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5))
        for ax, (scope, title) in zip(axes, panels):
            for arm, (_, color) in STRATEGIES.items():
                for method, ls in [('Non-coop', ':'), ('DiSheafFRL', '-')]:
                    d = w[(w.method == method) & (w.arm == arm)].sort_values(
                        'K'
                    )
                    mean = d[f'{scope} mean'].to_numpy()
                    sd = d[f'{scope} sd'].fillna(0).to_numpy()
                    ax.errorbar(
                        d.K,
                        mean,
                        # Accuracies cannot exceed 100%: clip the upper whisker there.
                        yerr=np.stack([sd, np.minimum(sd, 100 - mean)]),
                        marker='o',
                        capsize=2,
                        color=color,
                        linestyle=ls,
                    )
            ax.set_xscale('log')
            ax.set_xticks(
                KS,
                [str(k) for k in KS],
                rotation=45,
                ha='right',
                rotation_mode='anchor',
            )
            ax.minorticks_off()
            ax.set(xlabel='Pilots / anchors $K$', title=title)
            ax.grid(axis='y', alpha=0.3, linewidth=0.5)
        axes[0].set_ylabel('Accuracy (%)')
        handles = [
            Line2D([], [], color=c, marker='o', label=n)
            for n, c in STRATEGIES.values()
        ]
        handles += [
            Line2D([], [], color='black', linestyle='-', label='DiSheafFRL'),
            Line2D([], [], color='black', linestyle=':', label='Non-coop'),
        ]
        fig.legend(
            handles=handles,
            loc='upper center',
            ncol=5,
            frameon=False,
            bbox_to_anchor=(0.5, 1.0),
            columnspacing=1.0,
            handlelength=1.6,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.9))
        for ext in ['png', 'pdf']:
            path = OUT / f'{stem}.{ext}'
            fig.savefig(path, dpi=300, bbox_inches='tight')
            print('wrote', path.relative_to(ROOT))
        plt.close(fig)


def main() -> None:
    """Aggregate the scored runs and write the tables and figures."""
    runs = load_runs()
    runs.to_csv(OUT / 'per_run.csv', index=False)
    w = summarise(runs)
    w.to_csv(OUT / 'summary_wide.csv', index=False, float_format='%.4f')
    write_markdown(w)
    write_provenance(runs)
    plot(w)
    print(len(runs), 'runs;', len(w), 'summary rows')


if __name__ == '__main__':
    main()
