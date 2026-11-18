"""Figures of the per-domain scores of scripts/domain_scope.py (one or more projects, several seeds).

Reads results/<project>/domain_scope.csv (one row per run, cell and test domain) and writes, to --out:

    lambda_delta.{pdf,png}        change vs non-coop (paired by seed, mean ± std) against lambda:
                                  comm and private, on the agent's own domain and on the unseen ones
    tradeoff.{pdf,png}            own-domain private (cost) vs comm over all test rows (gain),
                                  one point per method, mean ± std over seeds, joined in lambda order
    encdec_delta.{pdf,png}        encoder x decoder matrix of the best lambda minus non-coop, pooled
                                  over the test domains (diagonal: private, off-diagonal: comm)
    encdec_delta_by_domain_<project>.{pdf,png}
                                  the same matrix for each test domain (small multiples)

Heatmap cells show the paired change, mean ± std over seeds; a faded cell changes sign between
seeds. The best lambda is the tune_and_run.py rule: highest logged test/avg_comm_task_perf.

    uv run python scripts/plot_domain_scope.py domainnet_6agents domainnet_6agents_shift \
        --names "No label shift" "Label shift (7/10 classes)" --out results/domainnet_6agents_figures
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import (  # noqa: E402
    LinearSegmentedColormap,
    Normalize,
    to_rgb,
)
from matplotlib.patches import Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# Reference palette (dataviz skill, light mode); validated with scripts/validate_palette.js.
INK, INK2, MUTED = '#0b0b0b', '#52514e', '#898781'
GRID, AXIS, SURFACE = '#e1e0d9', '#c3c2b7', '#ffffff'
SERIES = [
    '#2a78d6',
    '#eb6834',
    '#1baf7a',
    '#eda100',
]  # blue, orange, aqua, yellow (fixed order)
DIVERGING = LinearSegmentedColormap.from_list(
    'red_gray_blue',
    [
        (0.0, '#8a2020'),
        (0.25, '#e34948'),
        (0.45, '#f5c4c3'),
        (0.5, '#f0efec'),
        (0.55, '#b7d3f6'),
        (0.75, '#2a78d6'),
        (1.0, '#104281'),
    ],
)

plt.rcParams.update(
    {
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 8,
        'axes.edgecolor': AXIS,
        'axes.labelcolor': INK2,
        'axes.titlesize': 9,
        'axes.titlecolor': INK,
        'xtick.color': MUTED,
        'ytick.color': MUTED,
        'xtick.labelcolor': INK2,
        'ytick.labelcolor': INK2,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'legend.frameon': False,
        'pdf.fonttype': 42,
        'figure.facecolor': SURFACE,
        'axes.facecolor': SURFACE,
    }
)


def load(project: str) -> pd.DataFrame:
    """Cells of one project (no @ pooled rows), with correct counts and the lambda of the run."""
    df = pd.read_csv(ROOT / 'results' / project / 'domain_scope.csv')
    df = df[~df.test_domain.str.startswith('@')].copy()
    df['correct'] = (df.acc.fillna(0) / 100 * df.n).round()
    df['lam'] = [
        0.0 if m == 'NonCooperativeLearning' else float(m.split('λ=')[1])
        for m in df.method
    ]
    return df


def agent_domains(df: pd.DataFrame) -> list[str]:
    """Domain of every agent, in agent order."""
    p = (
        df[df.kind == 'private']
        .drop_duplicates('receiver')
        .sort_values('receiver')
    )
    return list(p.receiver_domain)


def _pool(g: pd.DataFrame) -> float:
    n = g.n.sum()
    return 100 * g.correct.sum() / n if n else np.nan


def run_metrics(r: pd.DataFrame) -> dict[str, float]:
    """Return the own vs unseen domain accuracies of one run, and its comm over every test row.

    Private and comm (agent as receiver) are pooled per agent, then averaged over the agents;
    comm over every test row is averaged over the directed pairs.
    """
    out = {}
    for kind in ('private', 'comm'):
        k = r[r.kind == kind]
        for own, tag in ((True, 'own'), (False, 'unseen')):
            out[f'{kind}_{tag}'] = (
                k[k.own_domain == own]
                .groupby('receiver')
                .apply(_pool, include_groups=False)
                .mean()
            )
    c = r[r.kind == 'comm']
    out['comm_all'] = (
        c.groupby(['sender', 'receiver'])
        .apply(_pool, include_groups=False)
        .mean()
    )
    return out


def per_run(df: pd.DataFrame) -> pd.DataFrame:
    """One row per run: lambda, seed, logged comm and the run_metrics."""
    rows = [
        {
            'run_id': rid,
            'lam': g.lam.iloc[0],
            'seed': g.seed.iloc[0],
            'logged': g.logged_comm.iloc[0],
            **run_metrics(g),
        }
        for rid, g in df.groupby('run_id')
    ]
    return pd.DataFrame(rows)


def best_lam(runs: pd.DataFrame) -> float:
    """Lambda with the highest mean logged test comm (the tune_and_run.py rule)."""
    m = runs[runs.lam > 0].groupby('lam').logged.mean()
    return float(m.idxmax())


def matrices(
    r: pd.DataFrame, n: int, by_domain: bool
) -> dict[str, np.ndarray]:
    """{test domain or 'all': n x n accuracy, [encoder, decoder]} of one run."""
    keys = ['sender', 'receiver'] + (['test_domain'] if by_domain else [])
    pooled = r.groupby(keys).apply(_pool, include_groups=False)
    out = {}
    for key, v in pooled.items():
        s, rc, d = (
            (key[0], key[1], key[2]) if by_domain else (key[0], key[1], 'all')
        )
        out.setdefault(d, np.full((n, n), np.nan))[int(s), int(rc)] = v
    return out


def paired_delta(
    df: pd.DataFrame, lam: float, by_domain: bool
) -> dict[str, tuple]:
    """{domain: (mean, std, consistent)} of (method at lam) - non-coop, paired by seed."""
    n = len(agent_domains(df))
    per_seed = {}
    for seed in sorted(set(df.seed)):
        a, b = (
            df[(df.lam == lam) & (df.seed == seed)],
            df[(df.lam == 0) & (df.seed == seed)],
        )
        if a.empty or b.empty:
            continue
        ma, mb = matrices(a, n, by_domain), matrices(b, n, by_domain)
        per_seed[seed] = {d: ma[d] - mb[d] for d in ma}
    out = {}
    for d in next(iter(per_seed.values())):
        x = np.stack([v[d] for v in per_seed.values()])
        cons = (
            (x > 0).all(0) | (x < 0).all(0)
            if len(x) > 1
            else np.ones(x.shape[1:], bool)
        )
        out[d] = (
            x.mean(0),
            x.std(0, ddof=1) if len(x) > 1 else np.full(x.shape[1:], np.nan),
            cons,
        )
    return out, len(per_seed)


def save(fig, out: Path, name: str) -> None:
    """Write the figure as PDF and PNG and close it."""
    for ext in ('pdf', 'png'):
        fig.savefig(out / f'{name}.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out / name}.pdf/.png')


def lam_ticks(lams):
    """X tick labels: NC, then the lambdas."""
    return ['NC'] + [f'{x:g}' for x in lams[1:]]


def fig_lambda_delta(data, names, out):
    """Change vs NC against lambda, own vs unseen domains, private and comm."""
    series = [
        ('comm_own', 'comm, own domain'),
        ('comm_unseen', 'comm, unseen domains'),
        ('private_unseen', 'private, unseen domains'),
        ('private_own', 'private, own domain'),
    ]
    fig, axes = plt.subplots(
        1, len(data), figsize=(3.45 * len(data), 2.9), sharey=True
    )
    axes = np.atleast_1d(axes)
    for ax, (proj, (df, runs)), name in zip(axes, data.items(), names):
        lams = sorted(runs.lam.unique())
        x = np.arange(len(lams))
        base = runs[runs.lam == 0].set_index('seed')
        ax.axhline(0, color=AXIS, lw=0.8, zorder=1)
        ax.grid(axis='y', color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for (key, label), color in zip(series, SERIES):
            m, s = [], []
            for lam in lams:
                cur = runs[runs.lam == lam].set_index('seed')
                d = (cur[key] - base[key]).dropna()
                m.append(d.mean())
                s.append(d.std(ddof=1) if len(d) > 1 else 0.0)
            m, s = np.array(m), np.array(s)
            ax.fill_between(x, m - s, m + s, color=color, alpha=0.12, lw=0)
            ax.plot(
                x,
                m,
                color=color,
                lw=1.6,
                solid_capstyle='round',
                label=label,
                zorder=3,
            )
            ax.plot(
                x, m, 'o', color=color, ms=5, mec=SURFACE, mew=1.2, zorder=4
            )
            ax.annotate(
                label,
                (x[-1], m[-1]),
                xytext=(7, 0),
                textcoords='offset points',
                va='center',
                fontsize=7,
                color=INK2,
                annotation_clip=False,
            )
        ax.set_xticks(x, lam_ticks(lams))
        ax.set_xlabel('λ (DiSheafFRL); NC = non-cooperative')
        ax.set_title(name, loc='left')
        n = runs.groupby('lam').size().min()
        ax.text(
            0,
            -0.24,
            f'mean ± std over {n} seeds, paired by seed',
            transform=ax.transAxes,
            fontsize=6.5,
            color=MUTED,
        )
    axes[0].set_ylabel('change vs NC (accuracy points)')
    fig.subplots_adjust(wspace=0.75)
    save(fig, out, 'lambda_delta')


def fig_tradeoff(data, names, out):
    """Own-domain private (cost) against comm over all test rows (gain), per lambda."""
    fig, axes = plt.subplots(1, len(data), figsize=(3.45 * len(data), 2.9))
    axes = np.atleast_1d(axes)
    for ax, (proj, (df, runs)), name in zip(axes, data.items(), names):
        g = runs.groupby('lam')
        mx, sx = g.private_own.mean(), g.private_own.std(ddof=1)
        my, sy = g.comm_all.mean(), g.comm_all.std(ddof=1)
        best = best_lam(runs)
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.plot(
            mx.values, my.values, color=SERIES[0], lw=1.0, alpha=0.5, zorder=2
        )
        for lam in mx.index:
            color = MUTED if lam == 0 else SERIES[0]
            ax.errorbar(
                mx[lam],
                my[lam],
                xerr=sx[lam],
                yerr=sy[lam],
                fmt='none',
                ecolor=color,
                elinewidth=0.8,
                capsize=0,
                alpha=0.8,
                zorder=3,
            )
            ax.plot(
                mx[lam],
                my[lam],
                'o',
                color=color,
                ms=6,
                mec=SURFACE,
                mew=1.2,
                zorder=4,
            )
            lab = (
                'NC'
                if lam == 0
                else f'λ{lam:g}' + (' (best)' if lam == best else '')
            )
            low = lam == min(
                x for x in mx.index if x > 0
            )  # sits next to the λ above it: label below
            ax.annotate(
                lab,
                (mx[lam], my[lam]),
                xytext=(6, -10 if low else 4),
                textcoords='offset points',
                fontsize=7,
                color=INK if lam == best else INK2,
                fontweight='bold' if lam == best else 'normal',
            )
        ax.set_xlabel('private accuracy, own domain (%)')
        ax.set_title(name, loc='left')
        ax.margins(x=0.18, y=0.15)
    axes[0].set_ylabel('comm accuracy, all test domains (%)')
    fig.text(
        0.0,
        -0.06,
        'mean ± std over seeds; mean over agents (x) and over the 30 directed pairs (y)',
        fontsize=6.5,
        color=MUTED,
    )
    fig.subplots_adjust(wspace=0.3)
    save(fig, out, 'tradeoff')


def _text_color(rgb) -> str:
    """White or near-black ink for a fill, by its luminance."""
    r, g, b = rgb
    return SURFACE if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.45 else INK


def draw_matrix(ax, mean, std, cons, doms, norm, fs=7.0, ticks=(True, True)):
    """Draw one encoder x decoder matrix of changes, mean ± sd, faded where the sign flips."""
    n = len(doms)
    gap = 0.06
    for i in range(n):
        for j in range(n):
            v = mean[i, j]
            if np.isnan(v):
                continue
            rgb = np.array(to_rgb(DIVERGING(norm(v))))
            faded = not cons[i, j]
            if faded:
                rgb = 0.35 * rgb + 0.65 * np.ones(3)
            ax.add_patch(
                Rectangle(
                    (j + gap / 2, i + gap / 2),
                    1 - gap,
                    1 - gap,
                    facecolor=rgb,
                    lw=0,
                )
            )
            tc = MUTED if faded else _text_color(rgb)
            weight = 'bold' if i == j else 'normal'
            ax.text(
                j + 0.5,
                i + 0.42,
                f'{v:+.1f}',
                ha='center',
                va='center',
                fontsize=fs,
                color=tc,
                fontweight=weight,
            )
            if not np.isnan(std[i, j]):
                ax.text(
                    j + 0.5,
                    i + 0.72,
                    f'±{std[i, j]:.1f}',
                    ha='center',
                    va='center',
                    fontsize=fs - 1.6,
                    color=tc,
                )
    ax.set_xlim(0, n)
    ax.set_ylim(n, 0)
    ax.set_aspect('equal')
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_xticks(
        np.arange(n) + 0.5, doms if ticks[1] else [], rotation=45, ha='right'
    )
    ax.set_yticks(np.arange(n) + 0.5, doms if ticks[0] else [])


def colorbar(fig, axes, norm, label):
    """Shared colorbar of the change vs NC."""
    sm = plt.cm.ScalarMappable(norm=norm, cmap=DIVERGING)
    cb = fig.colorbar(sm, ax=axes, shrink=0.7, pad=0.02, aspect=25)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, labelsize=7)
    cb.set_label(label, color=INK2, fontsize=7.5)
    return cb


def fig_encdec(data, names, out):
    """Plot the encoder x decoder change at the best lambda, pooled over test domains."""
    deltas = {}
    for proj, (df, runs) in data.items():
        lam = best_lam(runs)
        d, n_seeds = paired_delta(df, lam, by_domain=False)
        deltas[proj] = (lam, d['all'], n_seeds)
    vmax = max(np.nanmax(np.abs(m)) for _, (m, _, _), _ in deltas.values())
    norm = Normalize(-vmax, vmax)
    fig, axes = plt.subplots(1, len(data), figsize=(3.6 * len(data), 3.6))
    axes = np.atleast_1d(axes)
    for k, (ax, (proj, (df, runs)), name) in enumerate(
        zip(axes, data.items(), names)
    ):
        lam, (m, s, c), n_seeds = deltas[proj]
        doms = agent_domains(df)
        draw_matrix(ax, m, s, c, doms, norm, ticks=(k == 0, True))
        ax.set_title(f'{name}: λ{lam:g} − NC', loc='left')
        ax.set_xlabel('decoder (receiver)')
        if k == 0:
            ax.set_ylabel('encoder (sender)')
    colorbar(fig, axes, norm, 'change vs NC (accuracy points)')
    fig.text(
        0.02,
        -0.04,
        f'Cell: paired change, mean ± std over {n_seeds} seeds, pooled over all test domains. '
        'Diagonal (bold) = private; off-diagonal = comm. Faded: the sign differs between seeds.',
        fontsize=6.5,
        color=MUTED,
    )
    save(fig, out, 'encdec_delta')


def fig_encdec_by_domain(proj, df, runs, name, out):
    """Plot the encoder x decoder change at the best lambda, one panel per test domain."""
    lam = best_lam(runs)
    d, n_seeds = paired_delta(df, lam, by_domain=True)
    doms = agent_domains(df)
    order = [x for x in doms if x in d] + [x for x in d if x not in doms]
    vmax = max(np.nanmax(np.abs(d[x][0])) for x in order)
    norm = Normalize(-vmax, vmax)
    cols = 3
    rows = int(np.ceil(len(order) / cols))
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(10.5, 3.3 * rows),
        gridspec_kw={'hspace': 0.28, 'wspace': 0.12},
    )
    axes = np.atleast_2d(axes)
    for k, dom in enumerate(order):
        ax = axes[k // cols, k % cols]
        m, s, c = d[dom]
        draw_matrix(
            ax,
            m,
            s,
            c,
            doms,
            norm,
            fs=6.8,
            ticks=(k % cols == 0, k // cols == rows - 1),
        )
        ax.set_title(f'test images: {dom}', loc='left')
        if k % cols == 0:
            ax.set_ylabel('encoder (sender)')
        if k // cols == rows - 1:
            ax.set_xlabel('decoder (receiver)')
    for k in range(len(order), rows * cols):
        axes[k // cols, k % cols].axis('off')
    colorbar(
        fig, axes.ravel().tolist(), norm, 'change vs NC (accuracy points)'
    )
    fig.suptitle(
        f'{name}: λ{lam:g} − NC, by test domain',
        x=0.06,
        y=0.95,
        ha='left',
        fontsize=10,
        color=INK,
    )
    fig.text(
        0.06,
        0.0,
        f'Cell: paired change, mean ± std over {n_seeds} seeds. Diagonal (bold) = private; '
        'off-diagonal = comm. Faded: the sign differs between seeds.',
        fontsize=7,
        color=MUTED,
        va='top',
    )
    save(fig, out, f'encdec_delta_by_domain_{proj}')


def main() -> None:
    """Read the projects and write every figure."""
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('projects', nargs='+')
    p.add_argument(
        '--names', nargs='+', default=None, help='panel title per project'
    )
    p.add_argument('--out', required=True)
    a = p.parse_args()
    names = a.names or a.projects
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    data = {}
    for proj in a.projects:
        df = load(proj)
        data[proj] = (df, per_run(df))
    fig_lambda_delta(data, names, out)
    fig_tradeoff(data, names, out)
    fig_encdec(data, names, out)
    for proj, name in zip(a.projects, names):
        fig_encdec_by_domain(proj, *data[proj], name, out)


if __name__ == '__main__':
    main()
