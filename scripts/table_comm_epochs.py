"""Paper table tab:comm-epochs: DiSheafFRL communicating on a share rho of the epochs (MNIST).

Rows: rho = 0% (non-cooperative), the CEDiSheafFRL runs of ``just ablation-comm-epochs`` (wandb
project mnist_4agents_shift, groups mnist4_ce_di_comm<rho>) and rho = 100% (DiSheafFRL). Columns:
training traffic in MB (the logged test/train_communication_kilobytes_cumulative / 1024, which
includes the one-off pilot exchange of the evaluation maps), the accuracies of tab:main and the
alignment diagnostics of tab:geom (MRR, TV, mu_min), mean and s.d. over seeds, ranked as in the
paper (src/utils/paper_tables.py): traffic, private accuracy and mu_min are not marked.

Accuracies and diagnostics come from results/mnist_4agents_shift/rescore/runs.csv
(scripts/rescore_hetero.py). The 0% and 100% rows are the MNIST non_cooperative / di_sheaf_frl
runs of tab:main when they are re-scored on this machine; otherwise the values printed in
tab:main and tab:geom are used (PAPER_BASELINES), and the script says so.

Writes results/mnist_4agents_shift/comm_epochs/{rows.tex,table.md,runs.csv}; rows.tex holds the
row bodies of the paper's table.

    uv run python scripts/table_comm_epochs.py [--shares 2,5,10,20,40,60,80]
"""

import argparse
import math
import sys
from pathlib import Path

import pandas as pd
import wandb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.paper_tables import (  # noqa: E402
    md_cell,
    mean_sd,
    rank_groups,
    sd_text,
    tex_cell,
)

PROJECT = 'mnist_4agents_shift'
GROUP = 'mnist4_ce_di_comm'
KB_KEY = 'test/train_communication_kilobytes_cumulative'
OUT = ROOT / 'results' / PROJECT / 'comm_epochs'

# column -> (markdown header, higher is better or None when not ranked)
COLS = {
    'mb': ('Traffic (MB)', None),
    'private_seen': ('Priv. C_i', None),
    'comm_rx_seen': ('Comm. C_i', True),
    'comm_rx_only': ('Comm. C_i∖C_j', True),
    'comm_both_seen': ('Comm. C_i∩C_j', True),
    'mrr': ('MRR ↑', True),
    'tv': ('TV ↓', False),
    'lmin': ('μ_min ↓', None),
}
PERCENT = {
    'private_seen',
    'comm_rx_seen',
    'comm_rx_only',
    'comm_both_seen',
    'mrr',
}
SD_DEC = 3  # decimals of the s.d. of the comm-efficient rows

# rho = 0% / 100% as printed in tab:main and tab:geom (MNIST), used when their runs are not
# re-scored here: (mean, s.d., s.d. decimals). Traffic: the logged KB of those 3-seed runs.
PAPER_BASELINES = {
    0: {
        'mb': (17408 / 1024, 0.0, 0),
        'private_seen': (99.6, 0.01, 2),
        'comm_rx_seen': (93.8, 0.17, 2),
        'comm_rx_only': (84.8, 0.47, 2),
        'comm_both_seen': (99.4, 0.04, 2),
        'mrr': (10.5, 0.32, 2),
        'tv': (3.70, 0.042, 3),
        'lmin': (1.88e-3, 0.32e-3, 2),
    },
    100: {
        'mb': (9229141.33 / 1024, 31409.4 / 1024, 0),
        'private_seen': (99.6, 0.03, 2),
        'comm_rx_seen': (99.2, 0.02, 2),
        'comm_rx_only': (99.1, 0.09, 2),
        'comm_both_seen': (99.3, 0.03, 2),
        'mrr': (33.4, 0.56, 2),
        'tv': (0.92, 0.014, 3),
        'lmin': (3.16e-6, 0.26e-6, 2),
    },
}
BASELINE_GROUPS = {0: 'non_cooperative', 100: 'di_sheaf_frl'}


def load(shares: list[int]) -> pd.DataFrame:
    """Load one row per run: rho, seed, run id, traffic and the re-scored values.

    The rho = 0 / 100 rows are included when their runs are re-scored on this machine.
    """
    path = ROOT / 'results' / PROJECT / 'rescore' / 'runs.csv'
    if not path.exists():
        raise SystemExit(
            f'{path.relative_to(ROOT)} missing: run `just table-comm-epochs` (it re-scores first)'
        )
    runs = pd.read_csv(path)
    rho = {f'{GROUP}{p}': p for p in shares} | {
        g: p for p, g in BASELINE_GROUPS.items()
    }
    runs = runs[runs.group.isin(rho)].copy()
    runs['rho'] = runs.group.map(rho)
    api = wandb.Api()
    runs['mb'] = [
        api.run(f'{api.default_entity}/{PROJECT}/{i}').summary.get(
            KB_KEY, math.nan
        )
        / 1024
        for i in runs.run_id
    ]
    for c in PERCENT:
        runs[c] = 100 * runs[c]
    return runs.sort_values(['rho', 'seed'])


def fmt(col: str, m: float, s: float, dec: int) -> str:
    """LaTeX math body of one cell."""
    if (
        col == 'mb'
    ):  # integer MB; no s.d. when the traffic is the same in every seed
        return f'{m:.0f}' if not s >= 0.5 else f'{m:.0f}_{{\\pm {s:.0f}}}'
    if col == 'tv':
        return f'{m:.2f}_{{\\pm {sd_text(s, dec)}}}'
    if col == 'lmin':  # mantissa x 10^e, the s.d. on the same exponent
        e = math.floor(math.log10(abs(m)))
        return f'{m / 10**e:.2f}_{{\\pm {sd_text(s / 10**e, dec)}}}{{\\times}}10^{{{e}}}'
    return f'{m:.1f}_{{\\pm {sd_text(s, dec)}}}'


def main() -> None:
    """Write the rows of tab:comm-epochs."""
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument(
        '--shares',
        default='2,5,10,20,40,60,80',
        help='comm-epoch shares (%%) of the table',
    )
    a = p.parse_args()
    shares = [int(x) for x in a.shares.split(',')]
    runs = load(shares)

    table = []  # (rho, {col: (mean, sd, decimals)})
    for rho in [0, *shares, 100]:
        g = runs[runs.rho == rho]
        if g.empty and rho in PAPER_BASELINES:
            print(
                f'rho = {rho}%: {BASELINE_GROUPS[rho]} runs not re-scored here, '
                'using the values printed in tab:main / tab:geom'
            )
            table.append((rho, PAPER_BASELINES[rho]))
        elif g.empty:
            print(f'rho = {rho}%: no finished, re-scored run, row left out')
        else:
            if len(g) < 3:
                print(f'rho = {rho}%: {len(g)} seed(s) only')
            table.append((rho, {c: (*mean_sd(g[c]), SD_DEC) for c in COLS}))

    ranks = {
        c: (
            rank_groups([st[c][:2] for _, st in table], higher)
            if higher is not None
            else [None] * len(table)
        )
        for c, (_, higher) in COLS.items()
    }
    name = {0: r'0\% (Non-coop)', 100: r'100\% (DiSheaf-FRL)'}
    tex, md = (
        [],
        [
            '# tab:comm-epochs: DiSheafFRL communicating on a share ρ of the 100 epochs (MNIST)',
            '',
            'Mean ± s.d. over seeds. Traffic: training payload in MB, including the one-off 17 MB pilot exchange '
            'of the evaluation maps. Ranked columns: **best group**, <u>second</u>, *third*.',
            '',
            '| ρ | ' + ' | '.join(h for h, _ in COLS.values()) + ' |',
            '|---|' + '---:|' * len(COLS),
        ],
    )
    for k, (rho, st) in enumerate(table):
        cells = [tex_cell(fmt(c, *st[c]), ranks[c][k]) for c in COLS]
        if rho == 100:
            tex.append(r'\midrule')
        tex.append(
            f'{name.get(rho, f"{rho}\\%")} & ' + ' & '.join(cells) + r' \\'
        )
        if rho == 0:
            tex.append(r'\midrule')
        md_cells = [
            md_cell(
                fmt(c, *st[c])
                .replace('_{\\pm ', ' ± ')
                .replace('{\\times}10^{', ' ×1e')
                .replace('}', ''),
                ranks[c][k],
            )
            for c in COLS
        ]
        md.append(f'| {rho}% | ' + ' | '.join(md_cells) + ' |')
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'rows.tex').write_text('\n'.join(tex) + '\n')
    (OUT / 'table.md').write_text('\n'.join(md) + '\n')
    runs.to_csv(OUT / 'runs.csv', index=False)
    print('\n'.join(md))
    print(f'wrote {OUT}/rows.tex, table.md, runs.csv')


if __name__ == '__main__':
    main()
