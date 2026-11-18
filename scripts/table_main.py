r"""Paper tables tab:main (task accuracy) and tab:geom (alignment) of the four 4-agent benchmarks.

Reads results/<project>/rescore/runs.csv, written by scripts/rescore_hetero.py (one post-hoc
protocol for every method, from the final checkpoints), for

    CIFAR-10 cifar10_4agents_full   FEMNIST femnist_4agents_shift
    MNIST mnist_4agents_shift       Fashion-MNIST fashionmnist_4agents_shift

and writes the table rows (mean and s.d. over seeds, ranked as in the paper:
src/utils/paper_tables.py) to results/paper/:

    tab_main.tex / .md   per dataset: private on the agent's classes C_i; comm on the receiver's
                         classes C_i, on the classes unseen by the sender C_i\\C_j and on the
                         shared ones C_i∩C_j (ranked per dataset and column; private unmarked)
    tab_geom.tex / .md   per dataset: MRR, sheaf total variation TV and smallest Laplacian
                         eigenvalue mu_min (x1e-4) of the post-hoc maps; "--" for ComFed

The .tex files hold the row bodies only; the table environments and captions are in the paper.

    uv run python scripts/table_main.py
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402

sys.path.insert(0, str(ROOT))
from src.utils.paper_tables import (  # noqa: E402
    md_cell,
    mean_sd,
    rank_groups,
    sd_text,
    tex_cell,
)

OUT = ROOT / 'results' / 'paper'
DATASETS = {  # label -> wandb project; the paper prints them in two blocks of two
    'CIFAR-10': 'cifar10_4agents_full',
    'FEMNIST': 'femnist_4agents_shift',
    'MNIST': 'mnist_4agents_shift',
    'Fashion-MNIST': 'fashionmnist_4agents_shift',
}
BLOCKS = [('CIFAR-10', 'FEMNIST'), ('MNIST', 'Fashion-MNIST')]
METHODS = {  # wandb group -> row label, in the paper's order
    'non_cooperative': 'Non-coop',
    'fedre': 'FedRE',
    'fedproto': 'FedProto',
    'fedmuscle': 'FedMuscle',
    'comfed': 'ComFed',
    'sheaf_fmtl': 'Sheaf-FMTL',
    'sheaf_frl': 'Sheaf-FRL',
    'di_sheaf_frl': 'DiSheaf-FRL',
}
# column -> (header, higher is better, ranked); accuracies in %
MAIN = {
    'private_seen': ('Priv. C_i', True, False),
    'comm_rx_seen': ('Comm. C_i', True, True),
    'comm_rx_only': ('Comm. C_i∖C_j', True, True),
    'comm_both_seen': ('Comm. C_i∩C_j', True, True),
}
GEOM = {
    'mrr': ('MRR ↑', True, True),
    'tv': ('TV ↓', False, True),
    'lmin': ('μ_min (×1e-4) ↓', False, True),
}
SCALE = {
    'private_seen': 100,
    'comm_rx_seen': 100,
    'comm_rx_only': 100,
    'comm_both_seen': 100,
    'mrr': 100,
    'tv': 1,
    'lmin': 1e4,
}


def load() -> dict[str, pd.DataFrame]:
    """Per dataset, one row per (method, seed) with the rescored values."""
    out = {}
    for label, project in DATASETS.items():
        path = ROOT / 'results' / project / 'rescore' / 'runs.csv'
        if not path.exists():
            print(
                f'{label}: {path.relative_to(ROOT)} missing (`just tables-benchmarks` re-scores it), left empty'
            )
            continue
        df = pd.read_csv(path)
        df = df[df.group.isin(METHODS)]
        dup = df.duplicated(['group', 'seed'], keep='first')
        for _, r in df[dup].iterrows():
            print(
                f'{label}: {r.group} seed {r.seed}: extra run {r.run_id} ignored'
            )
        out[label] = df[~dup]
    return out


def stats(df: pd.DataFrame | None, col: str) -> list[tuple[float, float]]:
    """(mean, sd) per method in METHODS order, nan when missing."""
    if df is None or col not in df:
        return [(float('nan'), float('nan'))] * len(METHODS)
    return [mean_sd(df.loc[df.group == g, col] * SCALE[col]) for g in METHODS]


def fmt(col: str, m: float, s: float) -> str:
    """LaTeX math body of one cell."""
    sd = f'_{{\\pm {sd_text(s, 2)}}}' if s == s else ''
    if col == 'tv':
        sd = f'_{{\\pm {s:.2g}}}' if s == s else ''
        return f'{m:.2f}{sd}'
    if col == 'lmin':
        sd = f'_{{\\pm {s:.2g}}}' if s == s else ''
        return f'{m:.3g}{sd}'
    if col == 'mrr':
        sd = f'_{{\\pm {s:.2f}}}' if s == s else ''
    return f'{m:.1f}{sd}'


def md_fmt(col: str, m: float, s: float) -> str:
    """Markdown text of one cell."""
    body = fmt(col, m, s).replace('_{\\pm ', ' ± ').replace('}', '')
    return body


def build(
    data: dict, columns: dict, ds_order: list[str], skip_comfed: tuple = ()
) -> tuple[list[str], list[str]]:
    """LaTeX row lines and markdown lines of one block of datasets."""
    cells = {g: [] for g in METHODS}
    md_cells = {g: [] for g in METHODS}
    for ds in ds_order:
        df = data.get(ds)
        for col, (_, higher, ranked) in columns.items():
            st = stats(df, col)
            if (
                col in skip_comfed
            ):  # not defined for ComFed (no isometric maps)
                st = [
                    (float('nan'), float('nan')) if g == 'comfed' else x
                    for g, x in zip(METHODS, st)
                ]
            ranks = rank_groups(st, higher) if ranked else [None] * len(st)
            for g, (m, s), r in zip(METHODS, st, ranks):
                missing = m != m
                cells[g].append(
                    '--' if missing else tex_cell(fmt(col, m, s), r)
                )
                md_cells[g].append(
                    '–' if missing else md_cell(md_fmt(col, m, s), r)
                )
    tex = [f'{METHODS[g]} & ' + ' & '.join(cells[g]) + r' \\' for g in METHODS]
    md = [
        f'| {METHODS[g]} | ' + ' | '.join(md_cells[g]) + ' |' for g in METHODS
    ]
    return tex, md


def main() -> None:
    """Write tab:main and tab:geom."""
    data = load()
    OUT.mkdir(parents=True, exist_ok=True)
    note = (
        'Mean ± s.d. over seeds; per dataset and ranked column: **best group**, <u>second</u>, '
        '*third* (overlapping mean ± s.d. intervals).'
    )

    tex, md = [], ['# tab:main: task accuracy (%)', '', note, '']
    for block in BLOCKS:
        t, m = build(data, MAIN, list(block))
        tex += [
            f'% --- {" | ".join(block)}: '
            + ', '.join(h for h, _, _ in MAIN.values())
            + ' per dataset',
            *t,
        ]
        head = ['method'] + [
            f'{ds} {h}' for ds in block for h, _, _ in MAIN.values()
        ]
        md += [
            '| ' + ' | '.join(head) + ' |',
            '|---|' + '---:|' * (len(head) - 1),
            *m,
            '',
        ]
    (OUT / 'tab_main.tex').write_text('\n'.join(tex) + '\n')
    (OUT / 'tab_main.md').write_text('\n'.join(md))

    tex, md = (
        [],
        [
            '# tab:geom: alignment diagnostics',
            '',
            note + ' "–": not defined (ComFed).',
            '',
        ],
    )
    for ds in DATASETS:
        t, m = build(data, GEOM, [ds], skip_comfed=('tv', 'lmin'))
        tex += [
            f'% --- {ds}: ' + ', '.join(h for h, _, _ in GEOM.values()),
            *t,
        ]
        head = ['method'] + [f'{ds} {h}' for h, _, _ in GEOM.values()]
        md += [
            '| ' + ' | '.join(head) + ' |',
            '|---|' + '---:|' * (len(head) - 1),
            *m,
            '',
        ]
    (OUT / 'tab_geom.tex').write_text('\n'.join(tex) + '\n')
    (OUT / 'tab_geom.md').write_text('\n'.join(md))
    print((OUT / 'tab_main.md').read_text())
    print((OUT / 'tab_geom.md').read_text())
    print(f'wrote {OUT}/tab_{{main,geom}}.{{tex,md}}')


if __name__ == '__main__':
    main()
