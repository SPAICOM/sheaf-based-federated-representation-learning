"""Heterogeneity table: every method on the four 4-agent shift-1.0 setups.

Reads the finished runs of ``just hetero <dataset>`` from the wandb remote
(one project per dataset, one group per method) and tabulates communication
and private test accuracy, mean and standard deviation over seeds.

Writes results/paper/hetero_table.{md,tex}.
"""

from pathlib import Path

import pandas as pd
import wandb

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'paper'
# column label -> wandb project (= experiment config name)
DATASETS = {
    'MNIST': 'mnist_4agents_shift',
    'FMNIST': 'fashionmnist_4agents_shift',
    'FEMNIST': 'femnist_4agents_shift',
    'CIFAR-10': 'cifar10_4agents_full',
}
# config name (= wandb group) -> row label, in table order
METHODS = {
    'non_cooperative': 'Non-coop',
    'sheaf_fmtl': 'Sheaf-FMTL',
    'comfed': 'ComFed',
    'fedproto': 'FedProto',
    'fedmuscle': 'FedMuscle',
    'fedre': 'FedRE',
    'sheaf_frl': 'SheafFRL',
    'di_sheaf_frl': 'DiSheafFRL',
}
METRICS = {'comm': 'Comm.', 'private': 'Private'}


def load_runs() -> pd.DataFrame:
    """One row per finished run: dataset, method, seed, accuracies in %."""
    api = wandb.Api()
    existing = {p.name for p in api.projects(api.default_entity)}
    rows = []
    for dataset, project in DATASETS.items():
        if project not in existing:
            print(f'{api.default_entity}/{project} does not exist yet: {dataset} column left empty')
            continue
        for run in api.runs(f'{api.default_entity}/{project}', filters={'state': 'finished'}):
            if run.group not in METHODS:
                continue
            rows.append(
                {
                    'dataset': dataset,
                    'method': run.group,
                    'seed': run.config['seed'],
                    'created': run.created_at,
                    **{k: 100 * run.summary[f'test/avg_{k}_task_perf'] for k in METRICS},
                }
            )
    if not rows:
        raise SystemExit('no finished runs with a method group in ' + ', '.join(DATASETS.values()))
    runs = pd.DataFrame(rows)
    # a re-run of the same cell replaces the older one
    return runs.sort_values('created').drop_duplicates(['dataset', 'method', 'seed'], keep='last')


def cell(values: pd.Series) -> str:
    """'mean ± std (n)' with the seed count shown only when it is not 3."""
    text = f'{values.mean():.1f} ± {values.std():.1f}' if len(values) > 1 else f'{values.mean():.1f}'
    return text if len(values) == 3 else f'{text} (n={len(values)})'


def to_markdown(table: pd.DataFrame) -> str:
    """Pipe table with one header row 'dataset metric'."""
    header = ['method'] + [f'{d} {m}' for d, m in table.columns]
    lines = ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    lines += ['| ' + ' | '.join([idx, *row]) + ' |' for idx, row in zip(table.index, table.values, strict=True)]
    return '\n'.join(lines) + '\n'


def main() -> None:
    """Build the method x (dataset, metric) table and write it as Markdown and LaTeX."""
    runs = load_runs()
    columns = pd.MultiIndex.from_product([list(DATASETS), list(METRICS.values())])
    table = pd.DataFrame('–', index=list(METHODS.values()), columns=columns)
    for (dataset, method), group in runs.groupby(['dataset', 'method']):
        for key, label in METRICS.items():
            table.loc[METHODS[method], (dataset, label)] = cell(group[key])

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'hetero_table.md').write_text(to_markdown(table))
    (OUT / 'hetero_table.tex').write_text(table.to_latex(multicolumn_format='c').replace('±', r'$\pm$'))
    print(table.to_string())
    print(f'wrote {OUT}/hetero_table.{{md,tex}}')


if __name__ == '__main__':
    main()
