"""Summary table for the 4-agent FashionMNIST pilot-count sweep (non-coop vs
DiSheafFRL, fixed lambda, 3 seeds): config/hydra/fashionmnist_4agents_shift.yaml,
scripts/run_fashionmnist_4agents_shift_sweep.sh.

Pulls every finished run of the sweep's wandb group from the cloud API, groups
by (dataset.pilot_num_samples, orchestrator_name), and prints/writes:

  1. pilot budget x method table of test/avg_private_task_perf and
     test/avg_comm_task_perf (mean +/- std over seeds), the DiSheafFRL -
     non-coop comm gap, and n finished seeds per cell (missing runs show as
     n < 3 instead of silently averaging over fewer runs).
  2. per-agent private/comm accuracy at each pilot budget (mean over seeds,
     one row per agent x method x budget).

Usage:
    uv run python scripts/fashionmnist_4agents_shift_summary.py
    uv run python scripts/fashionmnist_4agents_shift_summary.py --entity my-team
    uv run python scripts/fashionmnist_4agents_shift_summary.py --project other_proj --group "other group"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ENTITY = 'leonardo-dinino-sapienza-universit-di-roma'  # matches logger.entity
                                                        # in the fashionmnist
                                                        # config (mle-spaicom,
                                                        # the wandb default
                                                        # entity, rejects
                                                        # project creation)
PROJECT = 'fashionmnist_4agents_shift'
GROUP = 'FashionMNIST 4 agents shift 1.0 x pilots (non-coop vs DiSheafFRL)'
N_AGENTS = 4
PILOTS = [384, 768, 2048, 4096]
# orchestrator_name values: the orchestrator's CLASS name (from its _target_),
# not the Hydra override string — 'orchestrator=non_cooperative' logs as
# 'NonCooperativeLearning'.
METHODS = ['NonCooperativeLearning', 'DiSheafFRL']
LABELS = {'NonCooperativeLearning': 'non-coop', 'DiSheafFRL': 'DiSheafFRL'}  # display names
OUT_DIR = ROOT / 'results/fashionmnist_4agents_shift'


def _config_value(config: dict, key: str):
    """Dotted-key lookup into a wandb config, whatever way Hydra's dict was
    flattened ('a.b.c', 'a/b/c', or nested)."""
    for k in (key, key.replace('.', '/')):
        if k in config:
            return config[k]
    node = config
    for part in key.split('.'):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _run_row(run) -> dict | None:
    s, c = run.summary, run.config
    pilots = _config_value(c, 'dataset.pilot_num_samples')
    method = _config_value(c, 'orchestrator_name')
    seed = _config_value(c, 'seed')
    if pilots is None or method is None or seed is None:
        return None
    out = {
        'run_id': run.id,
        'pilots': int(pilots),
        'method': str(method),
        'seed': int(seed),
        'avg_private': s.get('test/avg_private_task_perf'),
        'avg_comm': s.get('test/avg_comm_task_perf'),
    }
    for i in range(N_AGENTS):
        out[f'private_agent_{i}'] = s.get(f'test/private_task_perf_agent_{i}')
        out[f'comm_agent_{i}'] = s.get(f'test/comm_task_perf_agent_{i}')
    return out


def fetch_runs(api, entity: str, project: str, group: str) -> pd.DataFrame:
    runs = api.runs(
        f'{entity}/{project}',
        filters={'group': group, 'state': 'finished'},
    )
    rows = [r for r in (_run_row(run) for run in runs) if r is not None]
    if not rows:
        return pd.DataFrame(
            columns=['run_id', 'pilots', 'method', 'seed', 'avg_private', 'avg_comm']
        )
    return pd.DataFrame(rows)


def _mean_std(s: pd.Series) -> str:
    s = s.dropna()
    if s.empty:
        return ''
    if len(s) == 1:
        return f'{s.iloc[0]:.3f}'
    return f'{s.mean():.3f} ± {s.std():.3f}'


def build_main_table(df: pd.DataFrame) -> pd.DataFrame:
    """Rows: pilot budget. Columns: method. Cells: mean+-std over seeds of
    avg_private / avg_comm, plus the DiSheafFRL - non-coop comm gap and n."""
    rows = []
    for pilots in PILOTS:
        row = {'pilots': pilots}
        comm_means = {}
        for method in METHODS:
            cell = df[(df.pilots == pilots) & (df.method == method)]
            n = cell['seed'].nunique()
            label = LABELS[method]
            row[f'{label} private'] = _mean_std(cell['avg_private'])
            row[f'{label} comm'] = _mean_std(cell['avg_comm'])
            row[f'{label} n_seeds'] = n
            if not cell['avg_comm'].dropna().empty:
                comm_means[method] = cell['avg_comm'].mean()
        if set(METHODS) <= set(comm_means):
            row['DiSheafFRL - non_coop comm gap'] = round(
                comm_means['DiSheafFRL'] - comm_means['NonCooperativeLearning'], 4
            )
        else:
            row['DiSheafFRL - non_coop comm gap'] = ''
        rows.append(row)
    return pd.DataFrame(rows)


def build_per_agent_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (pilots, method, agent): mean +- std over seeds."""
    rows = []
    for pilots in PILOTS:
        for method in METHODS:
            cell = df[(df.pilots == pilots) & (df.method == method)]
            n = cell['seed'].nunique()
            for i in range(N_AGENTS):
                rows.append(
                    {
                        'pilots': pilots,
                        'method': LABELS[method],
                        'agent': i,
                        'private': _mean_std(cell[f'private_agent_{i}']),
                        'comm': _mean_std(cell[f'comm_agent_{i}']),
                        'n_seeds': n,
                    }
                )
    return pd.DataFrame(rows)


def to_markdown(t: pd.DataFrame) -> str:
    head = [
        '| ' + ' | '.join(map(str, t.columns)) + ' |',
        '|' + '---|' * len(t.columns),
    ]
    return '\n'.join(
        head + ['| ' + ' | '.join(map(str, r)) + ' |' for r in t.itertuples(index=False)]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--entity', default=ENTITY, help=f'wandb entity (default: {ENTITY}, matching logger.entity in the config)')
    parser.add_argument('--project', default=PROJECT)
    parser.add_argument('--group', default=GROUP)
    args = parser.parse_args()

    import wandb

    api = wandb.Api()
    entity = args.entity or api.default_entity
    if entity is None:
        raise SystemExit('No wandb entity available (pass --entity, or run `wandb login`).')

    print(f'Fetching finished runs from {entity}/{args.project} (group={args.group!r}) …')
    df = fetch_runs(api, entity, args.project, args.group)
    print(f'{len(df)} finished runs found.')
    if df.empty:
        print('Nothing to summarise yet.')
        return

    main_table = build_main_table(df)
    per_agent_table = build_per_agent_table(df)

    print('\n== Table 1: pilot budget x method (mean ± std over seeds) ==\n')
    print(to_markdown(main_table))
    print('\n== Table 2: per-agent private / comm accuracy ==\n')
    print(to_markdown(per_agent_table))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    main_table.to_csv(OUT_DIR / 'summary.csv', index=False)
    per_agent_table.to_csv(OUT_DIR / 'summary_per_agent.csv', index=False)
    print(f'\nWrote {OUT_DIR / "summary.csv"} and {OUT_DIR / "summary_per_agent.csv"}')


if __name__ == '__main__':
    main()
