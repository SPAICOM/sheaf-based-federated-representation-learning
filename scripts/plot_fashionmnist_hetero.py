"""Promoted summary of the Fashion-MNIST heterogeneity table (4 agents, shift 1.0, 40 epochs).

Pulls every *finished* run of the W&B project fashionmnist_4agents_shift (crashed runs of the first,
interrupted launch are ignored) and writes results/fashionmnist_4agents_shift/:

  summary.csv   one row per run: headline test metrics + pooled class-scope metrics
  table.csv/md  final table, seeds 42/43/44 at the tuned lambda: comm acc, gain over non-coop (paired by
                seed), class scopes, private accuracy, MRR, misalignment, communication cost, run ids
  grid.csv      lambda grid on seed 42 (comm acc %) used to pick the tuned lambda (selection uses test acc)

Groups: <method> (3 seeds at the tuned lambda; non-coop / FedRE are lambda-free) and <method>_lambda
(seed-42 grid). Launchers: tmp/2026-11-13 hetero pipeline (just hetero-pipeline fashionmnist ...) and
tmp/2026-11-14_fashionmnist_noncoop.sh. Sources: LEDGER rows fashionmnist_4agents_shift.

Usage: uv run python scripts/plot_fashionmnist_hetero.py
"""

from pathlib import Path

import pandas as pd

import wandb

ROOT = Path(__file__).resolve().parents[1]
PROJECT = 'fashionmnist_4agents_shift'
OUT = ROOT / 'results' / PROJECT
SEEDS = [42, 43, 44]
NONCOOP = 'NonCooperativeLearning'
SCOPE = 'test/scope/'
HEADLINE = [
    'test/avg_comm_task_perf',
    'test/avg_private_task_perf',
    'test/mean_reciprocal_rank',
    'test/misalignment_energy',
    'train/communication_kilobytes',
]
SCOPE_KEYS = [
    SCOPE + k
    for k in (
        'comm_all',
        'comm_both_seen',
        'comm_tx_unseen_rx_seen',
        'comm_tx_seen_rx_unseen',
        'comm_both_unseen',
        'private_seen',
        'private_unseen',
    )
]
COLUMNS = {
    'test/avg_comm_task_perf': 'comm',
    SCOPE + 'comm_both_seen': 'both_seen',
    SCOPE + 'comm_tx_unseen_rx_seen': 'txU_rxS',
    'test/avg_private_task_perf': 'private',
    SCOPE + 'private_seen': 'priv_target',
    'test/mean_reciprocal_rank': 'MRR',
    'test/misalignment_energy': 'misal',
}


def fetch():
    """All finished runs of the project, one row each (headline + scope metrics)."""
    api = wandb.Api()
    proj = f'{api.default_entity}/{PROJECT}'
    rows = []
    for stub in api.runs(proj):
        if stub.state != 'finished':
            continue
        r = api.run(f'{proj}/{stub.id}')  # listings carry empty configs
        orch = r.config.get('orchestrator') or {}
        row = {
            'run': r.id,
            'group': r.group,
            'method': str(orch.get('_target_', '')).split('.')[-1],
            'lambda': orch.get('max_lmb'),
            'seed': r.config.get('seed'),
            'epoch': r.summary.get('epoch'),
        }
        for k in HEADLINE + SCOPE_KEYS:
            row[k] = r.summary.get(k)
        rows.append(row)
    return (
        pd.DataFrame(rows)
        .sort_values(['group', 'lambda', 'seed'], na_position='first')
        .reset_index(drop=True)
    )


def main():
    """Write summary.csv, table.csv/md and grid.csv."""
    OUT.mkdir(parents=True, exist_ok=True)
    runs = fetch()
    assert (runs.epoch == 40).all(), runs[runs.epoch != 40][['run', 'epoch']]
    runs.to_csv(OUT / 'summary.csv', index=False)

    final = runs[~runs.group.str.endswith('_lambda')]
    for m, g in final.groupby('method'):
        assert sorted(g.seed) == SEEDS, (m, sorted(g.seed))
        assert g['lambda'].nunique() == 1, (m, g['lambda'].unique())

    g = final.groupby('method')
    t = g[list(COLUMNS)].mean().rename(columns=COLUMNS)
    for c in ('comm', 'both_seen', 'txU_rxS', 'private', 'priv_target'):
        t[c] *= 100
    paired = (
        final.pivot_table(
            index='seed',
            columns='method',
            values='test/avg_comm_task_perf',
        )
        * 100
    )
    gain = paired.sub(paired[NONCOOP], axis=0)
    t.insert(0, 'comm_sd', g['test/avg_comm_task_perf'].std() * 100)
    t.insert(0, 'lambda', g['lambda'].first())
    t.insert(3, 'gain_vs_noncoop', gain.mean().reindex(t.index))
    t.insert(4, 'gain_sd', gain.std().reindex(t.index))
    t['MB'] = g['train/communication_kilobytes'].first() / 1024
    t['run_ids'] = g.apply(
        lambda x: ' '.join(f'{s}:{r}' for s, r in zip(x.seed, x.run))
    )
    t = t.sort_values('comm', ascending=False)
    t.index.name = 'method'
    t.round(3).to_csv(OUT / 'table.csv')

    grid = runs[runs.group.str.endswith('_lambda')].pivot_table(
        index='method', columns='lambda', values='test/avg_comm_task_perf'
    )
    for (m, lmb), v in (
        final[final.seed == 42]
        .set_index(['method', 'lambda'])['test/avg_comm_task_perf']
        .items()
    ):
        grid.loc[m, lmb] = v
    grid = (grid.sort_index(axis=1) * 100).round(2)
    grid.to_csv(OUT / 'grid.csv')

    shown = t.drop(columns='run_ids').round(3)
    (OUT / 'table.md').write_text(
        'Fashion-MNIST, 4 agents, shift 1.0, 40 epochs. Seeds 42/43/44 at the tuned lambda '
        '(chosen on seed-42 test accuracy); %. gain = comm minus non-coop (pts), paired by seed, '
        'SD over seeds. Scope columns: both_seen = both agents trained the class; txU_rxS = sender '
        "unseen, receiver seen; priv_target = private accuracy on the agent's own target classes; "
        'MB = training communication.\n\n```\n'
        + shown.to_string()
        + '\n```\n\nLambda grid, seed 42, comm acc %:\n\n```\n'
        + grid.to_string()
        + '\n```\n\nRun IDs (seed:run): '
        + '; '.join(f'{m}: {r}' for m, r in t.run_ids.items())
        + '\n'
    )
    print((OUT / 'table.md').read_text())


if __name__ == '__main__':
    main()
