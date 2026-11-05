"""6v6 grayblur + CenterNorm (no whitening): comm accuracy vs λ for the SheafFRL flavours,
and a best-λ table with the main metrics, fetched from the remote wandb project.

Flavours (wandb groups of cifar10_2agents_grayblur_6v6):
  SheafFRL, deer/dog pilots      lmb_cnorm_resnet            (non-coop ref: cnorm_resnet)
  SheafFRL, shared pilots        sharedpilots_cnorm_lmb      (non-coop ref: sharedpilots_cnorm_noncoop)
  DiSheafFRL, shared pilots      sharedpilots_cnorm_disheaf  (non-coop ref: sharedpilots_cnorm_noncoop)

7v7 (classes 0-6 vs 3-9, project cifar10_2agents_grayblur_7v7): non-coop and DiSheafFRL,
both with shared pilots (tmp/2026-11-03_7v7_cnorm_sharedpilots_disheaf.sh).

    uv run python scripts/plot_6v6_cnorm_study.py [--setup 7v7]
"""

import argparse

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import wandb

ROOT = Path(__file__).resolve().parents[1]
SETUPS = {  # project, accuracy cap, flavours (label, group, colour), non-coop refs (group: label, ls, colour)
    '6v6': ('cifar10_2agents_grayblur_6v6', 0.6, [
        ('SheafFRL, deer/dog pilots', 'lmb_cnorm_resnet', 'tab:blue'),
        ('SheafFRL, shared pilots', 'sharedpilots_cnorm_lmb', 'tab:orange'),
        ('DiSheafFRL, shared pilots', 'sharedpilots_cnorm_disheaf', 'tab:green'),
    ], {'cnorm_resnet': ('non-coop, deer/dog pilots', ':', 'tab:blue'),
        'sharedpilots_cnorm_noncoop': ('non-coop, shared pilots', '--', 'tab:gray')}),
    '7v7': ('cifar10_2agents_grayblur_7v7', 0.7, [
        ('DiSheafFRL, shared pilots', 'sharedpilots_cnorm_disheaf', 'tab:green'),
    ], {'sharedpilots_cnorm_noncoop': ('non-coop, shared pilots', '--', 'tab:gray')}),
}
PROJECT, CAP, FLAVOURS, NONCOOP = SETUPS['6v6']
METRICS = {  # column -> summary key(s); tuples are averaged over the two agents
    'comm test': 'test/avg_comm_task_perf',
    'comm 0←1': 'test/comm_task_perf_agent_0',
    'comm 1←0': 'test/comm_task_perf_agent_1',
    'private test': 'test/avg_private_task_perf',
    'eff. rank test (a0 / a1)': ('test/effective_rank_agent_0', 'test/effective_rank_agent_1'),
    'eff. rank pilots (a0 / a1)': ('pilots/effective_rank_agent_0', 'pilots/effective_rank_agent_1'),
    'misalign test': 'test/misalignment_energy',
    'misalign pilots': 'pilots/misalignment_energy',
    'MRR test': 'test/mean_reciprocal_rank',
    'MRR pilots': 'pilots/mean_reciprocal_rank',
}


def fetch(group: str) -> pd.DataFrame:
    rows = []
    for r in wandb.Api().runs(PROJECT, filters={'group': group, 'state': 'finished'}):
        s = r.summary
        row = {'run': r.id, 'lmb': 0.0 if group in NONCOOP else float(r.config['max_lmb'])}
        for col, key in METRICS.items():
            if isinstance(key, tuple):
                v = [s.get(k) for k in key]
                row[col] = f'{v[0]:.0f} / {v[1]:.0f}' if None not in v else None
            else:
                row[col] = s.get(key)
        rows.append(row)
    return pd.DataFrame(rows).sort_values('lmb')


def main():
    global PROJECT, CAP, FLAVOURS, NONCOOP
    ap = argparse.ArgumentParser()
    ap.add_argument('--setup', choices=SETUPS, default='6v6')
    setup = ap.parse_args().setup
    proj, CAP, FLAVOURS, NONCOOP = SETUPS[setup]
    PROJECT = f'engrima18/{proj}'
    OUT = ROOT / f'results/cifar10_2agents_grayblur_{setup}/cnorm_study'
    OUT.mkdir(parents=True, exist_ok=True)
    data = {g: fetch(g) for _, g, _ in FLAVOURS}
    ref = {g: fetch(g) for g in NONCOOP}

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for label, g, c in FLAVOURS:
        d = data[g]
        axes[0].plot(d.lmb, d['comm test'], 'o-', color=c, label=label)
        axes[1].plot(d.lmb, d['private test'], 'o-', color=c, label=label)
    for ng, (nlabel, ls, c) in NONCOOP.items():
        for ax, col in zip(axes, ('comm test', 'private test')):
            ax.axhline(ref[ng][col].iloc[0], ls=ls, color=c, label=nlabel + ' (λ=0)')
    for ax, t in zip(axes, ('post-comm test accuracy (avg of both directions)', 'private test accuracy (avg)')):
        ax.set_xscale('log')
        ax.set_xlabel('λ (max_lmb)')
        ax.set_ylabel(f'accuracy (10 test classes, max {CAP})')
        ax.set_title(t, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle(f'CIFAR-10 {setup} grayscale/blur, two ResNets, CenterNorm, no whitening, single seed', fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / 'comm_vs_lambda.png', dpi=140)
    plt.close(fig)

    best = []
    for g, (label, _, _) in NONCOOP.items():
        best.append({'method': label, 'λ': 0.0, **ref[g].iloc[0].drop('lmb').to_dict()})
    for label, g, _ in FLAVOURS:
        d = data[g]
        b = d.loc[d['comm test'].idxmax()]
        best.append({'method': label, 'λ': b.lmb, **b.drop('lmb').to_dict()})
    t = pd.DataFrame(best)
    t.to_csv(OUT / 'best_lambda_table.csv', index=False)
    fmt = t.copy()
    for col in fmt.columns:
        if fmt[col].dtype.kind == 'f' and col != 'λ':
            fmt[col] = fmt[col].map(lambda v: f'{v:.3f}')
    fmt['λ'] = fmt['λ'].map(lambda v: f'{v:g}')
    md = '\n'.join(['| ' + ' | '.join(fmt.columns) + ' |', '|' + '---|' * len(fmt.columns)]
                   + ['| ' + ' | '.join(map(str, r)) + ' |' for r in fmt.itertuples(index=False)])
    (OUT / 'best_lambda_table.md').write_text(md + '\n')
    for g, d in data.items():
        d.to_csv(OUT / f'{g}.csv', index=False)
    print(md)
    print(f'-> {OUT}')


if __name__ == '__main__':
    main()
