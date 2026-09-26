"""Aggregate the union-vs-intersection regime re-run into a seed-averaged table.

Companion to ``comm_ablation_table.py``, but built for the controlled re-run in
the ``comm_ablation_regime`` project: every cell is averaged over seeds and
reported with its standard deviation, so differences can actually be read
against the noise floor (sigma ~= 0.007 on ``test/avg_comm_task_perf``,
measured from the 3-seed non-cooperative baseline).

Rows are keyed by (orchestrator, align_on_intersection, comm_percentage).
``comm_percentage=100`` is the SheafFRL cell: CESheafFRL at 100% reduces to
SheafFRL exactly (``_is_collab_epoch`` is unconditionally True, so the map
refresh and the both-live penalty match SheafFRL's), verified by a bit-identical
reproduction.

Usage:
    uv run scripts/regime_comparison_table.py
    uv run scripts/regime_comparison_table.py --project comm_ablation_regime --plot
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import wandb

METRICS = {
    'comm_acc': 'test/avg_comm_task_perf',
    'private': 'test/avg_private_task_perf',
    'mrr': 'pilots/mean_reciprocal_rank',
    'misalign': 'test/misalignment_loss',
    'whiten': 'test/whitening_cov_identity_dist',
    'kB': 'test/train_communication_kilobytes_cumulative',
}


def _unwrap(v):
    return v['value'] if isinstance(v, dict) and 'value' in v else v


def fetch(project: str) -> pd.DataFrame:
    api = wandb.Api()
    rows = []
    for r in api.runs(project):
        try:
            r.load(force=True)
        except Exception:
            continue
        if r.state != 'finished':
            continue
        cfg = dict(r.config)
        orch = _unwrap(cfg.get('orchestrator')) or {}
        s = dict(r.summary)
        # Only count runs that actually reached the end of training; a job that
        # died early still reports state='finished' to wandb.
        if s.get('epoch') is None:
            continue
        rows.append({
            'id': r.id,
            'target': str(orch.get('_target_', '')).split('.')[-1],
            'align_int': orch.get('align_on_intersection'),
            'ws': orch.get('whitening_source'),
            'p': orch.get('comm_percentage'),
            'seed': _unwrap(cfg.get('seed')),
            'epoch': s.get('epoch'),
            **{k: s.get(v) for k, v in METRICS.items()},
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project', default='comm_ablation_regime')
    ap.add_argument('--baseline-project', default='comm_ablation_seeds',
                    help='project holding the 3-seed non_cooperative baselines')
    ap.add_argument('--out-dir', type=Path,
                    default=Path('results/comm_ablation/regime'))
    args = ap.parse_args()

    d = fetch(args.project)
    if d.empty:
        print(f'no finished runs in {args.project} yet')
        return
    b = fetch(args.baseline_project)
    if not b.empty:
        d = pd.concat([d, b[b.target == 'NonCooperativeLearning']], ignore_index=True)

    d['regime'] = d.align_int.map({True: 'intersection', False: 'union'})
    d.loc[d.target == 'NonCooperativeLearning', 'regime'] = 'baseline'
    d['cell'] = d.apply(
        lambda r: (f"non_coop ({r.ws})" if r.target == 'NonCooperativeLearning'
                   else ('sheaf_frl (p=100)' if r.p == 100 else f'ce p={int(r.p)}')),
        axis=1,
    )

    num = [c for c in METRICS if c in d.columns]
    g = d.groupby(['regime', 'cell'], sort=False)
    agg = g[num].agg(['mean', 'std']).round(4)
    agg[('n', '')] = g.size()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    d.to_csv(args.out_dir / 'regime_runs.csv', index=False)
    agg.to_csv(args.out_dir / 'regime_summary.csv')

    pd.set_option('display.width', 220)
    print(f'\n{args.project}: seed-averaged cells (sigma_comm_acc ~ 0.007)')
    print('=' * 110)
    show = pd.DataFrame({
        'n': agg[('n', '')],
        'comm_acc': agg[('comm_acc', 'mean')].map('{:.4f}'.format)
                    + ' ± ' + agg[('comm_acc', 'std')].fillna(0).map('{:.4f}'.format),
        'misalign': agg[('misalign', 'mean')].map('{:.4f}'.format),
        'mrr': agg[('mrr', 'mean')].map('{:.4f}'.format),
        'private': agg[('private', 'mean')].map('{:.4f}'.format),
        'kB': agg[('kB', 'mean')].map('{:,.0f}'.format),
    })
    print(show.to_string())
    print(f'\nper-run CSV : {args.out_dir / "regime_runs.csv"}')


if __name__ == '__main__':
    main()
