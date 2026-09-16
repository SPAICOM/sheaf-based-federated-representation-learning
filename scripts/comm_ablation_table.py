"""Build the communication-efficiency comparison table from a wandb project.

Reads the finished runs of the ``comm_ablation`` project and assembles one table
comparing, per row:

* ``non_cooperative``      — zero training communication (single post-hoc fit),
* ``ce_sheaf_frl`` (local_reg=False) at each ``comm_percentage``,
* ``ce_sheaf_frl`` (local_reg=True)  at each ``comm_percentage``,
* ``sheaf_frl``            — communicates every step.

Columns: test comm accuracy, pilot instance-identity MRR, test misalignment loss,
and cumulative (one-directional) communication in kB.  Duplicate configs keep the
most recent finished run.

Usage:
    uv run scripts/comm_ablation_table.py --project comm_ablation
    just comm-ablation-table
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import wandb

# Summary keys pulled for each run.
METRICS = {
    'comm_acc': 'test/avg_comm_task_perf',
    'mrr': 'pilots/mean_reciprocal_rank',
    'misalign': 'test/misalignment_loss',
    'comm_kb': 'test/train_communication_kilobytes_cumulative',
    'comm_rounds': 'test/train_communication_rounds_cumulative',
    'whiten_dist': 'test/whitening_cov_identity_dist',
    'train_loss': 'train/total_loss_epoch',
    'private': 'test/avg_private_task_perf',
    'hetero': 'test/avg_heterophil_comm_task_perf',
}
# Metric columns shown in the printed/CSV table (label → metric key above).
# The 'comm%' column is handled separately (it is row identity, not a summary
# metric): CESheafFRL → its comm_percentage, non_cooperative → 0, sheaf_frl → 100.
# whiten_dist = mean ‖cov(whitened test latents) − I‖_F/√d (0 = perfectly white);
# comm_rounds = total training communication rounds.
COLUMNS = [
    ('comm_acc', 'comm_acc'),
    ('MRR', 'mrr'),
    ('misalign', 'misalign'),
    ('whiten_dist', 'whiten_dist'),
    ('private', 'private'),
    ('train_loss', 'train_loss'),
    ('comm_rounds', 'comm_rounds'),
    ('comm_kB', 'comm_kb'),
]

# CESheafFRL comm% cells to drop from the table (p=1 is the degenerate
# single-final-communication cell — see the ablation discussion).
_DROP_COMM_PCT = {1}


def _comm_pct_value(r: dict) -> float | None:
    """The comm% axis value for a row (baselines mapped to their equivalents)."""
    if r['target'] == 'CESheafFRL':
        return r['comm_pct']
    return {'NonCooperativeLearning': 0, 'SheafFRL': 100}.get(r['target'])


def _unwrap(cfg: dict, key: str) -> Any:
    """Return cfg[key], transparently unwrapping wandb's {'value': ...} form."""
    v = cfg.get(key)
    if isinstance(v, dict) and 'value' in v:
        return v['value']
    return v


def _read_run(run) -> dict[str, Any] | None:
    """Extract (config identity + metrics) for one finished run, or None."""
    try:
        run.load(force=True)
    except Exception:
        return None
    cfg = dict(run.config)
    orch = _unwrap(cfg, 'orchestrator') or {}
    target = str(orch.get('_target_', '')).split('.')[-1]
    if not target:
        return None
    summary = dict(run.summary)
    row = {
        'target': target,
        'comm_pct': orch.get('comm_percentage'),
        'local_reg': bool(orch.get('local_reg', False)),
        # learn_whitening drives the whitening (SWBN vs ZCA) and is the dominant
        # confound for MRR/misalignment — it MUST be part of the row identity,
        # else SWBN and ZCA runs of the same cell silently overwrite each other.
        'learn_whitening': orch.get('learn_whitening'),
        'created': str(run.created_at),
    }
    for short, key in METRICS.items():
        val = summary.get(key)
        row[short] = float(val) if isinstance(val, (int, float)) else None
    return row


def _whitening_tag(r: dict) -> str:
    """Whitening actually used: non_coop is always ZCA; others follow learn_whitening."""
    if r['target'] == 'NonCooperativeLearning':
        return 'ZCA'
    return 'SWBN' if r.get('learn_whitening') else 'ZCA'


def _row_key(r: dict) -> tuple:
    """Row identity: (target, local_reg, whitening, comm%) — whitening included."""
    wt = _whitening_tag(r)
    if r['target'] == 'CESheafFRL':
        return ('CESheafFRL', r['local_reg'], wt, r['comm_pct'])
    return (r['target'], None, wt, None)


def _sort_key(key: tuple) -> tuple:
    """Order: non_coop, CE (by whitening, reg, comm%), sheaf_frl."""
    target, local_reg, wt, comm_pct = key
    wt_order = 0 if wt == 'ZCA' else 1
    if target == 'NonCooperativeLearning':
        return (0, 0, 0, 0.0)
    if target == 'CESheafFRL':
        return (1, wt_order, 1 if local_reg else 0, float(comm_pct or 0))
    if target == 'SheafFRL':
        return (2, wt_order, 0, 0.0)
    return (3, 0, 0, 0.0)


def _row_label(key: tuple) -> str:
    # comm% is its own column; label carries method/variant + whitening.
    target, local_reg, wt, _comm_pct = key
    if target == 'CESheafFRL':
        return f'ce_sheaf_frl (reg={"T" if local_reg else "F"}, {wt})'
    return {
        'NonCooperativeLearning': f'non_cooperative ({wt})',
        'SheafFRL': f'sheaf_frl every-step ({wt})',
    }.get(target, target)


def _fmt(v: Any, kind: str) -> str:
    if v is None:
        return '—'
    if kind == 'comm_kb':
        return f'{v:,.1f}'
    if kind == 'comm_rounds':
        return f'{int(round(v))}'
    return f'{v:.4f}'


def build_table(project: str, entity: str | None, workers: int = 8) -> list[dict]:
    api = wandb.Api()
    entity = entity or api.default_entity
    if entity is None:
        raise SystemExit('No wandb entity available (pass --entity or run `wandb login`).')
    runs = list(api.runs(f'{entity}/{project}', filters={'state': 'finished'}))
    rows = [r for r in ThreadPoolExecutor(workers).map(_read_run, runs) if r]
    # Drop pre-refactor CESheafFRL runs (old collab_epochs schedule, no
    # comm_percentage) and the excluded comm% cells so the table shows only the
    # current-scheme cells of interest.
    rows = [
        r for r in rows
        if not (
            r['target'] == 'CESheafFRL'
            and (r['comm_pct'] is None or r['comm_pct'] in _DROP_COMM_PCT)
        )
    ]

    # Deduplicate: keep the most recent finished run per row identity.
    latest: dict[tuple, dict] = {}
    for r in rows:
        k = _row_key(r)
        if k not in latest or r['created'] > latest[k]['created']:
            latest[k] = r
    return [latest[k] for k in sorted(latest, key=_sort_key)]


def _fmt_comm_pct(v: Any) -> str:
    return '—' if v is None else f'{int(v)}'


def render_markdown(table: list[dict]) -> str:
    header = ['model', 'comm%'] + [c[0] for c in COLUMNS]
    lines = ['| ' + ' | '.join(header) + ' |',
             '|' + '|'.join(['---'] * len(header)) + '|']
    for r in table:
        cells = [_row_label(_row_key(r)), _fmt_comm_pct(_comm_pct_value(r))]
        cells += [_fmt(r[key], key) for _, key in COLUMNS]
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', default='comm_ablation', help='wandb project name')
    p.add_argument('--entity', default=None, help='wandb entity (default: your default)')
    p.add_argument(
        '--out', default='results/comm_ablation',
        help='output directory for table.md / table.csv',
    )
    args = p.parse_args()

    table = build_table(args.project, args.entity)
    if not table:
        raise SystemExit(f'No finished runs found in {args.project!r}.')

    md = render_markdown(table)
    print(md)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'table.md').write_text(md + '\n')
    with (out_dir / 'table.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['model', 'comm%'] + [c[0] for c in COLUMNS])
        for r in table:
            w.writerow(
                [_row_label(_row_key(r)), _comm_pct_value(r)]
                + [('' if r[key] is None else r[key]) for _, key in COLUMNS]
            )
    print(f'\nSaved {out_dir/"table.md"} and {out_dir/"table.csv"}')


if __name__ == '__main__':
    main()
