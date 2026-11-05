"""Score wandb runs against an objective and append them to its trial log.

    uv run python scripts/score_run.py 6v6_cnorm abc123 def456 --note "DiSheafFRL tau=0.2"
    uv run python scripts/score_run.py 6v6_cnorm abc123 --value 0.47   # offline-scored metric
    uv run python scripts/score_run.py 6v6_cnorm --show                # print the log

Objective: state/objectives/<name>.yaml.  Log: state/objectives/<name>.tsv.
A trial is a candidate when it passes all constraints and beats `best` by more than
`noise_floor`; only the user (or a confirmed multi-seed check) updates `best` in the yaml.
"""

from __future__ import annotations

import argparse
import csv
import operator
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path('state/objectives')
OPS = {
    '>=': operator.ge,
    '<=': operator.le,
    '>': operator.gt,
    '<': operator.lt,
}
FIELDS = [
    'date',
    'run',
    'seed',
    'metric',
    'delta_best',
    'constraints',
    'verdict',
    'note',
]


def fetch(project: str, run_id: str) -> tuple[dict, dict]:
    import wandb

    api = wandb.Api()
    run = api.run(f'{api.default_entity}/{project}/{run_id}')
    return dict(run.summary), dict(run.config)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('objective')
    ap.add_argument('runs', nargs='*')
    ap.add_argument('--note', default='')
    ap.add_argument('--value', type=float, help='metric computed offline')
    ap.add_argument('--show', action='store_true')
    a = ap.parse_args()

    obj = yaml.safe_load((ROOT / f'{a.objective}.yaml').read_text())
    log = ROOT / f'{a.objective}.tsv'
    if a.show or not a.runs:
        print(log.read_text() if log.exists() else 'no trials yet')
        return

    sign = 1 if obj['direction'] == 'max' else -1
    best = obj['best']['value']
    new = not log.exists()
    with log.open('a', newline='') as f:
        w = csv.DictWriter(f, FIELDS, delimiter='\t')
        if new:
            w.writeheader()
        for run_id in a.runs:
            summary, config = fetch(obj['project'], run_id)
            value = (
                a.value if a.value is not None else summary.get(obj['metric'])
            )
            if value is None:
                print(
                    f'{run_id}: metric {obj["metric"]} missing — pass --value'
                )
                continue
            fails = [
                f'{c["metric"]}={summary.get(c["metric"])}'
                for c in obj.get('constraints', [])
                if summary.get(c['metric']) is None
                or not OPS[c['op']](summary[c['metric']], c['value'])
            ]
            delta = sign * (value - best)
            verdict = (
                'rejected'
                if fails
                else 'candidate'
                if delta > obj['noise_floor']
                else 'within-noise'
                if delta > -obj['noise_floor']
                else 'worse'
            )
            row = {
                'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
                'run': run_id,
                'seed': config.get('seed'),
                'metric': round(value, 4),
                'delta_best': f'{delta:+.4f}',
                'constraints': 'ok' if not fails else ';'.join(fails),
                'verdict': verdict,
                'note': a.note,
            }
            w.writerow(row)
            print('\t'.join(str(row[k]) for k in FIELDS))


if __name__ == '__main__':
    main()
