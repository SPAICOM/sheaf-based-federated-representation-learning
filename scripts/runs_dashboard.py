"""Live terminal dashboard over logs/status/<sweep>/*.json (see src/utils/progress.py).

uv run python scripts/runs_dashboard.py            # all sweeps, refresh every 5 s
uv run python scripts/runs_dashboard.py SWEEP      # one sweep
uv run python scripts/runs_dashboard.py --once     # print once (for Claude / scripts)
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import time
from pathlib import Path

ROOT = Path('logs/status')
STALE_S = 180


def fmt_t(s: float | None) -> str:
    if s is None:
        return '--'
    s = int(s)
    h, m = divmod(s // 60, 60)
    return f'{h}h{m:02d}m' if h else f'{m}m{s % 60:02d}s'


def alive(pid: int, host: str) -> bool | None:
    if host != socket.gethostname():
        return None
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def bar(frac: float, width: int = 20) -> str:
    n = int(round(max(0.0, min(1.0, frac)) * width))
    return '█' * n + '·' * (width - n)


def render(sweep_filter: str | None) -> str:
    lines = [
        time.strftime('%H:%M:%S') + '  runs dashboard  (Ctrl-C to quit)',
        '',
    ]
    sweeps = (
        sorted(p for p in ROOT.glob('*') if p.is_dir())
        if ROOT.exists()
        else []
    )
    if sweep_filter:
        sweeps = [p for p in sweeps if p.name == sweep_filter]
    if not sweeps:
        return '\n'.join(lines + ['no status files under logs/status/'])
    now = time.time()
    for sp in sweeps:
        meta_p = sp / '_sweep.json'
        meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
        jobs = []
        for f in sorted(sp.glob('job_*.json')):
            try:
                jobs.append(json.loads(f.read_text()))
            except json.JSONDecodeError:  # noqa: PERF203 — file mid-rewrite
                continue
        if not jobs:
            continue
        done = [j for j in jobs if j['status'] == 'done']
        total = meta.get('total')
        head = f'■ {sp.name}   jobs {len(done)}/{total or "?"} done'
        if done and total:
            mean = sum(j['elapsed_s'] for j in done) / len(done)
            running = [j for j in jobs if j['status'] != 'done']
            left = (total - len(done) - len(running)) * mean + sum(
                (j.get('eta_s') or 0) for j in running
            )
            head += f'   sweep ETA ≈ {fmt_t(left)}   (mean job {fmt_t(mean)})'
        lines.append(head)
        for j in jobs[-12:]:
            status = j['status']
            quiet = now - j['updated']
            if status not in ('done', 'failed') and quiet > STALE_S:
                if alive(j['pid'], j['host']) is False:
                    status = 'DEAD'
                else:
                    status += f' ({fmt_t(quiet)} quiet)'
            max_ep = j['max_epochs'] or 1
            nb = j.get('n_batches') or 0
            frac = (j['epoch'] + (j['batch'] / nb if nb else 0)) / max_ep
            if status == 'done':
                frac = 1.0
            ov = ' '.join(j['overrides'])[:70]
            lines.append(
                f'  #{j["job"]:<3} {status:<12} {bar(frac)} ep {j["epoch"]:>3}/{max_ep:<3} '
                f'elap {fmt_t(j["elapsed_s"]):>7}  ETA {fmt_t(j.get("eta_s")):>7}  '
                f'{(str(j.get("last_epoch_s")) + "s/ep") if j.get("last_epoch_s") else ""}'
            )
            if ov:
                lines.append(f'        {ov}')
            if j['metrics'] and status != 'done':
                m = '  '.join(f'{k}={v}' for k, v in j['metrics'].items())
                lines.append(f'        {m[:140]}')
            if j.get('error'):
                lines.append(f'        ERROR {j["error"]}')
        lines.append('')
    return '\n'.join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('sweep', nargs='?')
    ap.add_argument('--once', action='store_true')
    ap.add_argument('--every', type=float, default=5.0)
    a = ap.parse_args()
    if a.once:
        print(render(a.sweep))
        return
    try:
        while True:
            out = render(a.sweep)
            print('\033[2J\033[H' + out, flush=True)
            time.sleep(a.every)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
