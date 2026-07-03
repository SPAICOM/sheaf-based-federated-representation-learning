"""List local WandB runs with the identifiers that match the wandb web UI.

Each local run lives in ``logs/wandb/run-<timestamp>-<run_id>``. The ``<run_id>``
suffix is exactly the id in the wandb URL (``wandb.ai/<entity>/<project>/runs/
<run_id>``) and ``display_name`` is the name shown in the UI — so this tool
prints run_id + name + project + group + state + size for every local run,
letting you cross-reference the UI and pick which dirs to delete.

Deleting a local ``run-*`` dir only reclaims local disk; it does NOT remove the
run from wandb.ai (and deleting in the UI does not remove the local dir).

Usage:
    python scripts/list_wandb_runs.py                       # all local runs
    python scripts/list_wandb_runs.py --project sfrl_bottleneck
    python scripts/list_wandb_runs.py --incomplete          # only runs that
                                                            # didn't finish cleanly
    python scripts/list_wandb_runs.py --incomplete --paths-only | xargs -r du -sh
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any


def _run_record(run_dir: Path) -> Any | None:
    from wandb.proto import wandb_internal_pb2 as pb
    from wandb.sdk.internal.datastore import DataStore

    wf = next(run_dir.glob('*.wandb'), None)
    if wf is None:
        return None
    ds = DataStore()
    try:
        ds.open_for_scan(str(wf))
    except Exception:
        return None
    for _ in range(50):
        try:
            data = ds.scan_data()
        except Exception:
            break
        if data is None:
            break
        rec = pb.Record()
        rec.ParseFromString(data)
        if rec.WhichOneof('record_type') == 'run':
            return rec.run
    return None


_EXIT_RE = None


def _exit_code(run_dir: Path) -> int | None:
    """Return the run's exit code (0 = clean finish) from debug-internal.log.

    wandb writes ``exit_code":N`` near the end of the internal debug log — an
    authoritative, project-agnostic finish signal (unlike task-specific metric
    keys). Reads only the tail of the log to stay cheap.
    """
    global _EXIT_RE
    if _EXIT_RE is None:
        import re

        _EXIT_RE = re.compile(rb'exit_code"?\s*:\s*(\d+)')
    log = run_dir / 'logs' / 'debug-internal.log'
    if not log.exists():
        return None
    try:
        size = log.stat().st_size
        with log.open('rb') as fh:
            if size > 200_000:
                fh.seek(size - 200_000)
            blob = fh.read()
    except OSError:
        return None
    matches = _EXIT_RE.findall(blob)
    return int(matches[-1]) if matches else None


def _dir_size(run_dir: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(run_dir):
        for f in files:
            try:
                total += os.stat(os.path.join(root, f)).st_size
            except OSError:
                pass
    return total


def _fmt_size(n: int) -> str:
    for unit in ('B', 'K', 'M', 'G'):
        if n < 1024 or unit == 'G':
            return f'{n:.0f}{unit}' if unit == 'B' else f'{n:.1f}{unit}'
        n /= 1024
    return f'{n:.1f}G'


def _created(run_dir: Path) -> str:
    # Dir name: run-YYYYMMDD_HHMMSS-<id>
    parts = run_dir.name.split('-')
    if len(parts) >= 3:
        try:
            return datetime.strptime(parts[1], '%Y%m%d_%H%M%S').strftime(
                '%Y-%m-%d %H:%M'
            )
        except ValueError:
            pass
    return '?'


def collect(run_dir: Path) -> dict[str, Any] | None:
    run = _run_record(run_dir)
    if run is None:
        return None
    summ_path = run_dir / 'files' / 'wandb-summary.json'
    summary: dict[str, Any] = {}
    if summ_path.exists():
        try:
            summary = json.loads(summ_path.read_text())
        except json.JSONDecodeError:
            summary = {}
    code = _exit_code(run_dir)
    if code == 0:
        state = 'ok'            # finished cleanly
    elif code is None:
        state = 'running?'      # no exit logged yet (live run or hard-killed)
    else:
        state = f'crashed({code})'
    return {
        'dir': run_dir,
        'run_id': run.run_id,
        'name': run.display_name,
        'project': run.project,
        'group': run.run_group,
        'created': _created(run_dir),
        'runtime_s': summary.get('_runtime'),
        'state': state,
        'size': _dir_size(run_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wandb_dir', type=Path, default=Path('logs/wandb'))
    parser.add_argument(
        '--project', type=str, default=None, help='Filter to this project.'
    )
    parser.add_argument(
        '--incomplete',
        action='store_true',
        help='Only runs that did not finish cleanly (crashed or no exit code). '
        'NOTE: includes "running?" runs — a live run looks the same, check dates.',
    )
    parser.add_argument(
        '--crashed',
        action='store_true',
        help='Only runs that errored out (exit_code > 0). Safe to delete.',
    )
    parser.add_argument(
        '--paths-only',
        action='store_true',
        help='Print only run dir paths (for piping to du/rm).',
    )
    args = parser.parse_args()

    rows = []
    for run_dir in sorted(args.wandb_dir.glob('run-*')):
        info = collect(run_dir)
        if info is None:
            continue
        if args.project is not None and info['project'] != args.project:
            continue
        if args.incomplete and info['state'] == 'ok':
            continue
        if args.crashed and not info['state'].startswith('crashed'):
            continue
        rows.append(info)

    if args.paths_only:
        for r in rows:
            print(r['dir'])
        return

    rows.sort(key=lambda r: (r['project'] or '', r['created']))
    print(
        f'{"STATE":<12}{"PROJECT":<26}{"RUN_ID":<12}{"CREATED":<18}'
        f'{"RUNTIME":>9}{"SIZE":>8}  NAME'
    )
    total = 0
    for r in rows:
        total += r['size']
        rt = f'{int(r["runtime_s"])}s' if r['runtime_s'] is not None else '—'
        print(
            f'{r["state"]:<12}{(r["project"] or "?")[:25]:<26}'
            f'{r["run_id"]:<12}{r["created"]:<18}{rt:>9}'
            f'{_fmt_size(r["size"]):>8}  {r["name"]}'
        )
    print(
        f'\n{len(rows)} runs, {_fmt_size(total)} on disk'
        + (f' (project={args.project})' if args.project else '')
    )
    print(
        'wandb URL for a run:  '
        'https://wandb.ai/<your-entity>/<project>/runs/<run_id>'
    )


if __name__ == '__main__':
    main()
