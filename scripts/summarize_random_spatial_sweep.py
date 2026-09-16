"""Summarize completed random-spatial reconstruction parquet results."""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import pandas as pd


VALUES = (0.10, 0.20, 0.30, 0.40, 0.50)
ORCHESTRATORS = ('SheafFRL', 'NonCooperativeLearning')


def _latest_results(results_dir: Path) -> dict[tuple[str, float], Path]:
    candidates: dict[tuple[str, float], list[Path]] = {}
    for path in results_dir.glob('reconstruction__*.parquet'):
        try:
            frame = pd.read_parquet(path)
        except Exception:
            continue
        if (
            frame.empty
            or 'mask_mode' not in frame
            or 'orchestrator' not in frame
        ):
            continue
        if frame['mask_mode'].iloc[0] != 'agent_random_spatial':
            continue
        if 'inactive_random_shared_visible_probability' not in frame:
            continue
        shared = float(frame['inactive_random_shared_visible_probability'].iloc[0])
        orch = str(frame['orchestrator'].iloc[0])
        key = (orch, round(shared, 2))
        candidates.setdefault(key, []).append(path)
    return {key: max(paths, key=lambda p: p.stat().st_mtime)
            for key, paths in candidates.items()}


def _fmt(value: float) -> str:
    return 'NA' if pd.isna(value) else f'{value:.6f}'


def _metric_from_log(path: Path, name: str) -> float | None:
    value = None
    for line in path.read_text(encoding='utf-8', errors='ignore').splitlines():
        if name not in line or '_agent_' in line:
            continue
        numbers = re.findall(
            r'[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?', line
        )
        if numbers:
            value = float(numbers[-1])
    return value


def _latest_log(logs_dir: Path, value: float, orchestrator: str) -> Path | None:
    tag = f's{int(round(value * 100)):03d}'
    token = 'sheaf_frl' if orchestrator == 'SheafFRL' else 'non_cooperative'
    required = (
        'test/avg_private_mse_missing',
        'test/avg_private_mse_visible',
        'test/avg_comm_task_perf',
    )
    candidates = []
    for path in logs_dir.glob(f'random_spatial_{tag}__*{token}*.log'):
        content = path.read_text(encoding='utf-8', errors='ignore')
        if 'Results saved ->' in content and all(
            _metric_from_log(path, name) is not None for name in required
        ):
            candidates.append(path)
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def build_summary(results_dir: Path, logs_dir: Path, output: Path) -> bool:
    latest = _latest_results(results_dir)
    required = {(orch, value) for orch in ORCHESTRATORS for value in VALUES}
    latest_logs = {
        key: _latest_log(logs_dir, key[1], key[0]) for key in required
    }
    missing = sorted(key for key, path in latest_logs.items() if path is None)
    if not required.issubset(latest) or missing:
        print(f'Waiting for test logs: {missing}', flush=True)
        return False

    rows: list[dict[str, object]] = []
    for value in VALUES:
        for orch in ORCHESTRATORS:
            path = latest[(orch, value)]
            log_path = latest_logs[(orch, value)]
            frame = pd.read_parquet(path)
            def mean_column(name: str) -> float:
                if name not in frame:
                    return float('nan')
                return float(frame[name].mean())

            rows.append({
                'shared_visible_probability': value,
                'orchestrator': orch,
                'avg_private_mse_missing': _metric_from_log(
                    log_path, 'test/avg_private_mse_missing'
                ),
                'avg_private_mse_visible': _metric_from_log(
                    log_path, 'test/avg_private_mse_visible'
                ),
                'avg_comm_task_perf': _metric_from_log(
                    log_path, 'test/avg_comm_task_perf'
                ),
                'source_log': log_path.name,
            })

    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        '# Random-Spatial Reconstruction Test Summary',
        '',
        'Averages are computed across agents 0 and 1 from the latest '
        'parquet for each setting.',
        '',
        '| shared visible probability | orchestrator | avg private MSE missing | '
        'avg private MSE visible | avg comm task performance | source log |',
        '|---:|---|---:|---:|---:|---|',
    ]
    for row in rows:
        lines.append(
            f"| {row['shared_visible_probability']:.2f} | "
            f"{row['orchestrator']} | "
            f"{_fmt(row['avg_private_mse_missing'])} | "
            f"{_fmt(row['avg_private_mse_visible'])} | "
            f"{_fmt(row['avg_comm_task_perf'])} | "
            f"{row['source_log']} |"
        )
    output.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'Summary saved -> {output}', flush=True)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=Path, default=Path('results/reconstruction'))
    parser.add_argument('--logs-dir', type=Path, default=Path('logs/reconstruction_random_spatial_sweep_20260901_090835'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--wait', action='store_true')
    parser.add_argument('--poll-seconds', type=int, default=30)
    args = parser.parse_args()

    while True:
        if build_summary(args.results_dir, args.logs_dir, args.output) or not args.wait:
            return
        time.sleep(max(1, args.poll_seconds))


if __name__ == '__main__':
    main()
