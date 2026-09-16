"""GPU queue capacity, isolated child environments and failure recovery."""
import json
import os
import sys

import pandas as pd
import pytest

from scripts.run_reconstruction_loss_ablation import METRICS, execute_runs


@pytest.mark.parametrize('slots', [('0', '1'), ('0', '0'),
                                  ('0', '0', '1'), ('0', '1', '1'),
                                  ('0', '0', '1', '1')])
def test_gpu_capacity_and_refill(tmp_path, slots):
    result = tmp_path / 'result.parquet'
    pd.DataFrame([dict.fromkeys(METRICS, 0.5)]).to_parquet(result)
    child = '''
import json, os, sys, time
from pathlib import Path
index, directory, result, slots = sys.argv[1:]
p = Path(directory)
start = time.monotonic()
(p / (index + '.started')).touch()
if int(index) < int(slots):
    deadline = start + 10
    while len(list(p.glob('*.started'))) < int(slots):
        if time.monotonic() > deadline:
            raise RuntimeError('slots did not start concurrently')
        time.sleep(0.01)
time.sleep(0.1)
(p / (index + '.json')).write_text(json.dumps({
    'start': start, 'end': time.monotonic(),
    'gpu': os.environ['CUDA_VISIBLE_DEVICES'],
}))
if index == '0':
    sys.exit(3)
print('Results saved -> ' + result)
'''
    manifest = {'runs': [
        {'variant': str(i), 'status': 'planned',
         'command': [sys.executable, '-c', child, str(i), str(tmp_path),
                     str(result), str(len(slots))],
         'log': str(tmp_path / f'{i}.log')}
        for i in range(len(slots) * 2 + 1)
    ]}
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '9'}
    with pytest.raises(RuntimeError, match='Failed runs'):
        execute_runs(manifest, tmp_path, env, gpu_slots=list(slots))
    assert env['CUDA_VISIBLE_DEVICES'] == '9'
    timings = [json.loads((tmp_path / f'{i}.json').read_text())
               for i in range(len(manifest['runs']))]
    for gpu in set(slots):
        events = sorted(event for t in timings if t['gpu'] == gpu
                        for event in [(t['start'], 1), (t['end'], -1)])
        active = peak = 0
        for _, delta in events:
            active += delta
            peak = max(peak, active)
        assert peak == slots.count(gpu)
    assert manifest['runs'][0]['status'] == 'failed'
    assert all(r['status'] == 'completed' for r in manifest['runs'][1:])
    for run, timing in zip(manifest['runs'], timings):
        assert run['gpu'] == timing['gpu'] == slots[run['gpu_slot']]
    assert json.loads((tmp_path / 'manifest.json').read_text()) == manifest
