"""Loss sweep isolation, result collection, and final image logging."""

import json
from types import SimpleNamespace

import pandas as pd
import torch
from omegaconf import OmegaConf

from scripts import run_reconstruction_loss_ablation as sweep
from scripts.reconstruction_experiment import (
    MaskedReconstructionDiagnosticsCallback,
)


def test_dry_run_freezes_independent_variants(tmp_path, monkeypatch):
    output = tmp_path / 'sweep'
    monkeypatch.setattr(
        'sys.argv', ['sweep', '--dry-run', '--out-dir', str(output)]
    )
    monkeypatch.setattr(
        sweep.subprocess,
        'Popen',
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError('training launched')
        ),
    )
    sweep.main()
    manifest = json.loads((output / 'manifest.json').read_text())
    assert len(manifest['runs']) == 5
    assert manifest['max_parallel'] == 2
    configs = {
        name: OmegaConf.load(output / 'configs' / f'{name}.yaml')
        for name in sweep.VARIANTS
    }
    for cfg in configs.values():
        assert cfg.trainer.max_epochs == 30
        assert cfg.orchestrator._target_.endswith('NonCooperativeLearning')
        assert cfg.orchestrator.comm_task_coeff == 0
        assert cfg.optimization.run_test
        assert cfg.dataset == configs['baseline'].dataset
    assert configs['baseline'].model.beta == 0
    assert configs['vae'].model.beta == 1e-4
    assert configs['vae'].model.sample_posterior
    assert configs['perceptual'].model.perceptual_weight == 0.01
    assert not configs['perceptual'].model.sample_posterior
    assert configs['sigma_vae'].model.sigma_vae
    assert not configs['sigma_vae'].model.sample_posterior
    assert configs['latent_256'].model.latent_dim == 256
    baseline = OmegaConf.to_container(configs['baseline'].model)
    larger = OmegaConf.to_container(configs['latent_256'].model)
    baseline['latent_dim'] = 256
    assert larger == baseline


def test_collect_completed_runs_only(tmp_path):
    result = tmp_path / 'result.parquet'
    pd.DataFrame(
        [dict.fromkeys(sweep.METRICS, value) for value in (1.0, 3.0)]
    ).to_parquet(result)
    sweep.collect_results(
        {
            'runs': [
                {
                    'status': 'completed',
                    'variant': 'baseline',
                    'results_file': str(result),
                },
                {'status': 'failed', 'variant': 'vae'},
            ]
        },
        tmp_path,
    )
    summary = pd.read_csv(tmp_path / 'summary.csv')
    assert summary['private_mse_missing'].tolist() == [2.0]
    assert len(pd.read_parquet(tmp_path / 'per_agent.parquet')) == 2


def test_final_test_images_include_composite_without_waiting_for_cadence():
    class Dataset:
        def __len__(self):
            return 1

        def _mask(self, idx):
            return torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])

        def __getitem__(self, idx):
            target = torch.ones(3, 2, 2)
            return target * self._mask(idx), target

    class Agent(torch.nn.Module):
        def forward(self, x):
            return torch.full_like(x, 0.25)

    dataset, agent = Dataset(), Agent()
    callback = MaskedReconstructionDiagnosticsCallback(every_n_epochs=5)
    rows = callback._sample_rows(dataset, agent, torch.device('cpu'), True)
    assert len(rows) == 5
    assert torch.all(rows[-1][:, :, 0] == 1)
    assert torch.all(rows[-1][:, :, 1] == 0.25)
    calls = []
    callback._log_image = lambda *args: calls.append(args)
    trainer = SimpleNamespace(
        current_epoch=29,
        datamodule=SimpleNamespace(
            train_datasets={}, models=[0], test_datasets={0: dataset}
        ),
    )
    module = SimpleNamespace(agents={'0': agent}, device=torch.device('cpu'))
    callback.on_test_end(trainer, module)
    assert len(calls) == 1
    assert calls[0][1] == 'test/masked_reconstruction_examples_agent_0'


def test_queue_runs_two_children_on_same_gpu_and_survives_failure(tmp_path):
    import os
    import sys

    import pytest

    result = tmp_path / 'result.parquet'
    pd.DataFrame([dict.fromkeys(sweep.METRICS, 0.5)]).to_parquet(result)
    child = """
import json, os, sys, time
from pathlib import Path
index, directory, result = sys.argv[1:]
p = Path(directory)
start = time.monotonic()
(p / (index + '.started')).touch()
# The first child cannot finish until a second child has actually started.
if index == '0':
    deadline = time.monotonic() + 5
    while not (p / '1.started').exists():
        if time.monotonic() > deadline:
            raise RuntimeError('second child never started concurrently')
        time.sleep(0.01)
time.sleep(0.15)
(p / (index + '.json')).write_text(json.dumps({
    'start': start, 'end': time.monotonic(),
    'gpu': os.environ['CUDA_VISIBLE_DEVICES'],
}))
if index == '1':
    sys.exit(3)
print('Results saved -> ' + result)
"""
    manifest = {
        'runs': [
            {
                'variant': str(i),
                'status': 'planned',
                'command': [
                    sys.executable,
                    '-c',
                    child,
                    str(i),
                    str(tmp_path),
                    str(result),
                ],
                'log': str(tmp_path / f'{i}.log'),
            }
            for i in range(4)
        ]
    }
    with pytest.raises(RuntimeError, match='Failed runs'):
        sweep.execute_runs(
            manifest, tmp_path, {**os.environ, 'CUDA_VISIBLE_DEVICES': '1'}
        )
    timings = [
        json.loads((tmp_path / f'{i}.json').read_text()) for i in range(4)
    ]
    events = sorted(
        [(t['start'], 1) for t in timings] + [(t['end'], -1) for t in timings]
    )
    active = peak = 0
    for _, delta in events:
        active += delta
        peak = max(peak, active)
    assert peak == 2
    assert all(t['gpu'] == '1' for t in timings)
    assert [run['status'] for run in manifest['runs']] == [
        'completed',
        'failed',
        'completed',
        'completed',
    ]
    assert len(pd.read_csv(tmp_path / 'summary.csv')) == 3
    saved = json.loads((tmp_path / 'manifest.json').read_text())
    assert saved == manifest
