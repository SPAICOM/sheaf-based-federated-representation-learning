"""Run independent 30-epoch reconstruction loss ablations and collect results.

Usage (from the activated project environment):
    python scripts/run_reconstruction_loss_ablation.py --gpu 1
    python scripts/run_reconstruction_loss_ablation.py --dry-run

Each run uses a frozen, fully composed copy of the current Hydra configuration.
Up to two runs share the selected GPU. No training or W&B connection occurs
with --dry-run.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from contextlib import suppress
from datetime import datetime
from pathlib import Path

import pandas as pd
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ('baseline', 'vae', 'perceptual', 'sigma_vae', 'latent_256')
METRICS = (
    'private_lpips_full',
    'comm_lpips_full',
    'private_mse_missing',
    'private_mse_visible',
    'private_mse_full',
    'private_psnr_full',
    'comm_mse',
    'comm_mse_sender_missing',
    'comm_psnr',
)


def build_configs(beta: float, perceptual_weight: float, overrides: list[str]):
    """Compose once, then change one loss family at a time (VAE also samples)."""
    with initialize_config_dir(
        config_dir=str(ROOT / 'config/hydra'), version_base='1.3'
    ):
        base = compose(
            config_name='masked_cifar10_vae',
            overrides=[
                'orchestrator=non_cooperative',
                *overrides,
            ],
        )
    # Enforce the requested protocol even if the working config changes later.
    base.trainer.max_epochs = 30
    base.optimization.run_test = True
    base.orchestrator.comm_task_coeff = 0.0
    base.diagnostics.masked_reconstruction.enabled = True
    base.logger.log_model = False
    configs = {}
    for variant in VARIANTS:
        cfg = OmegaConf.create(OmegaConf.to_container(base, resolve=False))
        cfg.loss_variant = variant
        cfg.model.beta = beta if variant == 'vae' else 0.0
        cfg.model.sample_posterior = variant == 'vae'
        cfg.model.perceptual_weight = (
            perceptual_weight if variant == 'perceptual' else 0.0
        )
        cfg.model.sigma_vae = variant == 'sigma_vae'
        if variant == 'latent_256':
            cfg.model.latent_dim = 256
        configs[variant] = cfg
    return configs


def collect_results(manifest: dict, out_dir: Path) -> None:
    frames = []
    for run in manifest['runs']:
        if run['status'] != 'completed':
            continue
        frame = pd.read_parquet(run['results_file'])
        frame['loss_variant'] = run['variant']
        frame['source_path'] = run['results_file']
        frames.append(frame)
    if not frames:
        return
    agents = pd.concat(frames, ignore_index=True)
    agents.to_csv(out_dir / 'per_agent.csv', index=False)
    agents.to_parquet(out_dir / 'per_agent.parquet', index=False)
    # One seed by default: these are means across agents, not seed uncertainty.
    summary = agents.reindex(columns=['loss_variant', *METRICS]).groupby(
        'loss_variant', sort=False
    ).mean()
    summary.to_csv(out_dir / 'summary.csv')
    print(
        '\nFinal test metrics (means across agents):\n' + summary.to_string()
    )


def execute_runs(
    manifest: dict, out_dir: Path, env: dict, max_parallel: int = 2,
    gpu_slots: list[str] | None = None,
):
    """Bounded process queue; only this parent writes the manifest and summary."""
    if gpu_slots is not None:
        if not gpu_slots:
            raise ValueError('gpu_slots must not be empty')
        max_parallel = len(gpu_slots)
    pending = iter(manifest['runs'])
    active = []
    exhausted = False
    manifest_path = out_dir / 'manifest.json'

    def save():
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')

    try:
        while active or not exhausted:
            while len(active) < max_parallel and not exhausted:
                run = next(pending, None)
                if run is None:
                    exhausted = True
                    break
                child_env = env.copy()
                if gpu_slots is not None:
                    occupied = {item['gpu_slot'] for _, item in active}
                    slot = next(i for i in range(len(gpu_slots)) if i not in occupied)
                    run.update(gpu_slot=slot, gpu=gpu_slots[slot])
                    child_env['CUDA_VISIBLE_DEVICES'] = gpu_slots[slot]
                print(f'Starting {run["variant"]} '
                      f'(GPU {child_env.get("CUDA_VISIBLE_DEVICES", "inherited")}) '
                      f'→ {run["log"]}', flush=True)
                with Path(run['log']).open('w') as log:
                    process = subprocess.Popen(
                        run['command'],
                        cwd=ROOT,
                        env=child_env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                active.append((process, run))
                run.update(status='running', pid=process.pid)
                save()
            finished = []
            for process, run in active:
                code = process.poll()
                if code is None:
                    continue
                run['returncode'] = code
                matches = re.findall(
                    r'^Results saved -> (.+)$',
                    Path(run['log']).read_text(errors='replace'),
                    re.MULTILINE,
                )
                path = Path(matches[-1].strip()) if matches else None
                if path is not None and not path.is_absolute():
                    path = ROOT / path
                if code == 0 and path is not None and path.is_file():
                    run.update(status='completed', results_file=str(path))
                else:
                    run['status'] = 'failed'
                print(f'{run["variant"]}: {run["status"]}', flush=True)
                finished.append((process, run))
            for item in finished:
                active.remove(item)
            if finished:
                save()
                collect_results(manifest, out_dir)
            if active and not finished:
                time.sleep(0.2)
    finally:
        # Stop only processes launched here, including their loader workers.
        for process, run in active:
            if process.poll() is None:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGTERM)
        for process, run in active:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            run.update(status='interrupted', returncode=process.returncode)
        save()
    failed = [
        run['variant'] for run in manifest['runs'] if run['status'] == 'failed'
    ]
    if failed:
        raise RuntimeError(
            f'Failed runs: {failed}. Inspect their logs in {out_dir}'
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--gpu',
        help='Physical GPU index; leaves CUDA visibility unchanged if omitted.',
    )
    parser.add_argument(
        '--beta',
        type=float,
        default=1e-4,
        help='Positive KL coefficient; current KL normalization is preserved.',
    )
    parser.add_argument('--perceptual-weight', type=float, default=0.01)
    parser.add_argument(
        '--wandb-mode', choices=('online', 'offline'), default='online'
    )
    parser.add_argument(
        '--variants', nargs='+', choices=VARIANTS, default=list(VARIANTS)
    )
    parser.add_argument(
        '--override',
        action='append',
        default=[],
        help='Common Hydra override, e.g. --override dataset.num_workers=2',
    )
    parser.add_argument(
        '--max-parallel',
        type=int,
        choices=(1, 2),
        default=2,
        help='Concurrent runs on the same GPU (default: 2).',
    )
    parser.add_argument('--out-dir', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.beta <= 0 or args.perceptual_weight <= 0:
        parser.error('--beta and --perceptual-weight must be positive')
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out_dir = (
        args.out_dir or ROOT / 'results/reconstruction/loss_ablations' / stamp
    ).resolve()
    out_dir.mkdir(parents=True, exist_ok=False)
    configs_dir = out_dir / 'configs'
    configs_dir.mkdir()
    configs = build_configs(args.beta, args.perceptual_weight, args.override)
    group = f'reconstruction-loss-ablation-{stamp}'
    manifest = {
        'group': group,
        'epochs': 30,
        'max_parallel': args.max_parallel,
        'gpu': args.gpu,
        'wandb_mode': args.wandb_mode,
        'runs': [],
    }
    env = os.environ.copy()
    env['WANDB_MODE'] = args.wandb_mode
    env['PYTHONUNBUFFERED'] = '1'
    if args.gpu is not None:
        env['CUDA_VISIBLE_DEVICES'] = args.gpu
    env.setdefault('MPLCONFIGDIR', str(out_dir / 'matplotlib'))
    for variant in dict.fromkeys(args.variants):
        cfg = configs[variant]
        cfg.logger.group = group
        cfg.logger.name = variant
        OmegaConf.save(cfg, configs_dir / f'{variant}.yaml')
        command = [
            sys.executable,
            str(ROOT / 'scripts/reconstruction_experiment.py'),
            '--config-path',
            str(configs_dir),
            '--config-name',
            variant,
            f'hydra.run.dir={out_dir / variant / "hydra"}',
        ]
        manifest['runs'].append(
            {
                'variant': variant,
                'command': command,
                'status': 'planned',
                'log': str(out_dir / f'{variant}.log'),
            }
        )
    (out_dir / 'manifest.json').write_text(
        json.dumps(manifest, indent=2) + '\n'
    )
    print(f'Configs, manifest and summaries: {out_dir}', flush=True)
    for run in manifest['runs']:
        print(f'{run["variant"]}: {" ".join(run["command"])}', flush=True)
    if args.dry_run:
        print('Dry run: no training launched and no W&B connection made.')
        return
    execute_runs(manifest, out_dir, env, max_parallel=args.max_parallel)


if __name__ == '__main__':
    main()
