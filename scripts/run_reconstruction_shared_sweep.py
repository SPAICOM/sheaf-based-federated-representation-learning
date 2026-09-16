"""Sweep shared visibility for Sheaf-FRL, non-cooperative and FedAvg.

Uses the current masked_cifar10_vae config; --dry-run freezes configurations
without training. Each sweep collects only its own runs, including FedAvg's
deliberately different (identity) communication protocol.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from run_reconstruction_loss_ablation import ROOT, execute_runs

METHODS = {
    'sheaf_frl': 'Sheaf-FRL',
    'non_cooperative': 'Non-cooperative',
    'federated': 'Federated',
}
METRICS = {
    'private_mse_full': 'MSE full (full image)',
    'private_lpips_full': 'LPIPS (full image)',
    'private_mse_visible': 'MSE visible (visible blocks only)',
    'comm_mse': 'Communication MSE (full image)',
    'comm_lpips_full': 'Communication LPIPS (full image)',
}


def plot_results(manifest: dict, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    frames = []
    for run in manifest['runs']:
        if run['status'] != 'completed':
            continue
        frame = pd.read_parquet(run['results_file'])
        frame['method'] = run['method']
        frame['shared_probability'] = run['shared_probability']
        frame['source_path'] = run['results_file']
        frames.append(frame)
    if not frames:
        return
    data = pd.concat(frames, ignore_index=True)
    data.to_parquet(out_dir / 'per_agent.parquet', index=False)
    data.to_csv(out_dir / 'per_agent.csv', index=False)
    # Agents first, then independent seeds; bands represent seed variability.
    per_seed = data.groupby(['method', 'shared_probability', 'seed'])[
        list(METRICS)
    ].mean()
    summary = per_seed.groupby(['method', 'shared_probability']).agg(
        ['mean', 'std']
    )
    summary.to_csv(out_dir / 'summary.csv')
    for metric, label in METRICS.items():
        fig, ax = plt.subplots(figsize=(6, 4))
        for method, name in METHODS.items():
            if method not in summary.index.get_level_values('method'):
                continue
            sub = summary.loc[method].sort_index()[metric]
            line, = ax.plot(sub.index, sub['mean'], marker='o', label=name)
            if sub['std'].notna().any():
                err = sub['std'].fillna(0)
                ax.fill_between(sub.index, sub['mean'] - err,
                                sub['mean'] + err, alpha=0.15,
                                color=line.get_color())
        ax.set(xlabel='Shared fraction of visible blocks', ylabel=label)
        ax.set_xticks(sorted(data['shared_probability'].unique()))
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        for extension in ('png', 'pdf'):
            fig.savefig(out_dir / f'{metric}_vs_shared.{extension}', dpi=150)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shared-values', nargs='+', type=float,
                        default=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    parser.add_argument(
        '--methods', nargs='+', choices=tuple(METHODS),
        default=list(METHODS),
        help='Methods to run (default: all three).',
    )
    parser.add_argument('--seeds', nargs='+', type=int, default=None,
                        help='Default: seed in the YAML.')
    parser.add_argument('--epochs', type=int, help='Default: epochs in the YAML.')
    parser.add_argument('--gpu', nargs='+',
                        help='One or two physical GPU indices; otherwise inherited.')
    parser.add_argument('--max-parallel', nargs='+', type=int, choices=(1, 2), default=[1],
                        help='Concurrent runs per GPU: one value for all GPUs, '
                             'or one value per GPU (e.g. --gpu 0 1 --max-parallel 2 1).')
    parser.add_argument('--wandb-mode', choices=('online', 'offline'), default='offline')
    parser.add_argument('--override', action='append', default=[],
                        help='Common Hydra override, repeatable.')
    parser.add_argument('--out-dir', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.epochs is not None and args.epochs < 1:
        parser.error('--epochs must be positive')
    if args.gpu is not None and (
        len(args.gpu) > 2 or len(set(args.gpu)) != len(args.gpu)
        or any(not gpu.isdecimal() for gpu in args.gpu)
    ):
        parser.error('--gpu requires one or two distinct nonnegative GPU indices')
    gpu_count = len(args.gpu or [None])
    capacities = args.max_parallel
    if len(capacities) == 1:
        capacities = capacities * gpu_count
    if len(capacities) != gpu_count:
        parser.error('--max-parallel requires one value, or one value per GPU')
    gpu_slots = ([gpu for gpu, count in zip(args.gpu, capacities)
                  for _ in range(count)] if args.gpu is not None else None)

    # Import after the script directory is available for reusable runner imports.
    sys.path.insert(0, str(ROOT))
    from src.datamodules.mask_generators import ConstantVisibleSharedMaskGenerator

    configs = []
    with initialize_config_dir(config_dir=str(ROOT / 'config/hydra'), version_base='1.3'):
        for method in dict.fromkeys(args.methods):
            base = compose(config_name='masked_cifar10_vae', overrides=[
                *args.override, f'orchestrator={method}',
            ])
            for seed in dict.fromkeys(args.seeds or [int(base.seed)]):
                for shared in dict.fromkeys(args.shared_values):
                    cfg = OmegaConf.create(OmegaConf.to_container(base, resolve=False))
                    cfg.seed = seed
                    cfg.dataset.mask_mode = 'constant_visible_shared'
                    cfg.dataset.constant_shared_visible_probability = shared
                    cfg.optimization.run_test = True
                    if args.gpu is not None:
                        cfg.trainer.accelerator = 'gpu'
                        cfg.trainer.devices = 1
                    if args.epochs is not None:
                        cfg.trainer.max_epochs = args.epochs
                    if method == 'federated':
                        cfg.orchestrator.alignment_method = None
                    try:
                        ConstantVisibleSharedMaskGenerator(
                            n_agents=cfg.dataset.n_agents,
                            focus_regions=cfg.dataset.random_focus_regions,
                            visible_fraction=cfg.dataset.constant_visible_fraction,
                            shared_visible_probability=shared,
                            block_size=cfg.dataset.random_block_size,
                            image_size=32, seed=seed,
                        )
                    except ValueError as exc:
                        parser.error(f'Invalid shared value {shared}: {exc}')
                    configs.append((method, seed, shared, cfg))

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out_dir = (args.out_dir or ROOT / 'results/reconstruction/shared_sweeps' / stamp).resolve()
    out_dir.mkdir(parents=True, exist_ok=False)
    configs_dir = out_dir / 'configs'
    configs_dir.mkdir()
    manifest = {'config': 'masked_cifar10_vae', 'methods': args.methods,
                'gpu': args.gpu,
                'max_parallel': sum(capacities), 'parallel_per_gpu': capacities,
                'gpu_slots': gpu_slots, 'wandb_mode': args.wandb_mode,
                'runs': []}
    for index, (method, seed, shared, cfg) in enumerate(configs):
        name = f'{index:03d}_{method}_shared{shared:g}_seed{seed}'
        cfg.logger.group = f'reconstruction-shared-sweep-{stamp}'
        cfg.logger.name = name
        OmegaConf.save(cfg, configs_dir / f'{name}.yaml')
        manifest['runs'].append({
            'variant': name, 'method': method, 'seed': seed,
            'shared_probability': shared, 'status': 'planned',
            'log': str(out_dir / f'{name}.log'),
            'command': [sys.executable, str(ROOT / 'scripts/reconstruction_experiment.py'),
                        '--config-path', str(configs_dir), '--config-name', name,
                        f'hydra.run.dir={out_dir / name / "hydra"}'],
        })
    (out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'{len(configs)} runs; configs, logs, metrics and plots: {out_dir}', flush=True)
    print(f'GPUs: {args.gpu or "inherited"}; concurrent runs per GPU: {capacities}', flush=True)
    if args.dry_run:
        print('Dry run: configuration validation only; no training started.')
        return
    env = os.environ.copy()
    env.update(WANDB_MODE=args.wandb_mode, PYTHONUNBUFFERED='1')
    env.setdefault('MPLCONFIGDIR', str(out_dir / 'matplotlib'))
    os.environ.setdefault('MPLCONFIGDIR', env['MPLCONFIGDIR'])
    try:
        execute_runs(manifest, out_dir, env, max_parallel=sum(capacities),
                     gpu_slots=gpu_slots)
    finally:
        plot_results(manifest, out_dir)


if __name__ == '__main__':
    main()
