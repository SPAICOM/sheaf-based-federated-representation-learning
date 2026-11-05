"""Post-hoc (non-cooperative) alignment evaluated on trained SheafFRL encoders.

For each SheafFRL checkpoint of a shift x pilots sweep, rebuild the exact
datamodule of that run (same config, seed, shift and pilot budget), load the
trained agent weights into BOTH

  * the SheafFRL orchestrator  -> its own test pipeline (SWBN whitening +
    closed-form maps), a sanity check that must reproduce the wandb number;
  * the NonCooperativeLearning orchestrator -> the post-hoc pipeline
    (whitening_source=pilots, one-shot Procrustes on the same pilots),

and run ``trainer.test`` on each. If post-hoc-on-sheaf-encoders matches the
Non-coop baseline, the encoders are fine and the gap comes from the
learnable (SWBN) vs closed-form pilot whitening.

Usage:
    uv run python scripts/posthoc_on_sheaf_encoders.py
    uv run python scripts/posthoc_on_sheaf_encoders.py --runs d0k3l99a:0.2:128
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from lightning import Trainer, seed_everything
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent))
from multi_agent_experiment import (  # noqa: E402
    _build_agents,
    _build_orchestrator,
    _parse_agent_target_classes,
    _parse_per_agent_cfg,
)

from src.utils.graph_generator import generate_neighbors  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
# SheafFRL runs of cifar10_4agents_shift: run_id -> (shift, pilots)
DEFAULT_RUNS = {
    'd0k3l99a': (0.2, 128),
    'dut6zi09': (0.2, 256),
    'xii4rh0r': (0.2, 2048),
    'ro84k324': (0.8, 128),
    '6v92qsqs': (0.8, 256),
    '1apuudtw': (0.8, 2048),
}


def _setup(cfg):
    seed_everything(cfg.seed, workers=True)
    dataset_cfg = OmegaConf.to_container(cfg.dataset, resolve=True)
    if 'agent_classes' in cfg:
        dataset_cfg['agent_classes'] = OmegaConf.to_container(cfg.agent_classes, resolve=True)
    dm = instantiate(dataset_cfg)
    dm.prepare_data()
    dm.setup()
    n_agents = len(dm.models)
    tc = _parse_agent_target_classes(cfg)
    if tc is not None:
        for i in range(n_agents):
            tc.setdefault(i, set())
    neighbors = generate_neighbors(
        mode=cfg.graph.neighbors_mode,
        n_agents=n_agents,
        seed=cfg.graph.get('seed', 42),
        p=cfg.graph.get('p', 0.3),
        m=cfg.graph.get('m', 3),
        manual=cfg.graph.get('neighbors', {}),
        target_classes=tc,
        max_edge_frac=cfg.graph.get('max_edge_frac', 0.4),
        similarity=cfg.graph.get('similarity', 'intersection'),
    )
    return dm, neighbors


def _evaluate(cfg, dm, neighbors, state_dict, strict):
    per_agents_cfg = _parse_per_agent_cfg(cfg)
    agents, latent_dims = _build_agents(cfg, dm, per_agents_cfg)
    orch = _build_orchestrator(cfg, agents, neighbors, latent_dims, per_agents_cfg)
    if strict:
        orch.load_state_dict(state_dict, strict=True)
    else:
        agent_sd = {k: v for k, v in state_dict.items() if k.startswith('agents.')}
        missing, unexpected = orch.load_state_dict(agent_sd, strict=False)
        missing = [k for k in missing if k.startswith('agents.')]
        if missing or unexpected:
            raise RuntimeError(f'agent weights mismatch: missing={missing[:5]} unexpected={unexpected[:5]}')
    trainer = Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    trainer.test(orch, datamodule=dm, verbose=False)
    cb = trainer.callback_metrics
    return float(cb['test/avg_private_task_perf']), float(cb['test/avg_comm_task_perf'])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config-name', default='cifar10_4agents_shift')
    parser.add_argument('--project_dir', type=Path, default=ROOT / 'logs/cifar10_4agents_shift')
    parser.add_argument('--runs', nargs='+', default=None, help='run_id:shift:pilots entries')
    parser.add_argument('--skip_sanity', action='store_true', help='skip the SheafFRL re-evaluation')
    args = parser.parse_args()

    runs = (
        {r.split(':')[0]: (float(r.split(':')[1]), int(r.split(':')[2])) for r in args.runs}
        if args.runs
        else DEFAULT_RUNS
    )
    rows = []
    with initialize_config_dir(config_dir=str(ROOT / 'config/hydra'), version_base='1.3'):
        for run_id, (shift, pilots) in runs.items():
            ckpt = sorted((args.project_dir / run_id / 'checkpoints').glob('*.ckpt'))[-1]
            state_dict = torch.load(ckpt, map_location='cpu', weights_only=False)['state_dict']
            base = [f'dataset.shift_strength={shift}', f'dataset.pilot_num_samples={pilots}']
            res = {'run': run_id, 'shift': shift, 'pilots': pilots}
            for name, orch, strict in (('sheaf', 'sheaf_frl', True), ('posthoc', 'non_cooperative', False)):
                if name == 'sheaf' and args.skip_sanity:
                    continue
                cfg = compose(args.config_name, overrides=[f'orchestrator={orch}', *base])
                dm, neighbors = _setup(cfg)
                res[f'{name}_priv'], res[f'{name}_comm'] = _evaluate(cfg, dm, neighbors, state_dict, strict)
                print(f'[{run_id} s={shift} p={pilots}] {name}: priv={res[f"{name}_priv"]:.4f} comm={res[f"{name}_comm"]:.4f}', flush=True)
                del dm
                torch.cuda.empty_cache()
            rows.append(res)

    print('\nrun       shift pilots | sheaf priv  sheaf comm | posthoc priv  posthoc comm')
    for r in rows:
        print(
            f"{r['run']:9s} {r['shift']:5.1f} {r['pilots']:6d} | "
            f"{r.get('sheaf_priv', float('nan')):10.4f} {r.get('sheaf_comm', float('nan')):11.4f} | "
            f"{r['posthoc_priv']:12.4f} {r['posthoc_comm']:13.4f}"
        )


if __name__ == '__main__':
    main()
