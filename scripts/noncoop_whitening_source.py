"""Re-score the non-cooperative baseline under both whitening sources.

``PostTrainingAlignmentMixin`` fits each agent's whitening operator on the full
**training** split, while SheafFRL/CESheafFRL's SWBN layers only ever see
**pilot** rows.  That makes the two families' normalisation incomparable: a
difference in ``misalignment_loss`` / MRR / comm accuracy could come from the
alignment map or merely from where the whitening statistics were estimated.

``orchestrator.whitening_source`` ('train' | 'pilots') switches the baseline's
operator to the pilot split.  This script re-scores an already-trained
non-cooperative checkpoint under both settings and prints them side by side --
no retraining needed, since a non-cooperative run's encoders never communicate
and are therefore identical under either choice.

Usage:
    uv run scripts/noncoop_whitening_source.py                # default run id
    uv run scripts/noncoop_whitening_source.py --run gplav88p
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(sys.path[0]).parent))

import pandas as pd
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from diagnose_mrr_collapse import (  # noqa: E402
    CONFIG_DIR,
    CONFIG_NAME,
    build_datamodule,
    load_orchestrator,
)

KEYS = {
    'comm_acc': 'test/avg_comm_task_perf',
    'hetero': 'test/avg_heterophil_comm_task_perf',
    'private': 'test/avg_private_task_perf',
    'mrr': 'pilots/mean_reciprocal_rank',
    'misalign': 'test/misalignment_loss',
    'whiten_dist': 'test/whitening_cov_identity_dist',
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', default='gplav88p', help='wandb run id of a non-cooperative run')
    ap.add_argument('--scope', choices=('run', 'on', 'off'), default='run',
                    help="class-intersection scoping of the post-hoc map fit and "
                         "the misalignment/MRR eval: 'run' honours the checkpoint's "
                         "own eval_on_class_intersection, 'on'/'off' force it. Use "
                         "'off' to match union-regime SheafFRL runs.")
    ap.add_argument('--out', type=Path,
                    default=Path('results/comm_ablation/diagnostics/noncoop_whitening_source.csv'))
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name=CONFIG_NAME)
    OmegaConf.set_struct(cfg, False)
    cfg.pop('logger', None)
    dm = build_datamodule(cfg)

    rows = []
    for source in ('train', 'pilots'):
        orch = load_orchestrator(args.run, cfg, dm, device)
        # hparams drives _fit_alignment_maps; set it before any fit happens.
        orch.hparams['whitening_source'] = source
        if args.scope != 'run':
            # Scopes BOTH the Procrustes fit and the misalignment/MRR eval,
            # exactly as the training-time flag does.
            orch._eval_on_class_intersection = args.scope == 'on'
            orch.hparams['eval_on_class_intersection'] = args.scope == 'on'
        logs = orch.evaluate_communication_accuracy(dm, prefix='test')
        logs.update(orch.evaluate_whitening_quality(dm, prefix='test'))
        rows.append({'whitening_source': source, 'scope': args.scope,
                     'run': args.run,
                     **{k: logs.get(v) for k, v in KEYS.items()}})
        del orch
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    d = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(args.out, index=False)
    pd.set_option('display.width', 200)
    print(f'\nnon_cooperative run {args.run} — post-hoc alignment, two whitening sources')
    print('=' * 100)
    print(d.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    print(f'\nCSV: {args.out}')


if __name__ == '__main__':
    main()
