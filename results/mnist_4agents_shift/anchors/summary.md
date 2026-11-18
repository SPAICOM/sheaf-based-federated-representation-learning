# Anchor-selection ablation: original vs proto_kmeans vs stratified

4-agent MNIST (7 classes/agent, fully connected), lambda .01, CenterNorm, bs256, 100 epochs. Accuracy %,
mean ± sample SD over seeds, pooled over the 12 directed edges (i = receiver, j = sender), rescored offline
from final checkpoints (overall Comm asserted vs remote W&B, 2e-3). `n` = number of seeds. Per-run numbers:
`per_run.csv`; wide stats: `summary_wide.csv`; run ids: `provenance.md`.

## Non-coop

| arm | K | n | Private | Private on C_i | Comm | Comm on C_i\C_j | Comm on C_i | Comm on C_i∩C_j |
|---|---|---|---|---|---|---|---|---|
| original | 128 | 3 | 69.57 ± 0.02 | 99.60 ± 0.03 | 64.14 ± 0.33 | 80.28 ± 1.08 | 91.83 ± 0.47 | 98.98 ± 0.25 |
| original | 384 | 3 | 69.56 ± 0.01 | 99.58 ± 0.01 | 64.78 ± 0.24 | 82.19 ± 1.09 | 92.75 ± 0.35 | 99.29 ± 0.11 |
| original | 768 | 3 | 69.57 ± 0.01 | 99.59 ± 0.01 | 65.21 ± 0.17 | 83.62 ± 0.68 | 93.35 ± 0.24 | 99.38 ± 0.06 |
| original | 1536 | 3 | 69.57 ± 0.01 | 99.60 ± 0.01 | 65.60 ± 0.38 | 85.01 ± 1.41 | 93.92 ± 0.54 | 99.44 ± 0.05 |
| kmeans (cold) | 128 | 3 | 69.56 ± 0.01 | 99.58 ± 0.02 | 65.91 ± 0.22 | 86.81 ± 0.88 | 94.36 ± 0.32 | 99.04 ± 0.19 |
| kmeans (cold) | 384 | 3 | 69.56 ± 0.01 | 99.58 ± 0.02 | 66.15 ± 0.32 | 87.85 ± 0.61 | 94.70 ± 0.46 | 98.94 ± 0.44 |
| kmeans (cold) | 768 | 3 | 69.56 ± 0.01 | 99.58 ± 0.02 | 66.20 ± 0.29 | 87.59 ± 1.02 | 94.77 ± 0.41 | 99.22 ± 0.04 |
| kmeans (cold) | 1536 | 3 | 69.56 ± 0.01 | 99.58 ± 0.02 | 66.41 ± 0.24 | 88.42 ± 0.98 | 95.07 ± 0.34 | 99.19 ± 0.07 |
| strat | 128 | 3 | 69.56 ± 0.01 | 99.58 ± 0.02 | 64.84 ± 0.28 | 87.36 ± 1.07 | 92.83 ± 0.40 | 96.23 ± 0.77 |
| strat | 384 | 3 | 69.56 ± 0.01 | 99.58 ± 0.02 | 65.96 ± 0.30 | 89.01 ± 0.94 | 94.43 ± 0.43 | 97.79 ± 0.83 |
| strat | 768 | 3 | 69.56 ± 0.01 | 99.58 ± 0.02 | 66.19 ± 0.13 | 89.51 ± 1.02 | 94.76 ± 0.19 | 98.02 ± 0.47 |
| strat | 1536 | 3 | 69.56 ± 0.01 | 99.58 ± 0.02 | 66.28 ± 0.22 | 89.70 ± 0.99 | 94.89 ± 0.32 | 98.11 ± 0.34 |

## DiSheafFRL

| arm | K | n | Private | Private on C_i | Comm | Comm on C_i\C_j | Comm on C_i | Comm on C_i∩C_j |
|---|---|---|---|---|---|---|---|---|
| original | 128 | 3 | 69.54 ± 0.03 | 99.55 ± 0.04 | 66.31 ± 0.81 | 88.35 ± 2.03 | 94.93 ± 1.15 | 99.00 ± 0.62 |
| original | 384 | 3 | 69.56 ± 0.03 | 99.58 ± 0.04 | 68.79 ± 0.21 | 96.95 ± 0.75 | 98.48 ± 0.31 | 99.43 ± 0.03 |
| original | 768 | 3 | 69.58 ± 0.01 | 99.61 ± 0.02 | 69.15 ± 0.01 | 98.36 ± 0.04 | 99.00 ± 0.01 | 99.39 ± 0.01 |
| original | 1536 | 3 | 69.58 ± 0.00 | 99.61 ± 0.00 | 69.27 ± 0.01 | 98.81 ± 0.05 | 99.17 ± 0.02 | 99.39 ± 0.00 |
| kmeans (cold) | 128 | 3 | 69.55 ± 0.02 | 99.56 ± 0.02 | 69.19 ± 0.03 | 99.15 ± 0.08 | 99.05 ± 0.04 | 98.99 ± 0.08 |
| kmeans (warm) | 128 | 1 | 69.55 | 99.57 | 69.23 | 99.09 | 99.11 | 99.13 |
| kmeans (cold) | 384 | 3 | 69.55 ± 0.02 | 99.57 ± 0.03 | 69.24 ± 0.01 | 99.16 ± 0.08 | 99.12 ± 0.02 | 99.09 ± 0.02 |
| kmeans (warm) | 768 | 3 | 69.55 ± 0.03 | 99.57 ± 0.04 | 69.25 ± 0.00 | 99.19 ± 0.07 | 99.14 ± 0.01 | 99.10 ± 0.05 |
| kmeans (warm) | 1536 | 3 | 69.56 ± 0.02 | 99.59 ± 0.03 | 69.26 ± 0.01 | 99.19 ± 0.05 | 99.15 ± 0.01 | 99.12 ± 0.04 |
| strat | 128 | 3 | 69.55 ± 0.01 | 99.57 ± 0.01 | 67.98 ± 0.08 | 96.26 ± 0.27 | 97.32 ± 0.11 | 97.98 ± 0.32 |
| strat | 384 | 3 | 69.54 ± 0.02 | 99.56 ± 0.02 | 68.95 ± 0.11 | 98.91 ± 0.13 | 98.71 ± 0.16 | 98.59 ± 0.22 |
| strat | 768 | 3 | 69.56 ± 0.01 | 99.59 ± 0.01 | 69.05 ± 0.02 | 99.17 ± 0.02 | 98.85 ± 0.02 | 98.66 ± 0.05 |
| strat | 1536 | 3 | 69.56 ± 0.02 | 99.59 ± 0.03 | 69.07 ± 0.04 | 99.28 ± 0.03 | 98.88 ± 0.06 | 98.64 ± 0.11 |

## DiSheafFRL − non-coop gap (Comm, points)

| arm | K=128 | K=384 | K=768 | K=1536 |
|---|---|---|---|---|
| original | +2.16 | +4.01 | +3.94 | +3.67 |
| kmeans | +3.28 | +3.09 | +3.05 | +2.85 |
| strat | +3.13 | +2.99 | +2.86 | +2.79 |

## Notes

- The k-means DiSheafFRL arm mixes two fit variants: cold start at K=128/384, warm start (kmeans_warm_start=true) at
  K=768/1536; a warm K=128 seed-42 row is listed separately.
- Original K=384/768/1536 come from results/mnist_4agents_shift/swbn_scopes/per_run.csv; original K=128 is scored by
  scripts/score_mnist4_anchors.py.
- Original non-coop = checkpoints trained with K pilots; k-means/stratified non-coop = 4096-pilot checkpoints refit
  with the K-anchor selection (private accuracy is therefore identical across those two arms).
- Single-seed noise floor ~±1.5 pt (3-seed SDs here are far below it).
