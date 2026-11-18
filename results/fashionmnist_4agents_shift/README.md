# Fashion-MNIST heterogeneity table (4 agents, shift 1.0)

Four heterogeneous CNNs, each trained on 7 of 10 classes, 40 epochs, 4096 pilots. W&B project
`fashionmnist_4agents_shift`; only finished runs are used. Final rows are seeds 42/43/44 at the
tuned lambda, which was picked on seed-42 *test* accuracy from the grid
.001/.003/.01/.03/.1/.3/1 (not on held-out validation data). FedRE and non-coop have no lambda.

- `table.md` / `table.csv`: final table (comm accuracy, gain over non-coop paired by seed,
  both-seen and sender-unseen/receiver-seen scopes, private accuracy, MRR, misalignment,
  communication cost, run ids).
- `grid.csv`: seed-42 lambda grid (comm accuracy %).
- `summary.csv`: every finished run (51) with headline and pooled class-scope metrics.

Regenerate (reads the W&B remote): `just fmnist-hetero-summary`
(`scripts/plot_fashionmnist_hetero.py`). No checkpoints are needed; all values are W&B summaries.
ComFed logs no scope / MRR / misalignment metrics.
Training: `just hetero-pipeline fashionmnist ...` (config `fashionmnist_4agents_shift.yaml`) and
`tmp/2026-11-14_fashionmnist_noncoop.sh` for the non-coop runs.
