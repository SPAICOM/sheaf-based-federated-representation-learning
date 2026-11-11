# Sheaf-Based Federated Representation Learning


<h5 align="center">
    
[![ieee](https://img.shields.io/static/v1?label=IEEE+Paper&message=ID-HERE&color=0057b7&logo=ieee)](https://ieeexplore.ieee.org/document/ID-HERE)
[![arXiv](https://img.shields.io/badge/Arxiv-ID.HERE-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/CODE.HERE)
[![License](https://img.shields.io/badge/Code%20License-MIT-yellow)](https://github.com/SPAICOM/REPO-NAME-HERE/blob/main/LICENSE)

 <br>

</h5>

> [!TIP]
> Heterogeneous federated systems require agents to learn and exchange informative representations despite differences in data distributions, model architectures, and latent dimensionalities. We propose Sheaf-based Federated Representation Learning (SFRL), a framework that jointly learns and aligns heterogeneous latent representations without sharing parameters, labels, or a single latent space. SFRL relates agent-specific latent spaces through learnable orthogonal transformations and isometric embeddings, promoting consistency via a sheaf-Laplacian gluing regularizer evaluated on a shared set of pilot samples. We develop Sheaf-FRL, a decentralized alternating algorithm combining local gradient updates with closed-form Procrustes updates of the alignment maps, and establish convergence to first-order stationary points in deterministic and stochastic settings. Applied to collaborative classification in semantic communication under model and data heterogeneity, Sheaf-FRL improves private and communication accuracy over federated baselines and is more robust to latent-space compression.

## Dependencies

This project uses [`uv`](https://github.com/astral-sh/uv) for Python dependency management and [`just`](https://github.com/casey/just) as the task runner.

### Install prerequisites

Install the required tools:

- [`uv`](https://docs.astral.sh/uv/getting-started/installation/)
- [`just`](https://github.com/casey/just)

Follow the installation instructions from their official documentation.

### Setup the development environment

From the project root, run:

```bash
just setup
```

The `setup` recipe will:

- Create the `.venv` virtual environment (if it does not exist)
- Install all project dependencies using `uv`

After the command completes, the development environment will be ready to use. 🚀

## Reproducing the experiments and figures

### Retained MNIST 7-vs-7 studies

The original successful analyses are promoted from `tmp/` into
`scripts/plot_mnist_7v7.py`, `scripts/plot_mnist4_pilots.py`,
`scripts/plot_mnist4_lambda.py`, and `scripts/plot_mnist4_maps.py`.
The retained [summary results](results/mnist_4agents_shift/README.md) cover
pilot counts 384/768/1536/2048/4096, lambda .003/.01/.03/.1/.3/1, and
CenterNorm/ZCA/OLS with posthoc Procrustes evaluation. The
[two-agent comparison](results/mnist_2agents_7v7/main/table.md) is preliminary
(two seeds). These results exclude p128, SimCLR, shift .8/.9, BN ablations,
and saved-map evaluation experiments.

Training uses the existing `scripts/multi_agent_experiment.py` and configs
`mnist_2agents_7v7`, `mnist_4agents_shift`, and
`mnist_4agents_mapstudy_{base,zca,ols}`. No new scheduler or shell script is
needed. Examples below each run **one** experiment; use the matrix to select
other seeds, pilot counts, and lambdas. Before a long run, check running jobs,
smoke-test the config with `just smoke`, and launch with `just launch`.

```bash
# Two agents: seed 42/43, non_cooperative (lambda 0) or di_sheaf_frl (lambda .03).
just mnist-7v7 seed=42 orchestrator=di_sheaf_frl orchestrator.max_lmb=.03 study_name=mnist_7v7_disheaf
# Four-agent CenterNorm: batch 256 is an essential override.
just mnist-4agents clean seed=42 orchestrator=di_sheaf_frl dataset.batch_size=256 dataset.pilot_num_samples=4096 orchestrator.max_lmb=.01 study_name=mnist4_di_clean_bs256_check_di_sheaf_frl_p4096
# OLS or ZCA: the inherited base config already fixes batch 256.
just mnist4-mapstudy ols seed=42 dataset.pilot_num_samples=4096 orchestrator.max_lmb=.01 logger.group=ols_p4096_l.01
```

| Study | Seeds | Pilot counts | Lambda | Historical W&B groups |
|---|---|---|---|---|
| Two-agent noncoop / CenterNorm | 42,43 | 4096 | 0 / .03 | `mnist_7v7_noncoop`, `mnist_7v7_disheaf` |
| Four-agent noncoop / CenterNorm pilots | 42,43,44 | 384,768,1536,2048,4096 | 0 / .01 | `mnist4_di_clean_bs256_pilots_<orchestrator>_p<N>`; 4096 uses `check` instead of `pilots` |
| Four-agent CenterNorm lambda | 42,43,44 | 4096 | .003,.01,.03,.1,.3,1 | `mnist4_di_clean_bs256_lambda4096_<lambda>` (e.g. `0.003`); .01 reuses the `check` baseline |
| Four-agent ZCA / OLS pilots | 42,43,44 | 384,768,1536,2048,4096 | .01 | `<method>_p<N>_l.01` |
| Four-agent ZCA / OLS lambda | 42,43,44 | 4096 | .003,.01,.03,.1,.3,1 | `<method>_p4096_l<lambda>` (e.g. `.003` or `1`); .01 reused |

All train for 100 epochs with shift strength 1 and exclusive pool pilots.
The two-agent batch size is 128; four-agent studies use 256. Metrics and
configs live in remote W&B, final checkpoints in `logs/<project>/<run_id>/checkpoints/`.
Rerunning historical groups adds replicates; analysis scripts deliberately
reject unexpected counts instead of silently pooling them. Historical runs
used dirty revisions, so the scripts/configs capture their protocol rather
than guarantee bitwise reproduction.

```bash
just mnist-7v7-summary
just mnist4-lambda-summary
just mnist4-map-summary
# Optional regeneration of the retained scoped pilot summary; requires all 30
# final checkpoints and a free GPU (defaults to the visible CUDA device).
just mnist4-pilot-summary
```

The map plot reads the promoted pilot/lambda aggregates and refreshes ZCA/OLS
metrics from remote W&B. If recomputing the pilot aggregates, run that recipe
before the map recipe. All promoted figures use the house style and produce
PDF and PNG under `results/<project>/`.

All experiments log to Weights & Biases (`wandb login` once), and every plotting
script reads the runs back from the cloud — so the figures can be rebuilt on any
machine, without the training runs sitting on disk.

### 1. Distribution-shift study (15 heterogeneous agents)

Everything the runs need is pinned in
`config/hydra/multiagent_mnist_shift_distr.yaml` (fixed λ, Optuna off, 100
epochs, `lr 1e-2`); it sweeps the five shift strengths `s ∈ {0.5 … 0.9}` and
logs to the **`shift_distr`** project. `seed=1,2,3` adds the seeds the error
bars are computed over — drop it for a single seed-42 run.

```bash
# Sheaf-FRL (the sweeper's default orchestrator), 5 shifts x 3 seeds
uv run python scripts/multi_agent_experiment.py \
  --config-name=multiagent_mnist_shift_distr seed=1,2,3

# Non-cooperative, Sheaf-FMTL and ComFed, same shifts and seeds.
# Sheaf-FMTL and ComFed need a per-shift λ (see below), which is why this is a
# script rather than one command; it is resumable and skips finished cells.
SEEDS="1 2 3" scripts/run_shift_distr_seeds.sh
```

Two caveats worth knowing:

- **λ is only pinned for Sheaf-FRL and Non-cooperative.** The reference
  Sheaf-FMTL and ComFed runs each tuned `orchestrator.max_lmb` with the
  built-in Optuna study, so re-running with the search enabled re-tunes it and
  changes the curve. The tuned values per shift are listed in the header of
  `config/hydra/multiagent_mnist_shift_distr.yaml` and applied by the script
  above.
- **FedProto and FedMuscle are not run here.** Their runs live in the team
  project `avino-1905974-sapienza-universit-di-roma/shift_distr`, which the
  plotting script below merges in automatically.

### 2. Bottleneck study (2 heterogeneous agents, matched latent dimension)

`config/hydra/hetero_rate_2agents_mnist_hetero_bottleneck.yaml` sweeps the
bottleneck dimension through the `model=hetero_bottleneck_{16…512}` configs at a
fixed shift `s = 0.7`, 20 epochs, seed 42, and logs to the
**`sfrl_hetero_bottleneck`** project. Run it once per method:

```bash
for orch in sheaf_frl non_cooperative comfed sheaf_fmtl fedproto fedmuscle; do
  uv run python scripts/multi_agent_experiment.py \
    --config-name=hetero_rate_2agents_mnist_hetero_bottleneck \
    hydra.sweeper.params.orchestrator=$orch
done
```

Here `lmb_study` is enabled for every method, so each (method, dimension) cell
tunes its own λ over 15 trials: a re-run searches again and will not land on the
published λ exactly. To pin a cell, read its `orchestrator.max_lmb` off the
wandb run and pass it with `lmb_study.enabled=false`.

### 3. Figures

```bash
# Composed 4-panel figure: (a,b) shift strength, (c,d) bottleneck dimension.
# Averages seeds 1-3 in (a,b) with seaborn's default error bars, and merges
# FedProto/FedMuscle from the team project. Writes one PDF+PNG per legend
# placement (top / inside panel b / inside panel c) to results/combined/plots/.
uv run python scripts/plot_bottleneck_shift_row.py
uv run python scripts/plot_bottleneck_shift_row.py --legend top   # just one
uv run python scripts/plot_bottleneck_shift_row.py --seeds 1 2 3 42

# The same data as standalone figures.
uv run python scripts/plot_multiagent_metrics.py --project shift_distr --together
uv run python scripts/plot_hetero_bottleneck_metrics.py --together
```

`plot_multiagent_metrics.py` keeps one run per (method, shift) — the most
recent — so with several seeds in a project it plots whichever ran last, and
also writes the per-shift summary tables. Use the composed script for the
seed-averaged view.

### 4. Two heterogeneous agents on CIFAR-10 (non-cooperative vs Sheaf-FRL)

Two ResNet agents with different latent sizes: agent 0 has d=512, agent 1 has
d=256. Each agent trains on its own label-shifted draw of CIFAR-10. Agent 0's
draw favours classes {0,1,2}, agent 1's favours {5,6,7}, by a fraction s, the
shift. Both agents are tested on the full official test split. The pilots are
images that both agents hold in their training sets. Each agent sees every
image, pilots included, through its own corruption (`dataset.agent_input_noise`,
see `src/datamodules/input_noise.py`).

All shared settings live in `config/hydra/cifar10_2agents_base.yaml`. Each
setup is one small config on top of it:

| setup | config | agent 0 sees | agent 1 sees | shift |
|---|---|---|---|---|
| (i) sensors | `cifar10_2agents_grayblur` | grayscale | Gaussian blur 5x5, σ=1 | 0.4 |
| (i) sensors, harsh | `cifar10_2agents_grayblur_harsh` | grayscale | Gaussian blur 9x9, σ=2 | 0.8 |
| (ii) portions | `cifar10_2agents_crops` | grayscale zoom on the central 16x16 crop | periphery only (central half of the area blacked out) | 0.4 |
| (iii) shift only | `cifar10_2agents_shift` | clean | clean | 0.8 |

Orchestrators:
- `non_cooperative`: independent training, then post-hoc whitening and a
  Procrustes map fitted on the pilots.
- `sheaf_frl`: joint training with the sheaf penalty, learnable SWBN whitening.
- `sheaf_frl_zca`: as `sheaf_frl`, but with closed-form ZCA whitening, refit
  each epoch.

Whitening statistics are fitted on the pilots, the rows the alignment map is
fitted on. That holds for the post-hoc pipeline (`orchestrator.whitening_source`),
SWBN (`swbn_fit_source`) and ZCA (`zca_refit_source`). Setting them to `train`
fits on each agent's own training set instead, which costs about 28 points on
grayblur_harsh (see the status note).

Other SheafFRL knobs used by the λ study:
- `orchestrator.swbn_momentum`: EMA retention of the SWBN running stats (default 0.95).
- `orchestrator.penalty_lr`: learning rate of the penalty gradient (default: the
  optimizer lr). Under SGD it acts like λ·penalty_lr/lr.
- `orchestrator.warmup_epochs`: number of epochs before the penalty switches on.

Both SheafFRL variants are evaluated with the same post-hoc pipeline as
non-cooperative. λ is fixed by `orchestrator.max_lmb` (default 0.01). Add
`lmb_study.enabled=true` to grid-search it on a training holdout instead.

```bash
# 1. Non-cooperative: ONE run per setup. The pilots don't affect its training,
#    so the budget only matters at alignment time. Note the wandb run id it prints.
just two-agents-noncoop crops
# 2. Non-coop comm accuracy at 8 ... 2048 pilots and at all of them, on the same encoders
just two-agents-sweep crops <run_id>     # -> results/cifar10_2agents_crops/noncoop_sweep.csv
# 3. Sheaf-FRL at a given pilot budget and λ: <setup> <sheaf_frl|sheaf_frl_zca> <pilots> <λ>
just two-agents-sheaf crops sheaf_frl 2048 0.01
just two-agents-sheaf crops sheaf_frl_zca 128 0.01
```

Each run prints per-agent `self_accuracy` (private) and `comm_accuracy` (the
receiver decoding the other agent's mapped latents). wandb logs the average
under `test/avg_comm_task_perf`. Compare Sheaf-FRL at n pilots with the
non-coop sweep row at the same n. Each setup logs to its own wandb project,
`cifar10_2agents_<setup>`. Running jobs in parallel needs the thread cap the
recipes already set, otherwise each job slows down about 25x. Other corruptions
can be set inline, e.g. left vs right half:
`'dataset.agent_input_noise={0:{type:mask,region:right,fraction:0.5},1:{type:mask,region:left,fraction:0.5}}'`.

**Status (seed 42, 2026-10-27).** Comm accuracy at 128 / 2048 pilots.
- **(ii) portions:**
  - non-coop: private 0.706, comm 0.483 / 0.588
  - Sheaf-FRL SWBN λ=0.01: 0.474 / 0.596
  - Sheaf-FRL ZCA λ=0.01: 0.450 / 0.577
  - λ=0.1 is worse for both.
- **(i) sensors:** non-coop private 0.871, comm 0.829 / 0.859.
- **(i) sensors, harsh:** non-coop private 0.700, comm 0.555 / 0.660.
- **(i), both variants:** these non-coop numbers come from 60-epoch runs; the
  configs now default to 40.
- **(iii) shift only:** not run yet.
- **(i) harsh, 40 epochs, train-fitted SWBN/ZCA statistics.**
  - Private accuracy is about 0.69 for every method.
  - Comm accuracy at 128 / 2048 pilots:

    | | post-hoc fit on train (default) | post-hoc fit on pilots |
    |---|---|---|
    | non-coop | 0.217 / 0.348 | 0.546 / 0.626 |
    | SWBN | 0.202 / 0.342 | 0.537 / 0.614 |
    | ZCA | 0.187 / 0.341 | 0.536 / 0.614 |

  - Re-score a checkpoint with
    `scripts/pilot_count_sweep.py --overrides orchestrator.whitening_source=pilots`.
  - Why: the train mixes differ from the pilot mix, so train-fitted whitening
    leaves the pilots off-centre and anisotropic, which a rotation can't fix.
    With an unconstrained linear map every whitening source scores 0.609.
    The defaults are back to `pilots`.
- **λ study (grayblur_harsh, 100 epochs, 2048 pilots).** Five SheafFRL
  variants × 10 λ in [0, 1]: `tmp/2026-10-28_lmb_study.sh`. Plots:
  `uv run python scripts/plot_lambda_study.py` →
  `results/cifar10_2agents_grayblur_harsh_lmb/`.
  - λ=0 (≈ non-coop): comm 0.699, private 0.712.
  - SWBN peaks at λ=0.03 with comm 0.721 (+2.1pt), private unchanged; SWBN momentum doesn't matter.
  - At λ≥0.1 both comm and private drop.
  - ZCA never helps and collapses at λ=1 (comm 0.18).
  - `swbn_vicreg` (`orchestrator.vicreg_vc=true`) adds VICReg variance/covariance terms (see `config/hydra/orchestrator/sheaf_frl.yaml`). It hurts: peak 0.706, 0.661 at λ=0.03. With γ=1 the variance hinge mostly pushes up the scale of the latents (per-dim std ~0.06).
  - Single seed.

Sheaf-FRL runs from before 2026-10-26 on (i) and (ii) are invalid and must be
re-run. Until then, SheafFRL's training pilots ignored the agents'
corruptions, so the penalty aligned clean images.

## Citation

If you find this code useful for your research, please consider citing the following paper:

```
```

## Used Technologies

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
