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

## Reproducing the paper

Every result of the paper has two `just` routines: a **train** routine, which
runs the experiment (Hydra multirun of `scripts/multi_agent_experiment.py`;
metrics and configs to Weights & Biases, checkpoints to `logs/`), and a
**table/figure** routine, which rebuilds the result into `results/` from wandb
(`wandb login` once) and, where the paper re-scores runs, from the
checkpoints. `just --list` prints them all. Runs logged to a team entity are
read with `WANDB_ENTITY=<entity> just <routine>`.

| Paper result | Train | Table / figure | Output | Status |
|---|---|---|---|---|
| Tab. `tab:main`: private and comm accuracy, 4 benchmarks × 8 methods (Sec. 6) | `just train-benchmark <dataset>`, dataset ∈ `mnist`, `fashionmnist`, `femnist`, `cifar10` | `just tables-benchmarks` | `results/paper/tab_main.{tex,md}` | ✅ ¹ |
| Tab. `tab:geom`: MRR, TV, μ_min (App. Alignment) | same runs | `just tables-benchmarks` | `results/paper/tab_geom.{tex,md}` | ✅ ¹ |
| Fig. `fig:compression`: FEMNIST, 2 agents, d ∈ {8, 16, 32, 64} (App. Compression) | `just train-compression` | `just figure-compression` | `results/femnist_2agents_bottleneck/bottleneck_split.pdf` | ✅ |
| Tab. `tab:comm-epochs`: communication epochs ρ = 2–80 % (App. Ablation) | `just train-ablation-comm-epochs` | `just table-comm-epochs` | `results/mnist_4agents_shift/comm_epochs/rows.tex` | ✅ ² |
| Fig. `fig:ablation-pilots`: pilot budget K (App. Ablation) | `just train-ablation-pilots` | `just figure-ablation-pilots` | `results/mnist_4agents_shift/{pilots,maps}/` | ⚠️ ³ |
| Fig. `fig:ablation-lambda`: penalty weight λ (App. Ablation) | `just train-ablation-lambda` | `just figure-ablation-lambda` | `results/mnist_4agents_shift/{lambda,maps}/` | ⚠️ ³ |
| Fig. `fig:ablation-topology`: star / ring / full graph (App. Ablation) | – | – | – | ❌ ⁴ |
| Figs. `fig:spectrum`, `fig:tv_normalized`, `fig:low_energy` (App. Global sections) | – | – | – | ❌ ⁴ |
| Sheaf-FMTL compression factor γ (App. Experimental details) | – | `just sheaf-fmtl-gamma <configs>` | printed | ✅ |
| Tab. `tab:setups`: agents of the benchmarks (App. Experimental details) | – | – | the experiment configs | descriptive |
| DomainNet, 6 agents × 6 domains (not in the paper yet) | `just domainnet-data`, then `just train-domainnet [domainnet_6agents\|domainnet_6agents_shift]` | `just report-domainnet` | `results/domainnet_6agents{,_shift}/domain_scope_report.md`, `results/domainnet_6agents_figures/` | ✅ |

¹ Rebuilt and checked against the paper for CIFAR-10, the benchmark whose
checkpoints are on the development machine: every cell and rank group of
`tab:geom`, and of `tab:main` except the Sheaf-FMTL row, whose printed s.d.
are rounded (.20/.20/.70/.20 against .16/.19/.66/.17 from the runs; with the
exact values its shared-class cell falls in the second rank group).
`tables-benchmarks` re-scores every run from its checkpoint
(`logs/<project>/<run_id>/checkpoints/`), so it needs the checkpoints of all
four benchmarks.
² Rebuilt rows identical to the paper's. The ρ = 0 % and 100 % rows are the
non-cooperative and DiSheafFRL runs of `tab:main`; when they are not re-scored
on the machine, the script uses the values printed in `tab:main`/`tab:geom` and
says so.
³ The routines train and plot the non-cooperative baseline and DiSheafFRL with
CenterNorm, ZCA whitening and unconstrained (OLS) maps. The paper's figures
(`pilot_study_shared.pdf`, `lambda_study_shared.pdf`) also show DiSheafFRL with
SWBN whitening, and split the accuracy into C_i∖C_j, C_i∩C_j and private; the
SWBN runs and that three-panel figure have no routine in this repository.
⁴ No config or routine in this repository.

### Experiments

| Config (`config/hydra/`) | Experiment |
|---|---|
| `paper_base.yaml` | shared protocol: label shift 1.0, shared pilots, CenterNorm, fully connected graph, post-hoc evaluation |
| `mnist_4agents_shift.yaml`, `fashionmnist_4agents_shift.yaml`, `femnist_4agents_shift.yaml`, `cifar10_4agents_full.yaml` | the four benchmarks of `tab:main` |
| `mnist_4agents_shift_{zca,ols}.yaml` | map ablation (ZCA whitening, unconstrained maps) |
| `femnist_2agents_bottleneck.yaml` + `bottleneck_femnist/d{8,16,32,64}.yaml` | compression sweep |
| `domainnet_6agents.yaml`, `domainnet_6agents_shift.yaml` + `dataset/domainnet.yaml` | DomainNet, one visual domain and one architecture per agent, without and with label shift |
| `dataset/`, `model/`, `optimizer/`, `orchestrator/` | building blocks (`orchestrator/ce_di_sheaf_frl.yaml`: DiSheafFRL communicating on a share of the epochs) |

`paper_base.yaml` holds everything the experiments share. Label shift is 1.0:
every agent trains only on its own classes. The pilots are unlabelled rows over
all classes, shared by every agent and removed from all training draws. Each
experiment config sets the dataset, the class groups, the models, the pilot
budget and `tuned_lmb`, the penalty weight of each method, which
`just train-benchmark` fills by grid search on the first seed. Every method is
scored with the same post-hoc Procrustes maps fitted on the pilots (ComFed with
its own learned maps).

FEMNIST (LEAF, Hugging Face `flwrlabs/femnist`, digits only) also splits by
writer: each agent trains on its own writers, the pilots come from a further
group of writers, and the test set is 10 % of every agent's rows, shared by
all agents. DomainNet (Hugging Face `wltjr1007/DomainNet`, cached by
`just domainnet-data`) gives every agent one of the six visual domains; the
test set pools 20 % of every domain, and `just report-domainnet` scores private
and communication accuracy separately on every test domain and agent pair.

### Running

```bash
# one cell of an experiment (no multirun)
uv run python scripts/multi_agent_experiment.py --config-name femnist_4agents_shift orchestrator=di_sheaf_frl seed=42
# a training routine, restricted with Hydra overrides
just train-ablation-comm-epochs orchestrator.comm_percentage=20 seed=42
# smoke-test a config: 1 epoch, a few batches, no wandb
just smoke domainnet_6agents orchestrator=di_sheaf_frl
# long runs: detached in tmux with a live dashboard
just launch comm_epochs 21 "just train-ablation-comm-epochs"
just attach comm_epochs        # or: just dash comm_epochs --once
```

Runs are grouped in wandb by method (`logger.group = study_name`, by default
the method's config name); the table and figure scripts read those groups, and
a re-run of the same cell replaces the older run. On macOS, add
`dataset.num_workers=0` (spawned DataLoader workers cost ~400 MB each).

## Citation

If you find this code useful for your research, please consider citing the following paper:

```
```

## Used Technologies

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
