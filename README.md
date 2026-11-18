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

Every study is produced by a short sequence of `just` routines: the `train-*`
routines run the experiments (metrics and configs go to Weights & Biases,
`wandb login` once), and the last routine rebuilds the table or figure from
wandb and, where the paper re-scores runs, from the checkpoints. Run the
commands in the order given; `just --list` describes every routine. Runs logged
to a team entity are read with `WANDB_ENTITY=<entity> just <routine>`.

### Paper results

**Task accuracy on the four benchmarks (Tab. `tab:main`) and alignment
diagnostics (Tab. `tab:geom`).** Train every method on each benchmark (λ grid on
the first seed, then the other seeds at the best λ), then re-score all runs
and rank the methods:

```bash
just train-benchmark mnist
just train-benchmark fashionmnist
just train-benchmark femnist
just train-benchmark cifar10
just tables-benchmarks
```

**Robustness to semantic compression (Fig. `fig:compression`).** FEMNIST, two
agents with latent dimensions d and d/2:

```bash
just train-compression
just figure-compression
```

**Number of pilots (Fig. `fig:ablation-pilots`).** MNIST, non-cooperative and
DiSheafFRL with CenterNorm, ZCA, unconstrained (OLS) maps and SWBN:

```bash
just train-ablation-pilots
just figure-ablation-pilots
```

**Regularization coefficient (Fig. `fig:ablation-lambda`).** The λ = 0.01 points
are runs of the pilot study:

```bash
just train-ablation-pilots
just train-ablation-lambda
just figure-ablation-lambda
```

**Topology (Fig. `fig:ablation-topology`).** Star and ring; the fully
connected graph is the pilot study at 4096 pilots:

```bash
just train-ablation-pilots
just train-ablation-topology
just figure-ablation-topology
```

**Number of communication epochs (Tab. `tab:comm-epochs`).** The 0 % and 100 %
rows are the MNIST non-cooperative and DiSheafFRL runs of `tab:main`:

```bash
just train-benchmark mnist
just train-ablation-comm-epochs
just tables-benchmarks
just table-comm-epochs
```

**Compression factor of Sheaf-FMTL (App. Experimental details):**

```bash
just sheaf-fmtl-gamma mnist_4agents_shift fashionmnist_4agents_shift femnist_4agents_shift cifar10_4agents_full
```

**Emergence of global agreement (App. Global sections: `fig:spectrum`,
`fig:tv_normalized`, `fig:low_energy`).** No routine yet: the code of this
study is not in the repository.

### New experiments (not in the paper yet)

**Anchor selection.** MNIST, k-means and class-stratified anchors against the
original pilots; the original arm and the non-cooperative runs come from the
pilot study:

```bash
just train-ablation-pilots
just train-anchors
just figure-anchors
```

**MNIST parity: 10 digits vs even/odd.** Two agents on the same images, k-NN
probes of digit identity in the even/odd agent (see
`scripts/mnist_parity_knn.md`):

```bash
just train-parity
just figure-parity
```

**DomainNet, 6 agents × 6 domains,** without and with label shift:

```bash
just domainnet-data
just train-domainnet domainnet_6agents
just train-domainnet domainnet_6agents_shift
just report-domainnet
```

### Notes

- The figure routines of the MNIST ablations and of the anchor study redraw
  from the per-run scores kept in the repository; add `rescore` (e.g.
  `just figure-ablation-pilots rescore`) to re-score every checkpoint first
  (CUDA). `tables-benchmarks` and `table-comm-epochs` always re-score from the
  checkpoints (cached after the first time).
- Checked against the paper: `tables-benchmarks` reproduces every cell and
  rank group of the CIFAR-10 block of `tab:geom`, and of `tab:main` except the
  Sheaf-FMTL row, whose printed s.d. are rounded; `table-comm-epochs`
  reproduces the rows of `tab:comm-epochs` exactly.
- The "CenterNorm + OLS" curves of the pilot and λ figures score the OLS runs
  through the maps saved in training; that evaluation was made by a script
  that is not in the repository (`tmp/2026-11-12_mnist4_ols_learned.py`), so
  `rescore` reuses its logged results.
- The parity study's runs are not in the repository yet; its routines are
  smoke-tested (`just smoke-parity`).

### Experiments

| Config (`config/hydra/`) | Experiment |
|---|---|
| `paper_base.yaml` | shared protocol: label shift 1.0, shared pilots, CenterNorm, fully connected graph, post-hoc evaluation |
| `mnist_4agents_shift.yaml`, `fashionmnist_4agents_shift.yaml`, `femnist_4agents_shift.yaml`, `cifar10_4agents_full.yaml` | the four benchmarks of `tab:main` |
| `mnist_4agents_shift_{zca,ols}.yaml`, `mnist_4agents_mapstudy_swbn.yaml` | map ablation (ZCA whitening, unconstrained maps, learnable SWBN whitening) |
| `mnist_4agents_shift_{star,ring}.yaml` (+ `_topology_base.yaml`) | topology ablation |
| `mnist_4agents_shift_anchors_{kmeans,strat}.yaml` (+ `_anchors_base.yaml`) | anchor selection |
| `mnist_2agents_parity_d40x8.yaml` (+ `mnist_2agents_parity_base.yaml`) | MNIST parity: 10 digits vs even/odd |
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
