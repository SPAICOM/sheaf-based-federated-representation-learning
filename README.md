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

Every experiment is one Hydra config run by `scripts/multi_agent_experiment.py`
in multirun mode, over the methods and seeds listed in the config's
`hydra.sweeper.params`. Metrics and configs go to Weights & Biases
(`wandb login` once); the figure and table scripts read them back from the
wandb remote, so they can be rebuilt on any machine.

| Study | Config | Train | Result |
|---|---|---|---|
| Accuracy vs bottleneck dimension (2 CIFAR-10 agents) | `cifar10_2agents_bottleneck` | `just bottleneck` | `just plot-bottleneck` |
| Heterogeneity table, MNIST | `mnist_4agents_shift` | `just hetero mnist` | `just table-hetero` |
| Heterogeneity table, FashionMNIST | `fashionmnist_4agents_shift` | `just hetero fashionmnist` | `just table-hetero` |
| Heterogeneity table, FEMNIST | `femnist_4agents_shift` | `just hetero femnist` | `just table-hetero` |
| Heterogeneity table, CIFAR-10 | `cifar10_4agents_full` | `just hetero cifar10` | `just table-hetero` |
| MNIST ablation: pilot budget | `mnist_4agents_shift` | `just mnist-ablation-pilots` | `just mnist-pilots-summary` |
| MNIST ablation: penalty weight | `mnist_4agents_shift` | `just mnist-ablation-lambda` | `just mnist-lambda-summary` |
| MNIST ablation: restriction maps | `mnist_4agents_shift_{zca,ols}` | `just mnist-ablation-maps zca` (or `ols`) | `just mnist-maps-summary` |

Methods: non-cooperative training with post-hoc alignment, Sheaf-FMTL, ComFed,
FedProto, FedMuscle, FedRE, SheafFRL and DiSheafFRL. Every method is scored
with the same post-hoc Procrustes map fitted on the pilots.

### Configs

```
config/hydra/
  paper_base.yaml                  shared protocol: shift 1.0, shared pilots, CenterNorm, eval
  cifar10_2agents_bottleneck.yaml  + bottleneck/d{8..256}.yaml (agent latent dims d and d/2)
  mnist_4agents_shift.yaml         + mnist_4agents_shift_{zca,ols}.yaml (map ablation)
  fashionmnist_4agents_shift.yaml
  femnist_4agents_shift.yaml
  cifar10_4agents_full.yaml
  dataset/ model/ optimizer/ orchestrator/
```

`paper_base.yaml` holds everything the experiments share. Label shift is 1.0:
every agent trains only on its own classes and is tested on the full official
test split. The pilots are unlabelled rows over all classes, shared by every
agent and removed from all training draws. Encoders are heterogeneous with
CenterNorm at the bottleneck and no whitening. Each experiment config sets
the dataset, the class groups, the models, the pilot budget and `tuned_lmb`,
the penalty weight of each method.

FEMNIST (LEAF, from the Hugging Face Hub `flwrlabs/femnist`, digits only)
also splits by writer. Each agent trains on its own disjoint set of writers,
so no handwriting style is shared between agents. The pilots come from a
fifth group of writers, and the test set is 10% of every agent's rows,
shared by all agents.

### Running

```bash
# one cell of a study (single run, no multirun)
uv run python scripts/multi_agent_experiment.py --config-name femnist_4agents_shift orchestrator=di_sheaf_frl seed=42
# a whole study, restricted to some methods / seeds
just hetero cifar10 orchestrator=non_cooperative,di_sheaf_frl seed=42
# MNIST map ablation, lambda sweep instead of pilot sweep
just mnist-ablation-maps ols dataset.pilot_num_samples=4096 orchestrator.max_lmb=0.003,0.01,0.03,0.1,0.3,1
```

Before a long run, smoke-test the config (`just smoke <config> orchestrator=<method>`),
then start it detached with `just launch <sweep> <n_jobs> "just hetero mnist"`
and follow it with `just dash <sweep> --once`. Runs are grouped in wandb by
method (`logger.group = study_name`, default the method's config name). The
table and the bottleneck plot only read groups named after a method, and a
re-run of the same cell replaces the older run.

### Seen/unseen-class breakdown

Under label shift every agent misses some classes, so each run also logs its
test accuracies split by class scope under `test/scope/`
(`src/utils/class_scope.py`): private accuracy on the agent's own classes vs
the rest, and communication accuracy on classes both agents trained on
(`both_seen`), only the receiver trained on (`tx_unseen_rx_seen`), only the
sender (`tx_seen_rx_unseen`) or neither (`both_unseen`). Each is logged per
directed edge, per agent (as receiver `rx` and as sender `tx`) and pooled over
the network, with the sample counts (`n_*`).

```bash
just class-scope cifar10_4agents_full                  # network / agent / edge tables + plots → results/<project>/class_scope/
just class-scope cifar10_2agents_bottleneck --x latent_d
just rescore-class-scope cifar10_4agents_full 3mgkbida # older runs: recompute from the checkpoint (add --write to store in wandb)
```

`just table-hetero` also writes `hetero_table_scope.{md,tex}`, and
`just plot-bottleneck` has a third panel for the receiver-only classes.

### Retained results

- MNIST ablations: [summary](results/mnist_4agents_shift/README.md), three
  seeds (42/43/44), with the historical wandb groups the ablation recipes
  reuse. The summary scripts check the exact run counts, so new replicates
  must be added deliberately.
- FashionMNIST pilot sweep: `just fmnist-pilots-summary` reads the retained
  runs from the team entity named in the script.

## Citation

If you find this code useful for your research, please consider citing the following paper:

```
```

## Used Technologies

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
