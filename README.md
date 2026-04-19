# REPO TITLE


<h5 align="center">
    
[![ieee](https://img.shields.io/static/v1?label=IEEE+Paper&message=ID-HERE&color=0057b7&logo=ieee)](https://ieeexplore.ieee.org/document/ID-HERE)
[![arXiv](https://img.shields.io/badge/Arxiv-ID.HERE-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/CODE.HERE)
[![License](https://img.shields.io/badge/Code%20License-MIT-yellow)](https://github.com/SPAICOM/REPO-NAME-HERE/blob/main/LICENSE)

 <br>

</h5>

> [!TIP]
> 

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

## Current Implementation Notes

The most recent federated-learning changes are:

- `SheafFRL` now exposes `anchor_strategy` with two supported modes:
  - `pilots`: uses the shared pilot loaders already aligned by pilot sample ids; `use_prototypes=true` compresses each pilot batch to one prototype per observed class before alignment.
  - `batch_anchors`: reuses each agent's current task-batch latents, compresses them immediately to per-class prototypes, logs communication on those prototype payloads, and aligns neighboring agents by class labels across independent local batches.
- During training, `SheafFRL` computes the sheaf penalty with the current frozen Stiefel matrices `V`; the matrices themselves are updated only in `on_train_epoch_end()` from the cached training anchors accumulated during the epoch.
- `ClassificationDataModule` now supports `split_strategy=non_iid_with_margin`, which assigns every agent exactly `K` classes while ensuring every global class is assigned to at least one agent. For each assigned class, the partitioner reserves a safety margin of samples before applying the skewed allocation, and it uses the same partitioner for train, validation, and test. `starve_clients=true` still subsamples only the training split afterward.

For more detailed module-level documentation, see:

- [`src/orchestrators/README.md`](src/orchestrators/README.md)
- [`src/datamodules/README.md`](src/datamodules/README.md)
- [`src/utils/README.md`](src/utils/README.md)

## Citation

If you find this code useful for your research, please consider citing the following paper:

```
```

## Authors

- [EXAMPLE](https://scholar.google.com/citations?user=EXAMPLE)

## Used Technologies

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
