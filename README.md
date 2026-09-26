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

## Citation

If you find this code useful for your research, please consider citing the following paper:

```
```

## Used Technologies

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
