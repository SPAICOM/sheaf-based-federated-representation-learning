# Sheaf-Based Federated Representation Learning


<h5 align="center">
    
[![ieee](https://img.shields.io/static/v1?label=IEEE+Paper&message=ID-HERE&color=0057b7&logo=ieee)](https://ieeexplore.ieee.org/document/ID-HERE)
[![arXiv](https://img.shields.io/badge/Arxiv-ID.HERE-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/CODE.HERE)
[![License](https://img.shields.io/badge/Code%20License-MIT-yellow)](https://github.com/SPAICOM/REPO-NAME-HERE/blob/main/LICENSE)

 <br>

</h5>

> [!TIP]
> Heterogeneous federated systems require agents to learn and exchange informative representations despite differences in data distributions, sensing modalities, model architectures, latent dimensionalities, and local learning objectives. To address this challenge, we propose Sheaf-based Federated Representation Learning (SFRL), a general framework that jointly optimizes local objectives with a manifold-constrained geometric alignment regularizer based on learnable sheaf restriction maps. Unlike most existing approaches, SFRL does not assume a shared global latent space. Instead, global consistency emerges from the alignment of neighboring latent representations through orthogonal transformations and isometric embeddings. This alignment is enforced by a quadratic gluing regularizer induced by the sheaf Laplacian, whose learnable restriction maps adapt the geometry to the observed data.
The penalty is evaluated on a small set of shared pilot samples, ensuring scalability and communication efficiency. We develop a decentralized algorithm for solving SFRL, termed Sheaf-FRL, which alternates between gradient updates of the local models and closed-form Procrustes updates of the edge-wise restriction maps. We further establish convergence of Sheaf-FRL to first-order stationary points in both deterministic and stochastic settings. As an application, we consider a cooperative classification task in the context of semantic communication, under model and data heterogeneity.
Our results show that Sheaf-FRL outperforms baseline approaches in terms of local and post-communication classification accuracy across different levels of local distribution shift and exhibits greater robustness to latent-space dimensionality compression.

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

## Authors

- [EXAMPLE](https://scholar.google.com/citations?user=EXAMPLE)

## Used Technologies

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
