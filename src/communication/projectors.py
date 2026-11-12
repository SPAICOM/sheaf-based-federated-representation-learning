"""Per-agent linear maps to a common dimension, for methods that need one d.

FedProto and FedMuscle compare latents across agents, and FedRE shares one
whole decoder, so all three need every agent in the same dimension.  These
helpers give each agent its own ``Linear(d_i -> d)``; ComFed's ``P_i`` plays
the same role there (``src/orchestrators/comfed.py``).
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from collections.abc import Mapping


def first_linear(module: nn.Module) -> tuple[str, nn.Linear] | None:
    """Name and module of the first ``nn.Linear`` in ``module``, if any."""
    return next(
        (
            (name, m)
            for name, m in module.named_modules()
            if isinstance(m, nn.Linear)
        ),
        None,
    )


def latent_dim(agent: nn.Module) -> int | None:
    """Return the agent's latent dim, the input width of its decoder's first linear.

    ``None`` when the decoder does not start from a linear layer (e.g. a
    convolutional VAE decoder), so the dim cannot be read off the model.
    """
    found = first_linear(agent.decoder)
    return None if found is None else found[1].in_features


def common_dim(agents: Mapping[str, nn.Module], dim: int | None) -> int | None:
    """Resolve the common dimension of a network of agents.

    ``dim=None`` means the largest latent dim in the network, or ``None`` when
    every agent already has the same dim (no map needed, the original method).
    Raises ``TypeError`` if the dims differ but some cannot be read.
    """
    if dim is not None:
        if int(dim) < 1:
            raise ValueError(f'The common dim must be at least 1, got {dim}')
        return int(dim)
    dims = {key: latent_dim(agent) for key, agent in agents.items()}
    known = {d for d in dims.values() if d is not None}
    if len(known) <= 1:
        return None
    missing = sorted(key for key, d in dims.items() if d is None)
    if missing:
        raise TypeError(
            f'Cannot read the latent dim of agents {missing}: their decoder '
            'has no linear layer.'
        )
    return max(known)


def build_projectors(
    agents: Mapping[str, nn.Module], proj_dim: int | None
) -> nn.ModuleDict | None:
    """One bias-free ``Linear(d_i -> proj_dim)`` per agent, keyed like ``agents``.

    ``proj_dim=None`` projects to the largest latent dim in the network, and
    builds nothing (returns ``None``) when all dims are already equal.  The
    maps are unconstrained, as in the original papers.
    """
    proj_dim = common_dim(agents, proj_dim)
    if proj_dim is None:
        return None
    projectors = {}
    for key, agent in agents.items():
        d = latent_dim(agent)
        if d is None:
            raise TypeError(
                f'Cannot read the latent dim of agent {key}: its decoder has '
                'no linear layer.'
            )
        projectors[key] = nn.Linear(d, proj_dim, bias=False)
    return nn.ModuleDict(projectors)


def project(
    projectors: nn.ModuleDict | None, idx: int | str, z: torch.Tensor
) -> torch.Tensor:
    """``z`` through agent ``idx``'s projector, or unchanged without projectors."""
    return z if projectors is None else projectors[str(idx)](z)


def retarget_input(module: nn.Module, in_features: int) -> nn.Module:
    """Copy of ``module`` whose first linear layer takes ``in_features`` inputs.

    Every other layer and its weights are kept; the new first layer is freshly
    initialised.
    """
    found = first_linear(module)
    if found is None:
        raise TypeError('Module has no linear layer to retarget.')
    name, old = found
    out = copy.deepcopy(module)
    new = nn.Linear(
        int(in_features), old.out_features, bias=old.bias is not None
    )
    if not name:  # the module itself is the linear layer
        return new
    parent_name, _, child = name.rpartition('.')
    parent = out.get_submodule(parent_name) if parent_name else out
    setattr(parent, child, new)
    return out
