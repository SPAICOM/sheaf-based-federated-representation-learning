"""Per-agent linear maps to a common dimension, for methods that need one d.

FedProto and FedMuscle compare latents across agents, and FedRE shares one
whole decoder, so all three need every agent in the same dimension.  These
helpers give each agent its own ``Linear(d_i -> d)``; ComFed's ``P_i`` plays
the same role there (``src/orchestrators/comfed.py``).
"""

from __future__ import annotations

import copy
import warnings
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch.nn.utils import parametrizations

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


def build_projectors(
    agents: Mapping[str, nn.Module],
    proj_dim: int | None,
    *,
    orthogonal: bool = True,
) -> nn.ModuleDict | None:
    """One bias-free ``Linear(d_i -> proj_dim)`` per agent, keyed like ``agents``.

    ``orthogonal`` keeps each map semi-orthogonal (orthonormal columns when
    ``proj_dim >= d_i``, rows otherwise).  This matters when the loss is a
    squared distance (FedProto): an unconstrained map can shrink to zero and
    satisfy it without moving the latents.  Only ``proj_dim >= d_i`` makes the
    map an isometry.  Below that it is a projection that can rotate onto the
    directions where the loss is already small, and with few classes it can
    null a prototype loss outright, so a warning is raised.

    ``proj_dim=None`` builds nothing and returns ``None``.  It then raises if
    the latent dims are readable and differ, because the caller's
    cross-agent loss would crash on them later.
    """
    dims = {key: latent_dim(agent) for key, agent in agents.items()}
    if proj_dim is None:
        known = {d for d in dims.values() if d is not None}
        if len(known) > 1:
            raise ValueError(
                f'Agents have different latent dims {dims}; set proj_dim to '
                'project them to a common dimension.'
            )
        return None
    if int(proj_dim) < 1:
        raise ValueError(f'proj_dim must be at least 1, got {proj_dim}')
    missing = sorted(key for key, d in dims.items() if d is None)
    if missing:
        raise TypeError(
            f'Cannot read the latent dim of agents {missing}: their decoder '
            'has no linear layer.'
        )

    narrow = {key: d for key, d in dims.items() if d > int(proj_dim)}
    if narrow:
        warnings.warn(
            f'proj_dim={proj_dim} is below the latent dim of agents {narrow}: '
            'their projector can discard the directions the cross-agent loss '
            'acts on instead of moving the encoder. Use proj_dim >= '
            f'{max(dims.values())} for an isometric map.',
            UserWarning,
            stacklevel=2,
        )

    projectors = {}
    for key, d in dims.items():
        linear = nn.Linear(d, int(proj_dim), bias=False)
        projectors[key] = (
            parametrizations.orthogonal(linear) if orthogonal else linear
        )
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
