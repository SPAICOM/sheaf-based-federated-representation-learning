"""SimCLR building blocks (Chen et al. 2020): batch GPU augmentation, projection
head and NT-Xent loss, used as an optional local self-supervised term by every
orchestrator (``BaseOrchestrator.simclr_weight``).

Augmentation is applied per sample on the batch tensor (inputs in [0, 1], no
normalisation): random resized crop (scale 0.2-1, ratio 3/4-4/3) + horizontal
flip, colour jitter (brightness/contrast/saturation 0.4, p=0.8; hue omitted)
and random grayscale (p=0.2).
"""

from __future__ import annotations

import contextlib
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def simclr_augment(
    x: torch.Tensor,
    crop_scale: tuple[float, float] = (0.2, 1.0),
    jitter: float = 0.4,
    p_jitter: float = 0.8,
    p_gray: float = 0.2,
) -> torch.Tensor:
    """One random SimCLR view of each image in ``x`` (B, C, H, W), values in [0, 1]."""
    B, C = x.shape[0], x.shape[1]
    dev, dt = x.device, x.dtype
    u = lambda *s: torch.rand(*s, device=dev, dtype=dt)  # noqa: E731

    # random resized crop + flip as one affine resample (theta in [-1, 1] coords)
    area = crop_scale[0] + (crop_scale[1] - crop_scale[0]) * u(B)
    log_r = math.log(3 / 4) + (math.log(4 / 3) - math.log(3 / 4)) * u(B)
    ratio = log_r.exp()
    w = (area * ratio).sqrt().clamp(max=1.0)
    h = (area / ratio).sqrt().clamp(max=1.0)
    cx = (1 - w) * (2 * u(B) - 1)
    cy = (1 - h) * (2 * u(B) - 1)
    flip = torch.where(u(B) < 0.5, -1.0, 1.0).to(dt)
    theta = torch.zeros(B, 2, 3, device=dev, dtype=dt)
    theta[:, 0, 0] = w * flip
    theta[:, 0, 2] = cx
    theta[:, 1, 1] = h
    theta[:, 1, 2] = cy
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    out = F.grid_sample(x, grid, mode='bilinear', padding_mode='reflection',
                        align_corners=False)

    def gray(img):
        if img.shape[1] != 3:
            return img.mean(1, keepdim=True)
        wts = torch.tensor([0.299, 0.587, 0.114], device=dev, dtype=dt)
        return (img * wts.view(1, 3, 1, 1)).sum(1, keepdim=True)

    def factor():
        return (1 - jitter + 2 * jitter * u(B)).view(B, 1, 1, 1)

    apply = (u(B) < p_jitter).view(B, 1, 1, 1)
    j = (out * factor()).clamp(0, 1)                                    # brightness
    m = gray(j).mean(dim=(2, 3), keepdim=True)
    j = ((j - m) * factor() + m).clamp(0, 1)                            # contrast
    if C == 3:
        g = gray(j)
        j = ((j - g) * factor() + g).clamp(0, 1)                        # saturation
    out = torch.where(apply, j, out)
    if C == 3:
        to_gray = (u(B) < p_gray).view(B, 1, 1, 1)
        out = torch.where(to_gray, gray(out).expand_as(out), out)
    return out.contiguous()


def nt_xent(p1: torch.Tensor, p2: torch.Tensor, temperature: float) -> torch.Tensor:
    """NT-Xent over 2N projections; row i of ``p1`` is the positive of row i of ``p2``."""
    n = p1.shape[0]
    z = F.normalize(torch.cat([p1, p2]), dim=1)
    sim = z @ z.t() / temperature
    sim = sim.masked_fill(torch.eye(2 * n, device=z.device, dtype=torch.bool), float('-inf'))
    target = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, target)


def projection_head(in_dim: int, hidden_dim: int = 512, out_dim: int = 128) -> nn.Module:
    return nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True),
                         nn.Linear(hidden_dim, out_dim))


@contextlib.contextmanager
def frozen_running_stats(module: nn.Module):
    """Normalise with batch statistics but leave every running mean/var untouched
    (BatchNorm and CenterNorm), so augmented SimCLR views do not shift the
    statistics the encoder is evaluated with. Momentum 0 rather than restoring
    copies: a ``copy_`` would bump the buffers' version counter and break any
    graph that saved them (eval-mode BN on the pilots)."""
    mods = [m for m in module.modules()
            if m.training and hasattr(m, 'running_mean') and hasattr(m, 'momentum')]
    saved = [m.momentum for m in mods]
    for m in mods:
        m.momentum = 0.0
    try:
        yield
    finally:
        for m, mom in zip(mods, saved, strict=True):
            m.momentum = mom
