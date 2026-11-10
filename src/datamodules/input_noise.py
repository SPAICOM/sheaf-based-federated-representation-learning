"""Agent-specific input corruptions (a fixed per-agent "sensor").

Each agent sees every image -- train, val, test AND its copy of the shared
pilots -- through its own deterministic corruption, so the agents observe the
same scene through different modalities. Specs (dataset.agent_input_noise):

    {type: grayscale}                          # luminance, replicated to 3 ch
    {type: blur, kernel_size: 5, sigma: 1.0}   # fixed-sigma Gaussian blur
    {type: mask, region: left, fraction: 0.5}  # occlude a fixed region (fill 0)
        region: left | right | top | bottom | center | border
    {type: crop, size: 16}                     # centre crop, resized back (zoom)
    {type: pixelate, size: 8}                  # down- then up-sample (low res)
    {type: compose, steps: [{...}, {...}]}     # apply several in order

Applied to the PIL image before any augmentation, and kept separate from the
dataset's ``transform`` on purpose: code that turns augmentation off (e.g. the
ZCA refit / pilot sweeps set ``ds.transform = None``) must not remove it.
"""

from __future__ import annotations

from collections.abc import Callable

from PIL import Image, ImageDraw
from torchvision import transforms


class _FixedMask:
    """Blank a fixed region of the image (same pixels for every image)."""

    def __init__(self, region: str, fraction: float, fill: int):
        if region not in ('left', 'right', 'top', 'bottom', 'center', 'border'):
            raise ValueError(f'Unknown mask region {region!r}')
        if not 0.0 < fraction < 1.0:
            raise ValueError('mask fraction must be in (0, 1)')
        self.region, self.fraction, self.fill = region, fraction, fill

    def _box(self, w: int, h: int) -> tuple[int, int, int, int]:
        fw, fh = round(w * self.fraction), round(h * self.fraction)
        if self.region == 'left':
            return 0, 0, fw, h
        if self.region == 'right':
            return w - fw, 0, w, h
        if self.region == 'top':
            return 0, 0, w, fh
        if self.region == 'bottom':
            return 0, h - fh, w, h
        # centre square covering `fraction` of the area ('center'), or the
        # kept centre square covering 1 - fraction of it ('border')
        area = self.fraction if self.region == 'center' else 1.0 - self.fraction
        side = round((w * h * area) ** 0.5)
        x0, y0 = (w - side) // 2, (h - side) // 2
        return x0, y0, x0 + side, y0 + side

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        fill = (self.fill,) * len(img.getbands())
        x0, y0, x1, y1 = self._box(w, h)
        if self.region == 'border':
            # blank everything outside the centre square
            out = Image.new(img.mode, img.size, fill)
            out.paste(img.crop((x0, y0, x1, y1)), (x0, y0))
            return out
        out = img.copy()
        ImageDraw.Draw(out).rectangle((x0, y0, x1 - 1, y1 - 1), fill=fill)
        return out


class _Resample:
    """Centre crop to `crop` px (zoom) and/or downsample to `low` px, then back to full size."""

    def __init__(self, crop: int | None = None, low: int | None = None):
        self.crop, self.low = crop, low

    def __call__(self, img: Image.Image) -> Image.Image:
        size = img.size
        if self.crop is not None:
            img = transforms.functional.center_crop(img, self.crop)
        if self.low is not None:
            img = img.resize((self.low, self.low), Image.BILINEAR)
            return img.resize(size, Image.NEAREST)
        return img.resize(size, Image.BILINEAR)


def build_input_noise(spec) -> Callable | None:
    if spec is None:
        return None
    spec = dict(spec)
    kind = str(spec.pop('type', 'none')).lower()
    if kind == 'none':
        return None
    if kind == 'grayscale':
        return transforms.Grayscale(num_output_channels=3)
    if kind == 'blur':
        k = int(spec.pop('kernel_size', 5))
        sigma = float(spec.pop('sigma', 1.0))
        # A fixed sigma makes GaussianBlur deterministic.
        return transforms.GaussianBlur(kernel_size=k, sigma=sigma)
    if kind == 'mask':
        return _FixedMask(
            str(spec.pop('region', 'left')), float(spec.pop('fraction', 0.5)), int(spec.pop('fill', 0))
        )
    if kind == 'crop':
        return _Resample(crop=int(spec.pop('size', 16)))
    if kind == 'pixelate':
        return _Resample(low=int(spec.pop('size', 8)))
    if kind == 'compose':
        steps = [build_input_noise(step) for step in spec.pop('steps')]
        return transforms.Compose([t for t in steps if t is not None])
    raise ValueError(
        f"Unknown input noise type {kind!r} (grayscale | blur | mask | crop | pixelate | compose | none)"
    )
