"""CNN variational autoencoder agent for federated learning."""

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_agent import BaseAgent
from .utils import VGGPerceptualLoss


def _image_size_tuple(img_size: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(img_size, int):
        return (img_size, img_size)
    if len(img_size) != 2:
        raise ValueError(f'img_size must be an int or a pair, got {img_size}')
    return (int(img_size[0]), int(img_size[1]))


def _activation(activation: type[nn.Module] | nn.Module) -> nn.Module:
    return activation() if isinstance(activation, type) else activation


def _ssim(
    y_hat: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    kernel_size: int = 11,
) -> torch.Tensor:
    """Differentiable batch SSIM with a local average window."""
    spatial = min(y_hat.shape[-2], y_hat.shape[-1])
    kernel_size = min(kernel_size, spatial)
    if kernel_size % 2 == 0:
        kernel_size -= 1
    if kernel_size < 3:
        return y_hat.new_tensor(1.0) - F.mse_loss(y_hat, target)

    padding = kernel_size // 2
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    mu_x = F.avg_pool2d(target, kernel_size, stride=1, padding=padding)
    mu_y = F.avg_pool2d(y_hat, kernel_size, stride=1, padding=padding)
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x_sq = (
        F.avg_pool2d(target * target, kernel_size, stride=1, padding=padding)
        - mu_x_sq
    )
    sigma_y_sq = (
        F.avg_pool2d(y_hat * y_hat, kernel_size, stride=1, padding=padding)
        - mu_y_sq
    )
    sigma_xy = (
        F.avg_pool2d(target * y_hat, kernel_size, stride=1, padding=padding)
        - mu_xy
    )

    numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
    return (numerator / denominator.clamp_min(1e-8)).mean()


class PixelShuffleUpsample(nn.Module):
    """Double spatial resolution with sub-pixel convolution."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels * 4, kernel_size=3, padding=1
        )
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shuffle(self.conv(x))


class CNNVAEResidualBlock(nn.Module):
    """Residual block used inside VAE encoder/decoder stages."""

    def __init__(
        self,
        channels: int,
        activation: type[nn.Module] | nn.Module = nn.SiLU,
        use_batchnorm: bool = True,
    ):
        super().__init__()
        bias = not use_batchnorm
        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, bias=bias
        )
        self.bn1 = nn.BatchNorm2d(channels) if use_batchnorm else nn.Identity()
        self.act1 = _activation(activation)
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, bias=bias
        )
        self.bn2 = nn.BatchNorm2d(channels) if use_batchnorm else nn.Identity()
        self.act2 = _activation(activation)
        if use_batchnorm:
            nn.init.zeros_(self.bn2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act1(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return self.act2(h + x)


class CNNVAEEncoder(nn.Module):
    """Convolutional posterior encoder returning ``mu`` by default."""

    def __init__(
        self,
        in_features: int,
        img_size: int | Sequence[int],
        hidden_dims: list[int] | None = None,
        latent_dim: int = 256,
        activation: type[nn.Module] | nn.Module = nn.SiLU,
        use_batchnorm: bool = True,
        use_residual: bool = False,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 128, 256]

        self.in_features = int(in_features)
        self.img_size = _image_size_tuple(img_size)
        self.hidden_dims = list(hidden_dims)
        self.out_features = int(latent_dim)

        layers: list[nn.Module] = []
        in_ch = self.in_features
        for out_ch in self.hidden_dims:
            layers.append(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1)
            )
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(_activation(activation))
            if use_residual:
                layers.append(
                    CNNVAEResidualBlock(
                        out_ch,
                        activation=activation,
                        use_batchnorm=use_batchnorm,
                    )
                )
            in_ch = out_ch

        self.conv_stack = nn.Sequential(*layers)
        with torch.no_grad():
            feat = self.conv_stack(
                torch.zeros(1, self.in_features, *self.img_size)
            )
        self.feature_shape = tuple(feat.shape[1:])
        feature_dim = int(feat.flatten(1).shape[1])
        self.fc_mu = nn.Linear(feature_dim, self.out_features)
        self.fc_logvar = nn.Linear(feature_dim, self.out_features)

    def posterior(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.conv_stack(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu, _logvar = self.posterior(x)
        return mu


class CNNVAEDecoder(nn.Module):
    """PixelShuffle decoder from a flat latent vector to an image."""

    def __init__(
        self,
        out_features: int,
        img_size: int | Sequence[int],
        feature_shape: Sequence[int],
        hidden_dims: list[int],
        latent_dim: int = 256,
        activation: type[nn.Module] | nn.Module = nn.SiLU,
        use_batchnorm: bool = True,
        use_residual: bool = False,
    ):
        super().__init__()
        self.out_features = int(out_features)
        self.img_size = _image_size_tuple(img_size)
        self.feature_shape = tuple(int(v) for v in feature_shape)
        self.hidden_dims = list(hidden_dims)
        self.latent_dim = int(latent_dim)

        feature_dim = 1
        for dim in self.feature_shape:
            feature_dim *= dim
        self.unproject = nn.Linear(self.latent_dim, feature_dim)

        reversed_dims = list(reversed(self.hidden_dims))
        layers: list[nn.Module] = []
        for i in range(len(reversed_dims) - 1):
            layers.append(
                PixelShuffleUpsample(reversed_dims[i], reversed_dims[i + 1])
            )
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(reversed_dims[i + 1]))
            layers.append(_activation(activation))
            if use_residual:
                layers.append(
                    CNNVAEResidualBlock(
                        reversed_dims[i + 1],
                        activation=activation,
                        use_batchnorm=use_batchnorm,
                    )
                )
        layers.append(PixelShuffleUpsample(reversed_dims[-1], out_features))

        self.deconv_stack = nn.Sequential(*layers)
        self.output_conv = nn.Conv2d(
            out_features, out_features, kernel_size=3, padding=1
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.unproject(z).view(-1, *self.feature_shape)
        h = self.deconv_stack(h)
        h = F.interpolate(
            h, size=self.img_size, mode='bilinear', align_corners=False
        )
        return torch.sigmoid(self.output_conv(h))


class CNNVariationalAE(BaseAgent):
    """CNN beta-VAE with the standard agent training interface.

    ``encode`` returns the posterior mean, so sheaf alignment and
    communication use deterministic latents. ``forward`` samples only while
    the module is in training mode and returns the reconstruction tensor
    expected by orchestrators.
    """

    task_type = 'reconstruction'

    def __init__(
        self,
        in_features: int,
        img_size: int | Sequence[int],
        out_features: int | None = None,
        latent_dim: int = 256,
        encoder_hidden_dims: list[int] | None = None,
        beta: float = 1e-4,
        free_bits: float = 0.0,
        reconstruction_l1_weight: float = 0.0,
        sigma_vae: bool = False,
        sample_posterior: bool = True,
        missing_region_loss: bool = False,
        visible_loss_weight: float = 0.25,
        normalize_kl: bool = False,
        beta_warmup_steps: int = 0,
        pilot_loss_weight: float = 0.0,
        mmd_weight: float = 0.0,
        ssim_weight: float = 0.0,
        ssim_warmup_steps: int = 0,
        perceptual_weight: float = 0.0,
        perceptual_warmup_steps: int = 0,
        activation: type[nn.Module] | nn.Module = nn.SiLU,
        use_batchnorm: bool = True,
        use_residual: bool = False,
        weight_decay: float = 0.0,
        pixel_max: float = 1.0,
        num_classes: int | None = None,
    ):
        super().__init__()
        if encoder_hidden_dims is None:
            encoder_hidden_dims = [64, 128, 256]
        output_features = int(
            in_features if out_features is None else out_features
        )

        self._encoder = CNNVAEEncoder(
            in_features=in_features,
            img_size=img_size,
            hidden_dims=encoder_hidden_dims,
            latent_dim=latent_dim,
            activation=activation,
            use_batchnorm=use_batchnorm,
            use_residual=use_residual,
        )
        self._decoder = CNNVAEDecoder(
            out_features=output_features,
            img_size=img_size,
            feature_shape=self._encoder.feature_shape,
            hidden_dims=encoder_hidden_dims,
            latent_dim=latent_dim,
            activation=activation,
            use_batchnorm=use_batchnorm,
            use_residual=use_residual,
        )
        self.perceptual_loss = (
            VGGPerceptualLoss() if perceptual_weight > 0.0 else None
        )

        self.latent_dim = int(latent_dim)
        self.in_features = int(in_features)
        self.out_features = output_features
        self.beta = float(beta)
        self.free_bits = float(free_bits)
        self.reconstruction_l1_weight = float(reconstruction_l1_weight)
        self.sigma_vae = bool(sigma_vae)
        self.sample_posterior = bool(sample_posterior)
        self.missing_region_loss = bool(missing_region_loss)
        self.visible_loss_weight = float(visible_loss_weight)
        self.normalize_kl = bool(normalize_kl)
        self.beta_warmup_steps = int(beta_warmup_steps)
        self.pilot_loss_weight = float(pilot_loss_weight)
        self.mmd_weight = float(mmd_weight)
        self.ssim_weight = float(ssim_weight)
        self.ssim_warmup_steps = int(ssim_warmup_steps)
        self.perceptual_weight = float(perceptual_weight)
        self.perceptual_warmup_steps = int(perceptual_warmup_steps)
        self.weight_decay = float(weight_decay)
        self.pixel_max = float(pixel_max)
        self.register_buffer(
            '_loss_step', torch.zeros((), dtype=torch.long), persistent=False
        )
        self._last_input: torch.Tensor | None = None
        self._last_mu: torch.Tensor | None = None
        self._last_logvar: torch.Tensor | None = None
        self.last_loss_components: dict[str, torch.Tensor] = {}

    @property
    def encoder(self) -> nn.Module:
        return self._encoder

    @property
    def decoder(self) -> nn.Module:
        return self._decoder

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(x, torch.Tensor) and x.ndim == 3:
            x = x.unsqueeze(0)
        self._last_input = x
        mu, logvar = self._encoder.posterior(x)
        self._last_mu = mu
        self._last_logvar = logvar
        return mu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = self.encode(x)
        return self.decode_posterior(mu)

    def decode_posterior(self, mu: torch.Tensor) -> torch.Tensor:
        """Sample for local training; keep encode() deterministic for messages."""
        z = (
            self.reparameterize(mu, self._last_logvar)
            if self.training and self.sample_posterior
            else mu
        )
        return self._decoder(z)

    def kl_penalty(self) -> torch.Tensor:
        if self._last_mu is None or self._last_logvar is None:
            param = next(self.parameters(), None)
            return torch.tensor(0.0) if param is None else param.new_zeros(())
        mu, logvar = self._last_mu, self._last_logvar
        kl_per_dim = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).mean(
            dim=0
        )
        kl = torch.clamp(kl_per_dim, min=self.free_bits)
        return kl.mean() if self.normalize_kl else kl.sum()

    def weight_decay_penalty(self) -> torch.Tensor:
        if self.weight_decay <= 0.0:
            param = next(self.parameters(), None)
            return torch.tensor(0.0) if param is None else param.new_zeros(())
        return self.weight_decay * sum(
            p.pow(2).sum() for p in self.parameters() if p.requires_grad
        )

    @staticmethod
    def mmd_penalty(z: torch.Tensor) -> torch.Tensor:
        prior = torch.randn_like(z)
        dim = z.size(1)
        z_dist = ((z.unsqueeze(1) - z.unsqueeze(0)) ** 2).sum(-1)
        p_dist = ((prior.unsqueeze(1) - prior.unsqueeze(0)) ** 2).sum(-1)
        zp_dist = ((z.unsqueeze(1) - prior.unsqueeze(0)) ** 2).sum(-1)
        k_zz = torch.exp(-z_dist / (2.0 * dim)).mean()
        k_pp = torch.exp(-p_dist / (2.0 * dim)).mean()
        k_zp = torch.exp(-zp_dist / (2.0 * dim)).mean()
        return k_zz + k_pp - 2 * k_zp

    def _warmup_weight(self, target_weight: float, warmup_steps: int) -> float:
        if target_weight <= 0.0:
            return 0.0
        if warmup_steps <= 0:
            return target_weight
        progress = min(1.0, float(self._loss_step.item()) / warmup_steps)
        return target_weight * progress

    def _advance_loss_step(self) -> None:
        if self.training and torch.is_grad_enabled():
            self._loss_step += 1

    def reconstruction_mse(self, y_hat, target, eval_mask=None):
        """Region-normalized inpainting loss, or legacy full-image MSE."""
        if not self.missing_region_loss:
            return F.mse_loss(y_hat, target)
        if eval_mask is None:
            raise ValueError(
                'Inpainting loss requires the visible input mask.'
            )
        missing, visible = self.region_mses(y_hat, target, eval_mask)
        return missing + self.visible_loss_weight * visible

    @staticmethod
    def region_mses(y_hat, target, eval_mask):
        """Missing and visible errors, averaged per sample with empty regions zero."""
        mask = eval_mask.to(y_hat)
        if mask.ndim == y_hat.ndim - 1:
            mask = mask.unsqueeze(1)
        mask = mask.expand_as(y_hat)
        error = (y_hat - target).square()
        axes = tuple(range(1, error.ndim))

        def regional(weights):
            return (
                (error * weights).sum(axes) / weights.sum(axes).clamp_min(1)
            ).mean()

        return regional(1 - mask), regional(mask)

    def compute_loss(self, y_hat, y, eval_mask=None) -> torch.Tensor:
        """
        Reconstruction plus posterior regularization. Regional MSE is opt-in;
        auxiliary L1/SSIM/perceptual losses, when enabled, use the full image.
        """
        target = (
            y
            if isinstance(y, torch.Tensor) and y.shape == y_hat.shape
            else self._last_input
        )
        if target is None:
            raise RuntimeError(
                'CNNVariationalAE.compute_loss requires a cached input from '
                'the most recent encode/forward call.'
            )

        target = target.to(y_hat.device)
        mse = F.mse_loss(y_hat, target)

        task_mse = self.reconstruction_mse(y_hat, target, eval_mask)
        mse_term = (
            0.5 * torch.log(task_mse + 1e-8) if self.sigma_vae else task_mse
        )
        recon_loss = mse_term

        l1 = y_hat.new_zeros(())
        if self.reconstruction_l1_weight > 0.0:
            l1 = F.l1_loss(y_hat, target)
            recon_loss = recon_loss + self.reconstruction_l1_weight * l1

        ssim_weight = self._warmup_weight(
            self.ssim_weight, self.ssim_warmup_steps
        )
        ssim_loss = y_hat.new_zeros(())
        if ssim_weight > 0.0:
            ssim_loss = 1.0 - _ssim(
                y_hat,
                target,
                data_range=self.pixel_max,
            )
            recon_loss = recon_loss + ssim_weight * ssim_loss

        perceptual_weight = self._warmup_weight(
            self.perceptual_weight, self.perceptual_warmup_steps
        )
        perceptual = y_hat.new_zeros(())
        if self.perceptual_loss is not None and perceptual_weight > 0.0:
            perceptual = self.perceptual_loss(y_hat, target)
            recon_loss = recon_loss + perceptual_weight * perceptual

        mmd = (
            self.mmd_weight * self.mmd_penalty(self._last_mu)
            if self.mmd_weight > 0.0 and self._last_mu is not None
            else y_hat.new_zeros(())
        )
        kl = self.kl_penalty()
        weight_decay = self.weight_decay_penalty()
        beta = self._warmup_weight(self.beta, self.beta_warmup_steps)
        loss = recon_loss + beta * kl + mmd + weight_decay

        self.last_loss_components = {
            'mse': mse.detach(),
            'mse_term': mse_term.detach(),
            'l1': l1.detach(),
            'reconstruction': recon_loss.detach(),
            'kl': kl.detach(),
            'kl_weighted': (beta * kl).detach(),
            'beta': y_hat.new_tensor(beta),
            'ssim_loss': ssim_loss.detach(),
            'ssim_weight': y_hat.new_tensor(ssim_weight),
            'perceptual': perceptual.detach(),
            'perceptual_weight': y_hat.new_tensor(perceptual_weight),
            'mmd': mmd.detach(),
            'weight_decay': weight_decay.detach(),
            'total': loss.detach(),
        }
        if eval_mask is not None:
            missing, visible = self.region_mses(
                y_hat.detach(), target, eval_mask
            )
            self.last_loss_components.update(
                mse_missing=missing, mse_visible=visible
            )
        self._advance_loss_step()
        return loss

    def task_performance(self, y_hat, y, eval_mask=None) -> torch.Tensor:
        target = (
            y
            if isinstance(y, torch.Tensor) and y.shape == y_hat.shape
            else self._last_input
        )
        if target is None:
            return torch.tensor(float('nan'), device=y_hat.device)
        target = target.to(y_hat.device)
        mse = F.mse_loss(y_hat, target).detach().clamp_min(1e-10)
        return 10.0 * torch.log10((self.pixel_max**2) / mse)
