"""LPIPS preprocessing, sample weighting and official CIFAR-size inference."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from src.utils.reconstruction_metrics import ReconstructionLPIPS


def test_lpips_weighting_normalization_and_no_grad():
    class SpyMetric(torch.nn.Module):
        def forward(self, prediction, target, normalize):
            assert normalize is True
            assert not torch.is_grad_enabled()
            assert not self.training
            assert prediction.dtype == torch.float32
            return (prediction - target).square().mean((1, 2, 3))

    with patch.dict(
        'sys.modules',
        {'lpips': SimpleNamespace(LPIPS=lambda **kw: SpyMetric())},
    ):
        metric = ReconstructionLPIPS('cpu')
    images = torch.zeros(3, 3, 32, 32, requires_grad=True)
    target = torch.zeros_like(images)
    target[-1] = 1
    assert metric(images, target, batch_size=2) == pytest.approx(1 / 3)
    with pytest.raises(ValueError, match='RGB'):
        metric(images[:, :1], target[:, :1])


@pytest.mark.slow
def test_official_pretrained_lpips_on_cifar():
    metric = ReconstructionLPIPS('cpu')
    torch.manual_seed(12)
    images = torch.rand(3, 3, 32, 32)
    assert metric(images, images, batch_size=2) == pytest.approx(0, abs=1e-6)
    expected = (
        metric.metric(images * 2 - 1, (1 - images) * 2 - 1).mean().item()
    )
    actual = metric(images, 1 - images, batch_size=2)
    assert actual > 0
    assert actual == pytest.approx(expected, rel=1e-5)
