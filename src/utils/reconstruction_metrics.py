"""Test-only perceptual metrics on full, raw RGB reconstructions."""

import torch


class ReconstructionLPIPS:
    """Official calibrated LPIPS-Alex v0.1, averaged over images.

    Kept outside the orchestrator's modules: these frozen evaluation weights
    must not enter optimizers, FedAvg, or model checkpoints.
    """

    def __init__(self, device):
        import lpips

        self.device = device
        self.metric = (
            lpips.LPIPS(net='alex', version='0.1', verbose=False)
            .to(device)
            .eval()
            .requires_grad_(False)
        )

    @torch.no_grad()
    def __call__(self, prediction, target, batch_size=64):
        if prediction.shape != target.shape or prediction.ndim != 4:
            raise ValueError('LPIPS requires matching NCHW image tensors.')
        if prediction.shape[1] != 3 or prediction.shape[0] == 0:
            raise ValueError('LPIPS requires a nonempty batch of RGB images.')
        total = 0.0
        # normalize=True performs the official [0, 1] -> [-1, 1] conversion.
        # No compositing, resizing, or masking is applied.
        with torch.autocast(
            device_type=torch.device(self.device).type, enabled=False
        ):
            for start in range(0, len(prediction), batch_size):
                pred = (
                    prediction[start : start + batch_size]
                    .detach()
                    .to(self.device, dtype=torch.float32)
                )
                truth = (
                    target[start : start + batch_size]
                    .detach()
                    .to(self.device, dtype=torch.float32)
                )
                total += self.metric(pred, truth, normalize=True).sum().item()
        return total / len(prediction)
