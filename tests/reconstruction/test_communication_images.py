"""Communication diagnostics use the actual predictions before map cleanup."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from scripts.reconstruction_experiment import MaskedReconstructionDiagnosticsCallback
from src.communication.alignment_mixin import PostTrainingAlignmentMixin
from src.orchestrators.base_orchestrator import BaseOrchestrator


@pytest.mark.parametrize('aligned', [False, True])
def test_communication_lpips_and_wandb_images(aligned):
    class Dataset:
        def __len__(self):
            return 2

        def _mask(self, idx):
            return torch.tensor([[[1., 0.], [1., 0.]]])

        def __getitem__(self, idx):
            target = torch.ones(3, 2, 2)
            return target * self._mask(idx), target, self._mask(idx)

    class Agent(torch.nn.Module):
        task_type = 'reconstruction'

        def __init__(self):
            super().__init__()
            self.decoder = torch.nn.Identity()

        def encode(self, x):
            return x

    dataset = Dataset()
    dm = SimpleNamespace(train_datasets={}, models=[0, 1],
                         test_datasets={0: dataset, 1: dataset})
    images = []
    callback = MaskedReconstructionDiagnosticsCallback(num_samples=1)
    trainer = SimpleNamespace(datamodule=dm, callbacks=[callback])
    callback._wandb_experiments = lambda trainer: [(
        SimpleNamespace(log=lambda payload: images.append(payload)),
        SimpleNamespace(Image=lambda image, caption: (image, caption)),
    )]
    module = SimpleNamespace(
        agents={'0': Agent(), '1': Agent()}, device=torch.device('cpu'),
        hparams=SimpleNamespace(neighbors={0: {1}, 1: {0}}),
        _alignment_maps={0: {1: torch.eye(2)}}, _trainer=trainer,
        send_message=lambda sender_idx, receiver_idx, Z_sender: Z_sender * 0.5,
        evaluate_heterophil_communication_accuracy=lambda *args, **kwargs: {},
    )
    module._log_communication_reconstruction = lambda *args: (
        BaseOrchestrator._log_communication_reconstruction(module, *args)
    )
    metric = lambda prediction, target: float((prediction - target).square().mean())
    evaluate = (PostTrainingAlignmentMixin._comm_accuracy_with_fitted_maps
                if aligned else BaseOrchestrator.evaluate_communication_accuracy)
    with patch('src.utils.reconstruction_metrics.ReconstructionLPIPS', return_value=metric):
        logs = evaluate(module, dm)
    assert logs['test/avg_private_lpips_full'] == pytest.approx(0.5)
    assert logs['test/avg_comm_lpips_full'] == pytest.approx(0.625)
    assert len(images) == 2
    key = 'test/communication_reconstruction_sender_1_receiver_0'
    grid, caption = images[0][key]
    assert 'receiver reconstruction from sender latent' in caption
    # torchvision grid: 2px padding, 2px-wide images; fourth column is prediction.
    assert torch.equal(grid[:, 2:4, 14:16], dataset[0][0] * 0.5)
