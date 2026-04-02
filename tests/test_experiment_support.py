import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from scripts import experiment
from src.agents.timm_classifier import TimmClassifier


class _FakeConfigLogger:
    def update(self, _cfg) -> None:
        return None


class _FakeLogger:
    def __init__(self) -> None:
        self.experiment = types.SimpleNamespace(config=_FakeConfigLogger())


class _FakeTrainer:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.fit_calls = []
        self.test_calls = []
        self.callback_metrics = {
            'validation/total_loss_epoch': torch.tensor(1.234),
        }

    def fit(self, orchestrator, datamodule=None) -> None:
        self.fit_calls.append((orchestrator, datamodule))

    def test(self, orchestrator, datamodule=None) -> None:
        self.test_calls.append((orchestrator, datamodule))


class _FakeAgent(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg


class _FakeDataModule:
    def __init__(self) -> None:
        self.num_classes = {'label': 10, 0: 3, 1: 4}
        self.models = [0, 1]
        self.input_dims = {'0': 16, '1': 16}

    def prepare_data(self) -> None:
        return None

    def setup(self) -> None:
        return None


class _RecorderModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_features = 6
        self.last_input = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.last_input = x
        return x.mean(dim=(2, 3), keepdim=True).repeat(1, 2, 1, 1)


class ExperimentSupportTests(unittest.TestCase):
    def test_experiment_uses_global_num_classes_and_runs_test(self):
        datamodule = _FakeDataModule()
        trainer_holder = {}
        model_cfgs = []

        cfg = OmegaConf.create(
            {
                'seed': 7,
                'dataset': {
                    '_target_': 'src.datamodules.ClassificationDataModule',
                    'repo': 'dummy',
                    'name': 'dummy',
                    'data_key': 'image',
                    'attributes': ['label'],
                    'pilot_split': 0.1,
                },
                'model': {
                    '_target_': 'src.agents.LatentClassifier',
                    'in_features': None,
                    'num_classes': None,
                    'latent_dim': 8,
                    'encoder_hidden_dims': [16],
                },
                'orchestrator': {
                    '_target_': 'src.orchestrators.SheafFRL',
                    'anchor_strategy': 'semantic_pilots',
                    'lambda_sheaf': 0.1,
                    'num_anchors': 8,
                    'parseval_normalization': False,
                    'l2_normalization': False,
                },
                'optimizer': {'_target_': 'torch.optim.SGD', 'lr': 0.1},
                'logger': {
                    '_target_': 'lightning.pytorch.loggers.WandbLogger',
                    'project': 'dummy-project',
                },
                'callbacks': {
                    'checkpoint': {
                        '_target_': 'lightning.pytorch.callbacks.ModelCheckpoint',
                    }
                },
                'trainer': {'max_epochs': 1},
                'graph': {
                    'neighbors_mode': 'fully_connected',
                    'seed': 7,
                    'p': 0.3,
                    'm': 2,
                    'neighbors': {},
                },
                'agents': {
                    0: {},
                    1: {},
                },
                'agent_rotations': {0: 0, 1: 90},
            }
        )

        def _instantiate_side_effect(*args, **kwargs):
            target = args[0]
            target_name = target.get('_target_')
            if target_name == 'lightning.pytorch.loggers.WandbLogger':
                return _FakeLogger()
            if target_name == 'lightning.pytorch.callbacks.ModelCheckpoint':
                return object()
            if target_name == 'src.datamodules.ClassificationDataModule':
                return datamodule
            if target_name == 'src.agents.LatentClassifier':
                model_cfgs.append(target)
                return _FakeAgent(target)
            if target_name == 'src.orchestrators.SheafFRL':
                return types.SimpleNamespace(
                    agents=kwargs['agents'],
                    neighbors=kwargs['neighbors'],
                    latent_dims=kwargs['latent_dims'],
                )
            raise AssertionError(f'unexpected instantiate target: {target_name}')

        def _trainer_factory(**kwargs):
            trainer = _FakeTrainer(**kwargs)
            trainer_holder['trainer'] = trainer
            return trainer

        with patch('scripts.experiment.instantiate', side_effect=_instantiate_side_effect):
            with patch('scripts.experiment.Trainer', side_effect=_trainer_factory):
                with patch('scripts.experiment.generate_neighbors', return_value={0: {1}, 1: {0}}):
                    with patch('scripts.experiment.remove_non_empty_dir'):
                        with patch('scripts.experiment.seed_everything'):
                            objective = experiment.main.__wrapped__(cfg)

        self.assertEqual(len(model_cfgs), 2)
        self.assertTrue(all(model_cfg.num_classes == 10 for model_cfg in model_cfgs))
        trainer = trainer_holder['trainer']
        self.assertEqual(len(trainer.fit_calls), 1)
        self.assertEqual(len(trainer.test_calls), 1)
        self.assertIs(trainer.fit_calls[0][1], datamodule)
        self.assertIs(trainer.test_calls[0][1], datamodule)
        self.assertAlmostEqual(objective, 1.234, places=6)

    def test_experiment_sanitizes_orchestrator_config_for_baseline_target(self):
        datamodule = _FakeDataModule()
        orchestrator_cfgs = []

        cfg = OmegaConf.create(
            {
                'seed': 7,
                'dataset': {
                    '_target_': 'src.datamodules.ClassificationDataModule',
                    'repo': 'dummy',
                    'name': 'dummy',
                    'data_key': 'image',
                    'attributes': ['label'],
                    'pilot_split': 0.0,
                },
                'model': {
                    '_target_': 'src.agents.LatentClassifier',
                    'in_features': None,
                    'num_classes': None,
                    'latent_dim': 8,
                    'encoder_hidden_dims': [16],
                },
                'orchestrator': {
                    '_target_': 'src.orchestrators.FederatedLearning',
                    'anchor_strategy': 'semantic_pilots',
                    'lambda_sheaf': 0.1,
                    'num_anchors': 8,
                    'parseval_normalization': False,
                    'l2_normalization': False,
                },
                'optimizer': {'_target_': 'torch.optim.SGD', 'lr': 0.1},
                'logger': {
                    '_target_': 'lightning.pytorch.loggers.WandbLogger',
                    'project': 'dummy-project',
                },
                'callbacks': {
                    'checkpoint': {
                        '_target_': 'lightning.pytorch.callbacks.ModelCheckpoint',
                    }
                },
                'trainer': {'max_epochs': 1},
                'graph': {
                    'neighbors_mode': 'fully_connected',
                    'seed': 7,
                    'p': 0.3,
                    'm': 2,
                    'neighbors': {},
                },
            }
        )

        def _instantiate_side_effect(*args, **kwargs):
            target = args[0]
            target_name = target.get('_target_')
            if target_name == 'lightning.pytorch.loggers.WandbLogger':
                return _FakeLogger()
            if target_name == 'lightning.pytorch.callbacks.ModelCheckpoint':
                return object()
            if target_name == 'src.datamodules.ClassificationDataModule':
                return datamodule
            if target_name == 'src.agents.LatentClassifier':
                return _FakeAgent(target)
            if target_name == 'src.orchestrators.FederatedLearning':
                orchestrator_cfgs.append(target)
                return types.SimpleNamespace(
                    agents=kwargs['agents'],
                    neighbors=kwargs['neighbors'],
                )
            raise AssertionError(f'unexpected instantiate target: {target_name}')

        with patch('scripts.experiment.instantiate', side_effect=_instantiate_side_effect):
            with patch('scripts.experiment.Trainer', side_effect=_FakeTrainer):
                with patch('scripts.experiment.generate_neighbors', return_value={0: {1}, 1: {0}}):
                    with patch('scripts.experiment.remove_non_empty_dir'):
                        with patch('scripts.experiment.seed_everything'):
                            experiment.main.__wrapped__(cfg)

        self.assertEqual(len(orchestrator_cfgs), 1)
        self.assertNotIn('anchor_strategy', orchestrator_cfgs[0])
        self.assertNotIn('lambda_sheaf', orchestrator_cfgs[0])
        self.assertNotIn('num_anchors', orchestrator_cfgs[0])

    def test_persist_run_results_writes_json_summary(self):
        cfg = OmegaConf.create(
            {
                'dataset': {'name': 'dummy'},
                'optimization': {
                    'objective_metric': 'validation/total_loss_epoch',
                },
            }
        )
        datamodule = _FakeDataModule()

        with TemporaryDirectory() as tmpdir:
            result_file = experiment._persist_run_results(
                results_path=Path(tmpdir),
                cfg=cfg,
                objective_metric_name='validation/total_loss_epoch',
                objective_value=1.23,
                test_results=[{'test/acc': torch.tensor(0.9)}],
                datamodule=datamodule,
            )

            self.assertTrue(result_file.exists())
            payload = result_file.read_text(encoding='utf-8')
            self.assertIn('"objective_value": 1.23', payload)
            self.assertIn('"test/acc": 0.899', payload)
            self.assertIn('"resolved_agent_classes"', payload)

    def test_timm_classifier_expands_grayscale_batches_to_rgb(self):
        encoder = _RecorderModule()

        with patch('src.agents.utils.timm.create_model', return_value=encoder):
            model = TimmClassifier(
                model_name='dummy-model',
                num_classes=3,
                pretrained=False,
                decoder_hidden_dims=[],
            )

        x = torch.randn(2, 1, 28, 28)
        features = model.encode(x)

        self.assertEqual(encoder.last_input.shape[:2], (2, 3))
        self.assertEqual(features.shape, (2, 6))


if __name__ == '__main__':
    unittest.main()
