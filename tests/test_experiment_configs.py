import unittest
from pathlib import Path

from omegaconf import OmegaConf


CONFIG_ROOT = Path('config/hydra')


class ExperimentConfigTests(unittest.TestCase):
    def _load(self, relative_path: str):
        return OmegaConf.load(CONFIG_ROOT / relative_path)

    def test_rotated_mnist_experiment_has_four_agents_and_pilots(self):
        cfg = self._load('rotated_mnist_experiment.yaml')
        self.assertEqual(cfg.defaults[1].dataset, 'mnist')
        self.assertEqual(len(cfg.agent_rotations), 4)
        self.assertEqual(cfg.defaults[2].model, 'cnn_classifier')
        self.assertGreater(cfg.dataset.pilot_split, 0)

    def test_rotated_mnist_sweep_covers_all_anchor_strategies(self):
        cfg = self._load('rotated_mnist_anchor_sweep.yaml')
        params = cfg.hydra.sweeper.params['orchestrator.anchor_strategy']
        self.assertIn('semantic_pilots', params)
        self.assertIn('clustered_pilots', params)
        self.assertIn('dynamic', params)

    def test_manual_cifar10_experiment_uses_class_partition(self):
        cfg = self._load('hetero_cifar10_manual_experiment.yaml')
        self.assertEqual(cfg.defaults[1].dataset, 'cifar10')
        self.assertEqual(cfg.defaults[2].model, 'cnn_classifier')
        self.assertEqual(cfg.dataset.split_strategy, 'class_partition')
        self.assertEqual(len(cfg.agent_classes), 4)
        self.assertGreater(cfg.dataset.pilot_split, 0)

    def test_dirichlet_cifar10_experiment_uses_non_iid_split(self):
        cfg = self._load('hetero_cifar10_dirichlet_experiment.yaml')
        self.assertEqual(cfg.defaults[1].dataset, 'cifar10')
        self.assertEqual(cfg.defaults[2].model, 'cnn_classifier')
        self.assertEqual(cfg.dataset.split_strategy, 'non_iid')
        self.assertEqual(cfg.dataset.n_agents, 4)
        self.assertGreater(cfg.dataset.classes_per_agent, 0)
        self.assertGreater(cfg.dataset.pilot_split, 0)

    def test_base_cifar10_experiment_uses_cnn_agents(self):
        cfg = self._load('timm_agents_experiment.yaml')
        self.assertEqual(cfg.defaults[1].dataset, 'cifar10')
        self.assertEqual(cfg.defaults[2].model, 'timm_classifier')

    def test_cifar10_sweeps_cover_all_anchor_strategies(self):
        for config_name in [
            'hetero_cifar10_manual_anchor_sweep.yaml',
            'hetero_cifar10_dirichlet_anchor_sweep.yaml',
        ]:
            cfg = self._load(config_name)
            params = cfg.hydra.sweeper.params['orchestrator.anchor_strategy']
            self.assertIn('prototype', params)
            self.assertIn('semantic_pilots', params)
            self.assertIn('clustered_pilots', params)

    def test_overlap_cifar10_experiment_uses_shared_class_partition(self):
        cfg = self._load('hetero_cifar10_overlap_experiment.yaml')
        self.assertEqual(cfg.defaults[1].dataset, 'cifar10')
        self.assertEqual(cfg.defaults[2].model, 'cnn_classifier')
        self.assertEqual(cfg.dataset.split_strategy, 'class_partition')
        self.assertEqual(cfg.dataset.n_agents, 4)
        self.assertEqual(cfg.dataset.shared_classes, 2)
        self.assertEqual(len(cfg.agents), 4)

    def test_overlap_cifar10_optuna_sweep_tunes_sheaf_hyperparameters(self):
        cfg = self._load('hetero_cifar10_overlap_optuna.yaml')
        params = cfg.hydra.sweeper.params
        self.assertIn('orchestrator.lambda_sheaf', params)
        self.assertIn('orchestrator.anchor_strategy', params)
        self.assertIn('orchestrator.parseval_normalization', params)
        self.assertIn('orchestrator.l2_normalization', params)
        self.assertIn('orchestrator.num_anchors', params)
        self.assertIn('dataset.shared_classes', params)

    def test_overlap_cifar10_orchestrator_sweep_covers_baselines(self):
        cfg = self._load('hetero_cifar10_overlap_orchestrator_sweep.yaml')
        params = cfg.hydra.sweeper.params['orchestrator']
        self.assertIn('sheaf_frl', params)
        self.assertIn('federated', params)
        self.assertIn('dpsgd', params)
        self.assertIn('dfedu', params)
        self.assertIn('non_cooperative', params)
        self.assertIn(
            'dataset.shared_classes', cfg.hydra.sweeper.params
        )


if __name__ == '__main__':
    unittest.main()
