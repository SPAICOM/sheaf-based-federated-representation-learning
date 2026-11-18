"""The MNIST parity study config composes for both methods."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

CONFIG_DIR = str(Path(__file__).resolve().parents[2] / 'config/hydra')


@pytest.mark.parametrize('orch', ['non_cooperative', 'di_sheaf_frl'])
def test_parity_d40x8_composes(orch):
    """Child config: own project, parity label map, sgd_cosine, d=40/8."""
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base='1.3'):
        cfg = compose(
            config_name='mnist_2agents_parity_d40x8',
            overrides=[f'orchestrator={orch}'],
        )
    assert cfg.logger.project == 'mnist_2agents_parity_d40x8'
    assert cfg.optimizer.scheduler == 'cosine'
    assert cfg.optimizer.lr == pytest.approx(1e-2)
    assert list(cfg.dataset.agent_label_map[1]) == [0, 1] * 5
    dims = [cfg.model.agents[i].encoder_hidden_dims[-1] for i in (0, 1)]
    assert dims == [40, 8]
