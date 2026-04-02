import unittest
from unittest.mock import patch

import torch
from datasets import Dataset, DatasetDict

from src.datamodules.classification_datamodule import ClassificationDataModule
from src.datamodules.semantic_datamodule import SemanticDataModule


def _unwrap_batch(batch):
    if isinstance(batch, tuple):
        return batch[0]
    return batch


class ClassificationPilotLoaderTests(unittest.TestCase):
    def test_shared_pilot_batches_include_aligned_sample_ids(self):
        labels = [class_label % 10 for class_label in range(120)]
        dataset = Dataset.from_dict(
            {
                'image': [[float(idx), float(idx + 1)] for idx in range(len(labels))],
                'label': labels,
            }
        )
        dataset_dict = DatasetDict(
            {
                'train': dataset.select(range(80)),
                'test': dataset.select(range(80, len(dataset))),
            }
        )

        datamodule = ClassificationDataModule(
            repo='dummy',
            name='dummy',
            data_key='image',
            attributes=['label'],
            n_agents=2,
            split_strategy='uniform',
            batch_size=8,
            pilot_split=0.1,
            pilot_batch_size=4,
            num_workers=0,
            seed=3,
        )

        with patch(
            'src.datamodules.classification_datamodule.load_dataset',
            return_value=dataset_dict,
        ):
            datamodule.setup()

        self.assertTrue(datamodule.pilot_datasets)
        self.assertEqual(
            datamodule.pilot_datasets[0].sample_ids,
            datamodule.pilot_datasets[1].sample_ids,
        )

        batch = _unwrap_batch(next(iter(datamodule.train_dataloader())))
        self.assertIn('pilot_0', batch)
        self.assertIn('pilot_1', batch)

        _x0, _y0, ids0 = batch['pilot_0']
        _x1, _y1, ids1 = batch['pilot_1']
        self.assertTrue(torch.equal(ids0, ids1))


class SemanticPilotLoaderTests(unittest.TestCase):
    def test_semantic_datamodule_emits_per_agent_shared_pilot_streams(self):
        base_labels = [class_label % 5 for class_label in range(90)]
        base_dataset = Dataset.from_dict(
            {
                'embedding': [
                    [float(idx), float(idx + 1)]
                    for idx in range(len(base_labels))
                ],
                'label': base_labels,
            }
        )
        alt_dataset = Dataset.from_dict(
            {
                'embedding': [
                    [float(idx + 100), float(idx + 101)]
                    for idx in range(len(base_labels))
                ],
                'label': base_labels,
            }
        )
        dataset_a = DatasetDict(
            {
                'train': base_dataset.select(range(60)),
                'test': base_dataset.select(range(60, len(base_dataset))),
            }
        )
        dataset_b = DatasetDict(
            {
                'train': alt_dataset.select(range(60)),
                'test': alt_dataset.select(range(60, len(alt_dataset))),
            }
        )

        datamodule = SemanticDataModule(
            repo='dummy',
            name='dummy',
            attributes=['label'],
            agents={
                0: {'model': 'model_a'},
                1: {'model': 'model_b'},
            },
            batch_size=8,
            pilot_split=0.1,
            pilot_batch_size=4,
            num_workers=0,
            seed=4,
        )

        def _load_dataset_side_effect(path, config_name):
            if config_name == 'model_a':
                return dataset_a
            if config_name == 'model_b':
                return dataset_b
            raise AssertionError(f'unexpected config: {config_name}')

        with patch(
            'src.datamodules.semantic_datamodule.load_dataset',
            side_effect=_load_dataset_side_effect,
        ):
            datamodule.setup()

        self.assertTrue(datamodule.pilot_datasets)
        self.assertEqual(
            datamodule.pilot_datasets['0'].sample_ids,
            datamodule.pilot_datasets['1'].sample_ids,
        )

        batch = _unwrap_batch(next(iter(datamodule.train_dataloader())))
        self.assertIn('pilot_0', batch)
        self.assertIn('pilot_1', batch)

        x0, _y0, ids0 = batch['pilot_0']
        x1, _y1, ids1 = batch['pilot_1']
        self.assertTrue(torch.equal(ids0, ids1))
        self.assertFalse(torch.allclose(x0, x1))


if __name__ == '__main__':
    unittest.main()
