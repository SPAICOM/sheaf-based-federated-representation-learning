import unittest
from unittest.mock import patch

from datasets import Dataset, DatasetDict

from src.datamodules.classification_datamodule import ClassificationDataModule
from src.utils.data_partitioner import (
    build_shared_class_partition,
    partition_non_iid,
)


def _flatten(partition: dict[int, list[int]]) -> list[int]:
    return [index for indices in partition.values() for index in indices]


class PartitionNonIidTests(unittest.TestCase):
    def test_shared_class_partition_controls_global_overlap(self):
        agent_classes = build_shared_class_partition(
            unique_classes=list(range(10)),
            n_agents=4,
            shared_classes=2,
            seed=17,
        )

        shared = set.intersection(
            *[set(classes) for classes in agent_classes.values()]
        )
        covered = set().union(*[set(classes) for classes in agent_classes.values()])

        self.assertEqual(len(shared), 2)
        self.assertEqual(covered, set(range(10)))
        self.assertTrue(all(len(classes) >= 2 for classes in agent_classes.values()))

    def test_partition_covers_all_samples_once_and_keeps_agents_non_empty(self):
        labels = [class_label for class_label in range(10) for _ in range(20)]

        partition = partition_non_iid(
            labels=labels,
            n_agents=7,
            classes_per_agent=2,
            seed=123,
            alpha=0.3,
        )

        flattened = _flatten(partition)
        self.assertEqual(len(flattened), len(labels))
        self.assertEqual(sorted(flattened), list(range(len(labels))))
        self.assertTrue(all(len(indices) > 0 for indices in partition.values()))

    def test_partition_is_reproducible_and_handles_many_classes(self):
        labels = [class_label for class_label in range(100) for _ in range(4)]

        partition_a, agent_classes_a = partition_non_iid(
            labels=labels,
            n_agents=8,
            classes_per_agent=5,
            seed=7,
            alpha=0.2,
            return_agent_classes=True,
        )
        partition_b, agent_classes_b = partition_non_iid(
            labels=labels,
            n_agents=8,
            classes_per_agent=5,
            seed=7,
            alpha=0.2,
            return_agent_classes=True,
        )

        self.assertEqual(partition_a, partition_b)
        self.assertEqual(agent_classes_a, agent_classes_b)

        flattened = _flatten(partition_a)
        self.assertEqual(sorted(flattened), list(range(len(labels))))
        covered_classes = set().union(*[set(classes) for classes in agent_classes_a.values()])
        self.assertEqual(covered_classes, set(range(100)))

    def test_partition_reuses_explicit_agent_classes(self):
        labels = [class_label for class_label in range(10) for _ in range(12)]
        partition, agent_classes = partition_non_iid(
            labels=labels,
            n_agents=4,
            classes_per_agent=3,
            seed=11,
            alpha=0.5,
            return_agent_classes=True,
        )

        reused_partition = partition_non_iid(
            labels=labels,
            n_agents=4,
            classes_per_agent=3,
            seed=999,
            alpha=0.5,
            agent_classes=agent_classes,
        )

        for agent_id, indices in reused_partition.items():
            observed_labels = {labels[index] for index in indices}
            self.assertTrue(observed_labels.issubset(set(agent_classes[agent_id])))

        self.assertEqual(sorted(_flatten(partition)), list(range(len(labels))))
        self.assertEqual(sorted(_flatten(reused_partition)), list(range(len(labels))))


class ClassificationDataModuleNonIidTests(unittest.TestCase):
    def test_class_partition_can_generate_agent_classes_from_shared_labels(self):
        labels = [class_label for class_label in range(10) for _ in range(24)]
        dataset = Dataset.from_dict(
            {
                'image': [
                    [float(index), float(index + 1)]
                    for index in range(len(labels))
                ],
                'label': labels,
            }
        )
        dataset_dict = DatasetDict(
            {
                'train': dataset.select(range(120)),
                'test': dataset.select(range(120, len(dataset))),
            }
        )

        datamodule = ClassificationDataModule(
            repo='dummy',
            name='dummy',
            data_key='image',
            attributes=['label'],
            n_agents=4,
            split_strategy='class_partition',
            shared_classes=2,
            batch_size=8,
            num_workers=0,
            val_split=0.2,
            test_split=0.2,
            seed=5,
        )

        with patch(
            'src.datamodules.classification_datamodule.load_dataset',
            return_value=dataset_dict,
        ):
            datamodule.setup()

        shared = set.intersection(
            *[
                set(datamodule.agent_classes[agent_id])
                for agent_id in range(datamodule.n_agents)
            ]
        )
        covered = set().union(
            *[
                set(datamodule.agent_classes[agent_id])
                for agent_id in range(datamodule.n_agents)
            ]
        )

        self.assertEqual(len(shared), 2)
        self.assertEqual(covered, set(range(10)))
        for agent_id in range(datamodule.n_agents):
            train_labels = set(datamodule.train_datasets[agent_id].dataset['label'])
            self.assertTrue(
                train_labels.issubset(set(datamodule.agent_classes[agent_id]))
            )

    def test_setup_uses_consistent_agent_classes_across_splits(self):
        labels = [class_label for class_label in range(10) for _ in range(24)]
        dataset = Dataset.from_dict(
            {
                'image': [[float(index), float(index + 1)] for index in range(len(labels))],
                'label': labels,
            }
        )
        dataset_dict = DatasetDict(
            {
                'train': dataset.select(range(120)),
                'test': dataset.select(range(120, len(dataset))),
            }
        )

        datamodule = ClassificationDataModule(
            repo='dummy',
            name='dummy',
            data_key='image',
            attributes=['label'],
            n_agents=4,
            split_strategy='non_iid',
            classes_per_agent=3,
            alpha=0.4,
            batch_size=8,
            num_workers=0,
            val_split=0.2,
            test_split=0.2,
            seed=5,
        )

        with patch(
            'src.datamodules.classification_datamodule.load_dataset',
            return_value=dataset_dict,
        ):
            datamodule.setup()

        global_classes = len(set(labels))
        for agent_id in range(datamodule.n_agents):
            train_labels = datamodule.train_datasets[agent_id].dataset['label']
            val_labels = datamodule.val_datasets[agent_id].dataset['label']
            test_labels = datamodule.test_datasets[agent_id].dataset['label']
            assigned_classes = set(datamodule.agent_classes[agent_id])

            self.assertTrue(train_labels)
            self.assertTrue(val_labels)
            self.assertTrue(test_labels)
            self.assertTrue(set(train_labels).issubset(assigned_classes))
            self.assertTrue(set(val_labels).issubset(assigned_classes))
            self.assertTrue(set(test_labels).issubset(assigned_classes))
            self.assertEqual(datamodule.num_classes[agent_id], len(set(train_labels)))
            self.assertLess(datamodule.num_classes[agent_id], global_classes)


if __name__ == '__main__':
    unittest.main()
