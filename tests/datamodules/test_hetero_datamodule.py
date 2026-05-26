"""Tests for src.datamodules.hetero_datamodule."""

from unittest.mock import patch

from datasets import Dataset, DatasetDict

from src.datamodules.hetero_datamodule import HeteroClassificationDataModule


def _build_mock_dataset(num_samples: int = 4000) -> DatasetDict:
    data = {
        'image': [
            [float(idx), float(idx + 1), float(idx + 2)]
            for idx in range(num_samples)
        ],
        'label': [idx % 4 for idx in range(num_samples)],
    }
    return DatasetDict({'train': Dataset.from_dict(data)})


def _dataset_ids(dataset: Dataset) -> set[int]:
    return {int(row[0]) for row in dataset['image']}


def _label_ratios(labels: list[int]) -> dict[int, float]:
    total = len(labels)
    return {
        class_label: labels.count(class_label) / total
        for class_label in sorted(set(labels))
    }


class TestHeteroClassificationDataModule:
    """Tests for client-first hetero classification splits."""

    def test_non_iid_with_margin_preserves_client_support_across_splits(self):
        mock_dataset = _build_mock_dataset()

        with patch(
            'src.datamodules.hetero_datamodule.load_dataset',
            return_value=mock_dataset,
        ):
            dm = HeteroClassificationDataModule(
                repo='test',
                name='toy',
                data_key='image',
                attributes=['label'],
                n_agents=4,
                split_strategy='non_iid_with_margin',
                classes_per_agent=2,
                alpha=0.5,
                val_split=0.2,
                test_split=0.1,
                seed=11,
            )
            dm.setup()

        client_totals = []
        for client_idx in range(dm.n_agents):
            expected_labels = set(dm.agent_classes[client_idx])

            train_labels = dm.train_datasets[client_idx].dataset['label']
            val_labels = dm.val_datasets[client_idx].dataset['label']
            test_labels = dm.test_datasets[client_idx].dataset['label']

            assert len(expected_labels) == dm.classes_per_agent
            assert set(train_labels) == expected_labels
            assert set(val_labels) == expected_labels
            assert set(test_labels) == expected_labels

            train_ids = _dataset_ids(dm.train_datasets[client_idx].dataset)
            val_ids = _dataset_ids(dm.val_datasets[client_idx].dataset)
            test_ids = _dataset_ids(dm.test_datasets[client_idx].dataset)

            assert train_ids.isdisjoint(val_ids)
            assert train_ids.isdisjoint(test_ids)
            assert val_ids.isdisjoint(test_ids)

            client_totals.append(
                len(train_labels) + len(val_labels) + len(test_labels)
            )

        assert max(client_totals) > min(client_totals)

    def test_non_iid_with_margin_keeps_local_label_ratios_close(self):
        mock_dataset = _build_mock_dataset(num_samples=8000)

        with patch(
            'src.datamodules.hetero_datamodule.load_dataset',
            return_value=mock_dataset,
        ):
            dm = HeteroClassificationDataModule(
                repo='test',
                name='toy',
                data_key='image',
                attributes=['label'],
                n_agents=4,
                split_strategy='non_iid_with_margin',
                classes_per_agent=2,
                alpha=0.5,
                val_split=0.2,
                test_split=0.1,
                seed=17,
            )
            dm.setup()

        for client_idx in range(dm.n_agents):
            train_ratios = _label_ratios(
                dm.train_datasets[client_idx].dataset['label']
            )
            val_ratios = _label_ratios(
                dm.val_datasets[client_idx].dataset['label']
            )
            test_ratios = _label_ratios(
                dm.test_datasets[client_idx].dataset['label']
            )

            for class_label in dm.agent_classes[client_idx]:
                assert (
                    abs(train_ratios[class_label] - val_ratios[class_label])
                    <= 0.08
                )
                assert (
                    abs(train_ratios[class_label] - test_ratios[class_label])
                    <= 0.08
                )
