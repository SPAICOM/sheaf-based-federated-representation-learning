"""Tests for src.datamodules.classification_datamodule."""

from unittest.mock import patch

from datasets import Dataset, DatasetDict

from src.datamodules.classification_datamodule import ClassificationDataModule


def _build_mock_dataset(num_samples: int = 100) -> DatasetDict:
    data = {
        'image': [
            [float(idx), float(idx + 1), float(idx + 2)]
            for idx in range(num_samples)
        ],
        'label': [idx % 4 for idx in range(num_samples)],
    }
    return DatasetDict({'train': Dataset.from_dict(data)})


def _count_label_occurrences(labels: list[int], target_label: int) -> int:
    return sum(1 for label in labels if label == target_label)


class TestClassificationDataModule:
    """Tests for ClassificationDataModule."""

    def test_starve_clients_only_reduces_train_data(self):
        mock_dataset = _build_mock_dataset()

        with patch(
            'src.datamodules.classification_datamodule.load_dataset',
            return_value=mock_dataset,
        ):
            base_dm = ClassificationDataModule(
                repo='test',
                name='toy',
                data_key='image',
                attributes=['label'],
                n_agents=4,
                split_strategy='uniform',
                val_split=0.1,
                test_split=0.1,
                seed=7,
            )
            base_dm.setup()

            starved_dm = ClassificationDataModule(
                repo='test',
                name='toy',
                data_key='image',
                attributes=['label'],
                n_agents=4,
                split_strategy='uniform',
                val_split=0.1,
                test_split=0.1,
                starve_clients=True,
                seed=7,
            )
            starved_dm.setup()

        assert starved_dm.starve_clients is True

        base_train_sizes = {
            client_idx: len(base_dm.train_datasets[client_idx])
            for client_idx in range(base_dm.n_agents)
        }
        starved_train_sizes = {
            client_idx: len(starved_dm.train_datasets[client_idx])
            for client_idx in range(starved_dm.n_agents)
        }
        assert {
            client_idx: len(base_dm.val_datasets[client_idx])
            for client_idx in range(base_dm.n_agents)
        } == {
            client_idx: len(starved_dm.val_datasets[client_idx])
            for client_idx in range(starved_dm.n_agents)
        }
        assert {
            client_idx: len(base_dm.test_datasets[client_idx])
            for client_idx in range(base_dm.n_agents)
        } == {
            client_idx: len(starved_dm.test_datasets[client_idx])
            for client_idx in range(starved_dm.n_agents)
        }

        reduced_clients = [
            client_idx
            for client_idx in range(starved_dm.n_agents)
            if starved_train_sizes[client_idx] != base_train_sizes[client_idx]
        ]
        assert len(reduced_clients) == starved_dm.n_agents // 2

        for client_idx in range(starved_dm.n_agents):
            if client_idx in reduced_clients:
                assert starved_train_sizes[client_idx] == max(
                    1,
                    int(round(base_train_sizes[client_idx] * 0.2)),
                )
            else:
                assert starved_train_sizes[client_idx] == base_train_sizes[
                    client_idx
                ]

    def test_non_iid_with_margin_uses_exact_k_classes_and_train_margin(self):
        mock_dataset = _build_mock_dataset(num_samples=2000)

        with patch(
            'src.datamodules.classification_datamodule.load_dataset',
            return_value=mock_dataset,
        ):
            dm = ClassificationDataModule(
                repo='test',
                name='toy',
                data_key='image',
                attributes=['label'],
                n_agents=4,
                split_strategy='non_iid_with_margin',
                classes_per_agent=2,
                alpha=0.5,
                val_split=0.1,
                test_split=0.1,
                seed=11,
            )
            dm.setup()

        assert all(
            len(dm.agent_classes[client_idx]) == dm.classes_per_agent
            for client_idx in range(dm.n_agents)
        )

        for client_idx in range(dm.n_agents):
            for split_datasets in (
                dm.train_datasets,
                dm.val_datasets,
                dm.test_datasets,
            ):
                split_labels = split_datasets[client_idx].dataset['label']
                for class_label in dm.agent_classes[client_idx]:
                    assert (
                        _count_label_occurrences(split_labels, class_label)
                        >= 10
                    )
