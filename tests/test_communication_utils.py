import unittest

import torch

from src.utils import calculate_communication_cost


class CommunicationCostTests(unittest.TestCase):
    def test_tensor_payload_uses_tensor_dtype(self):
        payload = torch.zeros(10, dtype=torch.float32)

        cost = calculate_communication_cost(payload)

        self.assertEqual(cost['bits'], 320.0)
        self.assertEqual(cost['bytes'], 40.0)
        self.assertEqual(cost['kilobytes'], 40.0 / 1024.0)

    def test_nested_payload_sums_all_tensor_bits(self):
        payload = {
            'anchors': torch.zeros(2, 3, dtype=torch.float32),
            'ids': [torch.zeros(4, dtype=torch.int64)],
        }

        cost = calculate_communication_cost(payload)

        expected_bits = (2 * 3 * 32) + (4 * 64)
        self.assertEqual(cost['bits'], float(expected_bits))

    def test_scalar_count_payload_honors_transmission_multiplier(self):
        cost = calculate_communication_cost(
            128,
            bits_per_scalar=16,
            n_transmissions=6,
        )

        self.assertEqual(cost['bits'], float(128 * 16 * 6))
        self.assertEqual(cost['bytes'], 128 * 16 * 6 / 8.0)

    def test_invalid_scalar_count_raises(self):
        with self.assertRaises(ValueError):
            calculate_communication_cost(-1)


if __name__ == '__main__':
    unittest.main()
