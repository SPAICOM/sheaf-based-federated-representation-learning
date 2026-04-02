"""Communication-cost utilities for federated experiments.

These helpers provide a shared definition of communication cost across
orchestrators so experiment plots can compare methods using the same unit.
Convention: communication cost equals the total number of
transmitted scalar values multiplied by their bit-width.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from numbers import Integral
from typing import Any

import torch


def _payload_num_bits(
    payload: Any,
    *,
    bits_per_scalar: int | None = None,
) -> int:
    """Return the number of transmitted bits required for a payload.

    Supported payload types are:
    - ``torch.Tensor`` and ``torch.nn.Parameter``: bit-width is inferred from
      the tensor dtype.
    - ``int``: interpreted as a scalar-count, using ``bits_per_scalar`` or the
      default 32-bit floating-point convention.
    - nested ``Mapping`` / ``Iterable`` containers of the types above.
    """
    if isinstance(payload, torch.Tensor):
        return int(payload.numel()) * int(payload.element_size()) * 8

    if isinstance(payload, Integral):
        if int(payload) < 0:
            raise ValueError('payload scalar counts must be non-negative')
        resolved_bits = 32 if bits_per_scalar is None else int(bits_per_scalar)
        if resolved_bits < 1:
            raise ValueError('bits_per_scalar must be at least 1')
        return int(payload) * resolved_bits

    if isinstance(payload, Mapping):
        return sum(
            _payload_num_bits(value, bits_per_scalar=bits_per_scalar)
            for value in payload.values()
        )

    if isinstance(payload, Iterable) and not isinstance(payload, (str, bytes)):
        return sum(
            _payload_num_bits(item, bits_per_scalar=bits_per_scalar)
            for item in payload
        )

    raise TypeError(
        'payload must be a tensor, an integer scalar-count, or a nested '
        'container of those types'
    )


def calculate_communication_cost(
    payload: Any,
    *,
    n_transmissions: int = 1,
    bits_per_scalar: int | None = None,
    bytes_per_kilobyte: int = 1024,
) -> dict[str, float]:
    """Calculate communication cost in bits, bytes, and kilobytes.

    Parameters
    ----------
    payload : Any
        Tensor-like payload, scalar-count, or nested container of payloads.
        When an integer is provided it is interpreted as the number of
        transmitted scalar values.
    n_transmissions : int, optional
        Number of times the payload is sent. Count directed sends here, so
        bidirectional exchange over one edge corresponds to ``2``.
    bits_per_scalar : int | None, optional
        Bit-width to use when ``payload`` is given as a scalar-count integer.
        Ignored for tensors, whose dtype determines the bit-width.
    bytes_per_kilobyte : int, optional
        Conversion factor for the reported ``kilobytes`` field. The default
        uses 1024 bytes per KB for experiment logging.

    Returns
    -------
    dict[str, float]
        Dictionary with ``bits``, ``bytes``, and ``kilobytes`` entries.
    """
    if int(n_transmissions) < 0:
        raise ValueError('n_transmissions must be non-negative')
    if int(bytes_per_kilobyte) < 1:
        raise ValueError('bytes_per_kilobyte must be at least 1')

    bits = _payload_num_bits(payload, bits_per_scalar=bits_per_scalar)
    bits *= int(n_transmissions)
    bytes_sent = bits / 8.0
    kilobytes_sent = bytes_sent / float(bytes_per_kilobyte)

    return {
        'bits': float(bits),
        'bytes': bytes_sent,
        'kilobytes': kilobytes_sent,
    }


__all__ = ['calculate_communication_cost']
