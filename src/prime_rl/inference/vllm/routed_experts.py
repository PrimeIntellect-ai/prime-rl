from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import numpy as np
import pybase64
from vllm.outputs import RequestOutput


def serialize_routed_experts(routed_experts: Any) -> dict[str, Any] | None:
    if routed_experts is None:
        return None

    array = np.asarray(routed_experts)
    assert array.ndim == 3
    assert np.issubdtype(array.dtype, np.integer)
    if array.size:
        assert array.min() >= 0
        assert array.max() <= np.iinfo(np.uint8).max

    compact = np.ascontiguousarray(array.astype(np.uint8, copy=False))
    return {
        "data": pybase64.b64encode(memoryview(compact)).decode("ascii"),
        "shape": list(compact.shape),
    }


class RoutedExpertsCapture:
    def __init__(self, generator: AsyncIterator[RequestOutput]):
        self._generator = generator
        self.routed_experts: dict[int, dict[str, Any]] = {}

    async def __aiter__(self):
        async for request_output in self._generator:
            for output in request_output.outputs:
                encoded = serialize_routed_experts(getattr(output, "routed_experts", None))
                if encoded is not None:
                    self.routed_experts[output.index] = encoded
            yield request_output
