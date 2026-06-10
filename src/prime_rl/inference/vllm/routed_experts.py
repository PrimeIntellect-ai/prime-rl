from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import numpy as np
import pybase64
from vllm.outputs import RequestOutput


def serialize_routed_experts(routed_experts: Any, start: int = 0) -> dict[str, Any] | None:
    if routed_experts is None:
        return None

    array = np.asarray(routed_experts)
    assert array.ndim == 3
    assert np.issubdtype(array.dtype, np.integer)
    # Narrow to the smallest int that holds the expert ids, matching vLLM's
    # RoutedExpertsManager (uint8 for <=256 experts, uint16 otherwise). Keeps
    # the wire payload compact while supporting >256-expert MoEs (e.g. Kimi-K2)
    # that overflow uint8. The dtype rides the payload so the consumer decodes
    # with the right element type.
    if array.size:
        assert array.min() >= 0
        max_id = int(array.max())
    else:
        max_id = 0
    if max_id <= np.iinfo(np.uint8).max:
        target_dtype = np.uint8
    elif max_id <= np.iinfo(np.uint16).max:
        target_dtype = np.uint16
    else:
        # Beyond uint16 (>65535 experts) astype(uint16) would wrap and corrupt
        # routing; fall back to int32 (consumer decodes via the dtype field).
        target_dtype = np.int32

    compact = np.ascontiguousarray(array.astype(target_dtype, copy=False))
    return {
        "data": pybase64.b64encode(memoryview(compact)).decode("ascii"),
        "shape": list(compact.shape),
        "start": start,
        "dtype": np.dtype(target_dtype).name,
    }


class RoutedExpertsCapture:
    def __init__(self, generator: AsyncIterator[RequestOutput], start: int = 0):
        self._generator = generator
        self._start = start
        self.routed_experts: dict[int, dict[str, Any]] = {}

    async def __aiter__(self):
        async for request_output in self._generator:
            for output in request_output.outputs:
                encoded = serialize_routed_experts(getattr(output, "routed_experts", None), start=self._start)
                if encoded is not None:
                    self.routed_experts[output.index] = encoded
            yield request_output
