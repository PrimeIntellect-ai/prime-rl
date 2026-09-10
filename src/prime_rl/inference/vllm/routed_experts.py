from __future__ import annotations

import os
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
    total_recall = os.environ.get("PRIME_TOTAL_ROUTER_RECALL") == "1"
    dtype = np.int32 if total_recall else np.uint8
    if total_recall:
        if array.dtype != np.int32 or array.shape[-1] % 2:
            raise ValueError("TRR export requires int32 IDs and FP32 weight bits")
    elif array.size:
        assert array.min() >= 0
        if array.max() > np.iinfo(np.uint8).max:
            # Models with >256 experts (e.g. NemotronH Super/Ultra: 512) need wider
            # indices. The payload self-describes via "dtype" so consumers pick it up.
            assert array.max() <= np.iinfo(np.uint16).max
            dtype = np.uint16

    compact = np.ascontiguousarray(array.astype(dtype, copy=False))
    return {
        "data": pybase64.b64encode(memoryview(compact)).decode("ascii"),
        "shape": list(compact.shape),
        "start": start,
        "dtype": np.dtype(dtype).name,
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
