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
    if array.size:
        assert array.min() >= 0
    # Preserve vLLM's per-MODEL dtype (RoutedExpertsManager uses uint8 for
    # <=256 experts, uint16 otherwise), so every sample of a model shares one
    # dtype. Re-narrowing per-sample on the observed max id would emit uint8
    # for one sample and uint16 for another of the SAME >256-expert model,
    # which the trainer's same-dtype routed-experts packing rejects. Only an
    # unexpectedly wide capture dtype (e.g. an int64 buffer) is capped to int32
    # (still consistent per model). The dtype rides the payload.
    if array.dtype in (np.uint8, np.uint16, np.int16, np.int32):
        compact = np.ascontiguousarray(array)
    else:
        compact = np.ascontiguousarray(array.astype(np.int32, copy=False))
    return {
        "data": pybase64.b64encode(memoryview(compact)).decode("ascii"),
        "shape": list(compact.shape),
        "start": start,
        "dtype": compact.dtype.name,
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
