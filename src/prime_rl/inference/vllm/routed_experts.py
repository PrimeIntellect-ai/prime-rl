from __future__ import annotations

import base64
from collections.abc import AsyncIterator
from io import BytesIO
from typing import Any

import numpy as np
from vllm.outputs import RequestOutput


def serialize_routed_experts(routed_experts: Any) -> str | None:
    if routed_experts is None:
        return None

    array = np.asarray(routed_experts)
    assert array.ndim == 3
    assert np.issubdtype(array.dtype, np.integer)

    if array.size == 0:
        compact = array.astype(np.uint8, copy=False)
    else:
        min_value = array.min()
        max_value = array.max()
        if min_value >= 0 and max_value <= np.iinfo(np.uint8).max:
            compact = array.astype(np.uint8, copy=False)
        elif min_value >= np.iinfo(np.int16).min and max_value <= np.iinfo(np.int16).max:
            compact = array.astype(np.int16, copy=False)
        else:
            compact = array.astype(np.int32, copy=False)

    buffer = BytesIO()
    np.save(buffer, np.ascontiguousarray(compact), allow_pickle=False)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class RoutedExpertsCapture:
    def __init__(self, generator: AsyncIterator[RequestOutput]):
        self._generator = generator
        self.routed_experts: dict[int, str] = {}

    async def __aiter__(self):
        async for request_output in self._generator:
            for output in request_output.outputs:
                encoded = serialize_routed_experts(getattr(output, "routed_experts", None))
                if encoded is not None:
                    self.routed_experts[output.index] = encoded
            yield request_output
