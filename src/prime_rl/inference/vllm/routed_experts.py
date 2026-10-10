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
    dtype = np.uint8
    if array.size:
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


def pad_to_start(routed_experts: Any, num_rows: int) -> Any:
    """Left-pad routing rows with zero rows up to ``num_rows``.

    A P/D decode instance returns rows only from its first forward (the last prompt token). The
    router (v0.2.2) drops as many leading decode rows as the prefill instance returned, so the
    decode rows must start at ``routed_experts_prompt_start`` like the prefill rows. The padding
    is never used: the router replaces it with the prefill rows.
    """
    if routed_experts is None or len(routed_experts) >= num_rows:
        return routed_experts
    array = np.asarray(routed_experts)
    padding = np.zeros((num_rows - len(array), *array.shape[1:]), dtype=array.dtype)
    return np.concatenate((padding, array))


class RoutedExpertsCapture:
    def __init__(self, generator: AsyncIterator[RequestOutput], start: int = 0, remote_prefill: bool = False):
        self._generator = generator
        self._start = start
        self._remote_prefill = remote_prefill
        self.routed_experts: dict[int, dict[str, Any]] = {}

    async def __aiter__(self):
        async for request_output in self._generator:
            for output in request_output.outputs:
                routed_experts = getattr(output, "routed_experts", None)
                if self._remote_prefill:
                    num_rows = len(request_output.prompt_token_ids) + len(output.token_ids) - 1 - self._start
                    routed_experts = pad_to_start(routed_experts, num_rows)
                encoded = serialize_routed_experts(routed_experts, start=self._start)
                if encoded is not None:
                    self.routed_experts[output.index] = encoded
            yield request_output
