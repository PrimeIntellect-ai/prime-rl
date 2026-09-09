from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

import numpy as np
import pybase64

if TYPE_CHECKING:
    from vllm.outputs import RequestOutput


def serialize_routed_experts(
    routed_experts: Any,
    start: int = 0,
    *,
    weights: Any = None,
    num_experts: int | None = None,
) -> dict[str, Any] | None:
    if routed_experts is None:
        if weights is not None:
            raise ValueError("Routing coefficients require paired expert IDs")
        return None

    array = np.asarray(routed_experts)
    if array.ndim != 3 or any(n <= 0 for n in array.shape[1:]):
        raise ValueError("Routing IDs must have shape [tokens, layers > 0, top_k > 0]")
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError("Routing IDs must have an integer dtype")
    if type(start) is not int or start < 0:
        raise ValueError("Routing start must be a nonnegative integer")
    if num_experts is not None and (type(num_experts) is not int or not 0 < num_experts <= 65536):
        raise ValueError("Routing payloads support 1..65536 logical experts")
    # Read each bound at most once before narrowing. Unsigned IDs cannot be negative.
    minimum = int(array.min()) if array.size and np.issubdtype(array.dtype, np.signedinteger) else 0
    maximum = int(array.max()) if array.size else 0
    if minimum < 0 or maximum > 65535:
        raise ValueError("Routing payload contains an invalid expert ID")
    if num_experts is not None and (maximum >= num_experts or array.shape[-1] > num_experts):
        raise ValueError("Routing expert ID/top_k exceeds the model's expert count")
    dtype = np.uint8 if (num_experts if num_experts is not None else maximum + 1) <= 256 else np.uint16

    compact = np.ascontiguousarray(array.astype(dtype, copy=False))
    payload = {
        "data": pybase64.b64encode(memoryview(compact)).decode("ascii"),
        "shape": list(compact.shape),
        "start": start,
        "dtype": np.dtype(dtype).name,
    }
    if weights is not None:
        coefficients = np.asarray(weights)
        if coefficients.shape != array.shape or coefficients.dtype != np.float32:
            raise ValueError("Routing coefficients must be FP32 with the same shape as expert IDs")
        if not (coefficients.min(initial=0) >= 0 and coefficients.max(initial=0) <= 1):
            raise ValueError("Routing v1 coefficients must be finite and in [0, 1]")
        coefficients = np.ascontiguousarray(coefficients, dtype="<f4")
        payload["format_version"] = 1
        payload["weights"] = {
            "data": pybase64.b64encode(memoryview(coefficients)).decode("ascii"),
            "dtype": "float32",
        }
    return payload


class RoutedExpertsCapture:
    def __init__(
        self,
        generator: AsyncIterator[RequestOutput],
        start: int = 0,
        *,
        require_weights: bool = False,
        num_experts: int | None = None,
    ):
        self._generator = generator
        self._start = start
        self._require_weights = require_weights
        self._num_experts = num_experts
        self.routed_experts: dict[int, dict[str, Any]] = {}

    async def __aiter__(self):
        async for request_output in self._generator:
            for output in request_output.outputs:
                ids = getattr(output, "routed_experts", None)
                weights = getattr(output, "routed_expert_weights", None) if self._require_weights else None
                if self._require_weights and output.finished() and (ids is None or weights is None):
                    raise ValueError("Total Router Recall response is missing paired routing IDs/coefficients")
                encoded = serialize_routed_experts(
                    ids, start=self._start, weights=weights, num_experts=self._num_experts
                )
                if encoded is not None:
                    self.routed_experts[output.index] = encoded
                    # The upstream response's .npy/base64 field is replaced below.
                    # Keep the already serialized private payload and avoid encoding it twice.
                    output.routed_experts = None
                    if self._require_weights:
                        output.routed_expert_weights = None
            yield request_output
