from __future__ import annotations

import base64
import io
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

import numpy as np
import pybase64

if TYPE_CHECKING:
    from vllm.outputs import RequestOutput


def normalize_native_routed_experts(result: dict[str, Any], start: int = 0) -> None:
    """Convert vLLM's native base64 ``.npy`` value at Prime's client boundary."""
    encoded = result.get("routed_experts")
    if encoded is None or isinstance(encoded, dict):
        return
    if not isinstance(encoded, str):
        raise TypeError(f"unsupported routed_experts payload: {type(encoded).__name__}")
    raw = base64.b64decode(encoded, validate=True)
    array = np.load(io.BytesIO(raw), allow_pickle=False)
    if array.ndim != 3 or not np.issubdtype(array.dtype, np.integer):
        raise ValueError("native routed_experts must be a rank-3 integer array")
    array = np.ascontiguousarray(array)
    result["routed_experts"] = {
        "data": base64.b64encode(memoryview(array)).decode("ascii"),
        "shape": list(array.shape),
        "start": start,
        "dtype": array.dtype.name,
    }


def install_native_routed_experts_normalizer() -> None:
    """Teach Prime's renderer boundary to accept native vLLM opaque ``.npy`` data."""
    import renderers.client as renderer_client

    original = renderer_client.generate
    if getattr(original, "_prime_rl_normalizes_native_routed_experts", False):
        return

    async def generate(**kwargs):
        result = await original(**kwargs)
        sampling_params = kwargs.get("sampling_params") or {}
        start = int(sampling_params.get("routed_experts_prompt_start", 0) or 0)
        normalize_native_routed_experts(result, start=start)
        return result

    generate._prime_rl_normalizes_native_routed_experts = True  # type: ignore[attr-defined]
    renderer_client.generate = generate


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
