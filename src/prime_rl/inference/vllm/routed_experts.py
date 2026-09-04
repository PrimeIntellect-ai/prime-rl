from __future__ import annotations

import base64
import io
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

import numpy as np
import pybase64

if TYPE_CHECKING:
    from vllm.outputs import RequestOutput

MAX_NATIVE_ROUTED_EXPERTS_BYTES = 64 * 1024 * 1024


def _decode_native_routed_experts(raw: bytes) -> np.ndarray:
    buffer = io.BytesIO(raw)
    version = np.lib.format.read_magic(buffer)
    if version == (1, 0):
        shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(buffer)
    elif version == (2, 0):
        shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(buffer)
    else:
        raise ValueError(f"unsupported native routed_experts .npy version: {version}")

    dtype = np.dtype(dtype)
    if len(shape) != 3 or dtype.kind not in ("i", "u") or dtype.hasobject or fortran_order:
        raise ValueError("native routed_experts must be a C-contiguous rank-3 integer array")

    element_count = int(np.prod(shape, dtype=object))
    payload_bytes = element_count * dtype.itemsize
    if payload_bytes > MAX_NATIVE_ROUTED_EXPERTS_BYTES:
        raise ValueError("native routed_experts payload exceeds the 64 MiB limit")

    payload = buffer.read()
    if len(payload) != payload_bytes:
        raise ValueError("native routed_experts payload size does not match its .npy header")

    return np.frombuffer(payload, dtype=dtype, count=element_count).reshape(shape)


def normalize_native_routed_experts(result: dict[str, Any], start: int = 0) -> None:
    encoded = result.get("routed_experts")
    if encoded is None or isinstance(encoded, dict):
        return
    if not isinstance(encoded, str):
        raise TypeError(f"unsupported routed_experts payload: {type(encoded).__name__}")
    if len(encoded) > ((MAX_NATIVE_ROUTED_EXPERTS_BYTES + 16 * 1024 + 2) // 3) * 4:
        raise ValueError("encoded native routed_experts payload exceeds the 64 MiB limit")

    raw = base64.b64decode(encoded, validate=True)
    array = _decode_native_routed_experts(raw)
    if array.size and (array.min() < 0 or array.max() > np.iinfo(np.uint16).max):
        raise ValueError("native routed_experts indices must be between 0 and 65535")
    target_dtype = np.uint8 if not array.size or array.max() <= np.iinfo(np.uint8).max else np.uint16
    array = np.ascontiguousarray(array, dtype=target_dtype)
    result["routed_experts"] = {
        "data": base64.b64encode(memoryview(array)).decode("ascii"),
        "shape": list(array.shape),
        "start": start,
        "dtype": array.dtype.name,
    }


def install_native_routed_experts_normalizer() -> None:
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
