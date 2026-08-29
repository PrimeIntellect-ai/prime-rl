from __future__ import annotations

import io
from typing import Any

import numpy as np
import pybase64


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


def compact_vllm_routed_experts(encoded: str | None, start: int = 0) -> dict[str, Any] | None:
    """Convert vLLM's base64 ``.npy`` payload to Prime's compact payload."""
    if encoded is None:
        return None
    array = np.load(io.BytesIO(pybase64.b64decode(encoded)), allow_pickle=False)
    return serialize_routed_experts(array, start=start)
