from __future__ import annotations

from typing import Any

import numpy as np
import pybase64


def serialize_routed_experts(routed_experts: Any, start: int = 0) -> dict[str, Any] | None:
    if routed_experts is None:
        return None

    # vLLM's capturer picks uint8 or uint16 from the model's expert count, so every request of a
    # model has the same dtype and the orchestrator can pack them into one micro batch.
    compact = np.ascontiguousarray(routed_experts)
    assert compact.ndim == 3
    assert compact.dtype in (np.uint8, np.uint16), compact.dtype
    return {
        "data": pybase64.b64encode(memoryview(compact)).decode("ascii"),
        "shape": list(compact.shape),
        "start": start,
        "dtype": compact.dtype.name,
    }
