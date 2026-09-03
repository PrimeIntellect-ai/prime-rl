from __future__ import annotations

from prime_rl.multimodal.base import MultimodalAdapter
from prime_rl.multimodal.kimi_k25 import KimiK25Adapter
from prime_rl.multimodal.qwen_vl import QwenVLAdapter

_ADAPTERS = (QwenVLAdapter(), KimiK25Adapter())
_BY_MODEL_TYPE: dict[str, MultimodalAdapter] = {
    model_type: adapter for adapter in _ADAPTERS for model_type in adapter.model_types
}


def get_multimodal_adapter(model_type: str) -> MultimodalAdapter:
    try:
        return _BY_MODEL_TYPE[model_type]
    except KeyError as exc:
        raise NotImplementedError(f"Raw image training is not implemented for model type {model_type!r}") from exc
