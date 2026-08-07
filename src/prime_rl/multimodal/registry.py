from __future__ import annotations

from prime_rl.multimodal.adapters.base import MultimodalAdapter
from prime_rl.multimodal.adapters.kimi_k25 import KimiK25Adapter
from prime_rl.multimodal.adapters.qwen_vl import QwenVLAdapter

_ADAPTERS: dict[str, MultimodalAdapter] = {
    QwenVLAdapter.family: QwenVLAdapter(),
    KimiK25Adapter.family: KimiK25Adapter(),
}


def get_multimodal_adapter(family: str) -> MultimodalAdapter:
    try:
        return _ADAPTERS[family]
    except KeyError as exc:
        raise NotImplementedError(f"No multimodal adapter registered for family {family!r}") from exc
