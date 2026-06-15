from types import SimpleNamespace

import torch.nn as nn

from prime_rl.utils.vlm import get_packed_mm_disabled_reasons, supports_packed_multimodal_training


class _Model(nn.Module):
    def __init__(self, *, supports_packed_mm=None):
        super().__init__()
        self.config = SimpleNamespace(model_type="qwen3_5_moe")
        if supports_packed_mm is not None:
            self.supports_packed_multimodal_training = supports_packed_mm


class _Wrapper(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module


def test_packed_mm_support_reads_model_capability():
    assert supports_packed_multimodal_training(_Model(supports_packed_mm=True))
    assert supports_packed_multimodal_training(_Wrapper(_Model(supports_packed_mm=True)))
    assert not supports_packed_multimodal_training(_Model(supports_packed_mm=False))
    assert not supports_packed_multimodal_training(_Model())


def test_packed_mm_gate_allows_only_supported_runtime():
    model = _Model(supports_packed_mm=True)

    assert (
        get_packed_mm_disabled_reasons(model, enabled=True, attn_impl="flash_attention_2", cp_enabled=False, cp_size=1)
        == []
    )
    assert get_packed_mm_disabled_reasons(model, enabled=True, attn_impl="sdpa", cp_enabled=False) == ["attn=sdpa"]
    assert get_packed_mm_disabled_reasons(model, enabled=True, attn_impl="fa4", cp_enabled=True, cp_size=2) == ["cp=2"]
    assert get_packed_mm_disabled_reasons(model, enabled=False, attn_impl="fa4", cp_enabled=False) == [
        "trainer.pack_multimodal=false"
    ]


def test_packed_mm_gate_rejects_models_without_capability():
    model = _Model()

    assert get_packed_mm_disabled_reasons(model, enabled=True, attn_impl="flash_attention_2", cp_enabled=False) == [
        "model_support=false"
    ]
