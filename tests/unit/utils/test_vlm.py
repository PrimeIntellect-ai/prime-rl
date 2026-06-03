from types import SimpleNamespace

import torch.nn as nn

from prime_rl.utils.vlm import get_packed_mm_disabled_reasons, get_packed_mm_position_strategy


class _Model(nn.Module):
    def __init__(self, *, strategy=None, model_type="qwen3_5_moe", is_vlm=True):
        super().__init__()
        self.config = SimpleNamespace(model_type=model_type)
        self._is_vlm = is_vlm
        if strategy is not None:
            self.packed_mm_position_strategy = strategy


def test_packed_mm_strategy_is_pass_1d_for_custom_qwen35_vlm():
    assert get_packed_mm_position_strategy(_Model()) == "pass_1d"


def test_packed_mm_strategy_none_for_text_only_custom_qwen35():
    assert get_packed_mm_position_strategy(_Model(is_vlm=False)) == "none"


def test_packed_mm_gate_allows_only_supported_runtime():
    model = _Model(strategy="pass_1d")

    assert (
        get_packed_mm_disabled_reasons(model, enabled=True, attn_impl="flash_attention_2", cp_enabled=False, cp_size=1)
        == []
    )
    assert get_packed_mm_disabled_reasons(model, enabled=True, attn_impl="sdpa", cp_enabled=False) == ["attn=sdpa"]
    assert get_packed_mm_disabled_reasons(model, enabled=True, attn_impl="fa4", cp_enabled=True, cp_size=2) == ["cp=2"]
    assert get_packed_mm_disabled_reasons(model, enabled=False, attn_impl="fa4", cp_enabled=False) == [
        "trainer.pack_multimodal=false"
    ]


def test_packed_mm_gate_rejects_hf_mrope_default_strategy():
    model = _Model(strategy="none", model_type="qwen3_vl")

    assert get_packed_mm_disabled_reasons(model, enabled=True, attn_impl="flash_attention_2", cp_enabled=False) == [
        "position_strategy=none"
    ]
