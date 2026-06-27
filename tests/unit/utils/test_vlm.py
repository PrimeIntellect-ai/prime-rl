from types import SimpleNamespace

import pytest
import torch.nn as nn

from prime_rl.utils.vlm import supports_packed_multimodal_training, validate_multi_modal_pack


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


def test_validate_multi_modal_pack_allows_supported_runtime():
    model = _Model(supports_packed_mm=True)

    validate_multi_modal_pack(model, attn_impl="flash_attention_2")


def test_validate_multi_modal_pack_rejects_models_without_capability():
    model = _Model()

    with pytest.raises(ValueError, match="model support"):
        validate_multi_modal_pack(model, attn_impl="flash_attention_2")


def test_validate_multi_modal_pack_rejects_non_varlen_attention():
    model = _Model(supports_packed_mm=True)

    with pytest.raises(ValueError, match="flash attention"):
        validate_multi_modal_pack(model, attn_impl="sdpa")
