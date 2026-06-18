from types import SimpleNamespace

import pytest
import torch.nn as nn

from prime_rl.utils.vlm import (
    supports_packed_multimodal_training,
    supports_ulysses_vlm_cp_training,
    validate_multi_modal_pack,
)


class _Model(nn.Module):
    def __init__(self, *, supports_packed_mm=None, supports_ulysses_vlm_cp=None):
        super().__init__()
        self.config = SimpleNamespace(model_type="qwen3_5_moe")
        if supports_packed_mm is not None:
            self.supports_packed_multimodal_training = supports_packed_mm
        if supports_ulysses_vlm_cp is not None:
            self.supports_ulysses_vlm_cp_training = supports_ulysses_vlm_cp

    def prepare_vlm_inputs_for_context_parallel(self):
        pass


def test_packed_mm_support_reads_model_capability():
    assert supports_packed_multimodal_training(_Model(supports_packed_mm=True))
    assert not supports_packed_multimodal_training(_Model(supports_packed_mm=False))
    assert not supports_packed_multimodal_training(_Model())


def test_validate_multi_modal_pack_allows_supported_runtime():
    model = _Model(supports_packed_mm=True)

    validate_multi_modal_pack(model, attn_impl="flash_attention_2")


def test_ulysses_vlm_cp_support_requires_capability_and_prepare_method():
    assert supports_ulysses_vlm_cp_training(_Model(supports_ulysses_vlm_cp=True))
    assert supports_ulysses_vlm_cp_training(_Wrapper(_Model(supports_ulysses_vlm_cp=True)))
    assert not supports_ulysses_vlm_cp_training(_Model(supports_ulysses_vlm_cp=False))
    assert not supports_ulysses_vlm_cp_training(_Model())

    model = _Model(supports_ulysses_vlm_cp=True)
    model.prepare_vlm_inputs_for_context_parallel = None
    assert not supports_ulysses_vlm_cp_training(model)


def test_validate_multi_modal_pack_rejects_models_without_capability():
    model = _Model()

    with pytest.raises(ValueError, match="model support"):
        validate_multi_modal_pack(model, attn_impl="flash_attention_2")


def test_validate_multi_modal_pack_rejects_non_varlen_attention():
    model = _Model(supports_packed_mm=True)

    with pytest.raises(ValueError, match="flash attention"):
        validate_multi_modal_pack(model, attn_impl="sdpa")
