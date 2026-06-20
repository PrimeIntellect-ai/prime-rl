from types import SimpleNamespace

import pytest

from prime_rl.utils.vlm import VLM_REGISTRY, get_final_logit_softcapping, get_layer_prefix, is_vlm_architecture


def test_gemma4_dispatches_as_vlm():
    cfg = SimpleNamespace(model_type="gemma4")
    assert is_vlm_architecture(cfg)
    assert get_layer_prefix(cfg) == "model.language_model.layers."
    assert VLM_REGISTRY["gemma4"].language_model_attr == "model.language_model"


def test_softcapping_read_from_nested_text_config():
    Gemma3Config = pytest.importorskip("transformers").Gemma3Config
    cfg = Gemma3Config(text_config={"final_logit_softcapping": 30.0})
    assert getattr(cfg, "final_logit_softcapping", None) is None  # absent at the top level
    assert get_final_logit_softcapping(cfg) == 30.0


def test_softcapping_read_from_text_only_config():
    LlamaConfig = pytest.importorskip("transformers").LlamaConfig
    assert get_final_logit_softcapping(LlamaConfig(final_logit_softcapping=50.0)) == 50.0
