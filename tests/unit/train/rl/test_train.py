import pytest

import prime_rl.trainer.rl.multimodal as rl_multimodal
from prime_rl.configs.trainer import TrainerConfig


def test_resolve_pack_multimodal_rejects_runtime_cp_model_support(monkeypatch: pytest.MonkeyPatch):
    config = TrainerConfig.model_validate({"model": {"cp": 2}})
    model = object()

    monkeypatch.setattr(rl_multimodal, "supports_packed_multimodal_training", lambda _model: True)
    monkeypatch.setattr(
        rl_multimodal,
        "validate_multi_modal_pack",
        lambda *_args, **_kwargs: pytest.fail("validate_multi_modal_pack should not run after CP rejection"),
    )

    with pytest.raises(ValueError, match="Multimodal packing.*context parallelism"):
        rl_multimodal.resolve_pack_multimodal(config, model)


def test_resolve_pack_multimodal_validates_supported_model(monkeypatch: pytest.MonkeyPatch):
    config = TrainerConfig.model_validate({})
    model = object()
    validate_calls = []

    monkeypatch.setattr(rl_multimodal, "supports_packed_multimodal_training", lambda _model: True)
    monkeypatch.setattr(
        rl_multimodal,
        "validate_multi_modal_pack",
        lambda model, attn_impl: validate_calls.append((model, attn_impl)),
    )

    assert rl_multimodal.resolve_pack_multimodal(config, model)
    assert validate_calls == [(model, config.model.attn)]
