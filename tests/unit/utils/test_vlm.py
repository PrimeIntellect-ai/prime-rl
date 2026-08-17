from types import SimpleNamespace

from prime_rl.utils.vlm import supports_packed_multimodal_training


def test_qwen3_vl_hf_model_supports_packed_multimodal_training() -> None:
    model = SimpleNamespace(config=SimpleNamespace(model_type="qwen3_vl"))

    assert supports_packed_multimodal_training(model) is True


def test_unknown_hf_model_does_not_claim_packed_multimodal_training() -> None:
    model = SimpleNamespace(config=SimpleNamespace(model_type="unknown"))

    assert supports_packed_multimodal_training(model) is False


def test_model_packed_multimodal_capability_takes_precedence() -> None:
    model = SimpleNamespace(
        config=SimpleNamespace(model_type="unknown"),
        supports_packed_multimodal_training=True,
    )

    assert supports_packed_multimodal_training(model) is True
