import pytest

from prime_rl.inference.config import InferenceConfig


def test_to_vllm_includes_fp8_fields() -> None:
    config = InferenceConfig(
        model={
            "quantization": "fp8",
            "kv_cache_dtype": "fp8_e4m3",
        },
        calculate_kv_scales=True,
    )

    namespace = config.to_vllm()
    assert namespace.quantization == "fp8"
    assert namespace.kv_cache_dtype == "fp8_e4m3"
    assert namespace.calculate_kv_scales is True


def test_to_vllm_omits_optional_none_fields() -> None:
    config = InferenceConfig()

    namespace = config.to_vllm()
    assert not hasattr(namespace, "quantization")
    assert not hasattr(namespace, "kv_cache_dtype")
    assert not hasattr(namespace, "reasoning_parser")
    assert not hasattr(namespace, "rope_scaling")


def test_validate_optimization_config_rejects_fp8_with_float32() -> None:
    with pytest.raises(ValueError, match="FP8 quantization requires model.dtype"):
        InferenceConfig(
            model={
                "dtype": "float32",
                "quantization": "fp8",
            }
        )


def test_validate_optimization_config_requires_fp8_kv_cache_for_scale_calc() -> None:
    with pytest.raises(ValueError, match="calculate_kv_scales requires model.kv_cache_dtype"):
        InferenceConfig(
            model={"kv_cache_dtype": "bfloat16"},
            calculate_kv_scales=True,
        )
