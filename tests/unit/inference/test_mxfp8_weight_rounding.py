import prime_kernels
import pytest
import torch

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
        reason="MXFP8 requires Blackwell",
    ),
]


@pytest.mark.parametrize("swizzled", [False, True])
def test_weight_patch_matches_trainer_and_preserves_activation_cast(monkeypatch, swizzled):
    from vllm.model_executor.layers.quantization.online import mxfp8
    from vllm.model_executor.layers.quantization.utils import mxfp8_utils

    from prime_rl.inference.vllm.mxfp8_weight_rounding import patch_mxfp8_weight_rounding

    original = mxfp8.mxfp8_e4m3_quantize
    activation_cast = mxfp8_utils.mxfp8_e4m3_quantize
    monkeypatch.setattr(mxfp8, "mxfp8_e4m3_quantize", original)
    monkeypatch.setenv("PRIME_MXFP8_STOCHASTIC_WEIGHT_ROUNDING", "1")
    weight = torch.nn.Parameter(torch.randn((129, 256), device="cuda", dtype=torch.bfloat16) * 0.02)
    nearest, rn_scales = activation_cast(weight.data, is_sf_swizzled_layout=swizzled)
    torch.manual_seed(47)
    seed = torch.empty((), device="cuda", dtype=torch.int64).random_()
    expected, expected_scales = prime_kernels.load("mxfp8").quantize(
        weight.data, rounding_seed=seed, rounding_salt=torch.zeros_like(seed)
    )
    if swizzled:
        expected_scales = mxfp8_utils.swizzle_mxfp8_scale(expected_scales, *weight.shape)
    patch_mxfp8_weight_rounding()
    patched = mxfp8.mxfp8_e4m3_quantize
    patch_mxfp8_weight_rounding()
    assert mxfp8.mxfp8_e4m3_quantize is patched
    torch.manual_seed(47)
    actual, scales = patched(weight, is_sf_swizzled_layout=swizzled)
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    assert torch.equal(scales, expected_scales)
    assert torch.equal(scales, rn_scales)
    assert not torch.equal(actual.view(torch.uint8), nearest.view(torch.uint8))
    assert mxfp8_utils.mxfp8_e4m3_quantize is activation_cast
    activations, _ = activation_cast(weight.data, is_sf_swizzled_layout=swizzled)
    assert torch.equal(activations.view(torch.uint8), nearest.view(torch.uint8))
