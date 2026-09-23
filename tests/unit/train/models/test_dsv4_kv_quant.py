import math

import pytest
import torch

from prime_rl.trainer.models.deepseek_v4.kv_quant import fake_quantize_kv_cache

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
        reason="the simulated KV cache casts to float8_e4m3fn, only supported on Hopper (SM90) and newer",
    ),
]

FP8_MAX = 448.0
QUANT_BLOCK = 64
HEAD_DIMS = [(512, 64), (32, 16), (192, 64)]


def reference_quantize(kv: torch.Tensor, rope_dim: int) -> tuple[torch.Tensor, list[int]]:
    """Literal transcription of vLLM's fused KV-insert kernel, one block at a time."""
    nope_dim = kv.shape[-1] - rope_dim
    flat = kv.reshape(-1, kv.shape[-1]).float()
    out = flat.clone()
    exponents = []
    for row in range(flat.shape[0]):
        for start in range(0, nope_dim, QUANT_BLOCK):
            block = flat[row, start : min(start + QUANT_BLOCK, nope_dim)].bfloat16().float()
            absmax = max(block.abs().max().item(), 1e-4)
            exponent = math.ceil(math.log2(absmax / FP8_MAX))
            scaled = (block * 2.0**-exponent).clamp(-FP8_MAX, FP8_MAX)
            out[row, start : start + block.numel()] = scaled.to(torch.float8_e4m3fn).float() * 2.0**exponent
            exponents.append(exponent)
    return out.to(kv.dtype).view_as(kv), exponents


@pytest.fixture
def kv_fp32() -> torch.Tensor:
    torch.manual_seed(1234)
    return torch.randn(2, 8, 1, 512, device="cuda") * 0.4


@pytest.mark.parametrize("head_dim,rope_dim", HEAD_DIMS)
def test_matches_kernel_reference(head_dim, rope_dim):
    torch.manual_seed(head_dim)
    kv = torch.randn(2, 8, 1, head_dim, device="cuda", dtype=torch.bfloat16) * 0.4

    out = fake_quantize_kv_cache(kv, rope_dim)
    expected, _ = reference_quantize(kv, rope_dim)

    assert torch.equal(out.view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize("head_dim,rope_dim", HEAD_DIMS)
def test_rope_slice_and_dtype_are_preserved(head_dim, rope_dim):
    torch.manual_seed(head_dim)
    kv = torch.randn(2, 8, 1, head_dim, device="cuda", dtype=torch.bfloat16) * 0.4

    out = fake_quantize_kv_cache(kv, rope_dim)

    assert out.dtype == kv.dtype
    assert out.shape == kv.shape
    assert torch.equal(
        out[..., head_dim - rope_dim :].view(torch.int16), kv[..., head_dim - rope_dim :].view(torch.int16)
    )
    assert not torch.equal(out[..., : head_dim - rope_dim], kv[..., : head_dim - rope_dim])


def test_backward_passes_gradient_through_unchanged(kv_fp32):
    kv = kv_fp32.to(torch.bfloat16).requires_grad_()
    grad_output = torch.randn_like(kv)

    fake_quantize_kv_cache(kv, 64).backward(grad_output)

    assert torch.equal(kv.grad, grad_output)


def test_dequantized_values_lie_on_a_power_of_two_grid(kv_fp32):
    kv = kv_fp32.to(torch.bfloat16)
    out = fake_quantize_kv_cache(kv, 64)
    _, exponents = reference_quantize(kv, 64)

    nope = out[..., :448].reshape(-1, 448 // QUANT_BLOCK, QUANT_BLOCK).float()
    # e4m3's smallest subnormal is 2**-9, so a power-of-two scale puts every dequantized
    # value on an integer multiple of 2 ** (exponent - 9).
    grid = torch.tensor(exponents, device=out.device, dtype=torch.float32).view(-1, 448 // QUANT_BLOCK, 1)
    on_grid = nope * torch.exp2(9.0 - grid)

    assert torch.equal(on_grid, on_grid.round())
    assert on_grid.abs().max() <= FP8_MAX * 2**9


def test_error_is_block_scaled_e4m3_regime_not_bfloat16(kv_fp32):
    nope_fp32 = kv_fp32[..., :448]
    kv = kv_fp32.to(torch.bfloat16)

    fp8_nope = fake_quantize_kv_cache(kv, 64)[..., :448].float()
    bf16_nope = kv[..., :448].float()

    reference_norm = nope_fp32.norm()
    fp8_error = ((fp8_nope - nope_fp32).norm() / reference_norm).item()
    bf16_error = ((bf16_nope - nope_fp32).norm() / reference_norm).item()

    assert 0.015 < fp8_error < 0.05
    assert bf16_error < 0.005
    assert fp8_error > 8 * bf16_error


def test_all_zero_block_uses_the_amax_floor():
    kv = torch.zeros(1, 4, 1, 512, device="cuda", dtype=torch.bfloat16)

    out = fake_quantize_kv_cache(kv, 64)

    assert torch.equal(out, kv)
