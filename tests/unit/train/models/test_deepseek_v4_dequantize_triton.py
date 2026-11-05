"""GPU correctness checks for the fused Triton dequantization path.

Mirrors `test_deepseek_v4_cpu.py`'s two worked examples exactly (same tensors, same expected
output) so the Triton kernel is checked against the identical ground truth the CPU reference
implementation already is, then adds randomized and batched (per-expert stacked) coverage
against `dequantize_weight` directly, since the real checkpoint's MoE expert weights are
3D-stacked, not the 2D-only shapes the worked examples use.
"""

import pytest
import torch

from prime_rl.trainer.models.deepseek_v4.dequantize import dequantize_state_dict_, dequantize_weight
from prime_rl.trainer.models.deepseek_v4.dequantize_triton import dequantize_weight_triton

pytestmark = [pytest.mark.gpu]


def test_dequantize_weight_triton_dense_fp8_matches_worked_example():
    weight = torch.tensor([[1.0, 2.0], [-1.0, 0.5]], dtype=torch.float32).to(torch.float8_e4m3fn).cuda()
    scale = torch.tensor([[128]], dtype=torch.uint8).view(torch.float8_e8m0fnu).cuda()

    result = dequantize_weight_triton(weight, scale)

    assert result.dtype == torch.bfloat16
    assert torch.equal(result.cpu(), torch.tensor([[2.0, 4.0], [-2.0, 1.0]], dtype=torch.bfloat16))


def test_dequantize_weight_triton_packed_mxfp4_matches_worked_example():
    packed = torch.tensor(
        [
            [(4 << 4) | 2, (6 << 4) | 10],
            [(7 << 4) | 0, (3 << 4) | 9],
        ],
        dtype=torch.int8,
    ).cuda()
    scale = torch.tensor([[127, 128], [129, 126]], dtype=torch.uint8).view(torch.float8_e8m0fnu).cuda()

    result = dequantize_weight_triton(packed, scale)

    expected = torch.tensor([[1.0, 2.0, -2.0, 8.0], [0.0, 24.0, -0.25, 0.75]], dtype=torch.bfloat16)
    assert result.dtype == torch.bfloat16
    assert torch.equal(result.cpu(), expected)


def test_dequantize_weight_triton_matches_reference_dense_fp8_random():
    torch.manual_seed(0)
    rows, cols = 256, 384  # 2x3 grid of 128x128 blocks
    weight_bf16 = torch.randn(rows, cols, dtype=torch.bfloat16)
    weight = weight_bf16.to(torch.float8_e4m3fn)
    scale = torch.randint(120, 135, (rows // 128, cols // 128), dtype=torch.uint8).view(torch.float8_e8m0fnu)

    reference = dequantize_weight(weight, scale)
    result = dequantize_weight_triton(weight.cuda(), scale.cuda())

    assert torch.equal(result.cpu(), reference)


def test_dequantize_weight_triton_matches_reference_packed_mxfp4_random():
    torch.manual_seed(0)
    rows, packed_cols = 16, 64  # unpacked_cols=128 -> 4 scale blocks of 32 per row
    packed = torch.randint(-128, 128, (rows, packed_cols), dtype=torch.int8)
    scale = torch.randint(120, 135, (rows, 4), dtype=torch.uint8).view(torch.float8_e8m0fnu)

    reference = dequantize_weight(packed, scale)
    result = dequantize_weight_triton(packed.cuda(), scale.cuda())

    assert torch.equal(result.cpu(), reference)


def test_dequantize_state_dict_streams_cpu_tensors_through_gpu():
    """The real entry point: `load_state_dict` always loads to CPU (the full checkpoint can't
    be GPU-resident all at once), so this must dispatch on CUDA *availability*, not on the
    weight already being a CUDA tensor -- and stream each pair through the GPU itself, not
    require the caller to have staged it there.
    """
    torch.manual_seed(0)
    packed = torch.randint(-128, 128, (16, 64), dtype=torch.int8)  # stays on CPU
    scale = torch.randint(120, 135, (16, 4), dtype=torch.uint8).view(torch.float8_e8m0fnu)  # stays on CPU
    expected = dequantize_weight(packed, scale)

    state_dict = {"layers.0.ffn.experts.0.w1.weight": packed, "layers.0.ffn.experts.0.w1.scale": scale}
    dequantize_state_dict_(state_dict)

    result = state_dict["layers.0.ffn.experts.0.w1.weight"]
    assert "layers.0.ffn.experts.0.w1.scale" not in state_dict
    assert result.device.type == "cpu"
    assert torch.equal(result, expected)


def test_dequantize_weight_triton_matches_reference_batched_experts():
    """Real MoE expert weights are 3D-stacked (num_experts, rows, cols): the shape this kernel
    actually has to run on in production, not just the 2D worked examples."""
    torch.manual_seed(0)
    num_experts, rows, packed_cols = 4, 8, 16  # unpacked_cols=32 -> 1 scale block of 32 per row
    packed = torch.randint(-128, 128, (num_experts, rows, packed_cols), dtype=torch.int8)
    scale = torch.randint(120, 135, (num_experts, rows, 1), dtype=torch.uint8).view(torch.float8_e8m0fnu)

    reference = dequantize_weight(packed, scale)
    result = dequantize_weight_triton(packed.cuda(), scale.cuda())

    assert torch.equal(result.cpu(), reference)
