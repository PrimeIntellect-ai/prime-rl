import torch

from prime_rl.trainer.models.dequantize import dequantize_state_dict_, dequantize_weight


def test_dequantize_weight_dense_fp8():
    """Dense fp8 case: one `float8_e8m0fnu` scale block covers the whole weight."""
    weight = torch.tensor([[1.0, 2.0], [-1.0, 0.5]], dtype=torch.float32).to(torch.float8_e4m3fn)
    scale = torch.tensor([[128]], dtype=torch.uint8).view(torch.float8_e8m0fnu)  # byte 128 -> 2**(128-127) = 2.0

    result = dequantize_weight(weight, scale)

    assert result.dtype == torch.bfloat16
    assert torch.equal(result, torch.tensor([[2.0, 4.0], [-2.0, 1.0]], dtype=torch.bfloat16))


def test_dequantize_weight_packed_mxfp4():
    """Packed MXFP4 expert case: unpack two e2m1 nibbles per byte, then a per-block scale."""
    # Nibble layout per byte is (high << 4) | low; e2m1 LUT indices used here:
    # 2->1.0, 4->2.0, 10->-1.0, 6->4.0, 0->0.0, 7->6.0, 9->-0.5, 3->1.5.
    packed = torch.tensor(
        [
            [(4 << 4) | 2, (6 << 4) | 10],  # row 0 -> unpacks to [1.0, 2.0, -1.0, 4.0]
            [(7 << 4) | 0, (3 << 4) | 9],  # row 1 -> unpacks to [0.0, 6.0, -0.5, 1.5]
        ],
        dtype=torch.int8,
    )
    # [2, 2] scale grid over the unpacked [2, 4] weight -> block_rows=1, block_cols=2.
    scale = torch.tensor([[127, 128], [129, 126]], dtype=torch.uint8).view(torch.float8_e8m0fnu)

    result = dequantize_weight(packed, scale)

    expected = torch.tensor([[1.0, 2.0, -2.0, 8.0], [0.0, 24.0, -0.25, 0.75]], dtype=torch.bfloat16)
    assert result.dtype == torch.bfloat16
    assert torch.equal(result, expected)


def test_dequantize_weight_blockwise_bf16_scale():
    """Qwen's layout: a `bfloat16` [2, 2] scale grid over a [4, 4] fp8 weight -> 2x2 blocks."""
    weight = torch.ones(4, 4).to(torch.float8_e4m3fn)
    scale = torch.tensor([[2.0, 0.5], [4.0, 1.0]], dtype=torch.bfloat16)

    result = dequantize_weight(weight, scale)

    expected = torch.tensor([[2.0, 2.0, 0.5, 0.5]] * 2 + [[4.0, 4.0, 1.0, 1.0]] * 2, dtype=torch.bfloat16)
    assert result.dtype == torch.bfloat16
    assert torch.equal(result, expected)


def test_dequantize_state_dict_merges_pairs_and_drops_scales():
    """Each `*.weight` with a scale sibling becomes its `bfloat16` product and the scale key is
    dropped; keys without a sibling pass through untouched."""
    state_dict = {
        "layers.0.mlp.up_proj.weight": torch.tensor([[1.0, -2.0]]).to(torch.float8_e4m3fn),
        "layers.0.mlp.up_proj.weight_scale_inv": torch.tensor([[4.0]], dtype=torch.bfloat16),
        "layers.0.input_layernorm.weight": torch.ones(2, dtype=torch.bfloat16),
    }

    dequantize_state_dict_(state_dict, ".weight_scale_inv")

    assert set(state_dict) == {"layers.0.mlp.up_proj.weight", "layers.0.input_layernorm.weight"}
    assert torch.equal(state_dict["layers.0.mlp.up_proj.weight"], torch.tensor([[4.0, -8.0]], dtype=torch.bfloat16))
    assert torch.equal(state_dict["layers.0.input_layernorm.weight"], torch.ones(2, dtype=torch.bfloat16))
