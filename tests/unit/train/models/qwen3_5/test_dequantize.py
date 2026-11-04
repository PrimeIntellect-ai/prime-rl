import pytest
import torch

from prime_rl.trainer.models.qwen3_5.converting_qwen3_5 import is_prime_state_dict
from prime_rl.trainer.models.qwen3_5.dequantize import dequantize_state_dict_, dequantize_weight


def test_dequantize_weight_blockwise_fp8():
    """Blockwise fp8 case: a [2, 4] scale grid over a [4, 8] weight -> 2x2 blocks."""
    weight = torch.tensor([[1.0, 2.0, -1.0, 0.5, 1.0, 1.0, 1.0, 1.0],
                           [2.0, 4.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                           [1.0, 1.0, 1.0, 1.0, 3.0, 6.0, 1.0, 1.0],
                           [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -2.0, 0.5]], dtype=torch.float32).to(torch.float8_e4m3fn)
    # 2x2 blocks: [2, 4] scale grid with distinct multipliers per block.
    scale_inv = torch.tensor([[2.0, 1.0, 4.0, 0.5], [3.0, 1.0, 1.0, 2.0]], dtype=torch.bfloat16)

    result = dequantize_weight(weight, scale_inv)

    assert result.dtype == torch.bfloat16
    assert result.shape == weight.shape
    # e4m3 represents these values exactly, so the dequantization is exact.
    expected = torch.tensor(
        [[2.0, 4.0, -1.0, 0.5, 4.0, 4.0, 0.5, 0.5],
         [4.0, 8.0, 1.0, 1.0, 4.0, 4.0, 0.5, 0.5],
         [3.0, 3.0, 1.0, 1.0, 3.0, 6.0, 2.0, 2.0],
         [3.0, 3.0, 1.0, 1.0, 1.0, 1.0, -4.0, 1.0]], dtype=torch.bfloat16
    )
    assert torch.equal(result, expected)


def test_dequantize_weight_rejects_non_fp8():
    with pytest.raises(ValueError, match="Unsupported quantized weight dtype"):
        dequantize_weight(torch.ones(2, 2, dtype=torch.bfloat16), torch.ones(1, 1, dtype=torch.bfloat16))


def test_dequantize_state_dict_merges_and_drops_scales():
    """`*.weight`/`*.weight_scale_inv` pairs are merged and the scale keys dropped, in place."""
    weight = torch.tensor([[1.0, -2.0]], dtype=torch.float32).to(torch.float8_e4m3fn)
    state_dict = {
        "model.layers.0.mlp.up_proj.weight": weight,
        "model.layers.0.mlp.up_proj.weight_scale_inv": torch.tensor([[4.0]], dtype=torch.bfloat16),
        "model.layers.0.input_layernorm.weight": torch.ones(4, dtype=torch.bfloat16),
    }

    dequantize_state_dict_(state_dict)

    assert set(state_dict) == {"model.layers.0.mlp.up_proj.weight", "model.layers.0.input_layernorm.weight"}
    assert state_dict["model.layers.0.mlp.up_proj.weight"].dtype == torch.bfloat16
    assert torch.equal(state_dict["model.layers.0.mlp.up_proj.weight"], torch.tensor([[4.0, -8.0]], dtype=torch.bfloat16))


def test_dequantize_state_dict_noop_without_scales():
    """Plain BF16 snapshots have no `*.weight_scale_inv` siblings and pass through untouched."""
    state_dict = {
        "model.layers.0.mlp.up_proj.weight": torch.ones(2, 2, dtype=torch.bfloat16),
        "model.layers.0.linear_attn.A_log": torch.ones(8, dtype=torch.bfloat16),
    }
    before = {k: v.clone() for k, v in state_dict.items()}

    dequantize_state_dict_(state_dict)

    assert state_dict.keys() == before.keys()
    for key, tensor in before.items():
        assert torch.equal(state_dict[key], tensor)


def test_is_prime_state_dict_rejects_fp8_snapshots():
    """Raw compressed-tensors fp8 snapshots carry `*.weight_scale_inv` siblings and must not
    be detected as PrimeRL format: they are only loadable after dequantization, which runs
    in the HF->PrimeRL auto-conversion that this detection gates."""
    fp8_snapshot = {
        "model.layers.0.linear_attn.A_log": torch.ones(8, dtype=torch.bfloat16),
        "model.layers.0.mlp.up_proj.weight": torch.ones(2, 2, dtype=torch.float8_e4m3fn),
        "model.layers.0.mlp.up_proj.weight_scale_inv": torch.ones(1, 1, dtype=torch.bfloat16),
    }
    bf16_snapshot = {
        "model.layers.0.linear_attn.A_log": torch.ones(8, dtype=torch.bfloat16),
        "model.layers.0.mlp.up_proj.weight": torch.ones(2, 2, dtype=torch.bfloat16),
    }

    assert is_prime_state_dict(fp8_snapshot) is False
    assert is_prime_state_dict(bf16_snapshot) is True
