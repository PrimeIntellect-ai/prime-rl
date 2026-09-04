"""DeepSeek V4 checks that need no GPU, kept out of the `gpu`-marked whole-model module.

The dequantization math is a pure function over hand-built tensors, and the config checks only
ever construct a `DeepseekV4Config`. Neither needs CUDA, and a module-level `pytest.mark.gpu`
cannot be undone per test, so they live here and run in the CPU job.
"""

import pytest
import torch

from prime_rl.trainer.models.deepseek_v4 import DeepseekV4Config
from prime_rl.trainer.models.deepseek_v4 import attention as dsv4_attention
from prime_rl.trainer.models.deepseek_v4.attention import DeepseekV4Attention
from prime_rl.trainer.models.deepseek_v4.dequantize import dequantize_weight

# The attention half of the toy config the GPU tests run: 4 heads over 32 channels, which the
# fused kernel cannot tile. Everything else is shrunk to whatever still builds one layer on a CPU.
_TOY_ATTENTION = dict(
    vocab_size=64,
    hidden_size=128,
    num_hidden_layers=1,
    num_hash_layers=0,
    num_attention_heads=4,
    head_dim=32,
    q_lora_rank=64,
    o_groups=2,
    o_lora_rank=16,
    sliding_window=6,
    layer_types=["sliding_attention"],
)


def test_deepseek_v4_config_translates_legacy_compress_ratios():
    """Real checkpoints ship the V3-flavoured legacy `compress_ratios`/`num_hash_layers` schema
    instead of `layer_types`/`mlp_layer_types`, which is what prime-rl's model code reads, so the
    config has to translate between them. Loading the real checkpoint without this built the
    wrong per-layer attention schedule outright.
    """
    config = DeepseekV4Config(num_hidden_layers=6, compress_ratios=[0, 0, 4, 128, 4, 128], num_hash_layers=2)

    assert config.layer_types == [
        "sliding_attention",
        "sliding_attention",
        "compressed_sparse_attention",
        "heavily_compressed_attention",
        "compressed_sparse_attention",
        "heavily_compressed_attention",
    ]
    assert config.mlp_layer_types == ["hash_moe", "hash_moe", "moe", "moe", "moe", "moe"]


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


def test_deepseek_v4_attn_impl_never_serializes():
    """`_attn_impl` is a test knob, so it must not survive into a checkpoint's `config.json`.

    HF's `to_dict` deep-copies `__dict__` and strips only an explicit list of names, so the
    underscore prefix buys nothing on its own. A checkpoint that shipped `_attn_impl="eager"` would
    load and train, silently on the dense reference path at a fraction of the throughput.
    """
    config = DeepseekV4Config(**_TOY_ATTENTION, _attn_impl="eager")

    assert config._attn_impl == "eager", "the knob must still be readable in memory"
    assert "_attn_impl" not in config.to_dict()


@pytest.mark.skipif(
    dsv4_attention.dsv4_sparse_attn is None, reason="without tilelang the gate names the missing import, not the shape"
)
def test_deepseek_v4_attention_rejects_a_config_the_kernel_cannot_tile():
    """Asking for the kernel at a shape it cannot serve has to fail where the config is chosen.

    The kernels tile the head axis up to a power of two of at least 16 and index the attention
    sinks over that padded block, so 4 heads would read past the end of a one-row tensor. Without
    the gate the first forward dies inside a tilelang compile, a long way from the config that
    caused it, and at 16 heads it is the backward that dies, after a step has already run. The
    message has to name the offending head count, or the reader is sent to the wrong knob.
    """
    with pytest.raises(ValueError, match=r"heads per group but this shape has 4\b"):
        DeepseekV4Attention(DeepseekV4Config(**_TOY_ATTENTION, _attn_impl="kernel"), layer_idx=0)
