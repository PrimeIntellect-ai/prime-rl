"""DeepSeek V4 checks that need no GPU, kept out of the `gpu`-marked whole-model module.

The quantization math is a pure function over hand-built tensors, the state-dict passes only
need a small CPU model, and the config checks only ever construct a `DeepseekV4Config`. None
of that needs CUDA, and a module-level `pytest.mark.gpu` cannot be undone per test, so they
live here and run in the CPU job. `_MODEL` is duplicated from `test_deepseek_v4.py` rather
than shared, because importing that module would pull in its `pytest.mark.gpu`.
"""

import re

import pytest
import torch

from prime_rl.trainer.models.deepseek_v4 import DeepseekV4Config, DeepseekV4ForCausalLM
from prime_rl.trainer.models.deepseek_v4.quantize import (
    _unpack_mxfp4,
    dequantize_state_dict_,
    dequantize_weight,
    quantize_fp8_block,
    quantize_mxfp4,
    quantize_state_dict_,
)

# Deliberately heterogeneous: one layer of every attention type, hash-routed bootstrap
# layers ahead of standard MoE ones, and a sliding window narrow enough that the compressed
# branches are what carries any long-range signal.
_MODEL = dict(
    vocab_size=64,
    hidden_size=128,
    moe_intermediate_size=64,
    num_hidden_layers=5,
    num_attention_heads=4,
    num_key_value_heads=1,
    head_dim=32,
    q_lora_rank=64,
    partial_rotary_factor=0.5,
    rope_theta=10000.0,
    compress_rope_theta=160000.0,
    max_position_embeddings=256,
    sliding_window=6,
    o_groups=2,
    o_lora_rank=16,
    layer_types=[
        "sliding_attention",
        "compressed_sparse_attention",
        "heavily_compressed_attention",
        "compressed_sparse_attention",
        "sliding_attention",
    ],
    compress_rates={"compressed_sparse_attention": 4, "heavily_compressed_attention": 8},
    index_n_heads=4,
    index_head_dim=24,
    # Smaller than the number of compressed entries the sequence yields, so the Lightning
    # Indexer's selection has to actually discard some of them.
    index_topk=2,
    n_routed_experts=8,
    num_experts_per_tok=3,
    n_shared_experts=1,
    scoring_func="sqrtsoftplus",
    routed_scaling_factor=1.5,
    swiglu_limit=10.0,
    num_hash_layers=2,
    hc_mult=4,
    hc_sinkhorn_iters=20,
    hc_eps=1e-6,
    rms_norm_eps=1e-6,
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


def test_dequantize_state_dict_pops_scale_and_leaves_other_keys_untouched():
    weight = torch.tensor([[1.0, 2.0], [-1.0, 0.5]], dtype=torch.float32).to(torch.float8_e4m3fn)
    scale = torch.tensor([[128]], dtype=torch.uint8).view(torch.float8_e8m0fnu)
    routing = torch.tensor([0, 1, 2], dtype=torch.int64)
    plain = torch.randn(3, dtype=torch.bfloat16)
    state_dict = {
        "layers.0.attn.wq_a.weight": weight,
        "layers.0.attn.wq_a.scale": scale,
        "layers.0.ffn.gate.tid2eid": routing,
        "embed.weight": plain,
    }

    dequantize_state_dict_(state_dict)

    assert set(state_dict) == {"layers.0.attn.wq_a.weight", "layers.0.ffn.gate.tid2eid", "embed.weight"}
    assert state_dict["layers.0.attn.wq_a.weight"].dtype == torch.bfloat16
    assert torch.equal(
        state_dict["layers.0.attn.wq_a.weight"], torch.tensor([[2.0, 4.0], [-2.0, 1.0]], dtype=torch.bfloat16)
    )
    assert torch.equal(state_dict["layers.0.ffn.gate.tid2eid"], routing)
    assert torch.equal(state_dict["embed.weight"], plain)


def test_quantize_fp8_block_round_trips_within_the_format():
    """The fp8 leg, and the two invariants a wrong scale would break.

    The error bound is the format's own: e4m3 carries three mantissa bits, so a value can be
    off by half an ulp, at most `2**-4` of the block's largest magnitude. Values already on
    the grid have to come back untouched.

    The scale assertions are what actually pin `_e8m0_scale`. Asserting the scales are powers
    of two would be vacuous, since `float8_e8m0fnu` cannot hold anything else. Bracketing the
    exponent from both sides is not: one halving too far and the block saturates at 448, one
    doubling too far and it throws away a mantissa bit for nothing.
    """
    torch.manual_seed(0)
    weight = torch.randn(256, 384, dtype=torch.bfloat16) * 3

    quantized, scale = quantize_fp8_block(weight)

    assert (quantized.dtype, scale.dtype) == (torch.float8_e4m3fn, torch.float8_e8m0fnu)
    assert quantized.shape == weight.shape
    assert scale.shape == (2, 3)

    # Both halves, because the two errors are opposite and a scale can only be wrong one way.
    block_amax = weight.float().view(2, 128, 3, 128).abs().amax(dim=(1, 3))
    assert (block_amax / scale.float() <= 448).all(), "a block clips: its scale was rounded up"
    assert (block_amax / (scale.float() / 2) > 448).all(), "the scale is coarser than it needs to be"

    restored = dequantize_weight(quantized, scale)
    assert (restored.float() - weight.float()).abs().max() <= 2**-4 * weight.float().abs().max()

    on_grid = torch.tensor([[1.0, 2.0, -4.0, 0.5]]).repeat(1, 32)
    assert torch.equal(dequantize_weight(*quantize_fp8_block(on_grid)), on_grid.bfloat16())


def test_quantize_mxfp4_round_trips_and_packs_low_nibble_first():
    """The MXFP4 leg, pinned against the decode path rather than against a restated layout.

    `_unpack_mxfp4` is what reads the real checkpoint, so asserting the packed bytes through
    it is what proves the nibble order is the checkpoint's and not merely self-consistent.
    e2m1 carries one mantissa bit, hence the `2**-2` bound.
    """
    grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    on_grid = torch.cat([grid, -grid]).repeat(4).view(2, 32)

    packed, scale = quantize_mxfp4(on_grid)

    assert (packed.dtype, scale.dtype) == (torch.int8, torch.float8_e8m0fnu)
    assert packed.shape == (2, 16), "two e2m1 values share a byte, halving the last dim"
    assert scale.shape == (2, 1), "one scale per 32 values along the input dim"
    assert torch.equal(_unpack_mxfp4(packed) * scale.float(), on_grid)
    assert torch.equal(dequantize_weight(packed, scale), on_grid.bfloat16())

    torch.manual_seed(0)
    weight = torch.randn(64, 128, dtype=torch.bfloat16)
    restored = dequantize_weight(*quantize_mxfp4(weight))
    assert (restored.float() - weight.float()).abs().max() <= 2**-2 * weight.float().abs().max()


# The families the checkpoint stores quantized, read off the real checkpoint's safetensors
# headers with the layer and expert indices folded away. Written out for the same reason
# `_VLLM_MAPPED_NAMES` is in `test_deepseek_v4.py`: the dispatch under test cannot be its own
# oracle, and the failure this guards against is a family silently going out unquantized.
_FP8_FAMILIES = {
    "layers.{i}.attn.wq_a",
    "layers.{i}.attn.wq_b",
    "layers.{i}.attn.wkv",
    "layers.{i}.attn.wo_a",
    "layers.{i}.attn.wo_b",
    "layers.{i}.attn.indexer.wq_b",
    "layers.{i}.ffn.shared_experts.w1",
    "layers.{i}.ffn.shared_experts.w2",
    "layers.{i}.ffn.shared_experts.w3",
}
_MXFP4_FAMILIES = {
    "layers.{i}.ffn.experts.{i}.w1",
    "layers.{i}.ffn.experts.{i}.w2",
    "layers.{i}.ffn.experts.{i}.w3",
}


@pytest.fixture(scope="module")
def on_disk_state_dict() -> dict[str, torch.Tensor]:
    """What a broadcast carries: `convert_to_hf` output, in the checkpoint's own key naming."""
    torch.manual_seed(0)
    model = DeepseekV4ForCausalLM._from_config(DeepseekV4Config(**_MODEL))
    return model.convert_to_hf(dict(model.state_dict()))


def _family(key: str) -> str:
    return re.sub(r"\.\d+\.", ".{i}.", key)


def test_quantize_state_dict_quantizes_exactly_the_families_the_checkpoint_stores_quantized(
    on_disk_state_dict,
):
    """Every quantized family, and nothing else.

    `layers.N.attn.wkv.weight` is fp8 while `layers.N.attn.compressor.wkv.weight` is
    `bfloat16` on disk, so the dispatch has to match anchored key patterns; a substring test
    would take the compressor's too. Untouched keys are compared by value, which covers the
    whole long tail of norms, compressors, gates and hyper-connection tensors at once.
    """
    state_dict = dict(on_disk_state_dict)

    quantize_state_dict_(state_dict, "fp4")

    fp8 = {_family(k).removesuffix(".weight") for k, v in state_dict.items() if v.dtype == torch.float8_e4m3fn}
    mxfp4 = {_family(k).removesuffix(".weight") for k, v in state_dict.items() if v.dtype == torch.int8}
    scales = {_family(k).removesuffix(".scale") for k, v in state_dict.items() if v.dtype == torch.float8_e8m0fnu}
    assert fp8 == _FP8_FAMILIES
    assert mxfp4 == _MXFP4_FAMILIES
    assert scales == _FP8_FAMILIES | _MXFP4_FAMILIES, "every quantized weight needs its `.scale` sibling"

    quantized = {k.removesuffix(".weight") for k in state_dict if _family(k).removesuffix(".weight") in scales}
    for key, before in on_disk_state_dict.items():
        if key.removesuffix(".weight") in quantized:
            continue
        assert torch.equal(state_dict[key], before), f"{key} was rewritten"


# Real (weight shape -> scale shape) pairs, read off the safetensors headers of
# `deepseek-ai/DeepSeek-V4-Flash-0731`. These are the shapes vLLM's `create_weights` allocates
# for this checkpoint, so they are an oracle for the block geometry in a way that recomputing
# `ceil(N / 128)` here would not be: that would only restate what the code already does.
_REAL_FP8_SHAPES = {
    "attn.wq_a": ((1024, 4096), (8, 32)),
    "attn.wq_b": ((32768, 1024), (256, 8)),
    "attn.wkv": ((512, 4096), (4, 32)),
    "attn.wo_a": ((8192, 4096), (64, 32)),
    "attn.wo_b": ((4096, 8192), (32, 64)),
    "attn.indexer.wq_b": ((8192, 1024), (64, 8)),
    "ffn.shared_experts.w1": ((2048, 4096), (16, 32)),
    "ffn.shared_experts.w2": ((4096, 2048), (32, 16)),
}
# For the experts the on-disk weight is already packed, so the unpacked input is twice as wide:
# `(unpacked shape) -> (packed shape, scale shape)`.
_REAL_MXFP4_SHAPES = {
    "ffn.experts.E.w1": ((2048, 4096), (2048, 2048), (2048, 128)),
    "ffn.experts.E.w2": ((4096, 2048), (4096, 1024), (4096, 64)),
}


def test_quantize_emits_the_block_geometry_the_real_checkpoint_ships():
    """Quantize at the real checkpoint's shapes and check the scale grids against its headers.

    `Fp8LinearMethod` and `Mxfp4MoEMethod` size their scale parameters from the same config
    the checkpoint was written under, so reproducing the checkpoint's grids is what makes the
    broadcast loadable. A shape mismatch here is the loud half of the defect this module fixes.
    """
    for name, (weight_shape, scale_shape) in _REAL_FP8_SHAPES.items():
        weight, scale = quantize_fp8_block(torch.zeros(weight_shape, dtype=torch.bfloat16))
        assert weight.shape == weight_shape, name
        assert scale.shape == scale_shape, name

    for name, (unpacked, packed_shape, scale_shape) in _REAL_MXFP4_SHAPES.items():
        packed, scale = quantize_mxfp4(torch.zeros(unpacked, dtype=torch.bfloat16))
        assert packed.shape == packed_shape, name
        assert scale.shape == scale_shape, name


def test_quantize_for_weight_transfer_only_fires_for_a_quantized_checkpoint(on_disk_state_dict):
    """The gate. A checkpoint with no `quantization_config` broadcasts exactly as it trained.

    That is what keeps the `bfloat16` mini checkpoint and every existing test on the plain
    path, mirroring how `dequantize_state_dict_` is already a no-op on the way in.
    """
    model = DeepseekV4ForCausalLM._from_config(DeepseekV4Config(**_MODEL))
    assert getattr(model.config, "quantization_config", None) is None

    unchanged = model.quantize_for_weight_transfer(dict(on_disk_state_dict))
    assert unchanged.keys() == on_disk_state_dict.keys()
    assert all(torch.equal(unchanged[key], value) for key, value in on_disk_state_dict.items())

    model.config.quantization_config = {"quant_method": "fp8", "weight_block_size": [128, 128]}
    quantized = model.quantize_for_weight_transfer(dict(on_disk_state_dict))
    assert quantized["layers.0.attn.wq_a.weight"].dtype == torch.float8_e4m3fn


def test_quantize_state_dict_refuses_the_fp8_expert_layout():
    """DeepSeek-V4-Flash-Base is not supported, and says so rather than shipping fp4 bytes
    into an fp8 expert parameter, which would be the same silent corruption in a new place."""
    with pytest.raises(NotImplementedError, match="expert_dtype='fp8'"):
        quantize_state_dict_({}, "fp8")

    with pytest.raises(ValueError, match="Unsupported DeepSeek V4 expert_dtype"):
        quantize_state_dict_({}, "nf4")
