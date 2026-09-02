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

from prime_rl.trainer.models.base import WEIGHT_TRANSFER_SCALE_SUFFIX
from prime_rl.trainer.models.deepseek_v4 import DeepseekV4Config, DeepseekV4ForCausalLM
from prime_rl.trainer.models.deepseek_v4.quantize import (
    _E8M0_BIAS,
    _e8m0_scale,
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
    # Eager, because the fused kernel cannot tile 4 attention heads. Nothing here reads
    # attention; the Flash-shaped config the kernel does accept shrinks the MoE fields
    # these tests quantize to nothing.
    _attn_impl="eager",
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


# The e2m1 grid, and the hand-rolled MXFP4 quantizer that `quantize_mxfp4` used before it
# delegated to `torchao`. Kept as an oracle rather than deleted: it was verified byte-identical
# against DeepSeek's own quantizer on real checkpoint tensors, so it is the one independent
# check that a torchao bump has not quietly changed the rounding or the scale derivation. That
# is a failure mode with no loud symptom -- the shapes and dtypes stay right and only the model's
# output degrades -- so it would otherwise surface on a training run, not here.
_E2M1_MAGNITUDES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
_E2M1_MAX = 6.0


def _reference_quantize_mxfp4(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rows, cols = weight.shape
    blocks = weight.float().view(rows, cols // 32, 32)
    scale = _e8m0_scale(blocks.abs().amax(dim=-1), _E2M1_MAX)
    scaled = (blocks / scale.float()[..., None]).clamp(-_E2M1_MAX, _E2M1_MAX).reshape(rows, cols)

    # Nearest grid point, ties to even, which is what a hardware e2m1 cast does. `torch.round`
    # cannot stand in: the grid is not uniform above 2.0.
    magnitude = scaled.abs()
    lower = (torch.bucketize(magnitude, _E2M1_MAGNITUDES) - 1).clamp(0, _E2M1_MAGNITUDES.numel() - 2)
    upper = lower + 1
    to_lower, to_upper = magnitude - _E2M1_MAGNITUDES[lower], _E2M1_MAGNITUDES[upper] - magnitude
    take_upper = (to_upper < to_lower) | ((to_upper == to_lower) & (upper % 2 == 0))
    index = torch.where(take_upper, upper, lower)

    nibbles = (index | (torch.signbit(scaled).to(torch.int64) << 3)).to(torch.uint8)
    return (nibbles[:, 0::2] | (nibbles[:, 1::2] << 4)).contiguous().view(torch.int8), scale


def _e2m1_ties_and_neighbours() -> torch.Tensor:
    """Every midpoint of the e2m1 grid, and the float32 either side of each."""
    ties = (_E2M1_MAGNITUDES[:-1] + _E2M1_MAGNITUDES[1:]) / 2
    below = torch.nextafter(ties, torch.tensor(0.0))
    above = torch.nextafter(ties, torch.tensor(float("inf")))
    return torch.cat([ties, below, above, -ties, -below, -above])


def test_quantize_mxfp4_matches_the_hand_rolled_reference_byte_for_byte():
    """`torchao`'s RCEIL mode against the transcription of DeepSeek's own kernel.

    The tie points are the subtle half. Ties-to-even is decided one ulp at a time, and a
    rounding mode that broke it would still round almost every random value correctly, so
    random data alone cannot see it: hence every midpoint of the grid and both of its float32
    neighbours, scaled so the block's own scale is exactly 1 and the grid is hit unshifted.

    The `logspace` row is the other half, exercising the scale derivation across 10^12 of
    dynamic range: `ceil(log2(amax / 6.0))` and `ceil(log2(amax)) - ceil(log2(6.0))` agree on
    most inputs and differ across a whole block whenever they do not.
    """
    ties = _e2m1_ties_and_neighbours()
    rows = {
        "e2m1 ties and their neighbours": ties.repeat(2, 32 * 2 // ties.numel() + 1)[:, :64],
        "exact grid points": torch.cat([_E2M1_MAGNITUDES, -_E2M1_MAGNITUDES]).repeat(2, 4),
        "an all-zero block": torch.zeros(2, 64),
        "logspace over 10^12": torch.randn(24, 64) * torch.logspace(-6, 6, 24)[:, None],
    }
    torch.manual_seed(0)
    rows["random bfloat16"] = torch.randn(64, 256, dtype=torch.bfloat16).float()

    for label, weight in rows.items():
        packed, scale = quantize_mxfp4(weight)
        reference_packed, reference_scale = _reference_quantize_mxfp4(weight)
        assert torch.equal(packed.view(torch.uint8), reference_packed.view(torch.uint8)), label
        assert torch.equal(scale.view(torch.uint8), reference_scale.view(torch.uint8)), label

    # An all-zero block is the one case where torchao alone disagrees: `log2(0)` sends it to
    # exponent -127 where DeepSeek's kernel floors amax and lands on -126. Pinned from the
    # value side so the fix cannot be silently reverted.
    _, zero_scale = quantize_mxfp4(torch.zeros(1, 32))
    assert zero_scale.view(torch.uint8).item() == _E8M0_BIAS - 126


def test_quantize_mxfp4_is_independent_along_the_leading_dims():
    """What lets a rank quantize its own expert shard before the gather.

    MXFP4's blocks run along the last dim, so an expert-batched `[E, I, H]` tensor has to
    quantize to exactly the rows the per-expert calls produce; if it did not, the shard the
    trainer broadcasts would not be the shard vLLM expects. This is also the invariant the
    internal chunking rests on.
    """
    torch.manual_seed(0)
    batched = torch.randn(6, 8, 96, dtype=torch.bfloat16)

    packed, scale = quantize_mxfp4(batched)

    assert packed.shape == (6, 8, 48) and scale.shape == (6, 8, 3)
    for expert in range(batched.shape[0]):
        expert_packed, expert_scale = quantize_mxfp4(batched[expert])
        assert torch.equal(packed[expert].view(torch.uint8), expert_packed.view(torch.uint8))
        assert torch.equal(scale[expert].view(torch.uint8), expert_scale.view(torch.uint8))


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


def test_pre_quantized_expert_shards_reach_the_same_bytes_as_the_cpu_path():
    """The wire format must not depend on where the experts were quantized.

    `gather_weights_parallel` hands the routed experts to
    `quantize_shard_for_weight_transfer` while they are still sharded and on the accelerator,
    so `convert_to_hf` receives a `(weight, scale)` pair under one prime key and has to walk
    the scale through the same expert unstack and `mlp.` -> `ffn.` rename as its weight. The
    check that matters is not that some scale key appears, but that the whole broadcast comes
    out byte-identical to quantizing everything on CPU afterwards, which is the path the
    branch already verified against the real checkpoint.
    """
    torch.manual_seed(0)
    model = DeepseekV4ForCausalLM._from_config(DeepseekV4Config(**_MODEL))
    model.config.quantization_config = {"quant_method": "fp8", "weight_block_size": [128, 128]}
    state_dict = dict(model.state_dict())

    expected = model.quantize_for_weight_transfer(model.convert_to_hf(dict(state_dict)))

    claimed, gathered = 0, {}
    for key, value in state_dict.items():
        quantized = model.quantize_shard_for_weight_transfer(key, value)
        if quantized is None:
            gathered[key] = value
            continue
        claimed += 1
        gathered[key], gathered[key + WEIGHT_TRANSFER_SCALE_SUFFIX] = quantized
    actual = model.quantize_for_weight_transfer(model.convert_to_hf(gathered))

    assert claimed == 3 * sum("mlp.experts.gate_proj" in key for key in state_dict), "not every MoE layer was claimed"
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        # `torch.equal` has no kernel for the float8 dtypes, and the bytes are the point anyway.
        if value.dtype.itemsize == 1:
            assert torch.equal(actual[key].view(torch.uint8), value.view(torch.uint8)), key
        else:
            assert torch.equal(actual[key], value), key

    # The hyper-connection parameters really are named `...scale`, so the borrowed suffix has
    # to be one no module path can produce, and they have to survive untouched.
    assert any(key.endswith("hc_attn_scale") for key in expected)


def test_quantize_shard_for_weight_transfer_claims_nothing_without_a_quantization_config():
    """Same gate as `quantize_for_weight_transfer`: the plain `bfloat16` mini checkpoint, and
    every other model, has to keep broadcasting exactly what it trained."""
    model = DeepseekV4ForCausalLM._from_config(DeepseekV4Config(**_MODEL))
    experts = "model.layers.2.mlp.experts.gate_proj"
    shard = model.state_dict()[experts]

    assert model.quantize_shard_for_weight_transfer(experts, shard) is None

    model.config.quantization_config = {"quant_method": "fp8"}
    assert model.quantize_shard_for_weight_transfer(experts, shard) is not None
    assert model.quantize_shard_for_weight_transfer("model.layers.2.mlp.router.gate.weight", shard) is None
    assert model.quantize_shard_for_weight_transfer("model.layers.2.attn_hc.scale", shard) is None


def test_quantize_state_dict_refuses_the_fp8_expert_layout():
    """DeepSeek-V4-Flash-Base is not supported, and says so rather than shipping fp4 bytes
    into an fp8 expert parameter, which would be the same silent corruption in a new place."""
    with pytest.raises(NotImplementedError, match="expert_dtype='fp8'"):
        quantize_state_dict_({}, "fp8")

    with pytest.raises(ValueError, match="Unsupported DeepSeek V4 expert_dtype"):
        quantize_state_dict_({}, "nf4")
