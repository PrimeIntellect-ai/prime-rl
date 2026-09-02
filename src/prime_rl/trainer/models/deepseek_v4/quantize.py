"""Convert DeepSeek V4 Flash weights between the checkpoint's on-disk formats and `bfloat16`.

The real `deepseek-ai/DeepSeek-V4-Flash-0731` checkpoint ships dense linear layers as
block-quantized FP8 (`float8_e4m3fn` weight + `float8_e8m0fnu` per-block scale, 128x128
blocks) and MoE expert weights as packed MXFP4 (two 4-bit e2m1 values per byte,
`float8_e8m0fnu` scale, 1x32 blocks after unpacking). Both directions live in one module so
the e2m1 grid, the nibble order and the block geometry are each written once and the
round-trip invariant reads in one place.

Dequantization runs on the way in. Neither prime-rl's model code nor its DCP loading path can
consume either format, so `convert_to_prime` unpacks everything to `bfloat16` before the
state dict reaches the model.

Quantization runs on the way out, and only onto the wire. vLLM serves this model from the
stock repo, so it builds its parameters from the same `config.json` the checkpoint ships:
`Fp8LinearMethod` with `block_quant=True` and `is_scale_e8m0=True` for the attention and
shared-expert linears, `Mxfp4MoEMethod` for the routed experts. A `bfloat16` broadcast lands
on those. The routed experts fail loudly (a `[2048, 2048]` `uint8` parameter slice against a
`[2048, 4096]` bf16 tensor), but the fp8 linears fail *silently*: bf16 `[1024, 4096]` matches
the fp8 parameter's shape exactly, so `copy_` succeeds as a numeric cast with no division by
a scale, and the layer then computes `fp8(W) * boot_time_scale` instead of `W`. Saved
training checkpoints never pass through the transport chain and stay bf16.

Naming needs no work here. vLLM's `_make_deepseek_v4_weights_mapper` rewrites the on-disk
`.scale` siblings this module writes to `.weight_scale` (routed experts) and
`.weight_scale_inv` (everything else), which is what its `create_weights` registered.

Which key family carries which format was read off the real checkpoint's safetensors headers
(72317 tensors across 48 shards), on the raw on-disk names:

| family                                                          | weight           | scale            | block  |
| --------------------------------------------------------------- | ---------------- | ---------------- | ------ |
| `layers.N.attn.{wq_a,wq_b,wkv,wo_a,wo_b}.weight`,                 | `float8_e4m3fn`  | `float8_e8m0fnu` | 128x128|
| `layers.N.attn.indexer.wq_b.weight`,                              |                  |                  |        |
| `layers.N.ffn.shared_experts.w{1,2,3}.weight`                     |                  |                  |        |
| `layers.N.ffn.experts.E.w{1,2,3}.weight`                          | `int8`, 2 nibbles| `float8_e8m0fnu` | 1x32   |
| norms, compressors, `attn.indexer.weights_proj`, `ffn.gate.weight`| `bfloat16`       | -                | -      |
| `hc_*`, `attn.*.ape`, `attn.attn_sink`, `ffn.gate.bias`           | `float32`        | -                | -      |
| `ffn.gate.tid2eid` (layers 0-2 only)                              | `int64`          | -                | -      |

Note that `layers.N.attn.wkv.weight` is fp8 while `layers.N.attn.compressor.wkv.weight` is
bf16, which is why the dispatch below matches anchored patterns rather than a substring.

The table covers `layers.*` only, and so does the dispatch. The checkpoint also ships 2329
quantized `mtp.*` tensors (the multi-token-prediction heads, including an `mtp.N.main_proj`
family with no `layers.*` counterpart), but neither HF nor prime-rl instantiates them:
`conversion_chain` drops the whole prefix, so `convert_to_hf` never emits one and there is
nothing here to match.

The scale arithmetic follows DeepSeek's own kernels, shipped inside the checkpoint at
`inference/kernel.py`: `fast_round_scale` derives a power-of-two scale as
`2 ** ceil(log2(amax / value_max))` from the IEEE 754 exponent field, floors `amax` so the
scale stays a normal float, and rounds the scaled values to the format's grid. Two
independent claims rest on reading vLLM's code rather than on a serving run: that
`create_weights` builds the parameter shapes reproduced here, and that
`initialize_layerwise_reload` restores the pre-`process_weights_after_loading` layout so
these tensors can be loaded into a running engine at all.

`prime_rl.trainer.models.fp8.quantize_to_fp8_blockwise` is deliberately not reused: it emits
a float32 `amax / fp8_max` scale rather than a power-of-two UE8M0 one, and is shaped around
the GLM kernel weight-transfer path.

This is intentionally plain preprocessing, not a `ConvOp`
(`prime_rl.trainer.models.conversion_ops`): that framework is for two-way *structural*
conversions between per-key HF and PrimeRL names, whereas these passes merge and split a
`(*.weight, *.scale)` pair against a single output key.

Not implemented, and not to be inferred as working:

- **NCCL broadcast, sender bug (fatal).** `preprocess_layer_checkpoint`
  (`prime_rl/transports/weights/nccl.py`) evaluates `is_prime_state_dict` on the per-layer
  bucket. DeepSeek V4's predicate matches `mlp.router.gate.weight` / `mlp.shared_expert.`,
  neither of which is in the non-layer bucket, so that bucket falls through to
  `transformers.core_model_loading.revert_weight_conversion`. `model.hc_head.hc_fn` /
  `hc_base` / `hc_scale` then reach vLLM unrenamed and `KeyError` at step 0. The fix is to
  decide from the model's full key set, as `prime_rl.utils.weights.convert_state_dict_to_hf`
  already does.
- **NCCL broadcast, `moe_backend` and `data_parallel_size`.** `moe_backend` must stay
  `"auto"`: `deep_gemm_mega_moe` selects `DeepseekV4MegaMoEExperts`, whose `finalize_weights`
  nulls the expert weights and early-returns forever after, so a second `load_weights`
  silently keeps serving the boot weights. `data_parallel_size` must stay 1: only DP rank 0
  gets a receiver.
- **NIXL broadcast.** Its `LazyWeight` tracing composes rename ops symbolically and cannot
  carry a value-transforming step.
- **`expert_dtype = "fp8"`** (the Flash-Base layout). vLLM's `DeepseekV4FP8Config` supports
  it, and DeepSeek's `inference/convert.py` reaches it by a lossless fp4 -> fp8 upcast
  (`cast_e2m1fn_to_e4m3fn`), which would cut the per-broadcast requantization error on the
  routed experts a long way at roughly double the expert bytes.
"""

from __future__ import annotations

import re

import torch
from torch import Tensor

from prime_rl.trainer.models.conversion_ops import StateDict

FP4_E2M1_LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)
_FP4_E2M1_MAGNITUDES = FP4_E2M1_LUT[:8]
_FP4_E2M1_MAX = 6.0
_FP8_E4M3_MAX = 448.0

_FP8_BLOCK = 128
_MXFP4_BLOCK = 32

_E8M0_BIAS = 127
_E8M0_MIN_EXP, _E8M0_MAX_EXP = -127, 127
# Floor on a block's amax, so `amax / value_max` stays a normal float32 and an all-zero block
# gets a well-defined scale instead of `log2(0)`. Matches `fp4_quant_kernel`'s
# `max(amax, 6 * 2**-126)` in the checkpoint's own `inference/kernel.py`.
_MIN_NORMAL = 2.0**-126


def _log2_ceil(x: Tensor) -> Tensor:
    """`ceil(log2(x))` for positive normal float32, read off the IEEE 754 fields.

    DeepSeek's `fast_log2_ceil`, transcribed. Taking `torch.log2(x).ceil()` instead would
    round the logarithm first, which can hand back an exponent one too small and silently
    clip the block that exponent scales.
    """
    bits = x.float().view(torch.int32)
    exponent = ((bits >> 23) & 0xFF) - _E8M0_BIAS
    return exponent + ((bits & 0x7FFFFF) != 0).to(torch.int32)


def _e8m0_scale(amax: Tensor, value_max: float) -> Tensor:
    """Smallest power-of-two scale with `amax / scale <= value_max`, as `float8_e8m0fnu`.

    The exponent is encoded straight into the e8m0 byte. Materializing a float scale and
    casting it would round to nearest, and e8m0 has no mantissa, so the scale could round
    *down* and clip every value in its block.
    """
    exponent = _log2_ceil(amax.float().clamp(min=value_max * _MIN_NORMAL) / value_max)
    return (exponent.clamp(_E8M0_MIN_EXP, _E8M0_MAX_EXP) + _E8M0_BIAS).to(torch.uint8).view(torch.float8_e8m0fnu)


def _unpack_mxfp4(packed: Tensor) -> Tensor:
    """Two packed e2m1 nibbles per `int8` byte -> `float32`, doubling the last dim."""
    lut = FP4_E2M1_LUT.to(packed.device)
    u8 = packed.contiguous().view(torch.uint8)
    low_nibble = (u8 & 0xF).long()
    high_nibble = ((u8 >> 4) & 0xF).long()
    unpacked = torch.stack([lut[low_nibble], lut[high_nibble]], dim=-1)
    return unpacked.reshape(*packed.shape[:-1], 2 * packed.shape[-1])


def _round_to_e2m1(magnitude: Tensor) -> Tensor:
    """Index into `_FP4_E2M1_MAGNITUDES` nearest each non-negative magnitude, ties to even.

    Ties-to-even is what a hardware e2m1 cast does, and what DeepSeek's `T.Cast(FP4, ...)`
    leans on. `torch.round` cannot be used: the grid is not uniform above 2.0.
    """
    grid = _FP4_E2M1_MAGNITUDES.to(magnitude.device)
    lower = (torch.bucketize(magnitude, grid) - 1).clamp(0, grid.numel() - 2)
    upper = lower + 1
    to_lower, to_upper = magnitude - grid[lower], grid[upper] - magnitude
    take_upper = (to_upper < to_lower) | ((to_upper == to_lower) & (upper % 2 == 0))
    return torch.where(take_upper, upper, lower)


def quantize_fp8_block(weight: Tensor, block_size: int = _FP8_BLOCK) -> tuple[Tensor, Tensor]:
    """Quantize a dense 2D weight to `float8_e4m3fn` with a `float8_e8m0fnu` block scale.

    The scale grid is `ceil(rows / 128) x ceil(cols / 128)`, matching what vLLM's
    `create_fp8_scale_parameter` builds; a weight whose shape is not a whole number of
    blocks is zero-padded for the block reduction only, and comes back at its own shape.
    """
    if weight.ndim != 2:
        raise ValueError(f"FP8 block quantization expects a 2D weight, got shape={tuple(weight.shape)}")

    rows, cols = weight.shape
    scale_rows = -(-rows // block_size)
    scale_cols = -(-cols // block_size)
    padded = weight.new_zeros(scale_rows * block_size, scale_cols * block_size, dtype=torch.float32)
    padded[:rows, :cols] = weight

    blocks = padded.view(scale_rows, block_size, scale_cols, block_size)
    scale = _e8m0_scale(blocks.abs().amax(dim=(1, 3)), _FP8_E4M3_MAX)
    scaled = (blocks / scale.float()[:, None, :, None]).clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX)

    quantized = scaled.reshape(padded.shape)[:rows, :cols].contiguous().to(torch.float8_e4m3fn)
    return quantized, scale


def quantize_mxfp4(weight: Tensor) -> tuple[Tensor, Tensor]:
    """Quantize a 2D expert weight to packed MXFP4 with a `float8_e8m0fnu` block scale.

    Returns the `int8` packing the checkpoint uses, halving the last dim, alongside a
    `[rows, cols / 32]` scale. Values are rounded onto the e2m1 grid and packed
    low-nibble-first, so `_unpack_mxfp4` is the exact inverse.
    """
    if weight.ndim != 2:
        raise ValueError(f"MXFP4 quantization expects a 2D weight, got shape={tuple(weight.shape)}")
    rows, cols = weight.shape
    if cols % _MXFP4_BLOCK:
        raise ValueError(f"MXFP4 input dim {cols} is not a multiple of the {_MXFP4_BLOCK}-wide scale block")

    blocks = weight.float().view(rows, cols // _MXFP4_BLOCK, _MXFP4_BLOCK)
    scale = _e8m0_scale(blocks.abs().amax(dim=-1), _FP4_E2M1_MAX)
    scaled = (blocks / scale.float()[..., None]).clamp(-_FP4_E2M1_MAX, _FP4_E2M1_MAX).reshape(rows, cols)

    nibbles = (_round_to_e2m1(scaled.abs()) | (torch.signbit(scaled).to(torch.int64) << 3)).to(torch.uint8)
    packed = nibbles[:, 0::2] | (nibbles[:, 1::2] << 4)
    return packed.contiguous().view(torch.int8), scale


def dequantize_weight(weight: Tensor, scale: Tensor) -> Tensor:
    """Dequantize one on-disk `(weight, scale)` pair to `bfloat16`.

    Dispatches on `weight.dtype`: `torch.int8` is a packed MXFP4 MoE expert weight
    (unpacked before scaling); `torch.float8_e4m3fn` is a dense fp8 weight. Both apply the
    same per-block scale multiply, with block size derived from the ratio between the
    (unpacked) weight's shape and the scale's shape, since dense layers use 128x128 blocks
    and MoE experts use 1x32 blocks. `scale` is `float8_e8m0fnu`, which decodes correctly via
    a plain `.float()` cast.
    """
    if weight.dtype == torch.int8:
        values = _unpack_mxfp4(weight)
    elif weight.dtype == torch.float8_e4m3fn:
        values = weight.float()
    else:
        raise ValueError(f"Unsupported quantized weight dtype: {weight.dtype}")

    rows, cols = values.shape[-2:]
    scale_rows, scale_cols = scale.shape[-2:]
    if rows % scale_rows or cols % scale_cols:
        raise ValueError(
            f"Weight shape {tuple(values.shape[-2:])} not divisible by scale grid {tuple(scale.shape[-2:])}"
        )
    block_rows, block_cols = rows // scale_rows, cols // scale_cols

    scale_expanded = scale.float().repeat_interleave(block_rows, dim=-2).repeat_interleave(block_cols, dim=-1)
    return (values * scale_expanded).bfloat16()


_FP8_WEIGHTS = re.compile(
    r"^layers\.\d+\.(?:attn\.(?:wq_a|wq_b|wkv|wo_a|wo_b|indexer\.wq_b)|ffn\.shared_experts\.w[123])\.weight$"
)
_MXFP4_WEIGHTS = re.compile(r"^layers\.\d+\.ffn\.experts\.\d+\.w[123]\.weight$")


def quantize_state_dict_(state_dict: StateDict, expert_dtype: str = "fp4") -> None:
    """Quantize every weight the checkpoint stores quantized, in place, adding its `.scale`.

    The inverse of `dequantize_state_dict_`, and like it must run on raw on-disk key names
    (e.g. `layers.0.attn.wq_a.weight`, `layers.0.ffn.experts.0.w1.weight`), which is what
    `convert_to_hf` emits. Keys outside the two patterns are left untouched. Safe on a
    rank-partial state dict: every weight is quantized on its own.
    """
    if expert_dtype == "fp8":
        raise NotImplementedError(
            "expert_dtype='fp8' (DeepSeek-V4-Flash-Base) is not implemented for weight transfer; "
            "emitting MXFP4 into an fp8 expert parameter would corrupt it silently"
        )
    if expert_dtype != "fp4":
        raise ValueError(f"Unsupported DeepSeek V4 expert_dtype={expert_dtype!r}")

    for key in [k for k in state_dict if k.endswith(".weight")]:
        if _FP8_WEIGHTS.match(key):
            weight, scale = quantize_fp8_block(state_dict[key])
        elif _MXFP4_WEIGHTS.match(key):
            weight, scale = quantize_mxfp4(state_dict[key])
        else:
            continue
        state_dict[key] = weight
        state_dict[key.removesuffix(".weight") + ".scale"] = scale


def dequantize_state_dict_(state_dict: StateDict) -> None:
    """Dequantize every on-disk `*.weight` / `*.scale` pair in place, dropping the scale.

    Must run on raw on-disk key names (e.g. `layers.0.attn.wq_a.weight`,
    `layers.0.ffn.experts.0.w1.weight`), before any renaming. Keys with no `.scale` sibling
    (plain `bfloat16`/`float32` params, the `int64` `tid2eid` routing table) have no sibling
    to pop and are left untouched.
    """
    for key in [k for k in state_dict if k.endswith(".weight")]:
        scale_key = key.removesuffix(".weight") + ".scale"
        scale = state_dict.pop(scale_key, None)
        if scale is None:
            continue
        state_dict[key] = dequantize_weight(state_dict[key], scale)
