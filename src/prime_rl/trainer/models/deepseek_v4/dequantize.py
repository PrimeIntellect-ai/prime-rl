"""Dequantize the real DeepSeek V4 Flash checkpoint's on-disk fp8/MXFP4 weights.

The real `deepseek-ai/DeepSeek-V4-Flash-0731` checkpoint ships dense linear layers as
block-quantized FP8 (`float8_e4m3fn` weight + `float8_e8m0fnu` per-block scale, 128x128
blocks) and MoE expert weights as packed MXFP4 (two 4-bit e2m1 values per `int8` byte,
`float8_e8m0fnu` scale, 1x32 blocks after unpacking). Neither prime-rl's model code nor its
DCP loading path can consume either format directly, so this module turns both back into
plain `bfloat16` tensors before the checkpoint's state dict reaches the model.

This is intentionally a one-way, plain preprocessing pass, not a `ConvOp`
(`prime_rl.trainer.models.conversion_ops`): that framework is for two-way *structural*
conversions between per-key HF and PrimeRL names, whereas dequantization merges two source
keys (`*.weight`, `*.scale`) into one output key and is never reversed.
"""

from __future__ import annotations

import torch
from torch import Tensor

from prime_rl.trainer.models.conversion_ops import StateDict

_FP4_E2M1_LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def _unpack_mxfp4(packed: Tensor) -> Tensor:
    """Two packed e2m1 nibbles per `int8` byte -> `float32`, doubling the last dim."""
    lut = _FP4_E2M1_LUT.to(packed.device)
    u8 = packed.contiguous().view(torch.uint8)
    low_nibble = (u8 & 0xF).long()
    high_nibble = ((u8 >> 4) & 0xF).long()
    unpacked = torch.stack([lut[low_nibble], lut[high_nibble]], dim=-1)
    return unpacked.reshape(*packed.shape[:-1], 2 * packed.shape[-1])


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
