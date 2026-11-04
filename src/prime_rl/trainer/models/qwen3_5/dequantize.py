"""Dequantize the compressed-tensors FP8 weights shipped by Qwen3.5-family checkpoints.

Official `Qwen/Qwen3.5-...-FP8` and `Qwen/Qwen3.8-...-FP8` snapshots store every
quantized linear as an on-disk `float8_e4m3fn` `*.weight` plus a `float16`/`bfloat16`
`*.weight_scale_inv` multiplier on a `weight_block_size` (128x128) grid, following the
compressed-tensors FP8 layout (`quantization_config.quant_method == "fp8"`). Neither
prime-rl's model code nor its DCP loading path consumes that layout: the raw fp8
tensors are loaded into `bfloat16` linears without their scales, silently producing a
model with garbage weights (near-uniform logits, mismatch-KL ~13 against the inference
server that serves the same files natively through vLLM).

This is intentionally a one-way, plain preprocessing pass, not a `ConvOp`
(`prime_rl.trainer.models.conversion_ops`): that framework is for two-way *structural*
conversions between per-key HF and PrimeRL names, whereas dequantization merges two
source keys (`*.weight`, `*.weight_scale_inv`) into one output key and is never reversed.
It mirrors `prime_rl.trainer.models.deepseek_v4.dequantize`, which does the same for
DeepSeek-V4's on-disk fp8/MXFP4 layout.
"""

from __future__ import annotations

import torch
from torch import Tensor

from prime_rl.trainer.models.conversion_ops import StateDict


def dequantize_weight(weight: Tensor, scale_inv: Tensor) -> Tensor:
    """Dequantize one on-disk `(weight, weight_scale_inv)` pair to `bfloat16`.

    `weight` is `torch.float8_e4m3fn`; `scale_inv` carries the per-block multiplier in
    `float16`/`bfloat16`, verified against the BF16 release of the same weights
    (multiply semantics: `weight * weight_scale_inv`). The block size is derived from
    the ratio between the weight's and the scale's shapes, since compressed-tensors
    uses 128x128 blocks by default but the derivation stays correct for other grids.
    """
    if weight.dtype != torch.float8_e4m3fn:
        raise ValueError(f"Unsupported quantized weight dtype: {weight.dtype}")

    values = weight.float()
    rows, cols = values.shape[-2:]
    scale_rows, scale_cols = scale_inv.shape[-2:]
    if rows % scale_rows or cols % scale_cols:
        raise ValueError(
            f"Weight shape {tuple(values.shape[-2:])} not divisible by scale grid {tuple(scale_inv.shape[-2:])}"
        )
    block_rows, block_cols = rows // scale_rows, cols // scale_cols

    scale_expanded = scale_inv.float().repeat_interleave(block_rows, dim=-2).repeat_interleave(block_cols, dim=-1)
    return (values * scale_expanded).bfloat16()


def dequantize_state_dict_(state_dict: StateDict) -> None:
    """Dequantize every on-disk `*.weight` / `*.weight_scale_inv` pair in place, dropping the scale.

    Must run on raw on-disk key names (e.g. `model.language_model.layers.0.mlp.up_proj.weight`),
    before any renaming. Keys with no `.weight_scale_inv` sibling (plain `bfloat16`
    parameters such as layernorms, GDN `A_log`/`dt_bias`, or the modules the release
    left unquantized — embeddings, `lm_head`, the vision tower) have no sibling to pop
    and are left untouched, so plain-BF16 snapshots pass through unchanged.
    """
    for key in [k for k in state_dict if k.endswith(".weight")]:
        scale_key = key.removesuffix(".weight") + ".weight_scale_inv"
        scale = state_dict.pop(scale_key, None)
        if scale is None:
            continue
        state_dict[key] = dequantize_weight(state_dict[key], scale)
