"""Fused Triton dequantization for DeepSeek V4 Flash's on-disk fp8/MXFP4 weights.

`dequantize.py`'s `dequantize_weight` is the correctness reference: plain PyTorch, runs on
CPU, and is what the unit tests in `tests/unit/train/models/test_deepseek_v4_cpu.py` check
against. It also runs unmodified as three-to-four separate op dispatches (nibble unpack,
two `repeat_interleave` calls, multiply, cast) per weight tensor, called once per key from
`dequantize_state_dict_`'s Python loop over the checkpoint's 72317 keys.

This module fuses that per-tensor sequence into a single Triton kernel launch: one read of
the packed/fp8 bytes and the block scale, one write of the bf16 result, no intermediate
tensors. It is only imported (from `dequantize.py`, lazily) when the state dict's tensors
are already on a CUDA device — Triton has nothing to run on CPU, so the plain path stays the
only path there, matching the existing CPU test module's module-level "no CUDA" contract.

Kept deliberately as two separate kernels rather than one branchy kernel, mirroring
`dequantize_weight`'s own dtype dispatch: MXFP4 experts always use 1x32 blocks (one scale
per 32 unpacked values, block_rows always 1), so the MXFP4 kernel parallelizes over rows and
32-wide column groups; dense fp8 uses 128x128 blocks, so that kernel tiles both dimensions.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

# Same 16-entry table as `dequantize._FP4_E2M1_LUT`, low nibble first for the low/high split
# a byte packs (`(high << 4) | low`, per `test_dequantize_weight_packed_mxfp4`'s worked
# example). Kept as a plain Python list: baked into the kernel as compile-time constants via
# a chain of `tl.where`, so no separate device tensor/pointer argument is needed.
_FP4_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


@triton.jit
def _e2m1_lookup(nibble):
    """Branchless 16-entry LUT: nibble (0-15) -> e2m1 float value."""
    v0: tl.constexpr = 0.0
    v1: tl.constexpr = 0.5
    v2: tl.constexpr = 1.0
    v3: tl.constexpr = 1.5
    v4: tl.constexpr = 2.0
    v5: tl.constexpr = 3.0
    v6: tl.constexpr = 4.0
    v7: tl.constexpr = 6.0
    v8: tl.constexpr = -0.0
    v9: tl.constexpr = -0.5
    v10: tl.constexpr = -1.0
    v11: tl.constexpr = -1.5
    v12: tl.constexpr = -2.0
    v13: tl.constexpr = -3.0
    v14: tl.constexpr = -4.0
    v15: tl.constexpr = -6.0
    result = tl.where(nibble == 0, v0, v15)
    result = tl.where(nibble == 1, v1, result)
    result = tl.where(nibble == 2, v2, result)
    result = tl.where(nibble == 3, v3, result)
    result = tl.where(nibble == 4, v4, result)
    result = tl.where(nibble == 5, v5, result)
    result = tl.where(nibble == 6, v6, result)
    result = tl.where(nibble == 7, v7, result)
    result = tl.where(nibble == 8, v8, result)
    result = tl.where(nibble == 9, v9, result)
    result = tl.where(nibble == 10, v10, result)
    result = tl.where(nibble == 11, v11, result)
    result = tl.where(nibble == 12, v12, result)
    result = tl.where(nibble == 13, v13, result)
    result = tl.where(nibble == 14, v14, result)
    return result


@triton.jit
def _dequant_mxfp4_kernel(
    packed_ptr,
    scale_ptr,
    out_ptr,
    scale_cols,
    stride_pb,
    stride_pr,
    stride_pc,
    stride_sb,
    stride_sr,
    stride_sc,
    stride_ob,
    stride_or,
    stride_oc,
    PACKED_BLOCK: tl.constexpr,
):
    """One program per (batch, row, scale block). `PACKED_BLOCK` is half the block's unpacked
    width (one byte packs two output values), so every program covers exactly one scale
    block and the scale load is a single scalar reused across it — whatever that block width
    actually is (32 for real 1x32 MXFP4 blocks, whatever the test vectors use for small
    shapes), never assumed.
    """
    pid_b = tl.program_id(0)
    pid_r = tl.program_id(1)
    pid_g = tl.program_id(2)

    packed_byte_offsets = pid_g * PACKED_BLOCK + tl.arange(0, PACKED_BLOCK)
    packed_base = packed_ptr + pid_b * stride_pb + pid_r * stride_pr
    raw = tl.load(packed_base + packed_byte_offsets * stride_pc).to(tl.uint8)

    low_nibble = (raw & 0xF).to(tl.int32)
    high_nibble = ((raw >> 4) & 0xF).to(tl.int32)
    low_val = _e2m1_lookup(low_nibble)
    high_val = _e2m1_lookup(high_nibble)

    scale_val = tl.load(
        scale_ptr + pid_b * stride_sb + pid_r * stride_sr + pid_g * stride_sc,
        mask=pid_g < scale_cols,
        other=0.0,
    ).to(tl.float32)

    low_out = (low_val * scale_val).to(tl.bfloat16)
    high_out = (high_val * scale_val).to(tl.bfloat16)

    out_base = out_ptr + pid_b * stride_ob + pid_r * stride_or
    tl.store(out_base + (packed_byte_offsets * 2) * stride_oc, low_out)
    tl.store(out_base + (packed_byte_offsets * 2 + 1) * stride_oc, high_out)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=2),
    ],
    key=["rows", "cols"],
)
@triton.jit
def _dequant_fp8_kernel(
    weight_ptr,
    scale_ptr,
    out_ptr,
    rows,
    cols,
    block_rows,
    block_cols,
    stride_wb,
    stride_wr,
    stride_wc,
    stride_sb,
    stride_sr,
    stride_sc,
    stride_ob,
    stride_or,
    stride_oc,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """One program per (batch, BLOCK_M x BLOCK_N output tile). block_rows/block_cols are the
    (row, col) scale-block sizes (128x128 for dense fp8), so every element in a program's
    tile that shares a scale block reads the same scale value.
    """
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    row_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (row_offsets[:, None] < rows) & (col_offsets[None, :] < cols)

    w_base = weight_ptr + pid_b * stride_wb
    w = tl.load(
        w_base + row_offsets[:, None] * stride_wr + col_offsets[None, :] * stride_wc,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    scale_row = row_offsets // block_rows
    scale_col = col_offsets // block_cols
    s_base = scale_ptr + pid_b * stride_sb
    s = tl.load(
        s_base + scale_row[:, None] * stride_sr + scale_col[None, :] * stride_sc,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    out = (w * s).to(tl.bfloat16)
    out_base = out_ptr + pid_b * stride_ob
    tl.store(
        out_base + row_offsets[:, None] * stride_or + col_offsets[None, :] * stride_oc,
        out,
        mask=mask,
    )


def _as_3d(t: Tensor) -> tuple[Tensor, tuple[int, ...]]:
    """View a >=2D tensor as 3D (batch, rows, cols), batch=1 if it was 2D. Returns the view
    and the original leading (non-row/col) shape, to restore on the way out."""
    if t.dim() < 2:
        raise ValueError(f"Expected a >=2D tensor, got shape {tuple(t.shape)}")
    lead = t.shape[:-2]
    rows, cols = t.shape[-2:]
    batch = 1
    for d in lead:
        batch *= d
    return t.reshape(batch, rows, cols), lead


def dequantize_weight_triton(weight: Tensor, scale: Tensor) -> Tensor:
    """CUDA-only fused equivalent of `dequantize.dequantize_weight`.

    Same dispatch, same block-size derivation, same output (`bfloat16`, same shape) as the
    reference. Callers must ensure `weight.is_cuda`; nothing here checks it, since the only
    caller (`dequantize_state_dict_`) already branches on device before importing this module.
    """
    weight3d, lead = _as_3d(weight.contiguous())
    scale3d, _ = _as_3d(scale.contiguous())
    batch, rows, _ = weight3d.shape
    _, scale_rows, scale_cols = scale3d.shape

    if weight.dtype == torch.int8:
        unpacked_cols = weight3d.shape[-1] * 2
        if rows % scale_rows or unpacked_cols % scale_cols:
            raise ValueError(
                f"Weight shape {(rows, unpacked_cols)} not divisible by scale grid {(scale_rows, scale_cols)}"
            )
        if rows != scale_rows:
            raise ValueError(f"MXFP4 dequant expects block_rows == 1 (rows {rows} != scale_rows {scale_rows})")
        block_cols = unpacked_cols // scale_cols
        if block_cols % 2 or (block_cols // 2) & (block_cols // 2 - 1):
            raise ValueError(
                f"MXFP4 dequant needs an even, power-of-2 block width in packed bytes; got block_cols={block_cols}"
            )
        out = torch.empty((batch, rows, unpacked_cols), device=weight.device, dtype=torch.bfloat16)
        grid = (batch, rows, scale_cols)
        _dequant_mxfp4_kernel[grid](
            weight3d,
            scale3d.view(torch.uint8),
            out,
            scale_cols,
            weight3d.stride(0),
            weight3d.stride(1),
            weight3d.stride(2),
            scale3d.stride(0),
            scale3d.stride(1),
            scale3d.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            PACKED_BLOCK=block_cols // 2,
        )
        return out.reshape(*lead, rows, unpacked_cols)

    if weight.dtype == torch.float8_e4m3fn:
        cols = weight3d.shape[-1]
        if rows % scale_rows or cols % scale_cols:
            raise ValueError(f"Weight shape {(rows, cols)} not divisible by scale grid {(scale_rows, scale_cols)}")
        block_rows, block_cols = rows // scale_rows, cols // scale_cols
        out = torch.empty((batch, rows, cols), device=weight.device, dtype=torch.bfloat16)
        grid = lambda meta: (batch, triton.cdiv(rows, meta["BLOCK_M"]), triton.cdiv(cols, meta["BLOCK_N"]))
        _dequant_fp8_kernel[grid](
            weight3d,
            scale3d.view(torch.uint8),
            out,
            rows,
            cols,
            block_rows,
            block_cols,
            weight3d.stride(0),
            weight3d.stride(1),
            weight3d.stride(2),
            scale3d.stride(0),
            scale3d.stride(1),
            scale3d.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
        )
        return out.reshape(*lead, rows, cols)

    raise ValueError(f"Unsupported quantized weight dtype: {weight.dtype}")
