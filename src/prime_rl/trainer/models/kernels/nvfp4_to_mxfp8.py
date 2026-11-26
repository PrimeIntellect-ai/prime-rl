"""NVFP4 operands requantized to MXFP8 for the gradient GEMMs, straight from their packed form."""

import torch
import triton
import triton.language as tl

_BLOCK_COLS = 64


@triton.jit
def _nvfp4_to_mxfp8_kernel(
    packed_ptr,
    block_scale_ptr,
    row_scale_ptr,
    data_ptr,
    scale_ptr,
    rows,
    cols,
    packed_batch_stride,
    block_scale_batch_stride,
    row_scale_batch_stride,
    row_scale_row_stride,
    BLOCK_COLS: tl.constexpr,
):
    row_block = tl.program_id(0)
    batch = tl.program_id(2).to(tl.int64)
    r = row_block.to(tl.int64) * 32 + tl.arange(0, 32)
    c = tl.program_id(1).to(tl.int64) * BLOCK_COLS + tl.arange(0, BLOCK_COLS)

    byte = tl.load(packed_ptr + batch * packed_batch_stride + r[:, None] * (cols // 2) + c[None, :] // 2)
    nibble = (byte.to(tl.int32) >> ((c[None, :] & 1) * 4).to(tl.int32)) & 0xF
    exponent = (nibble >> 1) & 3
    mantissa = (nibble & 1).to(tl.float32)
    power = tl.where(exponent == 1, 1.0, tl.where(exponent == 2, 2.0, 4.0))
    magnitude = tl.where(exponent == 0, mantissa * 0.5, (1.0 + mantissa * 0.5) * power)
    value = tl.where((nibble & 8) != 0, -magnitude, magnitude)
    block_scale = tl.load(
        block_scale_ptr + batch * block_scale_batch_stride + r[:, None] * (cols // 16) + c[None, :] // 16
    ).to(tl.float32)
    row_scale = tl.load(row_scale_ptr + batch * row_scale_batch_stride + r * row_scale_row_stride)
    x = value * block_scale * row_scale[:, None]

    # torchao's RCEIL scale: 2^ceil(log2(amax / 448)) as a biased E8M0 exponent (0 for zero or subnormal amax / 448)
    descale = tl.math.div_rn(tl.max(tl.abs(x), axis=0), 448.0)
    bits = descale.to(tl.int32, bitcast=True)
    exponent_field = (bits >> 23) & 0xFF
    ceil_log2 = exponent_field - 127 + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    biased = tl.where(exponent_field == 0, 0, tl.minimum(ceil_log2, 127) + 127)
    reciprocal = tl.where(biased == 0, 1.0, ((254 - biased) << 23).to(tl.float32, bitcast=True))
    scaled = tl.minimum(tl.maximum(x * reciprocal[None, :], -448.0), 448.0)
    data = scaled.to(tl.float8e4nv, fp_downcast_rounding="rtne")

    tl.store(data_ptr + batch * rows * cols + c[None, :] * rows + r[:, None], data)
    tl.store(scale_ptr + batch * cols * (rows // 32) + c * (rows // 32) + row_block, biased.to(tl.uint8))


def nvfp4_to_mxfp8_columns(
    packed: torch.Tensor, block_scales: torch.Tensor, row_scales: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dequantize ``[B, R, C]`` NVFP4 (packed E2M1 ``[B, R, C // 2]`` low nibble first, E4M3 block scales
    ``[B, R, C // 16]``, FP32 row scales ``[B, R]``) in FP32 and quantize it to MXFP8 along ``R``, as torchao's RCEIL
    recipe does (E8M0 scale ``2^ceil(log2(amax / 448))`` per 32 values, E4M3 round to nearest even). Returns the
    transposed E4M3 values ``[B, C, R]`` and E8M0 scales ``[B, C, R // 32]``."""
    batch, rows, half_cols = packed.shape
    cols = half_cols * 2
    if rows % 32 or cols % _BLOCK_COLS:
        raise ValueError(f"NVFP4 to MXFP8 needs rows divisible by 32 and columns by {_BLOCK_COLS}, got {rows}x{cols}")
    packed = packed.view(torch.uint8).contiguous()
    block_scales = block_scales.view(torch.float8_e4m3fn).reshape(batch, rows, cols // 16).contiguous()
    data = torch.empty(batch, cols, rows, dtype=torch.float8_e4m3fn, device=packed.device)
    scales = torch.empty(batch, cols, rows // 32, dtype=torch.uint8, device=packed.device)
    _nvfp4_to_mxfp8_kernel[(rows // 32, cols // _BLOCK_COLS, batch)](
        packed,
        block_scales,
        row_scales,
        data,
        scales,
        rows,
        cols,
        packed.stride(0),
        block_scales.stride(0),
        row_scales.stride(0),
        row_scales.stride(1),
        BLOCK_COLS=_BLOCK_COLS,
    )
    return data, scales.view(torch.float8_e8m0fnu)
