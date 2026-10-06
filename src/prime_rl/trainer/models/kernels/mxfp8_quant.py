"""MXFP8 (e4m3 data, e8m0 scale per 32 elements, rceil) operand preparation for cuDNN's block-scaled grouped GEMMs.

`mxfp8_quantize` reads a `(rows, cols)` operand once and writes both of its quantizations: along
the columns ("row-wise", the GEMM reduces over the columns) and along the rows ("column-wise",
the GEMM reduces over the rows, as a weight gradient does over tokens). Both keep the source
layout; cuDNN's kernels read the column-wise copy as an MN-major operand.

Scales are written straight into the 128x4-tile swizzle the tensor cores read: the row-wise
scales as one `(rows, cols / 32)` matrix, the column-wise scales as one `(cols, group_rows / 32)`
matrix per row group, stacked group after group, which is what cuDNN's grouped kernels index per
expert. Row groups end at `offsets` and must be multiples of 128 rows.

cuDNN's GLU kernels want the gate and up rows of the first-layer weight interleaved in 32-row
blocks. `mxfp8_quantize` can gather its source rows that way from the two separate tensors, and
`mxfp8_split_interleaved_columns` takes a column-wise quantized activation gradient in that
interleaved order apart again, so each weight gradient comes out in its parameter's layout.
"""

import torch
import triton
import triton.language as tl

INT32_MAX = tl.constexpr(2147483647)
# One warp per 32 x BLOCK_COLS tile: the column-wise reduction then stays within each thread.
BLOCK_COLS = 128


@triton.jit
def _e8m0_rceil(amax):
    """Biased e8m0 exponent of `2^ceil(log2(amax / 448))` and the fp32 reciprocal of that scale."""
    bits = (amax / 448.0).to(tl.int32, bitcast=True)
    biased = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    biased = tl.minimum(tl.maximum(biased, 1), 254)
    inverse = ((254 - biased) << 23).to(tl.float32, bitcast=True)
    return biased, inverse


@triton.jit
def _swizzled_offset(row, col, num_cols):
    """Byte offset of scale `(row, col)` in a 128x4-tile swizzled `(rows, num_cols)` scale matrix."""
    tile = (row // 128) * (num_cols // 4) + col // 4
    return tile * 512 + (row % 32) * 16 + ((row // 32) % 4) * 4 + col % 4


@triton.jit
def _row_group(offsets_ptr, num_groups, row, BLOCK_G: tl.constexpr):
    """`(start, end)` of the row group holding `row`; `end` is INT32_MAX past the last group."""
    ends = tl.load(offsets_ptr + tl.arange(0, BLOCK_G), mask=tl.arange(0, BLOCK_G) < num_groups, other=INT32_MAX)
    start = tl.max(tl.where(ends <= row, ends, 0))
    end = tl.min(tl.where(ends > row, ends, INT32_MAX))
    return start, end


@triton.jit
def _mxfp8_quantize_kernel(
    gate_ptr,
    up_ptr,
    row_data_ptr,
    row_scale_ptr,
    col_data_ptr,
    col_scale_ptr,
    offsets_ptr,
    num_groups,
    num_cols,
    source_group_rows,
    source_group_stride,
    source_row_stride,
    INTERLEAVE: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_G: tl.constexpr,
):
    # A (32 rows, BLOCK_C / 32 scale blocks, 32 columns) tile: both reductions are along an axis.
    row_start = tl.program_id(0) * 32
    col_start = tl.program_id(1) * BLOCK_C
    rows = row_start + tl.arange(0, 32)
    blocks = tl.arange(0, BLOCK_C // 32)
    cols = col_start + blocks[:, None] * 32 + tl.arange(0, 32)[None, :]

    source_row = rows % source_group_rows
    if INTERLEAVE:
        source_row = (source_row // 64) * 32 + source_row % 32
    source_offset = (rows // source_group_rows).to(tl.int64) * source_group_stride
    source_offset += source_row.to(tl.int64) * source_row_stride
    source_offset = source_offset[:, None, None] + cols[None, :, :]
    if INTERLEAVE:
        # Interleaved rows alternate 32 of `gate` and 32 of `up`.
        if row_start % 64 >= 32:
            x = tl.load(up_ptr + source_offset).to(tl.float32)
        else:
            x = tl.load(gate_ptr + source_offset).to(tl.float32)
    else:
        x = tl.load(gate_ptr + source_offset).to(tl.float32)
    abs_x = tl.abs(x)
    data_offset = rows.to(tl.int64)[:, None, None] * num_cols + cols[None, :, :]

    biased, inverse = _e8m0_rceil(tl.max(abs_x, axis=2))
    q = tl.clamp(x * inverse[:, :, None], -448.0, 448.0)
    tl.store(row_data_ptr + data_offset, q.to(tl.float8e4nv))
    scale_cols = col_start // 32 + blocks
    tl.store(row_scale_ptr + _swizzled_offset(rows[:, None], scale_cols[None, :], num_cols // 32), biased.to(tl.uint8))

    biased, inverse = _e8m0_rceil(tl.max(abs_x, axis=0))
    q = tl.clamp(x * inverse[None, :, :], -448.0, 448.0)
    tl.store(col_data_ptr + data_offset, q.to(tl.float8e4nv))
    group_start, group_end = _row_group(offsets_ptr, num_groups, row_start, BLOCK_G)
    # Rows past the last group (dispatch padding) have no column-wise scales.
    if group_end != INT32_MAX:
        offset = (group_start // 32).to(tl.int64) * num_cols
        offset += _swizzled_offset(cols, (row_start - group_start) // 32, (group_end - group_start) // 32)
        tl.store(col_scale_ptr + offset, biased.to(tl.uint8))


def mxfp8_quantize(
    x: torch.Tensor,
    offsets: torch.Tensor,
    *,
    interleave_with: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Row-wise and column-wise MXFP8 quantization of `x`.

    `x` is `(rows, cols)` or `(groups, group_rows, cols)` with unit column stride. With
    `interleave_with`, the source is `x` and `interleave_with` (both `(groups, n, cols)`)
    interleaved in 32-row blocks into `(groups, 2n, cols)`. `offsets` holds the int32 end row of
    every column-wise scale group.

    Returns `(row_data, row_scales, col_data, col_scales)`: fp8 data shaped `(rows, cols)` and
    flat e8m0 scales.
    """
    assert x.stride(-1) == 1 and x.shape[-1] % BLOCK_COLS == 0
    if x.dim() == 2:
        x = x.unsqueeze(0)
    groups, source_rows, cols = x.shape
    if interleave_with is not None:
        assert interleave_with.shape == x.shape and interleave_with.stride() == x.stride()
        assert source_rows % 32 == 0
        group_rows = 2 * source_rows
    else:
        group_rows = source_rows
    rows = groups * group_rows
    assert rows % 128 == 0

    row_data = torch.empty(rows, cols, dtype=torch.float8_e4m3fn, device=x.device)
    col_data = torch.empty_like(row_data)
    row_scales = torch.empty(rows * cols // 32, dtype=torch.uint8, device=x.device)
    col_scales = torch.empty_like(row_scales)
    _mxfp8_quantize_kernel[(rows // 32, cols // BLOCK_COLS)](
        x,
        interleave_with if interleave_with is not None else x,
        row_data,
        row_scales,
        col_data,
        col_scales,
        offsets,
        offsets.numel(),
        cols,
        group_rows,
        x.stride(0),
        x.stride(1),
        INTERLEAVE=interleave_with is not None,
        BLOCK_C=BLOCK_COLS,
        BLOCK_G=triton.next_power_of_2(offsets.numel()),
        num_warps=1,
    )
    return row_data, row_scales.view(torch.float8_e8m0fnu), col_data, col_scales.view(torch.float8_e8m0fnu)


@triton.jit
def _split_interleaved_columns_kernel(
    data_ptr,
    scale_ptr,
    gate_data_ptr,
    gate_scale_ptr,
    up_data_ptr,
    up_scale_ptr,
    offsets_ptr,
    num_groups,
    half_cols,
    BLOCK_C: tl.constexpr,
    BLOCK_G: tl.constexpr,
):
    row_start = tl.program_id(0) * 32
    group_start, group_end = _row_group(offsets_ptr, num_groups, row_start, BLOCK_G)
    if group_end == INT32_MAX:
        return
    col_start = tl.program_id(1) * BLOCK_C
    rows = row_start + tl.arange(0, 32)
    blocks = col_start // 32 + tl.arange(0, BLOCK_C // 32)
    lanes = tl.arange(0, 32)
    cols = blocks[:, None] * 32 + lanes[None, :]
    # Interleaved 32-column block `b` is block `b // 2` of the gate (even) or up (odd) half.
    half_col = (blocks[:, None] // 2) * 32 + lanes[None, :]
    is_gate = (blocks % 2 == 0)[:, None]

    row_offset = rows.to(tl.int64)[:, None, None]
    data = tl.load(data_ptr + row_offset * (2 * half_cols) + cols[None, :, :])
    half_offset = row_offset * half_cols + half_col[None, :, :]
    tl.store(gate_data_ptr + half_offset, data, mask=is_gate[None, :, :])
    tl.store(up_data_ptr + half_offset, data, mask=~is_gate[None, :, :])

    scale_col = (row_start - group_start) // 32
    group_scale_cols = (group_end - group_start) // 32
    scales = tl.load(
        scale_ptr
        + (group_start // 32).to(tl.int64) * (2 * half_cols)
        + _swizzled_offset(cols, scale_col, group_scale_cols)
    )
    half_offset = (group_start // 32).to(tl.int64) * half_cols + _swizzled_offset(half_col, scale_col, group_scale_cols)
    tl.store(gate_scale_ptr + half_offset, scales, mask=is_gate)
    tl.store(up_scale_ptr + half_offset, scales, mask=~is_gate)


def mxfp8_split_interleaved_columns(
    data: torch.Tensor, scales: torch.Tensor, offsets: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split a column-wise quantized `(rows, 2n)` operand with 32-column gate/up interleaving.

    `scales` are per row group as `mxfp8_quantize` writes them. Returns `(gate_data, gate_scales,
    up_data, up_scales)` for the two `(rows, n)` halves; rows past the last group are left unset.
    """
    rows, cols = data.shape
    assert cols % BLOCK_COLS == 0
    half_cols = cols // 2
    gate_data = data.new_empty(rows, half_cols)
    up_data = torch.empty_like(gate_data)
    gate_scales = scales.new_empty(scales.numel() // 2)
    up_scales = torch.empty_like(gate_scales)
    _split_interleaved_columns_kernel[(rows // 32, cols // BLOCK_COLS)](
        data.view(torch.uint8),
        scales.view(torch.uint8),
        gate_data.view(torch.uint8),
        gate_scales.view(torch.uint8),
        up_data.view(torch.uint8),
        up_scales.view(torch.uint8),
        offsets,
        offsets.numel(),
        half_cols,
        BLOCK_C=BLOCK_COLS,
        BLOCK_G=triton.next_power_of_2(offsets.numel()),
        num_warps=1,
    )
    return gate_data, gate_scales, up_data, up_scales
