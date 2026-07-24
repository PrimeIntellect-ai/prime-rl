from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

_FP4_MAX = 6.0
_FP8_MAX = 448.0
_GLOBAL_SCALE_DENOMINATOR = _FP4_MAX * _FP8_MAX
_NVFP4_BLOCK_SIZE = 16
_SCALE_ROW_TILE = 128
_SCALE_COL_TILE = 4
_QUANT_BLOCK_K = 256
_AMAX_BLOCK_K = 1024
_METADATA_BLOCK_M = 256


@dataclass(frozen=True)
class NVFP4GroupedTensor:
    """Packed NVFP4 values and their two levels of decode scales."""

    data: torch.Tensor
    block_scales: torch.Tensor
    global_scales: torch.Tensor


@triton.jit
def _build_activation_metadata_kernel(
    offsets_ptr,
    row_groups_ptr,
    scale_rows_ptr,
    ROWS: tl.constexpr,
    GROUP_COUNT: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    rows = tl.program_id(axis=0) * BLOCK_M + tl.arange(0, BLOCK_M)
    valid_rows = rows < ROWS

    selected_group = tl.zeros((BLOCK_M,), dtype=tl.int32)
    selected_start = tl.zeros((BLOCK_M,), dtype=tl.int32)
    selected_padded_start = tl.zeros((BLOCK_M,), dtype=tl.int32)
    group_start = 0
    padded_start = 0
    for group_index in range(GROUP_COUNT):
        group_end = tl.load(offsets_ptr + group_index)
        belongs_to_or_follows_group = rows >= group_start
        selected_group = tl.where(belongs_to_or_follows_group, group_index, selected_group)
        selected_start = tl.where(belongs_to_or_follows_group, group_start, selected_start)
        selected_padded_start = tl.where(
            belongs_to_or_follows_group,
            padded_start,
            selected_padded_start,
        )
        group_size = group_end - group_start
        padded_start += tl.where(group_size > 0, (group_size + 127) // 128 * 128, 0)
        group_start = group_end

    tl.store(row_groups_ptr + rows, selected_group, mask=valid_rows)
    tl.store(
        scale_rows_ptr + rows,
        rows - selected_start + selected_padded_start,
        mask=valid_rows,
    )


def _round_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


def _check_blackwell() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("NVFP4 grouped GEMM requires CUDA")
    capability = torch.cuda.get_device_capability()
    if capability < (10, 0):
        raise RuntimeError(
            f"NVFP4 grouped GEMM requires SM100 or newer, but the current device is SM{capability[0]}{capability[1]}"
        )
    if not hasattr(torch, "float4_e2m1fn_x2") or not hasattr(F, "scaled_grouped_mm"):
        raise RuntimeError("NVFP4 grouped GEMM requires PyTorch 2.11 or newer with scaled_grouped_mm support")


def _check_matrix(matrix: torch.Tensor, name: str) -> None:
    if matrix.device.type != "cuda":
        raise ValueError(f"{name} must be a CUDA tensor")
    if matrix.dtype != torch.bfloat16:
        raise ValueError(f"{name} must have dtype torch.bfloat16")
    if matrix.shape[-1] % _NVFP4_BLOCK_SIZE != 0:
        raise ValueError(f"{name}'s contraction dimension must be divisible by {_NVFP4_BLOCK_SIZE}")


@triton.jit
def _quantize_nvfp4_rows_kernel(
    input_ptr,
    output_ptr,
    scale_ptr,
    global_scale_ptr,
    row_group_ptr,
    scale_row_ptr,
    stride_input_row,
    stride_input_col,
    K: tl.constexpr,
    SCALE_COLS: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(axis=0)
    k_block = tl.program_id(axis=1)
    columns = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
    valid_columns = columns < K

    values = tl.load(
        input_ptr + row * stride_input_row + columns * stride_input_col,
        mask=valid_columns,
        other=0.0,
    ).to(tl.float32)
    values_by_scale_block = tl.reshape(values, (BLOCK_K // 16, 16))
    block_amax = tl.max(tl.abs(values_by_scale_block), axis=1)

    group = tl.load(row_group_ptr + row)
    global_scale = tl.load(global_scale_ptr + group).to(tl.float32)
    block_scale = block_amax / (6.0 * global_scale)
    block_scale = tl.minimum(tl.maximum(block_scale, 0.001953125), 448.0)
    block_scale_e4m3 = block_scale.to(tl.float8e4nv)
    rounded_block_scale = block_scale_e4m3.to(tl.float32)
    expanded_block_scale = tl.reshape(
        tl.broadcast_to(
            tl.reshape(rounded_block_scale, (BLOCK_K // 16, 1)),
            (BLOCK_K // 16, 16),
        ),
        (BLOCK_K,),
    )

    normalized = values / (global_scale * expanded_block_scale)
    magnitude = tl.abs(normalized)

    # E2M1 positive values are {0, .5, 1, 1.5, 2, 3, 4, 6}. The
    # alternating comparisons preserve round-to-nearest-even at midpoints.
    codes = (magnitude > 0.25).to(tl.uint8)
    codes += (magnitude >= 0.75).to(tl.uint8)
    codes += (magnitude > 1.25).to(tl.uint8)
    codes += (magnitude >= 1.75).to(tl.uint8)
    codes += (magnitude > 2.5).to(tl.uint8)
    codes += (magnitude >= 3.5).to(tl.uint8)
    codes += (magnitude > 5.0).to(tl.uint8)
    codes |= (normalized < 0).to(tl.uint8) << 3

    code_pairs = tl.reshape(codes, (BLOCK_K // 2, 2))
    low_codes, high_codes = tl.split(code_pairs)
    packed = low_codes | (high_codes << 4)
    packed_columns = k_block * (BLOCK_K // 2) + tl.arange(0, BLOCK_K // 2)
    tl.store(
        output_ptr + row * (K // 2) + packed_columns,
        packed,
        mask=packed_columns < K // 2,
    )

    scale_columns = k_block * (BLOCK_K // 16) + tl.arange(0, BLOCK_K // 16)
    scale_row = tl.load(scale_row_ptr + row)
    scale_row_in_tile = scale_row % 128
    swizzled_scale_offsets = (
        ((scale_row // 128) * (SCALE_COLS // 4) + scale_columns // 4) * 512
        + (scale_row_in_tile % 32) * 16
        + (scale_row_in_tile // 32) * 4
        + scale_columns % 4
    )
    tl.store(
        scale_ptr + swizzled_scale_offsets,
        block_scale_e4m3,
        mask=scale_columns < K // 16,
    )


@triton.jit
def _group_amax_kernel(
    input_ptr,
    row_group_ptr,
    active_rows_ptr,
    output_ptr,
    stride_input_row,
    stride_input_col,
    K: tl.constexpr,
    HAS_ACTIVE_ROW_LIMIT: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(axis=0)
    k_block = tl.program_id(axis=1)
    columns = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
    values = tl.load(
        input_ptr + row * stride_input_row + columns * stride_input_col,
        mask=columns < K,
        other=0.0,
    ).to(tl.float32)
    tile_amax = tl.max(tl.abs(values))
    group = tl.load(row_group_ptr + row)
    if HAS_ACTIVE_ROW_LIMIT:
        active = row < tl.load(active_rows_ptr)
        tl.atomic_max(output_ptr + group, tile_amax, mask=active)
    else:
        tl.atomic_max(output_ptr + group, tile_amax)


def _compute_global_scales(
    matrix: torch.Tensor,
    row_groups: torch.Tensor,
    group_count: int,
    active_rows: torch.Tensor | None = None,
) -> torch.Tensor:
    group_amax = torch.zeros(group_count, device=matrix.device, dtype=torch.float32)
    if matrix.shape[0] > 0:
        active_rows_ptr = row_groups if active_rows is None else active_rows
        grid = (matrix.shape[0], triton.cdiv(matrix.shape[1], _AMAX_BLOCK_K))
        _group_amax_kernel[grid](
            matrix,
            row_groups,
            active_rows_ptr,
            group_amax,
            matrix.stride(0),
            matrix.stride(1),
            K=matrix.shape[1],
            HAS_ACTIVE_ROW_LIMIT=active_rows is not None,
            BLOCK_K=_AMAX_BLOCK_K,
            num_warps=4,
        )
    return torch.where(
        group_amax > 0,
        group_amax / _GLOBAL_SCALE_DENOMINATOR,
        torch.ones_like(group_amax),
    )


def _quantize_rows(
    matrix: torch.Tensor,
    global_scales: torch.Tensor,
    row_groups: torch.Tensor,
    scale_rows: torch.Tensor,
    padded_scale_rows: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, contraction_size = matrix.shape
    active_rows = row_groups.numel()
    scale_columns = _round_up(contraction_size // _NVFP4_BLOCK_SIZE, _SCALE_COL_TILE)
    packed = torch.empty((rows, contraction_size // 2), device=matrix.device, dtype=torch.uint8)
    scales = torch.ones(
        (padded_scale_rows, scale_columns),
        device=matrix.device,
        dtype=torch.float8_e4m3fn,
    )
    if active_rows > 0:
        grid = (active_rows, triton.cdiv(contraction_size, _QUANT_BLOCK_K))
        _quantize_nvfp4_rows_kernel[grid](
            matrix,
            packed,
            scales,
            global_scales,
            row_groups,
            scale_rows,
            matrix.stride(0),
            matrix.stride(1),
            K=contraction_size,
            SCALE_COLS=scale_columns,
            BLOCK_K=_QUANT_BLOCK_K,
            num_warps=4,
        )
    return packed.view(torch.float4_e2m1fn_x2), scales


def quantize_nvfp4_activations(matrix: torch.Tensor, offsets: torch.Tensor) -> NVFP4GroupedTensor:
    """Quantize a 2D activation matrix independently for each expert group."""

    _check_blackwell()
    if matrix.ndim != 2:
        raise ValueError("matrix must be 2D")
    _check_matrix(matrix, "matrix")
    if not matrix.is_contiguous():
        matrix = matrix.contiguous()
    if offsets.ndim != 1 or offsets.dtype != torch.int32 or offsets.device != matrix.device:
        raise ValueError("offsets must be a 1D int32 tensor on the same CUDA device as matrix")

    group_count = offsets.numel()
    if group_count == 0:
        raise ValueError("offsets must contain at least one group")
    row_groups = torch.empty(matrix.shape[0], device=matrix.device, dtype=torch.int32)
    scale_rows = torch.empty_like(row_groups)
    _build_activation_metadata_kernel[(triton.cdiv(matrix.shape[0], _METADATA_BLOCK_M),)](
        offsets,
        row_groups,
        scale_rows,
        ROWS=matrix.shape[0],
        GROUP_COUNT=group_count,
        BLOCK_M=_METADATA_BLOCK_M,
        num_warps=4,
    )
    global_scales = _compute_global_scales(
        matrix,
        row_groups,
        group_count,
        active_rows=offsets[-1:],
    )

    padded_scale_rows = _round_up(
        matrix.shape[0] + group_count * (_SCALE_ROW_TILE - 1),
        _SCALE_ROW_TILE,
    )

    packed, block_scales = _quantize_rows(
        matrix,
        global_scales,
        row_groups,
        scale_rows,
        padded_scale_rows,
    )
    return NVFP4GroupedTensor(
        data=packed,
        block_scales=block_scales,
        global_scales=global_scales,
    )


def quantize_nvfp4_weights(weight: torch.Tensor) -> NVFP4GroupedTensor:
    """Quantize grouped right-hand weights with logical shape ``[G, K, N]``."""

    _check_blackwell()
    if weight.ndim != 3:
        raise ValueError("weight must be 3D")
    if weight.device.type != "cuda" or weight.dtype != torch.bfloat16:
        raise ValueError("weight must be a CUDA bfloat16 tensor")

    groups, contraction_size, output_size = weight.shape
    if contraction_size % _NVFP4_BLOCK_SIZE != 0:
        raise ValueError(f"weight's contraction dimension must be divisible by {_NVFP4_BLOCK_SIZE}")

    weight_rows = weight.transpose(-2, -1).contiguous()
    flat_row_count = groups * output_size
    flat_rows = torch.arange(flat_row_count, device=weight.device, dtype=torch.int32)
    row_groups = flat_rows // output_size
    global_scales = _compute_global_scales(
        weight_rows.view(flat_row_count, contraction_size),
        row_groups,
        groups,
    )
    padded_output_size = _round_up(output_size, _SCALE_ROW_TILE)
    scale_rows = row_groups * padded_output_size + flat_rows % output_size
    packed, block_scales = _quantize_rows(
        weight_rows.view(flat_row_count, contraction_size),
        global_scales,
        row_groups,
        scale_rows,
        groups * padded_output_size,
    )
    return NVFP4GroupedTensor(
        data=packed.view(groups, output_size, contraction_size // 2),
        block_scales=block_scales.view(groups, -1),
        global_scales=global_scales,
    )


def grouped_nvfp4_mm_quantized(
    activations: NVFP4GroupedTensor,
    weight: NVFP4GroupedTensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    """Run native grouped NVFP4 GEMM on already-packed inputs."""

    return F.scaled_grouped_mm(
        activations.data,
        weight.data.transpose(-2, -1),
        [activations.block_scales, activations.global_scales],
        [F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise],
        [weight.block_scales, weight.global_scales],
        [F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise],
        swizzle_a=F.SwizzleType.SWIZZLE_32_4_4,
        swizzle_b=F.SwizzleType.SWIZZLE_32_4_4,
        offs=offsets,
        output_dtype=torch.bfloat16,
    )


def _grouped_nvfp4_mm_forward(
    matrix: torch.Tensor,
    weight: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    activations_nvfp4 = quantize_nvfp4_activations(matrix, offsets)
    weight_nvfp4 = quantize_nvfp4_weights(weight)
    return grouped_nvfp4_mm_quantized(activations_nvfp4, weight_nvfp4, offsets)


class _GroupedNVFP4MM(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        matrix: torch.Tensor,
        weight: torch.Tensor,
        offsets: torch.Tensor,
    ) -> torch.Tensor:
        output = _grouped_nvfp4_mm_forward(matrix, weight, offsets)
        ctx.save_for_backward(matrix, weight, offsets)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        matrix, weight, offsets = ctx.saved_tensors
        grad_output = grad_output.contiguous().bfloat16()

        grad_matrix = grad_weight = None
        if ctx.needs_input_grad[0]:
            grad_matrix = F.grouped_mm(
                grad_output,
                weight.transpose(-2, -1),
                offs=offsets,
                out_dtype=torch.bfloat16,
            )
        if ctx.needs_input_grad[1]:
            grad_weight = F.grouped_mm(
                matrix.transpose(0, 1),
                grad_output,
                offs=offsets,
                out_dtype=torch.bfloat16,
            )
        return grad_matrix, grad_weight, None


@torch.compiler.disable
def grouped_nvfp4_mm(
    matrix: torch.Tensor,
    weight: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    """Grouped NVFP4 forward with a BF16 straight-through backward.

    ``matrix`` has shape ``[M, K]``, ``weight`` has shape ``[G, K, N]``, and
    ``offsets`` contains the cumulative row count for each of the ``G`` groups.
    """

    if matrix.ndim != 2 or weight.ndim != 3:
        raise ValueError("matrix must be 2D and weight must be 3D")
    if weight.shape[0] != offsets.numel():
        raise ValueError("weight and offsets must have the same number of groups")
    if matrix.shape[1] != weight.shape[1]:
        raise ValueError("matrix and weight contraction dimensions must match")
    _check_matrix(matrix, "matrix")
    if weight.device != matrix.device or weight.dtype != matrix.dtype:
        raise ValueError("weight must have the same CUDA device and dtype as matrix")
    return _GroupedNVFP4MM.apply(matrix, weight, offsets)
