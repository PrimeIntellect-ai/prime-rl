from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import deep_gemm
import torch
import torch.nn as nn
from cutlass.cute.runtime import make_ptr


@cute.kernel
def _activation_quant_1x128_kernel(
    x: cute.Tensor,
    y: cute.Tensor,
    scales: cute.Tensor,
    tiled_copy: cute.TiledCopy,
    s_x_layout: cute.Layout,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()

    row_tiles = cute.ceil_div(x.shape[0], 4)
    col_tiles = cute.ceil_div(x.shape[1], 128)

    if bidx < row_tiles and bidy < col_tiles:
        tiler_coord = (bidx, bidy)
        g_x = cute.local_tile(x, tiler=(4, 128), coord=tiler_coord)
        g_y = cute.local_tile(y, tiler=(4, 128), coord=tiler_coord)

        smem = cutlass.utils.SmemAllocator()
        s_x = smem.allocate_tensor(x.element_type, s_x_layout, 16)

        thr_copy = tiled_copy.get_slice(tidx)
        t_xg_x = thr_copy.partition_S(g_x)
        t_xs_x = thr_copy.partition_D(s_x)
        t_yg_y = thr_copy.partition_D(g_y)

        cute.copy(tiled_copy, t_xg_x[None, None, 0], t_xs_x[None, None, 0])
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()

        thread_vals = t_xs_x[(None, None), 0, 0].load().to(cutlass.Float32)
        max_per_thread = cutlass.Float32(0.0)
        for vid in cutlass.range_constexpr(4):
            v = thread_vals[vid]
            abs_v = v
            nv = -v
            if nv > abs_v:
                abs_v = nv
            if abs_v > max_per_thread:
                max_per_thread = abs_v

        max_per_warp = cute.arch.warp_reduction_max(max_per_thread, threads_in_group=32)
        max_per_warp_scalar = max_per_warp

        lane_id = tidx % 32
        warp_id = tidx // 32
        row_idx = bidx * 4 + warp_id

        fp8_max = cutlass.Float32(448.0)
        eps = cutlass.Float32(1e-4)
        safe_max = max_per_warp_scalar
        if safe_max < eps:
            safe_max = eps
        scale = safe_max / fp8_max
        inv_scale = fp8_max / safe_max

        if lane_id == 0 and row_idx < x.shape[0]:
            scales[row_idx, bidy] = scale

        out_vals = thread_vals * inv_scale
        t_yg_y[(None, None), 0, 0].store(out_vals.to(y.element_type))


@cute.jit
def _activation_quant_1x128_op(x: cute.Tensor, y: cute.Tensor, scales: cute.Tensor):
    copy_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        x.element_type,
        num_bits_per_copy=x.element_type.width * 4,
    )
    major_mode_size = 32
    t_a = cute.make_layout((4, major_mode_size), stride=(major_mode_size, 1))
    v_a = cute.make_layout((1, 4), stride=(4, 128))
    tiled_copy = cute.make_tiled_copy_tv(copy_atom, t_a, v_a)

    s_x_layout = cute.make_layout((4, 128), stride=(128, 1))
    block = (cute.size(t_a), 1, 1)
    grid = (*cute.ceil_div(x.shape, (4, 128)), 1)
    smem_size = cute.size_in_bytes(x.element_type, s_x_layout)

    _activation_quant_1x128_kernel(x, y, scales, tiled_copy, s_x_layout).launch(grid=grid, block=block, smem=smem_size)


@cute.kernel
def _weight_quant_128x128_kernel(
    x: cute.Tensor,
    y: cute.Tensor,
    scales: cute.Tensor,
    tiled_copy: cute.TiledCopy,
    s_x_layout: cute.Layout,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()

    warp_id = tidx // 32
    lane_id = tidx % 32
    row_block = bidx + warp_id
    col_block = bidy

    row_blocks = cute.ceil_div(x.shape[0], 128)
    col_blocks = cute.ceil_div(x.shape[1], 128)

    if row_block < row_blocks and col_block < col_blocks:
        thread_absmax = cutlass.Float32(0.0)

        smem = cutlass.utils.SmemAllocator()
        s_x = smem.allocate_tensor(x.element_type, s_x_layout, 16)
        s_x_warp = s_x[warp_id, None, None]

        tiler_coord = (row_block, col_block)
        g_x = cute.local_tile(x, tiler=(128, 128), coord=tiler_coord)
        g_y = cute.local_tile(y, tiler=(128, 128), coord=tiler_coord)

        lane_map = (lane_id % 16) * 2 + (lane_id // 16)
        thr_copy = tiled_copy.get_slice(lane_map)

        for col_group in cutlass.range_constexpr(4):
            for row_vec in cutlass.range_constexpr(16):
                g_x_sub = cute.local_tile(g_x, tiler=(8, 32), coord=(row_vec, col_group))
                s_x_sub = cute.local_tile(s_x_warp, tiler=(8, 32), coord=(row_vec, col_group))
                t_xg_x = thr_copy.partition_S(g_x_sub)
                t_xs_x = thr_copy.partition_D(s_x_sub)
                vals = t_xg_x[(None, None), 0, 0].load().to(cutlass.Float32)
                t_xs_x[(None, None), 0, 0].store(vals.to(x.element_type))
                for vid in cutlass.range_constexpr(8):
                    v = vals[vid]
                    abs_v = v
                    nv = -v
                    if nv > abs_v:
                        abs_v = nv
                    if abs_v > thread_absmax:
                        thread_absmax = abs_v

        block_absmax = cute.arch.warp_reduction_max(thread_absmax, threads_in_group=32)
        fp8_max = cutlass.Float32(448.0)
        eps = cutlass.Float32(1e-6)
        safe_absmax = block_absmax
        if safe_absmax < eps:
            safe_absmax = eps
        scale = safe_absmax / fp8_max

        if lane_id == 0:
            scales[row_block, col_block] = scale

        inv_scale = fp8_max / safe_absmax
        for col_group in cutlass.range_constexpr(4):
            for row_vec in cutlass.range_constexpr(16):
                g_y_sub = cute.local_tile(g_y, tiler=(8, 32), coord=(row_vec, col_group))
                s_x_sub = cute.local_tile(s_x_warp, tiler=(8, 32), coord=(row_vec, col_group))
                t_xs_x = thr_copy.partition_S(s_x_sub)
                t_yg_y = thr_copy.partition_D(g_y_sub)
                vals = t_xs_x[(None, None), 0, 0].load().to(cutlass.Float32)
                t_yg_y[(None, None), 0, 0].store((vals * inv_scale).to(y.element_type))


@cute.jit
def _weight_quant_128x128_op(x: cute.Tensor, y: cute.Tensor, scales: cute.Tensor):
    copy_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        x.element_type,
        num_bits_per_copy=x.element_type.width * 8,
    )
    t_a = cute.make_layout((1, 32), stride=(0, 1))
    v_a = cute.make_layout((8,), stride=(1,))
    tiled_copy = cute.make_tiled_copy_tv(copy_atom, t_a, v_a)

    s_x_layout = cute.make_layout((1, 128, 128), stride=(128 * 128, 1, 128))
    row_blocks = cute.ceil_div(x.shape[0], 128)
    col_blocks = cute.ceil_div(x.shape[1], 128)

    grid = (row_blocks, col_blocks, 1)
    block = (32, 1, 1)
    smem_size = cute.size_in_bytes(x.element_type, s_x_layout)
    _weight_quant_128x128_kernel(x, y, scales, tiled_copy, s_x_layout).launch(grid=grid, block=block, smem=smem_size)


@cute.kernel
def _weight_quant_dual_128x128_kernel(
    x: cute.Tensor,
    y_col: cute.Tensor,
    scales_col: cute.Tensor,
    y_row: cute.Tensor,
    scales_row: cute.Tensor,
    tiled_copy: cute.TiledCopy,
    s_x_layout: cute.Layout,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()

    warp_id = tidx // 32
    lane_id = tidx % 32
    row_block = bidx + warp_id
    col_block = bidy

    row_blocks = cute.ceil_div(x.shape[0], 128)
    col_blocks = cute.ceil_div(x.shape[1], 128)

    if row_block < row_blocks and col_block < col_blocks:
        thread_absmax = cutlass.Float32(0.0)

        smem = cutlass.utils.SmemAllocator()
        s_x = smem.allocate_tensor(x.element_type, s_x_layout, 16)
        s_x_warp = s_x[warp_id, None, None]

        tiler_coord = (row_block, col_block)
        g_x = cute.local_tile(x, tiler=(128, 128), coord=tiler_coord)
        g_y_col = cute.local_tile(y_col, tiler=(128, 128), coord=tiler_coord)
        g_y_row = cute.local_tile(y_row, tiler=(128, 128), coord=tiler_coord)

        lane_map = (lane_id % 16) * 2 + (lane_id // 16)
        thr_copy = tiled_copy.get_slice(lane_map)

        for col_group in cutlass.range_constexpr(4):
            for row_vec in cutlass.range_constexpr(16):
                g_x_sub = cute.local_tile(g_x, tiler=(8, 32), coord=(row_vec, col_group))
                s_x_sub = cute.local_tile(s_x_warp, tiler=(8, 32), coord=(row_vec, col_group))
                t_xg_x = thr_copy.partition_S(g_x_sub)
                t_xs_x = thr_copy.partition_D(s_x_sub)
                vals = t_xg_x[(None, None), 0, 0].load().to(cutlass.Float32)
                t_xs_x[(None, None), 0, 0].store(vals.to(x.element_type))
                for vid in cutlass.range_constexpr(8):
                    v = vals[vid]
                    abs_v = v
                    nv = -v
                    if nv > abs_v:
                        abs_v = nv
                    if abs_v > thread_absmax:
                        thread_absmax = abs_v

        block_absmax = cute.arch.warp_reduction_max(thread_absmax, threads_in_group=32)
        fp8_max = cutlass.Float32(448.0)
        eps = cutlass.Float32(1e-6)
        safe_absmax = block_absmax
        if safe_absmax < eps:
            safe_absmax = eps
        scale = safe_absmax / fp8_max

        if lane_id == 0:
            scales_col[row_block, col_block] = scale
            scales_row[row_block, col_block] = scale

        inv_scale = fp8_max / safe_absmax
        for col_group in cutlass.range_constexpr(4):
            for row_vec in cutlass.range_constexpr(16):
                g_y_col_sub = cute.local_tile(g_y_col, tiler=(8, 32), coord=(row_vec, col_group))
                g_y_row_sub = cute.local_tile(g_y_row, tiler=(8, 32), coord=(row_vec, col_group))
                s_x_sub = cute.local_tile(s_x_warp, tiler=(8, 32), coord=(row_vec, col_group))
                t_xs_x = thr_copy.partition_S(s_x_sub)
                t_yg_y_col = thr_copy.partition_D(g_y_col_sub)
                t_yg_y_row = thr_copy.partition_D(g_y_row_sub)
                vals = t_xs_x[(None, None), 0, 0].load().to(cutlass.Float32)
                out_vals = (vals * inv_scale).to(y_col.element_type)
                t_yg_y_col[(None, None), 0, 0].store(out_vals)
                t_yg_y_row[(None, None), 0, 0].store(out_vals)


@cute.jit
def _weight_quant_dual_128x128_op(
    x: cute.Tensor,
    y_col: cute.Tensor,
    scales_col: cute.Tensor,
    y_row: cute.Tensor,
    scales_row: cute.Tensor,
):
    copy_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        x.element_type,
        num_bits_per_copy=x.element_type.width * 8,
    )
    t_a = cute.make_layout((1, 32), stride=(0, 1))
    v_a = cute.make_layout((8,), stride=(1,))
    tiled_copy = cute.make_tiled_copy_tv(copy_atom, t_a, v_a)

    s_x_layout = cute.make_layout((1, 128, 128), stride=(128 * 128, 1, 128))
    row_blocks = cute.ceil_div(x.shape[0], 128)
    col_blocks = cute.ceil_div(x.shape[1], 128)

    grid = (row_blocks, col_blocks, 1)
    block = (32, 1, 1)
    smem_size = cute.size_in_bytes(x.element_type, s_x_layout)
    _weight_quant_dual_128x128_kernel(
        x,
        y_col,
        scales_col,
        y_row,
        scales_row,
        tiled_copy,
        s_x_layout,
    ).launch(grid=grid, block=block, smem=smem_size)


_ACTIVATION_QUANT_1X128_COMPILED = None
_WEIGHT_QUANT_128X128_COMPILED = None
_WEIGHT_QUANT_DUAL_128X128_COMPILED = None


@cute.jit
def _activation_quant_1x128_entry(
    x_ptr: cute.Pointer,
    y_ptr: cute.Pointer,
    scales_ptr: cute.Pointer,
    m: cutlass.Int32,
    k: cutlass.Int32,
):
    m = cute.assume(m, divby=128)
    k = cute.assume(k, divby=128)
    x = cute.make_tensor(x_ptr, cute.make_layout((m, k), stride=(k, 1)))
    y = cute.make_tensor(y_ptr, cute.make_layout((m, k), stride=(k, 1)))
    scales = cute.make_tensor(
        scales_ptr,
        cute.make_layout((m, k // 128), stride=(k // 128, 1)),
    )
    _activation_quant_1x128_op(x, y, scales)


@cute.jit
def _weight_quant_128x128_entry(
    x_ptr: cute.Pointer,
    y_ptr: cute.Pointer,
    scales_ptr: cute.Pointer,
    k: cutlass.Int32,
    n: cutlass.Int32,
):
    k = cute.assume(k, divby=128)
    n = cute.assume(n, divby=128)
    x = cute.make_tensor(x_ptr, cute.make_layout((k, n), stride=(1, k)))
    y = cute.make_tensor(y_ptr, cute.make_layout((k, n), stride=(1, k)))
    scales = cute.make_tensor(
        scales_ptr,
        cute.make_layout((k // 128, n // 128), stride=(1, k // 128)),
    )
    _weight_quant_128x128_op(x, y, scales)


@cute.jit
def _weight_quant_dual_128x128_entry(
    x_ptr: cute.Pointer,
    y_col_ptr: cute.Pointer,
    scales_col_ptr: cute.Pointer,
    y_row_ptr: cute.Pointer,
    scales_row_ptr: cute.Pointer,
    k: cutlass.Int32,
    n: cutlass.Int32,
):
    k = cute.assume(k, divby=128)
    n = cute.assume(n, divby=128)
    x = cute.make_tensor(x_ptr, cute.make_layout((k, n), stride=(1, k)))
    y_col = cute.make_tensor(y_col_ptr, cute.make_layout((k, n), stride=(1, k)))
    scales_col = cute.make_tensor(
        scales_col_ptr,
        cute.make_layout((k // 128, n // 128), stride=(1, k // 128)),
    )
    y_row = cute.make_tensor(y_row_ptr, cute.make_layout((k, n), stride=(n, 1)))
    scales_row = cute.make_tensor(
        scales_row_ptr,
        cute.make_layout((k // 128, n // 128), stride=(n // 128, 1)),
    )
    _weight_quant_dual_128x128_op(x, y_col, scales_col, y_row, scales_row)


def compile_activation_quant_1x128():
    global _ACTIVATION_QUANT_1X128_COMPILED
    if _ACTIVATION_QUANT_1X128_COMPILED is None:
        x_ptr = make_ptr(cutlass.BFloat16, 0, cute.AddressSpace.gmem, assumed_align=16)
        y_ptr = make_ptr(cutlass.Float8E4M3FN, 0, cute.AddressSpace.gmem, assumed_align=16)
        scales_ptr = make_ptr(cutlass.Float32, 0, cute.AddressSpace.gmem, assumed_align=16)
        _ACTIVATION_QUANT_1X128_COMPILED = cute.compile(
            _activation_quant_1x128_entry,
            x_ptr,
            y_ptr,
            scales_ptr,
            0,
            0,
        )
    return _ACTIVATION_QUANT_1X128_COMPILED


def compile_weight_quant_128x128():
    global _WEIGHT_QUANT_128X128_COMPILED
    if _WEIGHT_QUANT_128X128_COMPILED is None:
        x_ptr = make_ptr(cutlass.BFloat16, 0, cute.AddressSpace.gmem, assumed_align=16)
        y_ptr = make_ptr(cutlass.Float8E4M3FN, 0, cute.AddressSpace.gmem, assumed_align=16)
        scales_ptr = make_ptr(cutlass.Float32, 0, cute.AddressSpace.gmem, assumed_align=16)
        _WEIGHT_QUANT_128X128_COMPILED = cute.compile(
            _weight_quant_128x128_entry,
            x_ptr,
            y_ptr,
            scales_ptr,
            0,
            0,
        )
    return _WEIGHT_QUANT_128X128_COMPILED


def compile_weight_quant_dual_128x128():
    global _WEIGHT_QUANT_DUAL_128X128_COMPILED
    if _WEIGHT_QUANT_DUAL_128X128_COMPILED is None:
        x_ptr = make_ptr(cutlass.BFloat16, 0, cute.AddressSpace.gmem, assumed_align=16)
        y_col_ptr = make_ptr(cutlass.Float8E4M3FN, 0, cute.AddressSpace.gmem, assumed_align=16)
        scales_col_ptr = make_ptr(cutlass.Float32, 0, cute.AddressSpace.gmem, assumed_align=16)
        y_row_ptr = make_ptr(cutlass.Float8E4M3FN, 0, cute.AddressSpace.gmem, assumed_align=16)
        scales_row_ptr = make_ptr(cutlass.Float32, 0, cute.AddressSpace.gmem, assumed_align=16)
        _WEIGHT_QUANT_DUAL_128X128_COMPILED = cute.compile(
            _weight_quant_dual_128x128_entry,
            x_ptr,
            y_col_ptr,
            scales_col_ptr,
            y_row_ptr,
            scales_row_ptr,
            0,
            0,
        )
    return _WEIGHT_QUANT_DUAL_128X128_COMPILED


def activation_quant_1x128(x: torch.Tensor, y: torch.Tensor, scales: torch.Tensor) -> None:
    m, k = x.shape
    assert m % 4 == 0
    assert k % 128 == 0

    compiled = compile_activation_quant_1x128()
    compiled(
        make_ptr(cutlass.BFloat16, x.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Float8E4M3FN, y.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Float32, scales.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
        m,
        k,
    )


def weight_quant_128x128(x: torch.Tensor, y: torch.Tensor, scales: torch.Tensor) -> None:
    k, n = x.shape
    compiled = compile_weight_quant_128x128()
    compiled(
        make_ptr(cutlass.BFloat16, x.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Float8E4M3FN, y.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Float32, scales.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
        k,
        n,
    )


def weight_quant_dual_128x128(
    x: torch.Tensor,
    y_col: torch.Tensor,
    scales_col: torch.Tensor,
    y_row: torch.Tensor,
    scales_row: torch.Tensor,
) -> None:
    k, n = x.shape
    compiled = compile_weight_quant_dual_128x128()
    compiled(
        make_ptr(cutlass.BFloat16, x.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Float8E4M3FN, y_col.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Float32, scales_col.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Float8E4M3FN, y_row.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Float32, scales_row.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
        k,
        n,
    )


class _DeepGemmLinearAutograd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        output_dtype: str,
        w_q: torch.Tensor,
        w_s: torch.Tensor,
        w_t_q: torch.Tensor,
        w_t_s: torch.Tensor,
        act_q: torch.Tensor,
        act_s: torch.Tensor,
    ) -> torch.Tensor:
        x_bf16 = x if x.dtype == torch.bfloat16 else x.to(torch.bfloat16)
        activation_quant_1x128(x_bf16, act_q, act_s)

        out = torch.empty(x.shape[0], weight.shape[0], dtype=torch.bfloat16, device=x.device)
        deep_gemm.fp8_gemm_nt((act_q, act_s), (w_q, w_s), out, disable_ue8m0_cast=True)
        if bias is not None:
            out = out + bias

        ctx.save_for_backward(x_bf16)
        ctx.has_bias = bias is not None
        ctx.x_dtype = x.dtype

        ctx.weight_shape = tuple(weight.shape)
        ctx.weight_dtype = weight.dtype
        ctx.weight_device = weight.device

        ctx.w_t_q = w_t_q
        ctx.w_t_s = w_t_s

        if output_dtype == "input" and out.dtype != x.dtype:
            out = out.to(x.dtype)

        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_output_bf16 = grad_output if grad_output.dtype == torch.bfloat16 else grad_output.to(torch.bfloat16)

        go_q = torch.empty(
            grad_output.shape[0],
            grad_output.shape[1],
            dtype=torch.float8_e4m3fn,
            device=grad_output.device,
        )
        go_s = torch.empty(
            grad_output.shape[0],
            grad_output.shape[1] // 128,
            dtype=torch.float32,
            device=grad_output.device,
        )
        activation_quant_1x128(grad_output_bf16, go_q, go_s)

        grad_input_bf16 = torch.empty(
            grad_output.shape[0],
            ctx.weight_shape[1],
            dtype=torch.bfloat16,
            device=ctx.weight_device,
        )
        deep_gemm.fp8_gemm_nt((go_q, go_s), (ctx.w_t_q, ctx.w_t_s), grad_input_bf16, disable_ue8m0_cast=True)
        grad_input = grad_input_bf16 if ctx.x_dtype == torch.bfloat16 else grad_input_bf16.to(ctx.x_dtype)

        n, k = ctx.weight_shape
        grad_weight_bf16 = torch.empty(n, k, dtype=torch.bfloat16, device=ctx.weight_device)
        deep_gemm.bf16_gemm_tn(grad_output_bf16, x, grad_weight_bf16)
        grad_weight = grad_weight_bf16 if ctx.weight_dtype == torch.bfloat16 else grad_weight_bf16.to(ctx.weight_dtype)

        grad_bias = grad_output.sum(dim=0) if ctx.has_bias else None

        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None


@dataclass(frozen=True)
class FP8LinearConfig:
    output_dtype: str = "input"

    def __post_init__(self):
        if self.output_dtype not in {"input", "bfloat16"}:
            raise ValueError("FP8LinearConfig.output_dtype must be 'input' or 'bfloat16'")


class FP8LinearDeepGEMM(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str | None = None,
        config: FP8LinearConfig | None = None,
    ):
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        assert in_features % 128 == 0
        assert out_features % 128 == 0

        self.config = config or FP8LinearConfig()
        self.k_blocks = in_features // 128
        self.n_blocks = out_features // 128

        if self.weight.device.type == "meta":
            self.register_buffer("w_q", torch.empty(0), persistent=False)
            self.register_buffer("w_s", torch.empty(0), persistent=False)
            self.register_buffer("w_t_q", torch.empty(0), persistent=False)
            self.register_buffer("w_t_s", torch.empty(0), persistent=False)
        else:
            self.register_buffer(
                "w_q",
                torch.empty(out_features, in_features, dtype=torch.float8_e4m3fn, device=self.weight.device),
                persistent=False,
            )
            self.register_buffer(
                "w_s",
                torch.empty(self.n_blocks, self.k_blocks, dtype=torch.float32, device=self.weight.device),
                persistent=False,
            )
            self.register_buffer(
                "w_t_q",
                torch.empty(in_features, out_features, dtype=torch.float8_e4m3fn, device=self.weight.device),
                persistent=False,
            )
            self.register_buffer(
                "w_t_s",
                torch.empty(self.k_blocks, self.n_blocks, dtype=torch.float32, device=self.weight.device),
                persistent=False,
            )
        self.register_buffer("_act_q", torch.empty(0), persistent=False)
        self.register_buffer("_act_s", torch.empty(0), persistent=False)

    @classmethod
    def from_float(cls, linear: nn.Linear, config: FP8LinearConfig | None = None) -> "FP8LinearDeepGEMM":
        module = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            dtype=linear.weight.dtype,
            device=linear.weight.device,
            config=config,
        )
        module.weight = linear.weight
        if linear.bias is not None:
            module.bias = linear.bias
        return module

    @classmethod
    def from_linear(cls, linear: nn.Linear, config: FP8LinearConfig | None = None) -> "FP8LinearDeepGEMM":
        return cls.from_float(linear, config=config)

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, fp8_output_dtype={self.config.output_dtype}"

    def _ensure_weight_buffers(self, device: torch.device) -> None:
        if self.w_q.shape != (self.out_features, self.in_features) or self.w_q.device != device:
            self.w_q = torch.empty(self.out_features, self.in_features, dtype=torch.float8_e4m3fn, device=device)
            self.w_s = torch.empty(self.n_blocks, self.k_blocks, dtype=torch.float32, device=device)
            self.w_t_q = torch.empty(self.in_features, self.out_features, dtype=torch.float8_e4m3fn, device=device)
            self.w_t_s = torch.empty(self.k_blocks, self.n_blocks, dtype=torch.float32, device=device)

    def _ensure_act_buffers(self, m: int, device: torch.device) -> None:
        if self._act_q.shape != (m, self.in_features) or self._act_q.device != device:
            self._act_q = torch.empty(m, self.in_features, dtype=torch.float8_e4m3fn, device=device)
            self._act_s = torch.empty(m, self.k_blocks, dtype=torch.float32, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        x_2d = x.reshape(bsz * seq_len, self.in_features).contiguous()
        self._ensure_weight_buffers(x.device)
        self._ensure_act_buffers(bsz * seq_len, x.device)

        weight_quant = self.weight.detach()
        if weight_quant.dtype != torch.bfloat16:
            weight_quant = weight_quant.to(torch.bfloat16)
        weight_quant_dual_128x128(weight_quant.t(), self.w_q.t(), self.w_s.t(), self.w_t_q, self.w_t_s)

        out_2d = _DeepGemmLinearAutograd.apply(
            x_2d,
            self.weight,
            self.bias,
            self.config.output_dtype,
            self.w_q,
            self.w_s,
            self.w_t_q,
            self.w_t_s,
            self._act_q,
            self._act_s,
        )

        return out_2d.view(bsz, seq_len, self.out_features)


@dataclass
class FP8ConversionResult:
    converted: int
    skipped_non_multiple_128: int
    skipped_excluded: int


def _get_module_by_name(model: nn.Module, module_name: str) -> nn.Module:
    module = model
    for part in module_name.split("."):
        module = getattr(module, part)
    return module


def _set_module_by_name(model: nn.Module, module_name: str, new_module: nn.Module) -> None:
    parts = module_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def _is_excluded_linear(module_name: str, module: nn.Linear) -> bool:
    if module.__class__.__name__ in {
        "FusedOutputLinear",
        "VanillaOutputLinear",
        "GemmaFusedOutputLinear",
        "GemmaVanillaOutputLinear",
    }:
        return True
    if module_name == "lm_head" or module_name.endswith(".lm_head"):
        return True
    return False


def convert_to_fp8(model: nn.Module, config: FP8LinearConfig | None = None) -> FP8ConversionResult:
    config = config or FP8LinearConfig()
    target_names: list[str] = []
    skipped_non_multiple_128 = 0
    skipped_excluded = 0

    for module_name, module in model.named_modules():
        if not module_name:
            continue
        if not isinstance(module, nn.Linear):
            continue
        if isinstance(module, FP8LinearDeepGEMM):
            continue
        if _is_excluded_linear(module_name, module):
            skipped_excluded += 1
            continue
        if module.in_features % 128 != 0 or module.out_features % 128 != 0:
            skipped_non_multiple_128 += 1
            continue
        target_names.append(module_name)

    for module_name in target_names:
        module = _get_module_by_name(model, module_name)
        assert isinstance(module, nn.Linear)
        fp8_module = FP8LinearDeepGEMM.from_float(module, config=config)
        _set_module_by_name(model, module_name, fp8_module)

    return FP8ConversionResult(
        converted=len(target_names),
        skipped_non_multiple_128=skipped_non_multiple_128,
        skipped_excluded=skipped_excluded,
    )
