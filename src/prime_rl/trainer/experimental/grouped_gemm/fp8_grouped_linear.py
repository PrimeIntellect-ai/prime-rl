import torch
import torch.nn as nn
import torch.nn.functional as F

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils

from cutlass.cute.runtime import from_dlpack

import deep_gemm


def make_col_major_tensor(shape: tuple[int, int], dtype: torch.dtype) -> torch.Tensor:
    m, n = shape
    return torch.empty_strided((m, n), (1, m), dtype=dtype, device="cuda")


class ActivationQuantizer1x128:
    def __init__(self, tile_mk: tuple[int, int] = (4, 128), num_vectorized: int = 4):
        self._bM, self._bK = tile_mk
        self._num_vectorized = num_vectorized
        assert self._bM == 4
        assert self._bK == 128
        assert self._num_vectorized == 4

    @cute.jit
    def __call__(self, x: cute.Tensor, y: cute.Tensor, scales: cute.Tensor):
        x_major = utils.LayoutEnum.from_tensor(x)
        y_major = utils.LayoutEnum.from_tensor(y)
        s_major = utils.LayoutEnum.from_tensor(scales)

        assert x_major == utils.LayoutEnum.ROW_MAJOR
        assert y_major == utils.LayoutEnum.ROW_MAJOR
        assert s_major in (utils.LayoutEnum.ROW_MAJOR, utils.LayoutEnum.COL_MAJOR)
        assert x.shape == y.shape

        copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            x.element_type,
            num_bits_per_copy=x.element_type.width * self._num_vectorized,
        )

        major_mode_size = self._bK // self._num_vectorized
        tA = cute.make_layout((self._bM, major_mode_size), stride=(major_mode_size, 1))
        vA = cute.make_layout(
            (1, self._num_vectorized), stride=(self._num_vectorized, self._bK)
        )
        tiled_copy = cute.make_tiled_copy_tv(copy_atom, tA, vA)

        sX_layout = cute.make_layout((self._bM, self._bK), stride=(self._bK, 1))

        block = (cute.size(tA), 1, 1)
        grid = (*cute.ceil_div(x.shape, (self._bM, self._bK)), 1)
        smem_size = cute.size_in_bytes(x.element_type, sX_layout)

        self.kernel(x, y, scales, tiled_copy, sX_layout).launch(
            grid=grid,
            block=block,
            smem=smem_size,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        y: cute.Tensor,
        scales: cute.Tensor,
        tiled_copy: cute.TiledCopy,
        sX_layout: cute.Layout,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()

        row_tiles = cute.ceil_div(x.shape[0], self._bM)
        col_tiles = cute.ceil_div(x.shape[1], self._bK)

        if bidx < row_tiles and bidy < col_tiles:
            tiler_coord = (bidx, bidy)
            gX = cute.local_tile(x, tiler=(self._bM, self._bK), coord=tiler_coord)
            gY = cute.local_tile(y, tiler=(self._bM, self._bK), coord=tiler_coord)

            smem = cutlass.utils.SmemAllocator()
            sX = smem.allocate_tensor(x.element_type, sX_layout, 16)

            thr_copy = tiled_copy.get_slice(tidx)
            tXgX = thr_copy.partition_S(gX)
            tXsX = thr_copy.partition_D(sX)
            tYgY = thr_copy.partition_D(gY)

            cute.copy(
                tiled_copy,
                tXgX[None, None, 0],
                tXsX[None, None, 0],
            )
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()

            thread_vals = tXsX[(None, None), 0, 0].load().to(cutlass.Float32)
            max_per_thread = cutlass.Float32(0.0)
            for vid in cutlass.range_constexpr(self._num_vectorized):
                v = thread_vals[vid]
                abs_v = v
                nv = -v
                if nv > abs_v:
                    abs_v = nv
                if abs_v > max_per_thread:
                    max_per_thread = abs_v

            max_per_warp = cute.arch.warp_reduction_max(
                max_per_thread, threads_in_group=32
            )
            max_per_warp_scalar = max_per_warp

            lane_id = tidx % 32
            warp_id = tidx // 32
            row_idx = bidx * self._bM + warp_id

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
            tYgY[(None, None), 0, 0].store(out_vals.to(y.element_type))


class WeightQuantizer128x128:
    def __init__(
        self,
        block_shape: tuple[int, int] = (128, 128),
        warps_per_cta: int = 1,
        num_vectorized: int = 8,
    ):
        self._bM, self._bK = block_shape
        self._warps_per_cta = warps_per_cta
        self._num_vectorized = num_vectorized

        assert self._bM == 128
        assert self._bK == 128
        assert self._warps_per_cta >= 1
        assert self._num_vectorized == 8

    @cute.jit
    def __call__(self, x: cute.Tensor, y: cute.Tensor, scales: cute.Tensor):
        x_major = utils.LayoutEnum.from_tensor(x)
        y_major = utils.LayoutEnum.from_tensor(y)
        s_major = utils.LayoutEnum.from_tensor(scales)

        assert x_major == utils.LayoutEnum.COL_MAJOR
        assert y_major == utils.LayoutEnum.COL_MAJOR
        assert s_major in (utils.LayoutEnum.ROW_MAJOR, utils.LayoutEnum.COL_MAJOR)
        assert x.shape == y.shape

        copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            x.element_type,
            num_bits_per_copy=x.element_type.width * self._num_vectorized,
        )
        tA = cute.make_layout((1, 32), stride=(0, 1))
        vA = cute.make_layout((8,), stride=(1,))
        tiled_copy = cute.make_tiled_copy_tv(copy_atom, tA, vA)

        sX_layout = cute.make_layout(
            (self._warps_per_cta, self._bM, self._bK),
            stride=(self._bM * self._bK, 1, self._bM),
        )

        row_blocks = cute.ceil_div(x.shape[0], self._bM)
        col_blocks = cute.ceil_div(x.shape[1], self._bK)

        grid = (cute.ceil_div(row_blocks, self._warps_per_cta), col_blocks, 1)
        block = (self._warps_per_cta * 32, 1, 1)
        smem_size = cute.size_in_bytes(x.element_type, sX_layout)

        self.kernel(x, y, scales, tiled_copy, sX_layout).launch(
            grid=grid,
            block=block,
            smem=smem_size,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        y: cute.Tensor,
        scales: cute.Tensor,
        tiled_copy: cute.TiledCopy,
        sX_layout: cute.Layout,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()

        warp_id = tidx // 32
        lane_id = tidx % 32

        row_block = bidx * self._warps_per_cta + warp_id
        col_block = bidy

        row_blocks = cute.ceil_div(x.shape[0], self._bM)
        col_blocks = cute.ceil_div(x.shape[1], self._bK)

        if row_block < row_blocks and col_block < col_blocks:
            thread_absmax = cutlass.Float32(0.0)

            smem = cutlass.utils.SmemAllocator()
            sX = smem.allocate_tensor(x.element_type, sX_layout, 16)
            sX_warp = sX[warp_id, None, None]

            tiler_coord = (row_block, col_block)
            gX = cute.local_tile(x, tiler=(self._bM, self._bK), coord=tiler_coord)
            gY = cute.local_tile(y, tiler=(self._bM, self._bK), coord=tiler_coord)

            lane_map = (lane_id % 16) * 2 + (lane_id // 16)
            thr_copy = tiled_copy.get_slice(lane_map)

            for col_group in cutlass.range_constexpr(self._bK // 32):
                for row_vec in cutlass.range_constexpr(
                    self._bM // self._num_vectorized
                ):
                    gX_sub = cute.local_tile(
                        gX, tiler=(8, 32), coord=(row_vec, col_group)
                    )
                    sX_sub = cute.local_tile(
                        sX_warp, tiler=(8, 32), coord=(row_vec, col_group)
                    )
                    tXgX = thr_copy.partition_S(gX_sub)
                    tXsX = thr_copy.partition_D(sX_sub)
                    vals = tXgX[(None, None), 0, 0].load().to(cutlass.Float32)
                    tXsX[(None, None), 0, 0].store(vals.to(x.element_type))
                    for vid in cutlass.range_constexpr(self._num_vectorized):
                        v = vals[vid]
                        abs_v = v
                        nv = -v
                        if nv > abs_v:
                            abs_v = nv
                        if abs_v > thread_absmax:
                            thread_absmax = abs_v

            block_absmax = cute.arch.warp_reduction_max(
                thread_absmax, threads_in_group=32
            )

            fp8_max = cutlass.Float32(448.0)
            eps = cutlass.Float32(1e-6)
            safe_absmax = block_absmax
            if safe_absmax < eps:
                safe_absmax = eps
            scale = safe_absmax / fp8_max

            if lane_id == 0:
                scales[row_block, col_block] = scale

            inv_scale = fp8_max / safe_absmax

            for col_group in cutlass.range_constexpr(self._bK // 32):
                for row_vec in cutlass.range_constexpr(
                    self._bM // self._num_vectorized
                ):
                    gY_sub = cute.local_tile(
                        gY, tiler=(8, 32), coord=(row_vec, col_group)
                    )
                    sX_sub = cute.local_tile(
                        sX_warp, tiler=(8, 32), coord=(row_vec, col_group)
                    )
                    tXsX = thr_copy.partition_S(sX_sub)
                    tYgY = thr_copy.partition_D(gY_sub)
                    vals = tXsX[(None, None), 0, 0].load().to(cutlass.Float32)
                    tYgY[(None, None), 0, 0].store(
                        (vals * inv_scale).to(y.element_type)
                    )


class WeightQuantizer128x128DualOutput:
    def __init__(
        self,
        block_shape: tuple[int, int] = (128, 128),
        warps_per_cta: int = 1,
        num_vectorized: int = 8,
    ):
        self._bM, self._bK = block_shape
        self._warps_per_cta = warps_per_cta
        self._num_vectorized = num_vectorized

        assert self._bM == 128
        assert self._bK == 128
        assert self._warps_per_cta >= 1
        assert self._num_vectorized == 8

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        y_col: cute.Tensor,
        scales_col: cute.Tensor,
        y_row: cute.Tensor,
        scales_row: cute.Tensor,
    ):
        x_major = utils.LayoutEnum.from_tensor(x)
        y_col_major = utils.LayoutEnum.from_tensor(y_col)
        y_row_major = utils.LayoutEnum.from_tensor(y_row)
        s_col_major = utils.LayoutEnum.from_tensor(scales_col)
        s_row_major = utils.LayoutEnum.from_tensor(scales_row)

        assert x_major == utils.LayoutEnum.COL_MAJOR
        assert y_col_major == utils.LayoutEnum.COL_MAJOR
        assert y_row_major == utils.LayoutEnum.ROW_MAJOR
        assert s_col_major in (utils.LayoutEnum.ROW_MAJOR, utils.LayoutEnum.COL_MAJOR)
        assert s_row_major in (utils.LayoutEnum.ROW_MAJOR, utils.LayoutEnum.COL_MAJOR)
        assert x.shape == y_col.shape
        assert x.shape == y_row.shape

        copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            x.element_type,
            num_bits_per_copy=x.element_type.width * self._num_vectorized,
        )
        tA = cute.make_layout((1, 32), stride=(0, 1))
        vA = cute.make_layout((8,), stride=(1,))
        tiled_copy = cute.make_tiled_copy_tv(copy_atom, tA, vA)

        sX_layout = cute.make_layout(
            (self._warps_per_cta, self._bM, self._bK),
            stride=(self._bM * self._bK, 1, self._bM),
        )

        row_blocks = cute.ceil_div(x.shape[0], self._bM)
        col_blocks = cute.ceil_div(x.shape[1], self._bK)

        grid = (cute.ceil_div(row_blocks, self._warps_per_cta), col_blocks, 1)
        block = (self._warps_per_cta * 32, 1, 1)
        smem_size = cute.size_in_bytes(x.element_type, sX_layout)

        self.kernel(
            x,
            y_col,
            scales_col,
            y_row,
            scales_row,
            tiled_copy,
            sX_layout,
        ).launch(
            grid=grid,
            block=block,
            smem=smem_size,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        y_col: cute.Tensor,
        scales_col: cute.Tensor,
        y_row: cute.Tensor,
        scales_row: cute.Tensor,
        tiled_copy: cute.TiledCopy,
        sX_layout: cute.Layout,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()

        warp_id = tidx // 32
        lane_id = tidx % 32

        row_block = bidx * self._warps_per_cta + warp_id
        col_block = bidy

        row_blocks = cute.ceil_div(x.shape[0], self._bM)
        col_blocks = cute.ceil_div(x.shape[1], self._bK)

        if row_block < row_blocks and col_block < col_blocks:
            thread_absmax = cutlass.Float32(0.0)

            smem = cutlass.utils.SmemAllocator()
            sX = smem.allocate_tensor(x.element_type, sX_layout, 16)
            sX_warp = sX[warp_id, None, None]

            tiler_coord = (row_block, col_block)
            gX = cute.local_tile(x, tiler=(self._bM, self._bK), coord=tiler_coord)
            gY_col = cute.local_tile(
                y_col, tiler=(self._bM, self._bK), coord=tiler_coord
            )
            gY_row = cute.local_tile(
                y_row, tiler=(self._bM, self._bK), coord=tiler_coord
            )

            lane_map = (lane_id % 16) * 2 + (lane_id // 16)
            thr_copy = tiled_copy.get_slice(lane_map)

            for col_group in cutlass.range_constexpr(self._bK // 32):
                for row_vec in cutlass.range_constexpr(
                    self._bM // self._num_vectorized
                ):
                    gX_sub = cute.local_tile(
                        gX, tiler=(8, 32), coord=(row_vec, col_group)
                    )
                    sX_sub = cute.local_tile(
                        sX_warp, tiler=(8, 32), coord=(row_vec, col_group)
                    )
                    tXgX = thr_copy.partition_S(gX_sub)
                    tXsX = thr_copy.partition_D(sX_sub)
                    vals = tXgX[(None, None), 0, 0].load().to(cutlass.Float32)
                    tXsX[(None, None), 0, 0].store(vals.to(x.element_type))
                    for vid in cutlass.range_constexpr(self._num_vectorized):
                        v = vals[vid]
                        abs_v = v
                        nv = -v
                        if nv > abs_v:
                            abs_v = nv
                        if abs_v > thread_absmax:
                            thread_absmax = abs_v

            block_absmax = cute.arch.warp_reduction_max(
                thread_absmax, threads_in_group=32
            )

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

            for col_group in cutlass.range_constexpr(self._bK // 32):
                for row_vec in cutlass.range_constexpr(
                    self._bM // self._num_vectorized
                ):
                    gY_col_sub = cute.local_tile(
                        gY_col, tiler=(8, 32), coord=(row_vec, col_group)
                    )
                    gY_row_sub = cute.local_tile(
                        gY_row, tiler=(8, 32), coord=(row_vec, col_group)
                    )
                    sX_sub = cute.local_tile(
                        sX_warp, tiler=(8, 32), coord=(row_vec, col_group)
                    )
                    tXsX = thr_copy.partition_S(sX_sub)
                    tYgY_col = thr_copy.partition_D(gY_col_sub)
                    tYgY_row = thr_copy.partition_D(gY_row_sub)
                    vals = tXsX[(None, None), 0, 0].load().to(cutlass.Float32)
                    out_vals = (vals * inv_scale).to(y_col.element_type)
                    tYgY_col[(None, None), 0, 0].store(out_vals)
                    tYgY_row[(None, None), 0, 0].store(out_vals)


class _FP8LinearBase(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        assert in_features % 128 == 0
        assert out_features % 128 == 0

        self.in_features = in_features
        self.out_features = out_features
        self.k_blocks = in_features // 128
        self.n_blocks = out_features // 128

        # Keep identical parameter format to torch.nn.Linear: [out_features, in_features].
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=torch.bfloat16, device="cuda")
        )
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

        self._act_quant_kernel = ActivationQuantizer1x128()
        self._wgt_quant_kernel = WeightQuantizer128x128(warps_per_cta=1)
        self._act_quant_fn = None
        self._wgt_quant_fn = None

        self._has_prequantized_weight = False
        self._act_q = None
        self._act_s = None

    def _to_cute(self, t: torch.Tensor):
        return from_dlpack(t.detach(), assumed_align=16)

    def _run_act_quant(self, x: torch.Tensor, y: torch.Tensor, s: torch.Tensor):
        x_t = self._to_cute(x)
        y_t = self._to_cute(y)
        s_t = self._to_cute(s)
        if self._act_quant_fn is None:
            self._act_quant_fn = cute.compile[cute.GenerateLineInfo](
                self._act_quant_kernel, x_t, y_t, s_t
            )
        self._act_quant_fn(x_t, y_t, s_t)

    def _run_wgt_quant(self, x: torch.Tensor, y: torch.Tensor, s: torch.Tensor):
        x_t = self._to_cute(x)
        y_t = self._to_cute(y)
        s_t = self._to_cute(s)
        if self._wgt_quant_fn is None:
            self._wgt_quant_fn = cute.compile[cute.GenerateLineInfo](
                self._wgt_quant_kernel, x_t, y_t, s_t
            )
        self._wgt_quant_fn(x_t, y_t, s_t)

    def _check_input(self, x: torch.Tensor):
        assert x.dim() == 2
        assert x.shape[1] == self.in_features
        assert x.dtype == torch.bfloat16

    def _quantize_weight_from_linear(
        self,
        w_q_kn: torch.Tensor,
        w_s_kn: torch.Tensor,
    ):
        # `self.weight` is [N, K] (torch.nn.Linear format).
        # The custom kernel consumes [K, N] col-major blocks, so we pass a view.
        self._run_wgt_quant(self.weight.detach().t(), w_q_kn, w_s_kn)


class FP8LinearScaledMM(_FP8LinearBase):
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        self.k_blocks_scaledmm = max(4, self.k_blocks)

        self.register_buffer(
            "w_q_kn",
            make_col_major_tensor((in_features, out_features), torch.float8_e4m3fn),
            persistent=False,
        )
        self.register_buffer(
            "w_s_kn",
            make_col_major_tensor(
                (self.k_blocks_scaledmm, self.n_blocks), torch.float32
            ),
            persistent=False,
        )

        self._w_q_kn_tmp = make_col_major_tensor(
            (in_features, out_features), torch.float8_e4m3fn
        )
        self._w_s_kn_tmp = make_col_major_tensor(
            (self.k_blocks_scaledmm, self.n_blocks), torch.float32
        )

    def _pad_scaledmm_scales(self, w_s_kn: torch.Tensor):
        if self.k_blocks_scaledmm > self.k_blocks:
            w_s_kn[self.k_blocks :, :].copy_(
                w_s_kn[self.k_blocks - 1 : self.k_blocks, :]
            )

    def _quantize_weight_into(self, w_q_kn: torch.Tensor, w_s_kn: torch.Tensor):
        self._quantize_weight_from_linear(w_q_kn, w_s_kn)
        self._pad_scaledmm_scales(w_s_kn)

    def prepare_prequantized_weight(self):
        self._quantize_weight_into(self.w_q_kn, self.w_s_kn)
        self._has_prequantized_weight = True

    def _ensure_act_buffers(self, m: int):
        if self._act_q is not None and self._act_q.shape[0] == m:
            return
        self._act_q = torch.empty(
            m, self.in_features, dtype=torch.float8_e4m3fn, device="cuda"
        )
        # scaled_mm blockwise requires this stride: (1, M)
        self._act_s = torch.empty_strided(
            (m, self.k_blocks), (1, m), dtype=torch.float32, device="cuda"
        )

    def forward(
        self, x: torch.Tensor, quantize_weight_on_the_fly: bool = False
    ) -> torch.Tensor:
        self._check_input(x)
        x = x.contiguous()

        m = x.shape[0]
        self._ensure_act_buffers(m)
        self._run_act_quant(x, self._act_q, self._act_s)

        if quantize_weight_on_the_fly or not self._has_prequantized_weight:
            self._quantize_weight_into(self._w_q_kn_tmp, self._w_s_kn_tmp)
            w_q_kn = self._w_q_kn_tmp
            w_s_kn = self._w_s_kn_tmp
        else:
            w_q_kn = self.w_q_kn
            w_s_kn = self.w_s_kn

        return F.scaled_mm(
            self._act_q,
            w_q_kn,
            self._act_s,
            F.ScalingType.BlockWise1x128,
            w_s_kn,
            F.ScalingType.BlockWise128x128,
            output_dtype=torch.bfloat16,
        )


class _DeepGemmCustomQuant:
    def __init__(self):
        self._act_quant_kernel = ActivationQuantizer1x128()
        self._wgt_quant_kernel = WeightQuantizer128x128(warps_per_cta=1)
        self._wgt_quant_dual_kernel = WeightQuantizer128x128DualOutput(warps_per_cta=1)
        self._act_quant_fns = {}
        self._wgt_quant_fns = {}
        self._wgt_quant_dual_fns = {}

    def _to_cute(self, t: torch.Tensor):
        return from_dlpack(t.detach(), assumed_align=16)

    @staticmethod
    def _sig(t: torch.Tensor):
        return (tuple(t.shape), tuple(t.stride()), t.dtype)

    def _get_act_fn(self, x: torch.Tensor, y: torch.Tensor, s: torch.Tensor):
        key = (self._sig(x), self._sig(y), self._sig(s))
        fn = self._act_quant_fns.get(key)
        if fn is None:
            fn = cute.compile[cute.GenerateLineInfo](
                self._act_quant_kernel,
                self._to_cute(x),
                self._to_cute(y),
                self._to_cute(s),
            )
            self._act_quant_fns[key] = fn
        return fn

    def _get_wgt_fn(self, x: torch.Tensor, y: torch.Tensor, s: torch.Tensor):
        key = (self._sig(x), self._sig(y), self._sig(s))
        fn = self._wgt_quant_fns.get(key)
        if fn is None:
            fn = cute.compile[cute.GenerateLineInfo](
                self._wgt_quant_kernel,
                self._to_cute(x),
                self._to_cute(y),
                self._to_cute(s),
            )
            self._wgt_quant_fns[key] = fn
        return fn

    def _get_wgt_dual_fn(
        self,
        x: torch.Tensor,
        y_col: torch.Tensor,
        s_col: torch.Tensor,
        y_row: torch.Tensor,
        s_row: torch.Tensor,
    ):
        key = (
            self._sig(x),
            self._sig(y_col),
            self._sig(s_col),
            self._sig(y_row),
            self._sig(s_row),
        )
        fn = self._wgt_quant_dual_fns.get(key)
        if fn is None:
            fn = cute.compile[cute.GenerateLineInfo](
                self._wgt_quant_dual_kernel,
                self._to_cute(x),
                self._to_cute(y_col),
                self._to_cute(s_col),
                self._to_cute(y_row),
                self._to_cute(s_row),
            )
            self._wgt_quant_dual_fns[key] = fn
        return fn

    def run_act_quant(self, x: torch.Tensor, y: torch.Tensor, s: torch.Tensor):
        fn = self._get_act_fn(x, y, s)
        fn(self._to_cute(x), self._to_cute(y), self._to_cute(s))

    def run_wgt_quant(self, x: torch.Tensor, y: torch.Tensor, s: torch.Tensor):
        fn = self._get_wgt_fn(x, y, s)
        fn(self._to_cute(x), self._to_cute(y), self._to_cute(s))

    def run_wgt_quant_dual(
        self,
        x: torch.Tensor,
        y_col: torch.Tensor,
        s_col: torch.Tensor,
        y_row: torch.Tensor,
        s_row: torch.Tensor,
    ):
        fn = self._get_wgt_dual_fn(x, y_col, s_col, y_row, s_row)
        fn(
            self._to_cute(x),
            self._to_cute(y_col),
            self._to_cute(s_col),
            self._to_cute(y_row),
            self._to_cute(s_row),
        )

    def quantize_activation(self, x: torch.Tensor):
        m, k = x.shape
        assert m % 4 == 0
        assert k % 128 == 0
        y = torch.empty(m, k, dtype=torch.float8_e4m3fn, device=x.device)
        s = torch.empty(m, k // 128, dtype=torch.float32, device=x.device)
        self.run_act_quant(x, y, s)
        return y, s

    def quantize_activation_into(
        self, x: torch.Tensor, y: torch.Tensor, s: torch.Tensor
    ):
        m, k = x.shape
        assert m % 4 == 0
        assert k % 128 == 0
        self.run_act_quant(x, y, s)

    def quantize_weight_linear_into(
        self,
        w_nk: torch.Tensor,
        w_q_nk: torch.Tensor,
        w_s_nk: torch.Tensor,
    ):
        # Kernel quantizes [K, N] col-major into [K, N] col-major with [K/128, N/128] scales.
        # For linear-format tensors [N, K], use transposed views as adapters.
        self.run_wgt_quant(
            w_nk.t(),
            w_q_nk.t(),
            w_s_nk.t(),
        )

    def quantize_weight_linear_dual_into(
        self,
        w_nk: torch.Tensor,
        w_q_nk: torch.Tensor,
        w_s_nk: torch.Tensor,
        w_t_q_kn: torch.Tensor,
        w_t_s_kn: torch.Tensor,
    ):
        # Quantize once from [K, N] col-major adapter and write both:
        # - forward layout: w_q_nk / w_s_nk (via transposed adapter views)
        # - dgrad layout:   w_t_q_kn / w_t_s_kn (row-major)
        self.run_wgt_quant_dual(
            w_nk.t(),
            w_q_nk.t(),
            w_s_nk.t(),
            w_t_q_kn,
            w_t_s_kn,
        )

    def transpose_quantized_weight_into(
        self,
        w_q_nk: torch.Tensor,
        w_s_nk: torch.Tensor,
        w_t_q_kn: torch.Tensor,
        w_t_s_kn: torch.Tensor,
    ):
        # Reuse already-quantized linear weights for dgrad path.
        w_t_q_kn.copy_(w_q_nk.t())
        w_t_s_kn.copy_(w_s_nk.t())

    def quantize_col_matrix_to_row(self, x_col: torch.Tensor):
        # `x_col` is [M, K] col-major. Return [M, K] row-major data + [M/128, K/128] scales.
        m, k = x_col.shape
        assert m % 128 == 0
        assert k % 128 == 0
        y_col = make_col_major_tensor((m, k), torch.float8_e4m3fn)
        s_mk = torch.empty(m // 128, k // 128, dtype=torch.float32, device=x_col.device)
        self.run_wgt_quant(x_col, y_col, s_mk)
        return y_col.contiguous(), s_mk


class _DeepGemmLinearAutograd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        quant: _DeepGemmCustomQuant,
        w_q: torch.Tensor,
        w_s: torch.Tensor,
        w_t_q: torch.Tensor | None,
        w_t_s: torch.Tensor | None,
    ) -> torch.Tensor:
        need_grad_input = ctx.needs_input_grad[0]
        need_grad_weight = ctx.needs_input_grad[1]

        x_q, x_s = quant.quantize_activation(x)

        out = torch.empty(
            x.shape[0], weight.shape[0], dtype=torch.bfloat16, device=x.device
        )
        deep_gemm.fp8_gemm_nt(
            (x_q, x_s),
            (w_q, w_s),
            out,
            disable_ue8m0_cast=True,
        )
        if bias is not None:
            out = out + bias

        x_saved = (
            x if need_grad_weight else torch.empty(0, device=x.device, dtype=x.dtype)
        )
        ctx.save_for_backward(x_saved)
        ctx.has_saved_x = need_grad_weight
        ctx.weight_shape = tuple(weight.shape)
        ctx.weight_dtype = weight.dtype
        ctx.weight_device = weight.device

        if need_grad_input:
            assert w_t_q is not None and w_t_s is not None
            ctx.w_t_q = w_t_q
            ctx.w_t_s = w_t_s
        else:
            ctx.w_t_q = None
            ctx.w_t_s = None

        ctx.has_bias = bias is not None
        ctx.quant = quant
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        quant = ctx.quant
        (x_saved,) = ctx.saved_tensors
        x = x_saved if ctx.has_saved_x else None
        grad_output = grad_output.contiguous()

        grad_input = None
        grad_weight = None
        grad_bias = None

        if ctx.needs_input_grad[0]:
            go_q, go_s = quant.quantize_activation(grad_output)
            w_t_q = ctx.w_t_q
            w_t_s = ctx.w_t_s
            assert w_t_q is not None and w_t_s is not None
            grad_input = torch.empty_like(x)
            deep_gemm.fp8_gemm_nt(
                (go_q, go_s),
                (w_t_q, w_t_s),
                grad_input,
                disable_ue8m0_cast=True,
            )

        if ctx.needs_input_grad[1]:
            assert x is not None
            n, k = ctx.weight_shape
            grad_weight = torch.empty(
                n,
                k,
                dtype=ctx.weight_dtype,
                device=ctx.weight_device,
            )
            deep_gemm.bf16_gemm_tn(grad_output, x, grad_weight)

        if ctx.has_bias and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(dim=0)

        return grad_input, grad_weight, grad_bias, None, None, None, None, None


class FP8LinearDeepGEMM(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
    ):
        super().__init__()
        assert in_features % 128 == 0
        assert out_features % 128 == 0

        self.in_features = in_features
        self.out_features = out_features
        self.k_blocks = in_features // 128
        self.n_blocks = out_features // 128

        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=torch.bfloat16, device="cuda")
        )
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_features, dtype=torch.bfloat16, device="cuda")
            )
        else:
            self.register_parameter("bias", None)

        self.register_buffer(
            "w_q",
            torch.empty(
                out_features,
                in_features,
                dtype=torch.float8_e4m3fn,
                device="cuda",
            ),
            persistent=False,
        )
        self.register_buffer(
            "w_s",
            torch.empty(
                self.n_blocks, self.k_blocks, dtype=torch.float32, device="cuda"
            ),
            persistent=False,
        )
        self.register_buffer(
            "w_t_q",
            torch.empty(
                in_features,
                out_features,
                dtype=torch.float8_e4m3fn,
                device="cuda",
            ),
            persistent=False,
        )
        self.register_buffer(
            "w_t_s",
            torch.empty(
                self.k_blocks,
                self.n_blocks,
                dtype=torch.float32,
                device="cuda",
            ),
            persistent=False,
        )

        self._w_q_tmp = torch.empty(
            out_features,
            in_features,
            dtype=torch.float8_e4m3fn,
            device="cuda",
        )
        self._w_s_tmp = torch.empty(
            self.n_blocks,
            self.k_blocks,
            dtype=torch.float32,
            device="cuda",
        )

        self._act_q = None
        self._act_s = None
        self._has_prequantized_weight = False
        self._w_q_version = -1
        self._w_t_q_version = -1
        self._quant = _DeepGemmCustomQuant()

    def _ensure_act_buffers(self, m: int):
        if self._act_q is not None and self._act_q.shape[0] == m:
            return
        self._act_q = torch.empty(
            m, self.in_features, dtype=torch.float8_e4m3fn, device="cuda"
        )
        self._act_s = torch.empty(m, self.k_blocks, dtype=torch.float32, device="cuda")

    def _ensure_weight_cache(self, need_transposed: bool):
        v = self.weight._version
        if (not self._has_prequantized_weight) or (self._w_q_version != v):
            if need_transposed:
                self._quant.quantize_weight_linear_dual_into(
                    self.weight.detach(),
                    self.w_q,
                    self.w_s,
                    self.w_t_q,
                    self.w_t_s,
                )
                self._w_t_q_version = v
            else:
                self._quant.quantize_weight_linear_into(
                    self.weight.detach(), self.w_q, self.w_s
                )
                self._w_t_q_version = -1
            self._has_prequantized_weight = True
            self._w_q_version = v

        if need_transposed and self._w_t_q_version != self._w_q_version:
            self._quant.transpose_quantized_weight_into(
                self.w_q,
                self.w_s,
                self.w_t_q,
                self.w_t_s,
            )
            self._w_t_q_version = self._w_q_version

    def prepare_prequantized_weight(self):
        self._ensure_weight_cache(need_transposed=False)

    def _forward_inference(
        self, x: torch.Tensor, quantize_weight_on_the_fly: bool
    ) -> torch.Tensor:
        m = x.shape[0]
        self._ensure_act_buffers(m)
        self._quant.quantize_activation_into(x, self._act_q, self._act_s)

        if quantize_weight_on_the_fly or not self._has_prequantized_weight:
            self._quant.quantize_weight_linear_into(
                self.weight.detach(), self._w_q_tmp, self._w_s_tmp
            )
            w_q, w_s = self._w_q_tmp, self._w_s_tmp
        else:
            self._ensure_weight_cache(need_transposed=False)
            w_q, w_s = self.w_q, self.w_s

        out = torch.empty(
            m,
            self.out_features,
            dtype=torch.bfloat16,
            device=x.device,
        )
        deep_gemm.fp8_gemm_nt(
            (self._act_q, self._act_s),
            (w_q, w_s),
            out,
            disable_ue8m0_cast=True,
        )
        if self.bias is not None:
            out = out + self.bias
        return out

    def forward(
        self,
        x: torch.Tensor,
        quantize_weight_on_the_fly: bool = False,
    ) -> torch.Tensor:
        assert x.dim() == 2
        assert x.shape[1] == self.in_features
        assert x.shape[0] % 128 == 0
        assert x.dtype == torch.bfloat16
        x = x.contiguous()

        if torch.is_grad_enabled() and (
            x.requires_grad
            or self.weight.requires_grad
            or (self.bias is not None and self.bias.requires_grad)
        ):
            need_dgrad = x.requires_grad
            self._ensure_weight_cache(need_transposed=need_dgrad)
            return _DeepGemmLinearAutograd.apply(
                x,
                self.weight,
                self.bias,
                self._quant,
                self.w_q,
                self.w_s,
                self.w_t_q if need_dgrad else None,
                self.w_t_s if need_dgrad else None,
            )

        return self._forward_inference(x, quantize_weight_on_the_fly)


class _DeepGemmGroupedLinearAutograd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        group_offsets: torch.Tensor,
        grouped_layout: torch.Tensor,
        quant: _DeepGemmCustomQuant,
        w_q: torch.Tensor,
        w_s: torch.Tensor,
        w_t_q: torch.Tensor | None,
        w_t_s: torch.Tensor | None,
    ) -> torch.Tensor:
        need_grad_input = ctx.needs_input_grad[0]
        need_grad_weight = ctx.needs_input_grad[1]

        x_q, x_s = quant.quantize_activation(x)
        out = torch.empty(
            x.shape[0],
            weight.shape[1],
            dtype=torch.bfloat16,
            device=x.device,
        )
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            (x_q, x_s),
            (w_q, w_s),
            out,
            grouped_layout,
            disable_ue8m0_cast=True,
        )
        if bias is not None:
            out = out + bias[grouped_layout.to(torch.long)]

        x_saved = (
            x if need_grad_weight else torch.empty(0, device=x.device, dtype=x.dtype)
        )
        ctx.save_for_backward(x_saved, group_offsets, grouped_layout)
        ctx.has_saved_x = need_grad_weight
        ctx.has_bias = bias is not None

        ctx.weight_shape = tuple(weight.shape)
        ctx.weight_dtype = weight.dtype
        ctx.weight_device = weight.device
        ctx.quant = quant

        if need_grad_input:
            assert w_t_q is not None and w_t_s is not None
            ctx.w_t_q = w_t_q
            ctx.w_t_s = w_t_s
        else:
            ctx.w_t_q = None
            ctx.w_t_s = None

        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        quant = ctx.quant
        x_saved, group_offsets, grouped_layout = ctx.saved_tensors
        x = x_saved if ctx.has_saved_x else None
        grad_output = grad_output.contiguous()

        grad_input = None
        grad_weight = None
        grad_bias = None

        if ctx.needs_input_grad[0]:
            w_t_q = ctx.w_t_q
            w_t_s = ctx.w_t_s
            assert w_t_q is not None and w_t_s is not None
            go_q, go_s = quant.quantize_activation(grad_output)
            grad_input = torch.empty(
                grad_output.shape[0],
                ctx.weight_shape[2],
                dtype=ctx.weight_dtype,
                device=ctx.weight_device,
            )
            deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
                (go_q, go_s),
                (w_t_q, w_t_s),
                grad_input,
                grouped_layout,
                disable_ue8m0_cast=True,
            )

        if ctx.needs_input_grad[1]:
            assert x is not None
            grad_weight = torch._grouped_mm(
                grad_output.t().contiguous(),
                x,
                offs=group_offsets,
                out_dtype=torch.bfloat16,
            )

        if ctx.has_bias and ctx.needs_input_grad[2]:
            n_experts, n_out, _ = ctx.weight_shape
            grad_bias = torch.zeros(
                n_experts,
                n_out,
                dtype=grad_output.dtype,
                device=grad_output.device,
            )
            grad_bias.index_add_(0, grouped_layout.to(torch.long), grad_output)

        return (
            grad_input,
            grad_weight,
            grad_bias,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class FP8GroupedLinearDeepGEMM(nn.Module):
    def __init__(
        self,
        num_experts: int,
        in_features: int,
        out_features: int,
        bias: bool = False,
    ):
        super().__init__()
        assert num_experts > 0
        assert in_features % 128 == 0
        assert out_features % 128 == 0

        self.num_experts = num_experts
        self.in_features = in_features
        self.out_features = out_features
        self.k_blocks = in_features // 128
        self.n_blocks = out_features // 128

        self.weight = nn.Parameter(
            torch.empty(
                num_experts,
                out_features,
                in_features,
                dtype=torch.bfloat16,
                device="cuda",
            )
        )
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

        if bias:
            self.bias = nn.Parameter(
                torch.zeros(
                    num_experts,
                    out_features,
                    dtype=torch.bfloat16,
                    device="cuda",
                )
            )
        else:
            self.register_parameter("bias", None)

        self.register_buffer(
            "w_q",
            torch.empty(
                num_experts,
                out_features,
                in_features,
                dtype=torch.float8_e4m3fn,
                device="cuda",
            ),
            persistent=False,
        )
        self.register_buffer(
            "w_s",
            torch.empty(
                num_experts,
                self.n_blocks,
                self.k_blocks,
                dtype=torch.float32,
                device="cuda",
            ),
            persistent=False,
        )
        self.register_buffer(
            "w_t_q",
            torch.empty(
                num_experts,
                in_features,
                out_features,
                dtype=torch.float8_e4m3fn,
                device="cuda",
            ),
            persistent=False,
        )
        self.register_buffer(
            "w_t_s",
            torch.empty(
                num_experts,
                self.k_blocks,
                self.n_blocks,
                dtype=torch.float32,
                device="cuda",
            ),
            persistent=False,
        )

        self._w_q_tmp = torch.empty(
            num_experts,
            out_features,
            in_features,
            dtype=torch.float8_e4m3fn,
            device="cuda",
        )
        self._w_s_tmp = torch.empty(
            num_experts,
            self.n_blocks,
            self.k_blocks,
            dtype=torch.float32,
            device="cuda",
        )

        self._act_q = None
        self._act_s = None
        self._has_prequantized_weight = False
        self._w_q_version = -1
        self._w_t_q_version = -1
        self._quant = _DeepGemmCustomQuant()

    @staticmethod
    def _align128(x: int) -> int:
        return ((x + 127) // 128) * 128

    def _ensure_act_buffers(self, m: int):
        if self._act_q is not None and self._act_q.shape[0] == m:
            return
        self._act_q = torch.empty(
            m,
            self.in_features,
            dtype=torch.float8_e4m3fn,
            device="cuda",
        )
        self._act_s = torch.empty(
            m,
            self.k_blocks,
            dtype=torch.float32,
            device="cuda",
        )

    def _quantize_weight_into(self, w_q: torch.Tensor, w_s: torch.Tensor):
        for e in range(self.num_experts):
            self._quant.quantize_weight_linear_into(
                self.weight[e].detach(), w_q[e], w_s[e]
            )

    def _quantize_weight_dual_into(
        self,
        w_q: torch.Tensor,
        w_s: torch.Tensor,
        w_t_q: torch.Tensor,
        w_t_s: torch.Tensor,
    ):
        for e in range(self.num_experts):
            self._quant.quantize_weight_linear_dual_into(
                self.weight[e].detach(),
                w_q[e],
                w_s[e],
                w_t_q[e],
                w_t_s[e],
            )

    def _ensure_weight_cache(self, need_transposed: bool):
        v = self.weight._version
        if (not self._has_prequantized_weight) or (self._w_q_version != v):
            if need_transposed:
                self._quantize_weight_dual_into(
                    self.w_q,
                    self.w_s,
                    self.w_t_q,
                    self.w_t_s,
                )
                self._w_t_q_version = v
            else:
                self._quantize_weight_into(self.w_q, self.w_s)
                self._w_t_q_version = -1
            self._has_prequantized_weight = True
            self._w_q_version = v

        if need_transposed and self._w_t_q_version != self._w_q_version:
            for e in range(self.num_experts):
                self._quant.transpose_quantized_weight_into(
                    self.w_q[e],
                    self.w_s[e],
                    self.w_t_q[e],
                    self.w_t_s[e],
                )
            self._w_t_q_version = self._w_q_version

    def prepare_prequantized_weight(self):
        self._ensure_weight_cache(need_transposed=False)

    def _add_group_bias_(self, out: torch.Tensor, grouped_layout: torch.Tensor):
        if self.bias is None:
            return
        valid = grouped_layout >= 0
        if bool(valid.all()):
            out += self.bias[grouped_layout.to(torch.long)]
            return
        idx = grouped_layout[valid].to(torch.long)
        out_valid = out[valid]
        out_valid += self.bias[idx]
        out[valid] = out_valid

    def _forward_fp8_packed(
        self,
        x_packed: torch.Tensor,
        grouped_layout: torch.Tensor,
        quantize_weight_on_the_fly: bool,
    ) -> torch.Tensor:
        m = x_packed.shape[0]
        self._ensure_act_buffers(m)
        self._quant.quantize_activation_into(x_packed, self._act_q, self._act_s)

        if quantize_weight_on_the_fly or not self._has_prequantized_weight:
            self._quantize_weight_into(self._w_q_tmp, self._w_s_tmp)
            w_q, w_s = self._w_q_tmp, self._w_s_tmp
        else:
            self._ensure_weight_cache(need_transposed=False)
            w_q, w_s = self.w_q, self.w_s

        out = torch.empty(
            m,
            self.out_features,
            dtype=torch.bfloat16,
            device=x_packed.device,
        )
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            (self._act_q, self._act_s),
            (w_q, w_s),
            out,
            grouped_layout,
            disable_ue8m0_cast=True,
        )
        self._add_group_bias_(out, grouped_layout)
        return out

    def _forward_bf16_packed(
        self,
        x_packed: torch.Tensor,
        grouped_layout: torch.Tensor,
    ) -> torch.Tensor:
        out = torch.empty(
            x_packed.shape[0],
            self.out_features,
            dtype=torch.bfloat16,
            device=x_packed.device,
        )
        deep_gemm.m_grouped_bf16_gemm_nt_contiguous(
            x_packed,
            self.weight,
            out,
            grouped_layout,
        )
        self._add_group_bias_(out, grouped_layout)
        return out

    def _counts_from_offsets(self, group_offsets: torch.Tensor) -> torch.Tensor:
        assert group_offsets.dtype == torch.int32
        assert group_offsets.dim() == 1
        assert group_offsets.numel() == self.num_experts
        counts = torch.empty_like(group_offsets)
        counts[0] = group_offsets[0]
        if group_offsets.numel() > 1:
            counts[1:] = group_offsets[1:] - group_offsets[:-1]
        return counts

    def _grouped_layout_from_offsets_aligned(
        self,
        group_offsets: torch.Tensor,
    ) -> torch.Tensor:
        counts = self._counts_from_offsets(group_offsets).to(torch.long)
        experts = torch.arange(
            self.num_experts,
            dtype=torch.int32,
            device=group_offsets.device,
        )
        return torch.repeat_interleave(experts, counts)

    def _pack_from_offsets(self, x: torch.Tensor, group_offsets: torch.Tensor):
        assert group_offsets.dtype == torch.int32
        assert group_offsets.numel() == self.num_experts

        ends = group_offsets.detach().to(device="cpu", dtype=torch.int64).tolist()
        assert len(ends) == self.num_experts
        assert ends[-1] == x.shape[0]

        counts = []
        last = 0
        for end in ends:
            assert end >= last
            counts.append(end - last)
            last = end

        aligned_total = 0
        for c in counts:
            aligned_total += self._align128(c)

        if aligned_total == x.shape[0]:
            x_packed = x
        else:
            x_packed = torch.zeros(
                aligned_total,
                x.shape[1],
                dtype=x.dtype,
                device=x.device,
            )

        grouped_layout = torch.empty(
            aligned_total,
            dtype=torch.int32,
            device=x.device,
        )

        src_start = 0
        dst_start = 0
        for e, c in enumerate(counts):
            aligned_c = self._align128(c)
            if c > 0:
                if aligned_total != x.shape[0]:
                    x_packed[dst_start : dst_start + c].copy_(
                        x[src_start : src_start + c]
                    )
                grouped_layout[dst_start : dst_start + c] = e
            if aligned_c > c:
                grouped_layout[dst_start + c : dst_start + aligned_c] = -1
            src_start += c
            dst_start += aligned_c

        return x_packed, grouped_layout

    def forward(
        self,
        x: torch.Tensor,
        group_offsets: torch.Tensor | None = None,
        grouped_layout: torch.Tensor | None = None,
        quantize_weight_on_the_fly: bool = False,
    ) -> torch.Tensor:
        assert x.dim() == 2
        assert x.shape[1] == self.in_features
        assert x.shape[0] % 4 == 0
        assert x.dtype == torch.bfloat16
        x = x.contiguous()

        use_offsets = group_offsets is not None
        use_layout = grouped_layout is not None
        assert use_offsets != use_layout

        m_real = x.shape[0]
        if use_offsets:
            assert group_offsets is not None
            group_offsets = group_offsets.contiguous().to(torch.int32)
            assert group_offsets.shape[0] == self.num_experts
            assert int(group_offsets[-1].item()) == m_real
            counts = self._counts_from_offsets(group_offsets)
            is_aligned = bool((counts % 128 == 0).all().item())

            if is_aligned:
                x_packed = x
                grouped_layout = self._grouped_layout_from_offsets_aligned(
                    group_offsets
                )
            else:
                x_packed, grouped_layout = self._pack_from_offsets(x, group_offsets)
        else:
            assert grouped_layout is not None
            assert grouped_layout.dtype == torch.int32
            assert grouped_layout.dim() == 1
            assert grouped_layout.shape[0] == x.shape[0]
            grouped_layout = grouped_layout.contiguous()
            x_packed = x
            group_offsets = None

        needs_grad = torch.is_grad_enabled() and (
            x.requires_grad
            or self.weight.requires_grad
            or (self.bias is not None and self.bias.requires_grad)
        )
        if needs_grad:
            assert group_offsets is not None
            assert x_packed.shape[0] == m_real
            need_dgrad = x.requires_grad
            self._ensure_weight_cache(need_transposed=need_dgrad)
            out = _DeepGemmGroupedLinearAutograd.apply(
                x_packed,
                self.weight,
                self.bias,
                group_offsets,
                grouped_layout,
                self._quant,
                self.w_q,
                self.w_s,
                self.w_t_q if need_dgrad else None,
                self.w_t_s if need_dgrad else None,
            )
        else:
            out = self._forward_fp8_packed(
                x_packed,
                grouped_layout,
                quantize_weight_on_the_fly=quantize_weight_on_the_fly,
            )

        if use_offsets and out.shape[0] != m_real:
            out = out[:m_real]
        return out
