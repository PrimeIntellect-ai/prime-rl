"""
[DeepSeek V4 Sparse Attention: Forward, CuTe DSL port for sm90]

The same algorithm as the TileLang forward in `dsv4_sparse_attn_fwd.py`, written in NVIDIA's CuTe
Python DSL (`cutlass.cute`) and compiled for H100/H200 (`sm_90a`) only. That module's docstring
defines the shapes, the math, the sink-seeded online softmax and the caller's contract; this one
covers only how the port maps onto the hardware. It serves `heads = 64`, `dim = 512` and one KV
head, the production DeepSeek V4 shape.

The design is the one TileLang emits for that kernel, instruction flavor included:

  - one CTA of 256 threads (two warpgroups) per query position, holding all 64 heads;
  - `Q_shared[h,d]` loaded once, `KV_shared[k,d]` gathered per tile of 64 slots with `cp.async`,
    two stages deep, and a masked or out-of-range slot zero-filled through `cp.async`'s source-size
    operand;
  - the score GEMM `acc_s[h,k] = Q_shared[h,d] KV_shared[k,d]` as synchronous WGMMA
    `m64n32k16`, both operands in shared memory, each warpgroup taking 32 of the 64 slots;
  - the online softmax, whose row max and row sum combine each warpgroup's half through shared
    memory;
  - `P` rounded to bfloat16 and stored to shared memory with `stmatrix`;
  - the output GEMM `acc_o[h,d] += P[h,k] KV_shared[k,d]` as synchronous WGMMA `m64n256k16`, each
    warpgroup owning 256 of the 512 channels, so the fp32 accumulator costs 128 registers per thread.

Only the batch, the query count, the KV count and the slot count are runtime values, so one
compiled kernel serves every shape. It is compiled on first use and called through the TVM-FFI
executor, which takes torch tensors and the current torch stream directly.
"""

import functools

import cutlass
import cutlass.cute as cute
import cutlass.utils.hopper_helpers as sm90_utils
import torch
from cutlass import BFloat16, Float32, Int32, const_expr
from cutlass.cute.nvgpu import cpasync, warp, warpgroup
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cutlass.utils import LayoutEnum

LOG2E = 1.44269504

HEADS = 64
DIM = 512
BLOCK_I = 64
NUM_STAGES = 2
THREADS = 256
WARPGROUP_THREADS = 128
CHUNK = 8
CHUNKS_PER_ROW = DIM // CHUNK
ROWS_PER_PASS = THREADS // CHUNKS_PER_ROW


def _smem_layout(shape: tuple[int, ...]) -> cute.ComposedLayout:
    """A 128-byte-swizzled, channel-contiguous shared-memory layout that WGMMA reads by descriptor."""
    atom = warpgroup.make_smem_layout_atom(
        sm90_utils.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, BFloat16, shape[1]), BFloat16
    )
    return cute.tile_to_shape(atom, shape, tuple(range(len(shape))))


def _acc_rows_view(acc: cute.Tensor) -> cute.Tensor:
    """View a WGMMA accumulator `((2, 2, V), MMA_M, MMA_N)` as `((2, MMA_M), (2, V, MMA_N))`: heads by columns."""
    layout = cute.make_layout(acc.layout.shape)
    shape, stride = layout.shape, layout.stride
    rows_view = cute.make_layout(
        ((shape[0][1], shape[1]), (shape[0][0], shape[0][2], shape[2])),
        stride=((stride[0][1], stride[1]), (stride[0][0], stride[0][2], stride[2])),
    )
    return cute.make_tensor(acc.iterator, cute.composition(acc.layout, rows_view))


@cute.jit
def _reduce_rows(vals: cute.Tensor, sRed: cute.Tensor, head_of: cute.Tensor, wg_idx: Int32, is_max: cutlass.Constexpr):
    """Reduce each head's per-thread partial over the 4 threads sharing it, then over both warpgroups."""
    lane = cute.arch.lane_idx()
    for r in cutlass.range_constexpr(cute.size(vals)):
        v = vals[r]
        for step in cutlass.range_constexpr(2):
            other = cute.arch.shuffle_sync_bfly(v, offset=1 << step)
            v = cute.arch.fmax(v, other) if const_expr(is_max) else v + other
        vals[r] = v
        if lane % 4 == 0:
            sRed[wg_idx, head_of[r]] = v
    cute.arch.sync_threads()
    for r in cutlass.range_constexpr(cute.size(vals)):
        other = sRed[1 - wg_idx, head_of[r]]
        vals[r] = cute.arch.fmax(vals[r], other) if const_expr(is_max) else vals[r] + other


@cute.jit
def _gather_tile(
    mKV: cute.Tensor,
    mIndices: cute.Tensor,
    sKV: cute.Tensor,
    gather_atom: cute.CopyAtom,
    b_i: Int32,
    s_i: Int32,
    tile: Int32,
    stage: Int32,
    tidx: Int32,
):
    """Issue the `cp.async` copies of one tile of keys: 64 threads per row, 16 bytes each."""
    seq_len_kv = mKV.shape[1]
    chunk = tidx % CHUNKS_PER_ROW
    for i in cutlass.range_constexpr(BLOCK_I // ROWS_PER_PASS):
        row = i * ROWS_PER_PASS + tidx // CHUNKS_PER_ROW
        kv_idx = mIndices[b_i, s_i, 0, tile * BLOCK_I + row]
        in_range = kv_idx >= 0 and kv_idx < seq_len_kv
        src = cute.flat_divide(mKV[b_i, kv_idx if in_range else 0, 0, None], (CHUNK,))[None, chunk]
        src = cute.make_tensor(
            cute.make_ptr(BFloat16, src.iterator.toint(), cute.AddressSpace.gmem, assumed_align=16),
            cute.group_modes(src.layout, 0, 1),
        )
        dst = cute.group_modes(cute.flat_divide(sKV[row, None, stage], (CHUNK,))[None, chunk], 0, 1)
        # A false predicate lowers to `cp.async` with a source size of 0, which zero-fills the chunk.
        in_range_pred = cute.make_fragment_like(src, cutlass.Boolean)
        in_range_pred.fill(in_range)
        cute.copy(gather_atom, src, dst, pred=in_range_pred)


@cute.kernel
def _fwd_kernel(
    mQ: cute.Tensor,
    mKV: cute.Tensor,
    mIndices: cute.Tensor,
    mSinks: cute.Tensor,
    mTileCounts: cute.Tensor,
    mOut: cute.Tensor,
    mLse: cute.Tensor,
    sm_scale: Float32,
    sm_scale_log2: Float32,
    sQ_layout: cute.ComposedLayout,
    sKV_layout: cute.ComposedLayout,
    sP_layout: cute.ComposedLayout,
    tiled_mma_qk: cute.TiledMma,
    tiled_mma_pv: cute.TiledMma,
    tiled_copy_q: cute.TiledCopy,
):
    tidx, _, _ = cute.arch.thread_idx()
    s_i, b_i, _ = cute.arch.block_idx()
    wg_idx = cute.arch.make_warp_uniform(tidx // WARPGROUP_THREADS)

    smem = cutlass.utils.SmemAllocator()
    sQ = smem.allocate_tensor(BFloat16, sQ_layout.outer, 1024, sQ_layout.inner)
    sKV = smem.allocate_tensor(BFloat16, sKV_layout.outer, 1024, sKV_layout.inner)
    sP = smem.allocate_tensor(BFloat16, sP_layout.outer, 1024, sP_layout.inner)
    sMax = smem.allocate_tensor(Float32, cute.make_layout((2, HEADS), stride=(HEADS, 1)), 16)
    sSum = smem.allocate_tensor(Float32, cute.make_layout((2, HEADS), stride=(HEADS, 1)), 16)
    sKVt = cute.composition(sKV, cute.make_ordered_layout((DIM, BLOCK_I, NUM_STAGES), order=(1, 0, 2)))

    n_tiles = mIndices.shape[3] // BLOCK_I
    tile_count = cutlass.min(mTileCounts[b_i, s_i, 0], n_tiles)
    gather_atom = cute.make_copy_atom(
        cpasync.CopyG2SOp(cache_mode=cute.nvgpu.LoadCacheMode.GLOBAL), BFloat16, num_bits_per_copy=128
    )

    thr_copy_q = tiled_copy_q.get_slice(tidx)
    cute.copy(tiled_copy_q, thr_copy_q.partition_S(mQ[b_i, s_i, None, None]), thr_copy_q.partition_D(sQ))
    cute.arch.cp_async_commit_group()

    wg_mma_qk = tiled_mma_qk.get_slice(wg_idx * WARPGROUP_THREADS)
    thr_mma_qk = tiled_mma_qk.get_slice(tidx)
    tSrQ = wg_mma_qk.make_fragment_A(wg_mma_qk.partition_A(sQ))
    tSrK = wg_mma_qk.make_fragment_B(wg_mma_qk.partition_B(sKV))
    acc_s = cute.make_rmem_tensor(thr_mma_qk.partition_shape_C((HEADS, BLOCK_I)), Float32)
    acc_s_rows = _acc_rows_view(acc_s)
    tScS_rows = _acc_rows_view(thr_mma_qk.partition_C(cute.make_identity_tensor((HEADS, BLOCK_I))))

    wg_mma_pv = tiled_mma_pv.get_slice(wg_idx * WARPGROUP_THREADS)
    thr_mma_pv = tiled_mma_pv.get_slice(tidx)
    tOrP = wg_mma_pv.make_fragment_A(wg_mma_pv.partition_A(sP))
    tOrV = wg_mma_pv.make_fragment_B(wg_mma_pv.partition_B(sKVt))
    acc_o = cute.make_rmem_tensor(thr_mma_pv.partition_shape_C((HEADS, DIM)), Float32)
    acc_o.fill(0.0)
    acc_o_rows = _acc_rows_view(acc_o)

    smem_copy_p = cute.make_tiled_copy_C(
        cute.make_copy_atom(warp.StMatrix8x8x16bOp(transpose=False, num_matrices=4), BFloat16), tiled_mma_qk
    )
    thr_copy_p = smem_copy_p.get_slice(tidx)
    tPsP = thr_copy_p.partition_D(sP)
    rP = cute.make_fragment_like(acc_s, BFloat16)

    mma_atom_qk = cute.make_mma_atom(tiled_mma_qk.op)
    mma_atom_qk.set(warpgroup.Field.ACCUMULATE, True)
    mma_atom_pv = cute.make_mma_atom(tiled_mma_pv.op)
    mma_atom_pv.set(warpgroup.Field.ACCUMULATE, True)

    n_rows = cute.size(acc_s_rows, mode=[0])
    n_cols = cute.size(acc_s_rows, mode=[1])
    head_of = cute.make_rmem_tensor((n_rows,), Int32)
    m_i = cute.make_rmem_tensor((n_rows,), Float32)
    sumexp = cute.make_rmem_tensor((n_rows,), Float32)
    alpha = cute.make_rmem_tensor((n_rows,), Float32)
    tile_max = cute.make_rmem_tensor((n_rows,), Float32)
    tile_sum = cute.make_rmem_tensor((n_rows,), Float32)
    for r in cutlass.range_constexpr(n_rows):
        head_of[r] = tScS_rows[r, 0][0]
        m_i[r] = mSinks[head_of[r]] / sm_scale
        sumexp[r] = Float32(1.0)

    if tile_count > 0:
        _gather_tile(mKV, mIndices, sKV, gather_atom, b_i, s_i, 0, 0, tidx)
    cute.arch.cp_async_commit_group()

    for tile in cutlass.range(tile_count, unroll=1):
        stage = tile % NUM_STAGES
        cute.arch.sync_threads()
        if tile + 1 < tile_count:
            _gather_tile(mKV, mIndices, sKV, gather_atom, b_i, s_i, tile + 1, (tile + 1) % NUM_STAGES, tidx)
        cute.arch.cp_async_commit_group()

        for c in cutlass.range_constexpr(n_cols):
            slot_valid = mIndices[b_i, s_i, 0, tile * BLOCK_I + tScS_rows[0, c][1]] >= 0
            for r in cutlass.range_constexpr(n_rows):
                acc_s_rows[r, c] = Float32(0.0) if slot_valid else -Float32.inf

        cute.arch.cp_async_wait_group(1)
        cute.arch.fence_view_async_shared()
        cute.arch.sync_threads()

        warpgroup.fence()
        for k in cutlass.range_constexpr(cute.size(tSrQ, mode=[2])):
            cute.gemm(mma_atom_qk, acc_s, tSrQ[None, None, k], tSrK[None, None, k, stage], acc_s)
        warpgroup.commit_group()
        warpgroup.wait_group(0)

        for r in cutlass.range_constexpr(n_rows):
            tile_max[r] = acc_s_rows[r, None].load().reduce(cute.ReductionOp.MAX, -Float32.inf, 0)
        _reduce_rows(tile_max, sMax, head_of, wg_idx, True)
        for r in cutlass.range_constexpr(n_rows):
            m_prev = m_i[r]
            m_i[r] = cute.arch.fmax(m_prev, tile_max[r])
            alpha[r] = cute.math.exp2((m_prev - m_i[r]) * sm_scale_log2, fastmath=True)
            p = cute.math.exp2(acc_s_rows[r, None].load() * sm_scale_log2 - m_i[r] * sm_scale_log2, fastmath=True)
            acc_s_rows[r, None].store(p)
            tile_sum[r] = p.reduce(cute.ReductionOp.ADD, Float32(0.0), 0)
        _reduce_rows(tile_sum, sSum, head_of, wg_idx, False)
        for r in cutlass.range_constexpr(n_rows):
            sumexp[r] = sumexp[r] * alpha[r] + tile_sum[r]
            acc_o_rows[r, None].store(acc_o_rows[r, None].load() * alpha[r])

        rP.store(acc_s.load().to(BFloat16))
        cute.copy(smem_copy_p, thr_copy_p.retile(rP), tPsP)
        cute.arch.fence_view_async_shared()
        cute.arch.sync_threads()

        warpgroup.fence()
        for k in cutlass.range_constexpr(cute.size(tOrP, mode=[2])):
            cute.gemm(mma_atom_pv, acc_o, tOrP[None, None, k], tOrV[None, None, k, stage], acc_o)
        warpgroup.commit_group()
        warpgroup.wait_group(0)
    cute.arch.cp_async_wait_group(0)

    for r in cutlass.range_constexpr(n_rows):
        acc_o_rows[r, None].store(acc_o_rows[r, None].load() / sumexp[r])
    rO = cute.make_fragment_like(acc_o, BFloat16)
    rO.store(acc_o.load().to(BFloat16))
    cute.autovec_copy(rO, thr_mma_pv.partition_C(mOut[b_i, s_i, None, None]))

    if cute.arch.lane_idx() % 4 == 0 and wg_idx == 0:
        for r in cutlass.range_constexpr(n_rows):
            mLse[b_i, s_i, head_of[r]] = cute.math.log2(sumexp[r], fastmath=True) + m_i[r] * sm_scale_log2


@cute.jit
def _fwd(
    mQ: cute.Tensor,
    mKV: cute.Tensor,
    mIndices: cute.Tensor,
    mSinks: cute.Tensor,
    mTileCounts: cute.Tensor,
    mOut: cute.Tensor,
    mLse: cute.Tensor,
    sm_scale: Float32,
    sm_scale_log2: Float32,
    stream,
):
    sQ_layout = _smem_layout((HEADS, DIM))
    sKV_layout = _smem_layout((BLOCK_I, DIM, NUM_STAGES))
    sP_layout = _smem_layout((HEADS, BLOCK_I))
    tiled_mma_qk = sm90_utils.make_trivial_tiled_mma(
        BFloat16,
        BFloat16,
        cute.nvgpu.OperandMajorMode.K,
        cute.nvgpu.OperandMajorMode.K,
        Float32,
        atom_layout_mnk=(1, THREADS // WARPGROUP_THREADS, 1),
        tiler_mn=(HEADS, BLOCK_I // 2),
    )
    tiled_mma_pv = sm90_utils.make_trivial_tiled_mma(
        BFloat16,
        BFloat16,
        cute.nvgpu.OperandMajorMode.K,
        cute.nvgpu.OperandMajorMode.MN,
        Float32,
        atom_layout_mnk=(1, THREADS // WARPGROUP_THREADS, 1),
        tiler_mn=(HEADS, DIM // 2),
    )
    tiled_copy_q = cute.make_tiled_copy_tv(
        cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cute.nvgpu.LoadCacheMode.GLOBAL), BFloat16, num_bits_per_copy=128
        ),
        cute.make_ordered_layout((ROWS_PER_PASS, CHUNKS_PER_ROW), order=(1, 0)),
        cute.make_layout((1, CHUNK)),
    )
    smem_bytes = (
        sum(cute.size_in_bytes(BFloat16, layout) for layout in (sQ_layout, sKV_layout, sP_layout))
        + 2 * cute.size_in_bytes(Float32, cute.make_layout(2 * HEADS))
        + 1024
    )
    _fwd_kernel.set_name_prefix("dsv4_sparse_attn_fwd_cute", remove_cutlass_symbol=True, keep_mangled_name=False)
    _fwd_kernel(
        mQ,
        mKV,
        mIndices,
        mSinks,
        mTileCounts,
        mOut,
        mLse,
        sm_scale,
        sm_scale_log2,
        sQ_layout,
        sKV_layout,
        sP_layout,
        tiled_mma_qk,
        tiled_mma_pv,
        tiled_copy_q,
    ).launch(grid=(mQ.shape[1], mQ.shape[0], 1), block=(THREADS, 1, 1), smem=smem_bytes, stream=stream)


@functools.cache
def _compiled_fwd():
    batch, seq_len, seq_len_kv, n_slots = (cute.sym_int() for _ in range(4))
    row_major_4d = dict(stride_order=(3, 2, 1, 0))
    return cute.compile(
        _fwd,
        make_fake_compact_tensor(BFloat16, (batch, seq_len, HEADS, DIM), **row_major_4d, assumed_align=16),
        make_fake_compact_tensor(BFloat16, (batch, seq_len_kv, 1, DIM), **row_major_4d, assumed_align=16),
        make_fake_compact_tensor(Int32, (batch, seq_len, 1, n_slots), **row_major_4d, assumed_align=4),
        make_fake_compact_tensor(Float32, (HEADS,), assumed_align=4),
        make_fake_compact_tensor(Int32, (batch, seq_len, 1), stride_order=(2, 1, 0), assumed_align=4),
        make_fake_compact_tensor(BFloat16, (batch, seq_len, HEADS, DIM), **row_major_4d, assumed_align=16),
        make_fake_compact_tensor(Float32, (batch, seq_len, HEADS), stride_order=(2, 1, 0), assumed_align=4),
        Float32(1.0),
        Float32(1.0),
        make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def dsv4_sparse_attn_fwd_cute(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sinks: torch.Tensor,
    tile_counts: torch.Tensor,
    sm_scale: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the forward on contiguous inputs whose slot axis is a whole number of 64-slot tiles."""
    batch, seq_len, heads, dim = q.shape
    assert (heads, dim, kv.shape[2]) == (HEADS, DIM, 1), (
        f"the CuTe forward serves {HEADS} heads, head_dim {DIM} and one KV head, got {heads}, {dim}, {kv.shape[2]}"
    )
    assert indices.shape[-1] % BLOCK_I == 0, f"the slot axis must be a multiple of {BLOCK_I}"
    assert q.data_ptr() % 16 == 0 and kv.data_ptr() % 16 == 0, "q and kv must be 16-byte aligned"
    if sm_scale is None:
        sm_scale = dim**-0.5
    out = torch.empty_like(q)
    lse = q.new_empty((batch, seq_len, heads), dtype=torch.float32)
    _compiled_fwd()(q, kv, indices, sinks, tile_counts, out, lse, sm_scale, sm_scale * LOG2E)
    return out, lse
