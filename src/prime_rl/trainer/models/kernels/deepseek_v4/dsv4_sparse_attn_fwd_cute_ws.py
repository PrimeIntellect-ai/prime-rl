"""
[DeepSeek V4 Sparse Attention: Forward, warp-specialized CuTe DSL kernel for sm90]

Computes what `dsv4_sparse_attn_fwd.py` computes (its docstring defines the shapes, the math and
the caller's contract) with the structure of FlashMLA's sm90 sparse prefill forward instead of
TileLang's. It serves `heads = 64`, `dim = 512` and one KV head on `sm_90a` only.

One CTA of 384 threads (three warpgroups) per query position:

  - the producer warpgroup gathers keys with `cp.async`, zero-filling masked rows, into two
    64-slot buffers whose channel halves (0-255 and 256-511) each have a "ready" and a "free"
    mbarrier, so a half is refilled as soon as both consumers are done with it. It also publishes
    each slot's validity to shared memory. It keeps 72 registers per thread;
  - consumer warpgroup 0 owns the even tile of each pair of 64-slot tiles and output channels
    0-255, consumer warpgroup 1 the odd tile and channels 256-511; each keeps 216 registers.
    Each computes its own tile's scores `acc_s[h,k] = Q[h,d] K[k,d]` as one `m64n64k16` WGMMA
    chain, and the two agree on the running max through shared memory, warpgroup 0 first;
  - each consumer multiplies its own probabilities from registers into its channel half
    (`m64n256k16`, A in registers), and the other consumer's from shared memory (both operands in
    shared memory), so every probability tile feeds both channel halves;
  - WGMMA is asynchronous: the next pair's score GEMM is issued while the current output GEMM is
    still running, and a key half is released by waiting on all but the newest WGMMA batch.

The online softmax keeps the running max in log2 units, `m = max(logit) * scale * log2(e)`, seeded
with the sink `Sinks[h] * log2(e)`; the running sum of one thread per head (in warpgroup 0) is
seeded with the sink's own term, 1. The slot axis must be a whole number of 128-slot pairs; a
query reads the pairs that cover its `TileCounts` tiles.
"""

import functools

import cutlass
import cutlass.cute as cute
import cutlass.utils.hopper_helpers as sm90_utils
import torch
from cutlass import BFloat16, Float32, Int32
from cutlass.cute.nvgpu import cpasync, warp, warpgroup
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream

from prime_rl.trainer.models.kernels.deepseek_v4.dsv4_sparse_attn_fwd_cute import _acc_rows_view, _smem_layout

LOG2E = 1.44269504

HEADS = 64
DIM = 512
HALF = DIM // 2
BLOCK_I = 64
PAIR = 2 * BLOCK_I
WARPGROUP_THREADS = 128
THREADS = 3 * WARPGROUP_THREADS
CONSUMER_THREADS = 2 * WARPGROUP_THREADS
CONSUMER_REGS = 216
PRODUCER_REGS = 72
CHUNK = 8
GROUP_SIZE = 64 // CHUNK
NUM_GROUPS = WARPGROUP_THREADS // GROUP_SIZE
ROWS_PER_GROUP = BLOCK_I // NUM_GROUPS

BAR_WG0_MAX_READY = 1
BAR_WG1_MAX_READY = 2
BAR_S0_READY = 3
BAR_S1_READY = 4
BAR_SUM_READY = 5
BAR_Q_READY = 6

MBAR_K_READY = 0
MBAR_K_FREE = 4
MBAR_VALID_READY = 8
NUM_MBARS = 9


def _k_half_index(buf: int, half: int) -> int:
    return 2 * buf + half


@cute.jit
def _issue_gemm(
    tiled_mma: cute.TiledMma, acc: cute.Tensor, tA: cute.Tensor, tB: cute.Tensor, zero_init: cutlass.Constexpr
):
    """Issue one WGMMA chain over the K blocks of `tA`/`tB` without committing or waiting."""
    warpgroup.fence()
    mma_atom = cute.make_mma_atom(tiled_mma.op)
    mma_atom.set(warpgroup.Field.ACCUMULATE, not zero_init)
    for k in cutlass.range_constexpr(cute.size(tA, mode=[2])):
        cute.gemm(mma_atom, acc, tA[None, None, k], tB[None, None, k], acc)
        mma_atom.set(warpgroup.Field.ACCUMULATE, True)


@cute.jit
def _mask_scores(rP_rows: cute.Tensor, tScS_rows: cute.Tensor, sValid: cute.Tensor, buf: cutlass.Constexpr):
    for c in cutlass.range_constexpr(cute.size(rP_rows, mode=[1])):
        if sValid[buf, tScS_rows[0, c][1]] == 0:
            for r in cutlass.range_constexpr(cute.size(rP_rows, mode=[0])):
                rP_rows[r, c] = -Float32.inf


@cute.jit
def _online_softmax(
    rP_rows: cute.Tensor,
    rO_rows: cute.Tensor,
    rM: cute.Tensor,
    rL: cute.Tensor,
    old_max: cute.Tensor,
    sM: cute.Tensor,
    head_of: cute.Tensor,
    scale_log2: Float32,
):
    """Fold one tile into the running max and sum, rescale `rO`, and publish the new max to `sM`."""
    lane = cute.arch.lane_idx()
    for r in cutlass.range_constexpr(cute.size(rM)):
        tile_max = rP_rows[r, None].load().reduce(cute.ReductionOp.MAX, -Float32.inf, 0)
        for step in cutlass.range_constexpr(2):
            tile_max = cute.arch.fmax(tile_max, cute.arch.shuffle_sync_bfly(tile_max, offset=1 << step))
        new_max = cute.arch.fmax(old_max[r], tile_max * scale_log2)
        rescale = cute.math.exp2(rM[r] - new_max, fastmath=True)
        rO_rows[r, None].store(rO_rows[r, None].load() * rescale)
        p = cute.math.exp2(rP_rows[r, None].load() * scale_log2 - new_max, fastmath=True)
        rP_rows[r, None].store(p)
        rL[r] = rL[r] * rescale + p.reduce(cute.ReductionOp.ADD, Float32(0.0), 0)
        rM[r] = new_max
        if lane % 4 == 0:
            sM[head_of[r]] = new_max


@cute.jit
def _gather_half(
    mKV: cute.Tensor,
    sK: cute.Tensor,
    gather_atom: cute.CopyAtom,
    b_i: Int32,
    kv_idx: cute.Tensor,
    group: Int32,
    idx_in_group: Int32,
    buf: cutlass.Constexpr,
    half: cutlass.Constexpr,
):
    """Issue the producer's `cp.async` copies of one channel half of one key buffer."""
    seq_len_kv = mKV.shape[1]
    for r in cutlass.range_constexpr(ROWS_PER_GROUP):
        row = r * NUM_GROUPS + group
        idx = kv_idx[buf, r]
        in_range = idx >= 0 and idx < seq_len_kv
        gRow = cute.flat_divide(mKV[b_i, idx if in_range else 0, 0, None], (CHUNK,))
        sRow = cute.flat_divide(sK[row, None, buf], (CHUNK,))
        for tile in cutlass.range_constexpr(HALF // 64):
            chunk = (half * HALF + tile * 64) // CHUNK + idx_in_group
            src = gRow[None, chunk]
            src = cute.make_tensor(
                cute.make_ptr(BFloat16, src.iterator.toint(), cute.AddressSpace.gmem, assumed_align=16),
                cute.group_modes(src.layout, 0, 1),
            )
            dst = cute.group_modes(sRow[None, chunk], 0, 1)
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
    scale_log2: Float32,
    sQ_layout: cute.ComposedLayout,
    sK_layout: cute.ComposedLayout,
    sS_layout: cute.ComposedLayout,
    tiled_mma_qk: cute.TiledMma,
    tiled_mma_pv_rs: cute.TiledMma,
    tiled_mma_pv_ss: cute.TiledMma,
    tiled_copy_q: cute.TiledCopy,
):
    tidx, _, _ = cute.arch.thread_idx()
    s_i, b_i, _ = cute.arch.block_idx()
    wg_idx = cute.arch.make_warp_uniform(tidx // WARPGROUP_THREADS)
    idx_in_wg = tidx % WARPGROUP_THREADS

    smem = cutlass.utils.SmemAllocator()
    sQ = smem.allocate_tensor(BFloat16, sQ_layout.outer, 1024, sQ_layout.inner)
    sK = smem.allocate_tensor(BFloat16, sK_layout.outer, 1024, sK_layout.inner)
    sS = smem.allocate_tensor(BFloat16, sS_layout.outer, 1024, sS_layout.inner)
    sValid = smem.allocate_tensor(Int32, cute.make_layout((2, BLOCK_I), stride=(BLOCK_I, 1)), 16)
    sM = smem.allocate_tensor(Float32, cute.make_layout(HEADS), 16)
    sL = smem.allocate_tensor(Float32, cute.make_layout((2, HEADS), stride=(HEADS, 1)), 16)
    mbars = smem.allocate_array(cutlass.Int64, NUM_MBARS, byte_alignment=8)

    if tidx == 0:
        for i in cutlass.range_constexpr(4):
            cute.arch.mbarrier_init(mbars + MBAR_K_READY + i, WARPGROUP_THREADS)
            cute.arch.mbarrier_init(mbars + MBAR_K_FREE + i, WARPGROUP_THREADS)
        cute.arch.mbarrier_init(mbars + MBAR_VALID_READY, NUM_GROUPS)
        cute.arch.mbarrier_init_fence()
    cute.arch.sync_threads()

    n_tiles = mIndices.shape[3] // BLOCK_I
    tile_count = cutlass.min(mTileCounts[b_i, s_i, 0], n_tiles)
    num_pairs = (tile_count + 1) // 2

    if wg_idx == 2:
        cute.arch.setmaxregister_decrease(PRODUCER_REGS)
        gather_atom = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cute.nvgpu.LoadCacheMode.GLOBAL), BFloat16, num_bits_per_copy=128
        )
        idx_in_group = idx_in_wg % GROUP_SIZE
        group = idx_in_wg // GROUP_SIZE
        kv_idx = cute.make_rmem_tensor(cute.make_layout((2, ROWS_PER_GROUP), stride=(ROWS_PER_GROUP, 1)), Int32)
        free_phase = Int32(1)
        for pair in cutlass.range(num_pairs, unroll=1):
            for buf in cutlass.range_constexpr(2):
                for r in cutlass.range_constexpr(ROWS_PER_GROUP):
                    kv_idx[buf, r] = mIndices[b_i, s_i, 0, (2 * pair + buf) * BLOCK_I + r * NUM_GROUPS + group]
            for buf, half in ((0, 0), (1, 1), (0, 1), (1, 0)):
                cute.arch.mbarrier_wait(mbars + MBAR_K_FREE + _k_half_index(buf, half), free_phase)
                _gather_half(mKV, sK, gather_atom, b_i, kv_idx, group, idx_in_group, buf, half)
                cute.arch.cp_async_mbarrier_arrive_noinc(mbars + MBAR_K_READY + _k_half_index(buf, half))
            if idx_in_group == 0:
                for buf in cutlass.range_constexpr(2):
                    for r in cutlass.range_constexpr(ROWS_PER_GROUP):
                        sValid[buf, r * NUM_GROUPS + group] = Int32(kv_idx[buf, r] >= 0)
                cute.arch.mbarrier_arrive(mbars + MBAR_VALID_READY)
            free_phase ^= 1
    else:
        cute.arch.setmaxregister_increase(CONSUMER_REGS)
        thr_copy_q = tiled_copy_q.get_slice(tidx)
        cute.copy(tiled_copy_q, thr_copy_q.partition_S(mQ[b_i, s_i, None, None]), thr_copy_q.partition_D(sQ))
        cute.arch.cp_async_commit_group()

        sQ_halves = [cute.local_tile(sQ, (HEADS, HALF), (0, half)) for half in range(2)]
        sK_halves = [
            [cute.local_tile(sK[None, None, buf], (BLOCK_I, HALF), (0, half)) for half in range(2)] for buf in range(2)
        ]
        sV_halves = [
            [
                cute.composition(sK_halves[buf][half], cute.make_ordered_layout((HALF, BLOCK_I), order=(1, 0)))
                for half in range(2)
            ]
            for buf in range(2)
        ]

        thr_qk = tiled_mma_qk.get_slice(idx_in_wg)
        tSrQ = [thr_qk.make_fragment_A(thr_qk.partition_A(sQ_halves[half])) for half in range(2)]
        tSrK = [
            [thr_qk.make_fragment_B(thr_qk.partition_B(sK_halves[buf][half])) for half in range(2)] for buf in range(2)
        ]
        rP = cute.make_rmem_tensor(thr_qk.partition_shape_C((HEADS, BLOCK_I)), Float32)
        rP_rows = _acc_rows_view(rP)
        tScS_rows = _acc_rows_view(thr_qk.partition_C(cute.make_identity_tensor((HEADS, BLOCK_I))))

        thr_rs = tiled_mma_pv_rs.get_slice(idx_in_wg)
        thr_ss = tiled_mma_pv_ss.get_slice(idx_in_wg)
        rPb = cute.make_fragment_like(rP, BFloat16)
        tOrP = cute.make_tensor(rPb.iterator, cute.make_layout(thr_rs.partition_shape_A((HEADS, BLOCK_I))))
        tOrV_rs = [
            [thr_rs.make_fragment_B(thr_rs.partition_B(sV_halves[buf][half])) for half in range(2)] for buf in range(2)
        ]
        tOrV_ss = [
            [thr_ss.make_fragment_B(thr_ss.partition_B(sV_halves[buf][half])) for half in range(2)] for buf in range(2)
        ]
        tOrS_ss = [thr_ss.make_fragment_A(thr_ss.partition_A(sS[None, None, buf])) for buf in range(2)]
        rO = cute.make_rmem_tensor(thr_rs.partition_shape_C((HEADS, HALF)), Float32)
        rO.fill(0.0)
        rO_rows = _acc_rows_view(rO)

        smem_copy_p = cute.make_tiled_copy_C(
            cute.make_copy_atom(warp.StMatrix8x8x16bOp(transpose=False, num_matrices=4), BFloat16), tiled_mma_qk
        )
        thr_copy_p = smem_copy_p.get_slice(idx_in_wg)
        tPsP = [thr_copy_p.partition_D(sS[None, None, buf]) for buf in range(2)]

        n_rows = cute.size(rP_rows, mode=[0])
        head_of = cute.make_rmem_tensor((n_rows,), Int32)
        rM = cute.make_rmem_tensor((n_rows,), Float32)
        rL = cute.make_rmem_tensor((n_rows,), Float32)
        old_max = cute.make_rmem_tensor((n_rows,), Float32)
        rescale = cute.make_rmem_tensor((n_rows,), Float32)
        for r in cutlass.range_constexpr(n_rows):
            head_of[r] = tScS_rows[r, 0][0]
            rM[r] = mSinks[head_of[r]] * LOG2E
            rL[r] = Float32(1.0) if wg_idx == 0 and cute.arch.lane_idx() % 4 == 0 else Float32(0.0)

        cute.arch.cp_async_wait_group(0)
        cute.arch.fence_view_async_shared()
        cute.arch.barrier(barrier_id=BAR_Q_READY, number_of_threads=CONSUMER_THREADS)

        ready_phase = Int32(0)
        if wg_idx == 0:
            if num_pairs > 0:
                cute.arch.mbarrier_wait(mbars + MBAR_K_READY + _k_half_index(0, 0), ready_phase)
                _issue_gemm(tiled_mma_qk, rP, tSrQ[0], tSrK[0][0], True)
                cute.arch.mbarrier_wait(mbars + MBAR_K_READY + _k_half_index(0, 1), ready_phase)
                _issue_gemm(tiled_mma_qk, rP, tSrQ[1], tSrK[0][1], False)
                warpgroup.commit_group()
                warpgroup.wait_group(0)
            for pair in cutlass.range(num_pairs, unroll=1):
                cute.arch.mbarrier_wait(mbars + MBAR_VALID_READY, ready_phase)
                _mask_scores(rP_rows, tScS_rows, sValid, 0)
                for r in cutlass.range_constexpr(n_rows):
                    old_max[r] = rM[r]
                _online_softmax(rP_rows, rO_rows, rM, rL, old_max, sM, head_of, scale_log2)
                cute.arch.barrier_arrive(barrier_id=BAR_WG0_MAX_READY, number_of_threads=CONSUMER_THREADS)
                rPb.store(rP.load().to(BFloat16))
                _issue_gemm(tiled_mma_pv_rs, rO, tOrP, tOrV_rs[0][0], False)
                warpgroup.commit_group()
                warpgroup.wait_group(0)
                cute.arch.mbarrier_arrive(mbars + MBAR_K_FREE + _k_half_index(0, 0))

                cute.arch.barrier(barrier_id=BAR_WG1_MAX_READY, number_of_threads=CONSUMER_THREADS)
                for r in cutlass.range_constexpr(n_rows):
                    new_max = sM[head_of[r]]
                    rescale[r] = cute.math.exp2(rM[r] - new_max, fastmath=True)
                    rM[r] = new_max
                    rP_rows[r, None].store(rP_rows[r, None].load() * rescale[r])
                rPb.store(rP.load().to(BFloat16))
                cute.copy(smem_copy_p, thr_copy_p.retile(rPb), tPsP[0])
                cute.arch.fence_view_async_shared()
                cute.arch.barrier_arrive(barrier_id=BAR_S0_READY, number_of_threads=CONSUMER_THREADS)

                cute.arch.barrier(barrier_id=BAR_S1_READY, number_of_threads=CONSUMER_THREADS)
                for r in cutlass.range_constexpr(n_rows):
                    rO_rows[r, None].store(rO_rows[r, None].load() * rescale[r])
                    rL[r] = rL[r] * rescale[r]
                _issue_gemm(tiled_mma_pv_ss, rO, tOrS_ss[1], tOrV_ss[1][0], False)
                warpgroup.commit_group()

                ready_phase ^= 1
                if pair + 1 < num_pairs:
                    cute.arch.mbarrier_wait(mbars + MBAR_K_READY + _k_half_index(0, 0), ready_phase)
                    _issue_gemm(tiled_mma_qk, rP, tSrQ[0], tSrK[0][0], True)
                    warpgroup.commit_group()
                    warpgroup.wait_group(1)
                    cute.arch.mbarrier_arrive(mbars + MBAR_K_FREE + _k_half_index(1, 0))
                    cute.arch.mbarrier_wait(mbars + MBAR_K_READY + _k_half_index(0, 1), ready_phase)
                    _issue_gemm(tiled_mma_qk, rP, tSrQ[1], tSrK[0][1], False)
                    warpgroup.commit_group()
                    warpgroup.wait_group(0)
                else:
                    warpgroup.wait_group(0)
                    cute.arch.mbarrier_arrive(mbars + MBAR_K_FREE + _k_half_index(1, 0))
        else:
            for pair in cutlass.range(num_pairs, unroll=1):
                cute.arch.mbarrier_wait(mbars + MBAR_K_READY + _k_half_index(1, 1), ready_phase)
                _issue_gemm(tiled_mma_qk, rP, tSrQ[1], tSrK[1][1], True)
                cute.arch.mbarrier_wait(mbars + MBAR_K_READY + _k_half_index(1, 0), ready_phase)
                _issue_gemm(tiled_mma_qk, rP, tSrQ[0], tSrK[1][0], False)
                warpgroup.commit_group()
                warpgroup.wait_group(0)

                cute.arch.mbarrier_wait(mbars + MBAR_VALID_READY, ready_phase)
                _mask_scores(rP_rows, tScS_rows, sValid, 1)
                cute.arch.barrier(barrier_id=BAR_WG0_MAX_READY, number_of_threads=CONSUMER_THREADS)
                for r in cutlass.range_constexpr(n_rows):
                    old_max[r] = sM[head_of[r]]
                _online_softmax(rP_rows, rO_rows, rM, rL, old_max, sM, head_of, scale_log2)
                cute.arch.barrier_arrive(barrier_id=BAR_WG1_MAX_READY, number_of_threads=CONSUMER_THREADS)

                rPb.store(rP.load().to(BFloat16))
                _issue_gemm(tiled_mma_pv_rs, rO, tOrP, tOrV_rs[1][1], False)
                warpgroup.commit_group()
                cute.copy(smem_copy_p, thr_copy_p.retile(rPb), tPsP[1])
                cute.arch.barrier(barrier_id=BAR_S0_READY, number_of_threads=CONSUMER_THREADS)
                _issue_gemm(tiled_mma_pv_ss, rO, tOrS_ss[0], tOrV_ss[0][1], False)
                warpgroup.commit_group()
                cute.arch.fence_view_async_shared()
                cute.arch.barrier_arrive(barrier_id=BAR_S1_READY, number_of_threads=CONSUMER_THREADS)

                warpgroup.wait_group(1)
                cute.arch.mbarrier_arrive(mbars + MBAR_K_FREE + _k_half_index(1, 1))
                warpgroup.wait_group(0)
                cute.arch.mbarrier_arrive(mbars + MBAR_K_FREE + _k_half_index(0, 1))
                ready_phase ^= 1

        lane = cute.arch.lane_idx()
        for r in cutlass.range_constexpr(n_rows):
            v = rL[r]
            for step in cutlass.range_constexpr(2):
                v = v + cute.arch.shuffle_sync_bfly(v, offset=1 << step)
            rL[r] = v
            if lane % 4 == 0:
                sL[wg_idx, head_of[r]] = v
        cute.arch.barrier(barrier_id=BAR_SUM_READY, number_of_threads=CONSUMER_THREADS)
        for r in cutlass.range_constexpr(n_rows):
            rL[r] = rL[r] + sL[1 - wg_idx, head_of[r]]
            inv_sum = cute.arch.rcp_approx(rL[r])
            rO_rows[r, None].store(rO_rows[r, None].load() * inv_sum)
        rOb = cute.make_fragment_like(rO, BFloat16)
        rOb.store(rO.load().to(BFloat16))
        gO = cute.local_tile(mOut[b_i, s_i, None, None], (HEADS, HALF), (0, wg_idx))
        cute.autovec_copy(rOb, thr_rs.partition_C(gO))
        if lane % 4 == 0 and wg_idx == 0:
            for r in cutlass.range_constexpr(n_rows):
                mLse[b_i, s_i, head_of[r]] = cute.math.log2(rL[r], fastmath=True) + rM[r]


@cute.jit
def _fwd(
    mQ: cute.Tensor,
    mKV: cute.Tensor,
    mIndices: cute.Tensor,
    mSinks: cute.Tensor,
    mTileCounts: cute.Tensor,
    mOut: cute.Tensor,
    mLse: cute.Tensor,
    scale_log2: Float32,
    stream,
):
    sQ_layout = _smem_layout((HEADS, DIM))
    sK_layout = _smem_layout((BLOCK_I, DIM, 2))
    sS_layout = _smem_layout((HEADS, BLOCK_I, 2))
    tiled_mma_qk = sm90_utils.make_trivial_tiled_mma(
        BFloat16,
        BFloat16,
        cute.nvgpu.OperandMajorMode.K,
        cute.nvgpu.OperandMajorMode.K,
        Float32,
        atom_layout_mnk=(1, 1, 1),
        tiler_mn=(HEADS, BLOCK_I),
    )
    tiled_mma_pv_rs = sm90_utils.make_trivial_tiled_mma(
        BFloat16,
        BFloat16,
        cute.nvgpu.OperandMajorMode.K,
        cute.nvgpu.OperandMajorMode.MN,
        Float32,
        atom_layout_mnk=(1, 1, 1),
        tiler_mn=(HEADS, HALF),
        a_source=warpgroup.OperandSource.RMEM,
    )
    tiled_mma_pv_ss = sm90_utils.make_trivial_tiled_mma(
        BFloat16,
        BFloat16,
        cute.nvgpu.OperandMajorMode.K,
        cute.nvgpu.OperandMajorMode.MN,
        Float32,
        atom_layout_mnk=(1, 1, 1),
        tiler_mn=(HEADS, HALF),
    )
    tiled_copy_q = cute.make_tiled_copy_tv(
        cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cute.nvgpu.LoadCacheMode.GLOBAL), BFloat16, num_bits_per_copy=128
        ),
        cute.make_ordered_layout((CONSUMER_THREADS // (DIM // CHUNK), DIM // CHUNK), order=(1, 0)),
        cute.make_layout((1, CHUNK)),
    )
    smem_bytes = (
        sum(cute.size_in_bytes(BFloat16, layout) for layout in (sQ_layout, sK_layout, sS_layout))
        + 4 * (2 * BLOCK_I + 3 * HEADS)
        + 8 * NUM_MBARS
        + 1024
    )
    _fwd_kernel.set_name_prefix("dsv4_sparse_attn_fwd_cute_ws", remove_cutlass_symbol=True, keep_mangled_name=False)
    _fwd_kernel(
        mQ,
        mKV,
        mIndices,
        mSinks,
        mTileCounts,
        mOut,
        mLse,
        scale_log2,
        sQ_layout,
        sK_layout,
        sS_layout,
        tiled_mma_qk,
        tiled_mma_pv_rs,
        tiled_mma_pv_ss,
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
        make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def dsv4_sparse_attn_fwd_cute_ws(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sinks: torch.Tensor,
    tile_counts: torch.Tensor,
    sm_scale: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the forward on contiguous inputs whose slot axis is a whole number of 128-slot pairs."""
    batch, seq_len, heads, dim = q.shape
    assert (heads, dim, kv.shape[2]) == (HEADS, DIM, 1), (
        f"the CuTe forward serves {HEADS} heads, head_dim {DIM} and one KV head, got {heads}, {dim}, {kv.shape[2]}"
    )
    assert indices.shape[-1] % PAIR == 0, f"the slot axis must be a multiple of {PAIR}"
    assert q.data_ptr() % 16 == 0 and kv.data_ptr() % 16 == 0, "q and kv must be 16-byte aligned"
    if sm_scale is None:
        sm_scale = dim**-0.5
    out = torch.empty_like(q)
    lse = q.new_empty((batch, seq_len, heads), dtype=torch.float32)
    _compiled_fwd()(q, kv, indices, sinks, tile_counts, out, lse, sm_scale * LOG2E)
    return out, lse
