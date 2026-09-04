# Sparse attention backward kernels for the DeepSeek V4 CSA layers.
# The TileLang scaffolding is vendored from tile-ai/tilelang (Apache 2.0) and modified for dynamic
# shapes. The attention itself differs from tilelang's sparse MLA: there is no score-only channel
# tail (every channel feeds both the score and the output), and Delta is exposed as an output so
# the torch layer can form the per-head attention-sink gradient from it.

# TileLang ships a libcudart stub that proxies to the real CUDA runtime via
# dlsym(RTLD_DEFAULT, ...).  If the stub's own symbols are the first ones found
# (because nothing loaded the real libcudart globally yet), the self-check fails
# and the stub calls abort().  Pre-loading the real library with RTLD_GLOBAL
# ensures dlsym finds it before the stub's own exports.
import ctypes as _ctypes

try:
    _ctypes.CDLL("libcudart.so", mode=_ctypes.RTLD_GLOBAL)
except Exception:
    # This is expected on CPU-only machines
    pass

import tilelang
from tilelang import language as T


@tilelang.jit(out_idx=[-1])
def preprocess(
    H,
    D,
    block_ND=32,
    num_stages=5,
    dtype=T.bfloat16,
    accum_dtype=T.float32,
):
    assert dtype == T.bfloat16
    assert accum_dtype == T.float32
    B = T.dynamic("B")
    S = T.dynamic("S")
    shape = [B, S, H, D]

    @T.prim_func
    def preprocess_kernel(
        O: T.Tensor(shape, dtype),
        dO: T.Tensor(shape, dtype),
        Delta: T.Tensor([B, S, H], accum_dtype),
    ):
        with T.Kernel(H, T.ceildiv(S, block_ND), B) as (bx, by, bz):
            o = T.alloc_fragment([block_ND, block_ND], accum_dtype)
            do = T.alloc_fragment([block_ND, block_ND], accum_dtype)
            delta = T.alloc_fragment([block_ND], accum_dtype)
            acc = T.alloc_fragment([block_ND, block_ND], accum_dtype)
            T.clear(acc)
            for k in T.Pipelined(T.ceildiv(D, block_ND), num_stages=num_stages):
                T.copy(O[bz, by * block_ND : (by + 1) * block_ND, bx, k * block_ND : (k + 1) * block_ND], o)
                T.copy(dO[bz, by * block_ND : (by + 1) * block_ND, bx, k * block_ND : (k + 1) * block_ND], do)
                for i, j in T.Parallel(block_ND, block_ND):
                    acc[i, j] += o[i, j] * do[i, j]
            T.reduce_sum(acc, delta, 1)
            T.copy(delta, Delta[bz, by * block_ND : (by + 1) * block_ND, bx])

    return preprocess_kernel


@tilelang.jit(out_idx=[-1])
def postprocess(
    D,
    kv_group=1,
    block_N=64,
    threads=128,
    dtype=T.bfloat16,
    accum_dtype=T.float32,
):
    assert dtype == T.bfloat16
    assert accum_dtype == T.float32
    B = T.dynamic("B")
    S_kv = T.dynamic("S_kv")
    dkv_shape = [B, S_kv, kv_group, D]

    @T.prim_func
    def postprocess_kernel(
        dKV: T.Tensor(dkv_shape, accum_dtype),
        dKV_out: T.Tensor(dkv_shape, dtype),
    ):
        with T.Kernel(T.ceildiv(S_kv, block_N), kv_group, B, threads=threads) as (bx, by, bz):
            T.copy(
                dKV[bz, bx * block_N : (bx + 1) * block_N, by, :],
                dKV_out[bz, bx * block_N : (bx + 1) * block_N, by, :],
            )

    return postprocess_kernel


@tilelang.jit(
    out_idx=[-2],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        # Avoid TileLang MLA backward NaNs: https://github.com/tile-ai/tilelang/issues/2199
        tilelang.PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: False,
    },
)
def bwd(
    H,
    D,
    topk,
    kv_group=1,
    sm_scale=None,
    is_causal=True,
    block_size=32,
    num_stages=0,
    threads=256,
    indices_dtype=T.int32,
    dtype=T.bfloat16,
    accum_dtype=T.float32,
    n_sentinel=1,
    predicate=False,
):
    assert predicate in (False, "shared", "index"), f"unknown dKV store predicate {predicate!r}"
    assert is_causal is True, "non-casual is not supported now"
    assert topk % block_size == 0, "otherwise will load some index=0 thus causing wrong kv to be loaded"
    assert dtype == T.bfloat16
    assert accum_dtype == T.float32
    assert indices_dtype == T.int32

    if sm_scale is None:
        sm_scale = D ** (-0.5)
    sm_scale_mul_reciprocal_log2 = sm_scale * 1.44269504  # log2(e)

    B = T.dynamic("B")
    S = T.dynamic("S")
    S_kv = T.dynamic("S_kv")

    H_kv = H // kv_group
    q_shape = [B, S, H, D]
    k_shape = [B, S_kv, kv_group, D]
    o_shape = [B, S, H, D]
    indices_shape = [B, S, kv_group, topk]
    delta_shape = [B, S, H]
    lse_shape = [B, S, H]

    H = H_kv
    padded_H = max(tilelang.math.next_power_of_2(H_kv), 16)
    block_H = min(64, padded_H)
    assert padded_H % block_H == 0
    NH = padded_H // block_H
    BS = block_size
    NS = tilelang.cdiv(topk, block_size)

    split_store = 2
    # The acc_dkv accumulator's per-thread layout (from the GEMMs that produce it) is only
    # injective at this chunk granularity. TileLang's LayoutInference pass would reject a smaller
    # chunk if the store loop below were written over its natural extent, but that loop instead
    # iterates the full BS extent and masks with `if bi_i < chunk`, which bypasses the check: an
    # invalid chunk silently emits an aliased address formula and corrupts dKV rather than
    # raising. Verified this holds independent of warp specialization/TMA.
    chunk = BS // split_store
    if chunk < 8 or chunk & (chunk - 1) != 0:
        raise ValueError(
            f"block_size // split_store must be a power of two >= 8, got block_size={BS}, "
            f"split_store={split_store} (chunk={chunk}); see comment above for why this silently "
            "corrupts dKV instead of failing loudly if violated."
        )

    @T.prim_func
    def dsv4_sparse_attn_bwd_kernel(
        Q: T.Tensor(q_shape, dtype),
        KV: T.Tensor(k_shape, dtype),
        dO: T.Tensor(o_shape, dtype),
        Indices: T.Tensor(indices_shape, indices_dtype),
        Lse: T.Tensor(lse_shape, accum_dtype),
        Delta: T.Tensor(delta_shape, accum_dtype),
        dQ: T.Tensor(q_shape, dtype),
        dKV: T.Tensor(k_shape, accum_dtype),
    ):
        with T.Kernel(S, B, kv_group * NH, threads=threads) as (s_i, by, bz):
            Q_shared = T.alloc_shared([block_H, D], dtype)
            KV_shared = T.alloc_shared([BS, D], dtype)
            dO_shared = T.alloc_shared([block_H, D], dtype)
            mask = T.alloc_fragment([BS], "bool")
            # `predicate` picks how the dKV store below skips masked slots, whose atomics add
            # exact zeros. Reading `mask` directly in that loop also compiles, but LayoutInference
            # then pins the loop to the four threads owning those fragment elements instead of
            # spreading it over all `threads`. Both live modes dodge that: "shared" restages the
            # mask in shared memory, "index" re-reads from global the index the store address
            # already needs. Neither constrains the store loop's thread mapping.
            mask_shared = T.alloc_shared([BS], "bool") if predicate == "shared" else None

            P_shared_cast = T.alloc_shared([block_H, BS], dtype)
            dP_shared_cast = T.alloc_shared([block_H, BS], dtype)
            dQ_shared = T.alloc_shared([block_H, D], dtype)

            acc_p = T.alloc_fragment([block_H, BS], accum_dtype)
            acc_dp = T.alloc_fragment([block_H, BS], accum_dtype)
            acc_dq = T.alloc_fragment([block_H, D], accum_dtype)
            acc_dkv = T.alloc_fragment([BS, D], accum_dtype)
            acc_dkv_shared = T.alloc_shared([BS // split_store, D], accum_dtype)

            # See dsv4_sparse_attn_fwd: the last n_sentinel rows are zero KV, valid indices
            # live in [0, S_kv - n_sentinel). Using this single bound makes the kernel work for
            # both full and CP-sharded Q without needing a global Q offset.
            max_kv_i = S_kv - 1 - n_sentinel

            T.copy(Q[by, s_i, bz * block_H : (bz + 1) * block_H, :], Q_shared)
            T.copy(dO[by, s_i, bz * block_H : (bz + 1) * block_H, :], dO_shared)

            T.clear(acc_dq)

            for i_i in T.Pipelined(NS, num_stages=num_stages):
                for bi_i in T.Parallel(BS):
                    mask[bi_i] = Indices[by, s_i, bz // NH, i_i * BS + bi_i] <= max_kv_i
                    if predicate == "shared":
                        mask_shared[bi_i] = mask[bi_i]

                for h_i, bi_i in T.Parallel(block_H, BS):
                    acc_p[h_i, bi_i] = T.if_then_else(mask[bi_i], 0, -T.infinity(acc_p.dtype))

                for bi_i, d_i in T.Parallel(BS, D):
                    KV_shared[bi_i, d_i] = KV[by, Indices[by, s_i, bz // NH, i_i * BS + bi_i], bz // NH, d_i]

                T.gemm(Q_shared, KV_shared, acc_p, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)

                for h_i, bi_i in T.Parallel(block_H, BS):
                    acc_p[h_i, bi_i] = T.exp2(
                        acc_p[h_i, bi_i] * sm_scale_mul_reciprocal_log2 - Lse[by, s_i, bz * block_H + h_i]
                    )

                T.copy(acc_p, P_shared_cast)

                T.gemm(
                    dO_shared, KV_shared, acc_dp, transpose_B=True, policy=T.GemmWarpPolicy.FullCol, clear_accum=True
                )

                for h_i, bi_i in T.Parallel(block_H, BS):
                    acc_dp[h_i, bi_i] = (
                        acc_p[h_i, bi_i] * (acc_dp[h_i, bi_i] - Delta[by, s_i, bz * block_H + h_i]) * sm_scale
                    )

                T.copy(acc_dp, dP_shared_cast)
                T.gemm(dP_shared_cast, KV_shared, acc_dq, policy=T.GemmWarpPolicy.FullCol)

                T.gemm(
                    dP_shared_cast,
                    Q_shared,
                    acc_dkv,
                    transpose_A=True,
                    policy=T.GemmWarpPolicy.FullCol,
                    clear_accum=True,
                )
                T.gemm(P_shared_cast, dO_shared, acc_dkv, transpose_A=True, policy=T.GemmWarpPolicy.FullCol)

                for s in range(split_store):
                    for bi_i, d_i in T.Parallel(BS, D):
                        if bi_i < BS // split_store:
                            acc_dkv_shared[bi_i, d_i] = acc_dkv[bi_i + s * (BS // split_store), d_i]

                    for bi_i, d_i in T.Parallel(BS // split_store, D // 4):
                        slot_i = i_i * BS + bi_i + s * (BS // split_store)
                        if predicate == "shared":
                            store = mask_shared[bi_i + s * (BS // split_store)]
                        elif predicate == "index":
                            store = Indices[by, s_i, bz // NH, slot_i] <= max_kv_i
                        else:
                            store = True
                        if store:
                            T.atomic_addx4(
                                dKV[by, Indices[by, s_i, bz // NH, slot_i], bz // NH, d_i * 4],
                                acc_dkv_shared[bi_i, d_i * 4],
                            )

            T.copy(acc_dq, dQ_shared)

            T.copy(dQ_shared, dQ[by, s_i, bz * block_H : (bz + 1) * block_H, :])

    return dsv4_sparse_attn_bwd_kernel
