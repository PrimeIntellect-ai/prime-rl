"""
[DeepSeek V4 Sparse Attention: Backward]

Gradients for the forward in `dsv4_sparse_attn_fwd.py`, whose shapes, index letters and masking
rule carry over unchanged. Three kernels:

    preprocess    Delta, a per-query-head reduction over the channel axis
    bwd           dQ, and dKV scattered into a float32 buffer
    postprocess   casts that buffer back to bfloat16

Beyond the forward's tensors:

    dO[b, s, h, d]   (B, S, H, D)   incoming output gradient, bfloat16
    Delta[b, s, h]   (B, S, H)      float32
    dQ[b, s, h, d]   (B, S, H, D)   bfloat16
    dKV[b, n, g, d]  (B, N, G, D)   float32 until `postprocess`

[What it computes]

With `key[b,s,k,d] = KV[b, Indices[b,s,0,k], 0, d]` and `scale = D ** -0.5` as in the forward:

    Delta[b,s,h] = Output[b,s,h,d] dO[b,s,h,d]
    p[b,s,h,k]   = exp2(scale * Q[b,s,h,d] key[b,s,k,d] * log2(e) - Lse[b,s,h])
    dp[b,s,h,k]  = dO[b,s,h,d] key[b,s,k,d]
    ds[b,s,h,k]  = scale * p[b,s,h,k] * (dp[b,s,h,k] - Delta[b,s,h])

    dQ[b,s,h,d]   = ds[b,s,h,k] key[b,s,k,d]
    dkey[b,s,k,d] = ds[b,s,h,k] Q[b,s,h,d] + p[b,s,h,k] dO[b,s,h,d]

`dkey` has two terms because `V == K`: a gathered position is read once as a key inside the logit
and once as a value inside the output. Per tile of `block_size` slots the block runs five GEMMs,
two rebuilding `p` and `dp`, one for `dQ`, and two accumulating into the `dkey` tile.

`p` is recovered from `Lse` with a single `exp2`, never by a second max pass. The sink needs no
handling here: it is already inside the `Z` that `Lse` encodes, so this `p` is the shrunken
probability the forward produced. No sink gradient is formed in this file; `Delta` is returned so a
caller can build one from it.

[dQ is local, dKV is a scatter]

A block owns one query position, so that query's `dQ` accumulates in registers and is written once.
`dKV` is the opposite: many queries gather the same KV position `n` and they live in different
blocks, so the tile is scattered with `atomic_addx4`, four contiguous channels at a time. That is
why `dKV` is float32 while everything else here is bfloat16, and why `postprocess` exists.

Masked slots skip the store rather than adding their exact zero, and no predicate is written for
it: TileLang already guards the atomic on both bounds, `0 <= idx < N`, and a masked slot's index is
negative.

[How the loop runs]

The grid is `(query position, batch, head block)`, the third axis also carrying the KV head when
`G > 1`, and at `H <= 64` it is a single block covering every query head. The block walks the `K`
slots in tiles of `block_size`, half the forward's tile, and per tile:

  1. gather `KV_shared[k,d]`, and seed `acc_p` to `0` or `-inf` from the mask
  2. `acc_p[h,k] += Q_shared[h,d] KV_shared[k,d]` onto that seed, then `exp2(... - Lse)` in place,
     so a masked slot stays `-inf` and becomes exactly zero. This is `p`.
  3. `acc_dp[h,k] = dO_shared[h,d] KV_shared[k,d]`, then `p * (dp - Delta) * scale` in place: `ds`
  4. `acc_dq[h,d] += ds[h,k] KV_shared[k,d]`
  5. `acc_dkv[k,d] = ds[h,k] Q_shared[h,d] + p[h,k] dO_shared[h,d]`, two GEMMs into one tile
  6. scatter `acc_dkv` into `dKV`, in `split_store` passes through a staging buffer that holds
     `block_size // split_store` slots

`acc_dq` accumulates over every tile and is written once after the loop, while `acc_dkv` is cleared
each tile because step 6 has already sent it to memory. Steps 2 and 3 leave float32 fragments,
which are cast to bfloat16 and staged in shared memory before feeding the GEMMs of steps 4 and 5.

The TileLang scaffolding is vendored from tile-ai/tilelang (Apache 2.0) and modified for dynamic
shapes. As in the forward there is no score-only channel tail, and `Delta` is exposed as an output
rather than kept internal so the caller can form the per-head sink gradient from it.
"""

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
):
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

            P_shared_cast = T.alloc_shared([block_H, BS], dtype)
            dP_shared_cast = T.alloc_shared([block_H, BS], dtype)
            dQ_shared = T.alloc_shared([block_H, D], dtype)

            acc_p = T.alloc_fragment([block_H, BS], accum_dtype)
            acc_dp = T.alloc_fragment([block_H, BS], accum_dtype)
            acc_dq = T.alloc_fragment([block_H, D], accum_dtype)
            acc_dkv = T.alloc_fragment([BS, D], accum_dtype)
            acc_dkv_shared = T.alloc_shared([BS // split_store, D], accum_dtype)

            # See dsv4_sparse_attn_fwd: a negative index marks an absent key and is the only
            # thing masked on, which makes the kernel work for both full and CP-sharded Q
            # without needing a global Q offset. TileLang's guarded gather zero-fills the shared
            # tile for such a slot, so no clamp is needed here either.

            T.copy(Q[by, s_i, bz * block_H : (bz + 1) * block_H, :], Q_shared)
            T.copy(dO[by, s_i, bz * block_H : (bz + 1) * block_H, :], dO_shared)

            T.clear(acc_dq)

            for i_i in T.Pipelined(NS, num_stages=num_stages):
                for bi_i in T.Parallel(BS):
                    mask[bi_i] = Indices[by, s_i, bz // NH, i_i * BS + bi_i] >= 0

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

                    # A masked slot accumulates exactly zero, and issuing its atomic would only
                    # serialize the scatter. No predicate is written here because TileLang already
                    # emits `if (0 <= idx)` and `if (idx < S_kv)` around the store, so a negative
                    # index skips it for free and an out-of-range one cannot write past the end of
                    # dKV. Do not add a predicate back: reading the `mask` fragment here compiles
                    # cleanly but pins this loop to the four threads owning those elements.
                    for bi_i, d_i in T.Parallel(BS // split_store, D // 4):
                        slot_i = i_i * BS + bi_i + s * (BS // split_store)
                        T.atomic_addx4(
                            dKV[by, Indices[by, s_i, bz // NH, slot_i], bz // NH, d_i * 4],
                            acc_dkv_shared[bi_i, d_i * 4],
                        )

            T.copy(acc_dq, dQ_shared)

            T.copy(dQ_shared, dQ[by, s_i, bz * block_H : (bz + 1) * block_H, :])

    return dsv4_sparse_attn_bwd_kernel
