"""`dsv4_sparse_attn_fwd` with the slot axis split into a dynamic tile count and a static tile: `Indices` is `(B, S, G, n_tiles, block_I)`."""

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

LOG2E = 1.44269504


@tilelang.jit(
    out_idx=[-2, -1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def dsv4_sparse_attn_fwd_tiled(
    heads,
    dim,
    kv_group=1,
    sm_scale=None,
    is_causal=True,
    block_I=64,
    num_stages=2,
    threads=256,
):
    assert dim == tilelang.math.next_power_of_2(dim), f"haven't check padding correctness yet, dim={dim}"
    assert is_causal is True, "non-casual is not supported"
    if sm_scale is None:
        sm_scale = (1.0 / dim) ** 0.5
    # Both names are kept: the sink logit enters the softmax unscaled, so seeding the running max
    # with it means dividing by the raw scale that every `exp2` site below multiplies back in.
    sm_scale_mul_reciprocal_log2 = sm_scale * LOG2E

    batch = T.dynamic("batch")
    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")
    n_tiles = T.dynamic("n_tiles")

    head_kv = heads // kv_group
    q_shape = [batch, seq_len, heads, dim]
    kv_shape = [batch, seq_len_kv, kv_group, dim]
    o_shape = [batch, seq_len, heads, dim]
    indices_shape = [batch, seq_len, kv_group, n_tiles, block_I]
    sinks_shape = [heads]
    lse_shape = [batch, seq_len, heads]
    indices_dtype = T.int32
    dtype = T.bfloat16
    accum_dtype = T.float32

    H = head_kv
    padded_H = max(tilelang.math.next_power_of_2(head_kv), 16)
    if padded_H != H:
        assert kv_group == 1, (
            "here we solve the H padding automatically, other wise you should handle Q copy and Output copy"
            " with your mask (when kv_group == 1, use g_i * padded_H:(g_i+1) * padded_H would be handled"
            " automatically)"
        )
    BI = block_I
    NI = n_tiles
    D = dim

    if head_kv > 64:
        assert head_kv % 64 == 0, "head_kv should be a multiple of 64"
        REPLICATE_H = head_kv // 64
    else:
        REPLICATE_H = 1

    H_per_block = padded_H if REPLICATE_H == 1 else 64

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),  # type: ignore
        KV: T.Tensor(kv_shape, dtype),  # type: ignore
        Indices: T.Tensor(indices_shape, indices_dtype),  # type: ignore
        Sinks: T.Tensor(sinks_shape, accum_dtype),  # type: ignore
        Output: T.Tensor(o_shape, dtype),  # type: ignore
        Lse: T.Tensor(lse_shape, accum_dtype),  # type: ignore
    ):
        with T.Kernel(seq_len * REPLICATE_H, batch, kv_group, threads=threads) as (
            bx,
            by,
            bz,
        ):
            Q_shared = T.alloc_shared([H_per_block, D], dtype)
            KV_shared = T.alloc_shared([BI, D], dtype)
            mask = T.alloc_fragment([BI], "bool")

            acc_o = T.alloc_fragment([H_per_block, D], accum_dtype)
            acc_s = T.alloc_fragment([H_per_block, BI], accum_dtype)
            S_shared = T.alloc_shared([H_per_block, BI], dtype)
            sumexp = T.alloc_fragment([H_per_block], accum_dtype)
            sumexp_i = T.alloc_fragment([H_per_block], accum_dtype)
            alpha = T.alloc_fragment([H_per_block], accum_dtype)
            m_i = T.alloc_fragment([H_per_block], accum_dtype)
            m_i_prev = T.alloc_fragment([H_per_block], accum_dtype)

            b_i, g_i = by, bz
            s_i = bx if REPLICATE_H == 1 else (bx // REPLICATE_H)
            # A negative index marks an absent key, and it is the only thing this kernel masks
            # on. That preserves causality and varlen masking for both full and CP-sharded Q
            # (where local q_i no longer matches the global K position), because the caller has
            # already resolved all of it into the index values.
            #
            # No clamp guards the gather below. TileLang lowers it to `cp_async_gs_conditional`,
            # whose condition is `0 <= idx < seq_len_kv` and whose `cp.async` src-size operand is
            # 0 when that fails, so PTX zero-fills the shared tile. A masked slot therefore reads
            # as a zero key. The backward relies on the same guard; the sibling
            # `kernels/sparse_mla_{fwd,bwd}.py` still use a trailing zero sentinel row instead.

            H0 = g_i * padded_H + (0 if REPLICATE_H == 1 else (bx % REPLICATE_H) * 64)
            H1 = H0 + H_per_block

            T.fill(acc_o, 0)
            # The online softmax starts from the sink term alone rather than from an empty sum:
            # `m_i` carries raw dot-product units, so `m_i * sm_scale_mul_reciprocal_log2` equals
            # `sink * log2(e)` and the seed `sumexp` of 1 is exactly that term's own exponential.
            # `T.reduce_max(..., clear=False)` then keeps the sink as a floor on the running max.
            # A row whose slots are all masked therefore emits `out = 0` and a finite
            # `Lse = sink * log2(e)`, where a zero-seeded denominator would divide by zero.
            for h_i in T.Parallel(H_per_block):
                sumexp[h_i] = 1.0
            for h_i in T.Parallel(H_per_block):
                m_i[h_i] = Sinks[H0 + h_i] / sm_scale

            T.copy(Q[b_i, s_i, H0:H1, :], Q_shared)

            for i_i in T.Pipelined(NI, num_stages=num_stages):
                for bi_i in T.Parallel(BI):
                    mask[bi_i] = Indices[b_i, s_i, g_i, i_i, bi_i] >= 0

                for bi_i, d_i in T.Parallel(BI, D):
                    KV_shared[bi_i, d_i] = KV[b_i, Indices[b_i, s_i, g_i, i_i, bi_i], g_i, d_i]

                for h_i, bi_i in T.Parallel(H_per_block, BI):
                    acc_s[h_i, bi_i] = T.if_then_else(mask[bi_i], 0, -T.infinity(acc_s.dtype))
                T.gemm(
                    Q_shared,
                    KV_shared,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                T.copy(m_i, m_i_prev)
                T.reduce_max(acc_s, m_i, dim=1, clear=False)
                for h_i in T.Parallel(H_per_block):
                    m_i[h_i] = T.max(m_i[h_i], m_i_prev[h_i])
                for h_i in T.Parallel(H_per_block):
                    alpha[h_i] = T.exp2((m_i_prev[h_i] - m_i[h_i]) * sm_scale_mul_reciprocal_log2)
                for h_i, bi_i in T.Parallel(H_per_block, BI):
                    acc_s[h_i, bi_i] = T.exp2(
                        acc_s[h_i, bi_i] * sm_scale_mul_reciprocal_log2 - m_i[h_i] * sm_scale_mul_reciprocal_log2
                    )
                T.reduce_sum(acc_s, sumexp_i, dim=1)
                for h_i in T.Parallel(H_per_block):
                    sumexp[h_i] = sumexp[h_i] * alpha[h_i] + sumexp_i[h_i]
                for h_i, d_i in T.Parallel(H_per_block, D):
                    acc_o[h_i, d_i] = acc_o[h_i, d_i] * alpha[h_i]

                T.copy(acc_s, S_shared)
                T.gemm(S_shared, KV_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            for h_i, d_i in T.Parallel(H_per_block, D):
                acc_o[h_i, d_i] /= sumexp[h_i]
            for h_i in T.Parallel(H_per_block):
                sumexp[h_i] = T.log2(sumexp[h_i]) + m_i[h_i] * sm_scale_mul_reciprocal_log2

            T.copy(acc_o, Output[b_i, s_i, H0:H1, :])
            T.copy(sumexp, Lse[b_i, s_i, H0:H1])

    return main
