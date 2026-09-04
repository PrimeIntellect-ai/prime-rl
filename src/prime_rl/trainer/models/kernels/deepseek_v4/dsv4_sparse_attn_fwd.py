"""
[DeepSeek V4 Sparse Attention: Forward]

Each query attends to an explicit list of `K` KV positions supplied by the caller, rather than to
a contiguous causal span. This kernel only reads that list; it has no opinion on how the positions
were chosen. Because every query carries a different list, the kernel runs one block per query
position instead of tiling over a block of queries the way dense flash attention does.

Shapes, with the capital of an index letter naming that axis's size:

    Q[b, s, h, d]        (B, S, H, D)   queries, bfloat16
    KV[b, n, g, d]       (B, N, G, D)   keys, bfloat16; V == K here, so one buffer is both
    Indices[b, s, g, k]  (B, S, G, K)   int32 positions into KV's `n` axis
    Sinks[h]             (H,)           float32, one learnable logit per query head
    Output[b, s, h, d]   (B, S, H, D)   bfloat16
    Lse[b, s, h]         (B, S, H)      float32

  - `b`  batch index
  - `s`  query position
  - `n`  KV position
  - `k`  gather slot, one of the `K` a query reads
  - `d`  head dimension
  - `h`  query head index
  - `g`  KV head index, always of size 1; see below

`G` is always 1: every query head reads the same single KV head. The axis survives from the
vendored scaffolding, where `G > 1` would split the `H` query heads into `G` contiguous blocks of
`H / G`, each reading its own KV head. Nothing exercises that path, and the head-padding case
asserts `G == 1` outright, so the equations below fix `g = 0` and drop it. Note that the parameter
is spelled `kv_group` but counts KV heads; the query heads per KV head are `H / G`, spelled
`head_kv`.

[What it computes]

Write `key[b,s,k,d] = KV[b, Indices[b,s,0,k], 0, d]` for the gathered keys. Summing over repeated
indices, and with `scale = D ** -0.5`:

    logit[b,s,h,k]  = scale * Q[b,s,h,d] key[b,s,k,d]
    Z[b,s,h]        = exp(Sinks[h]) + sum_k exp(logit[b,s,h,k])
    p[b,s,h,k]      = exp(logit[b,s,h,k]) / Z[b,s,h]
    Output[b,s,h,d] = p[b,s,h,k] key[b,s,k,d]
    Lse[b,s,h]      = log2(Z[b,s,h])

Note: the sink logit is unscaled; `scale` multiplies the dot products and not `Sinks`. And `Lse`
is a base-2 logarithm of a partition function built from natural exponentials, not `ln Z`; that is
what lets the backward recover `p` as `exp2(logit * log2(e) - Lse)` with a single `exp2`.

The sink contributes to the denominator but owns no key, so `sum_k p[b,s,h,k] < 1` and `Output` is
a shrunken combination of the gathered keys. That is how a head attends to nothing in particular.

[What the caller must guarantee]

`K` is fixed when the kernel compiles and is part of its compilation key, so it is sized for the
most keys any query could need and a given query will often have fewer. Every unused slot must
hold `-1`. A slot is masked when

    Indices[b,s,0,k] < 0

and that is the only masking the kernel performs. It applies no causality test and never compares
`k` against `s`: whatever the caller means by a valid key is encoded entirely in the index values.
That is also what keeps the kernel correct when `s` is a local index that does not correspond to a
global KV position.

Nothing clamps the gather, and nothing needs to. TileLang lowers it to `cp_async_gs_conditional`
under the condition `0 <= Indices < N`, passing a `cp.async` src-size of 0 when that fails, which
PTX zero-fills. A masked slot therefore reads as a zero key rather than reading out of bounds.
That behavior is load-bearing: a fully masked query relies on it to emit exactly `Output = 0`
rather than a product against garbage.

Every other entry must be a real position in `[0, N)`. Nothing checks this at runtime, since
checking would cost the caller a device sync, so an index at or past `N` is masked by the same
guard and silently contributes nothing rather than raising.

[How the loop runs]

The grid is `(query position, batch, KV head)`, so with `G == 1` that is one block per query
position per batch index, holding all `H` query heads of that query at once, or a 64-head chunk
when `H > 64` (`REPLICATE_H`). The block walks the `K` slots in tiles of `block_I`,
software-pipelined `num_stages` deep, and per tile:

  1. gather `KV_shared[k,d] = KV[b, Indices[b,s,0,k], 0, d]` into one shared tile of keys
  2. `acc_s[h,k] = Q_shared[h,d] KV_shared[k,d]`, pre-seeded to `-inf` at masked slots
  3. rescale the running max, the denominator and `acc_o` (online softmax)
  4. `acc_o[h,d] += acc_s[h,k] KV_shared[k,d]`, reusing that same tile as values

Step 4 reusing step 1's tile is the `V == K` property: the gather is paid for once and feeds both
GEMMs.

The online softmax is seeded from the sink rather than from an empty sum. `m_i` is carried in raw
dot-product units, so seeding it with `Sinks[h] / scale` makes `m_i * scale * log2(e)` equal
`Sinks[h] * log2(e)`, and seeding `sumexp` to `1` is exactly that term's own exponential relative
to that max. `T.reduce_max(..., clear=False)` then keeps the sink as a floor on the running max. A
query whose slots are all masked therefore emits `Output = 0` and a finite
`Lse = Sinks[h] * log2(e)`, where a zero-seeded denominator would divide by zero.

The TileLang scaffolding is vendored from tile-ai/tilelang (Apache 2.0) and modified for dynamic
shapes. The attention itself differs from tilelang's sparse MLA: there is no score-only channel
tail (every channel feeds both the score and the output), and the per-head learnable sink logit is
folded into the online softmax as described above.
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

_LOG2E = 1.44269504


@tilelang.jit(
    out_idx=[-2, -1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def dsv4_sparse_attn_fwd(
    heads,
    dim,
    topk,
    kv_group=1,
    sm_scale=None,
    is_causal=True,
    block_I=64,
    num_stages=2,
    threads=256,
):
    assert dim == tilelang.math.next_power_of_2(dim), f"haven't check padding correctness yet, dim={dim}"
    assert is_causal is True, "non-casual is not supported"
    assert topk % block_I == 0, "otherwise will load some index=0 thus causing wrong kv to be loaded"
    if sm_scale is None:
        sm_scale = (1.0 / dim) ** 0.5
    # Both names are kept: the sink logit enters the softmax unscaled, so seeding the running max
    # with it means dividing by the raw scale that every `exp2` site below multiplies back in.
    sm_scale_mul_reciprocal_log2 = sm_scale * _LOG2E

    batch = T.dynamic("batch")
    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")

    head_kv = heads // kv_group
    q_shape = [batch, seq_len, heads, dim]
    kv_shape = [batch, seq_len_kv, kv_group, dim]
    o_shape = [batch, seq_len, heads, dim]
    indices_shape = [batch, seq_len, kv_group, topk]
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
    NI = tilelang.cdiv(topk, block_I)
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
                    mask[bi_i] = Indices[b_i, s_i, g_i, i_i * BI + bi_i] >= 0

                for bi_i, d_i in T.Parallel(BI, D):
                    KV_shared[bi_i, d_i] = KV[b_i, Indices[b_i, s_i, g_i, i_i * BI + bi_i], g_i, d_i]

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
