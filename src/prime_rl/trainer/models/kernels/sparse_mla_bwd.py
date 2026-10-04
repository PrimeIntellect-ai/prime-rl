# Vendored from tile-ai/tilelang (Apache 2.0), modified for dynamic shapes.

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

import math

import tilelang
import torch
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
    D_tail,
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
    dkv_shape = [B, S_kv, kv_group, D + D_tail]

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
    D_tail,
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
        sm_scale = (D + D_tail) ** (-0.5)
    sm_scale_mul_reciprocal_log2 = sm_scale * 1.44269504  # log2(e)

    B = T.dynamic("B")
    S = T.dynamic("S")
    S_kv = T.dynamic("S_kv")

    H_kv = H // kv_group
    q_shape = [B, S, H, D + D_tail]
    k_shape = [B, S_kv, kv_group, D + D_tail]
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

    @T.prim_func
    def sparse_mla_bwd_kernel(
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
            Q_tail_shared = T.alloc_shared([block_H, D_tail], dtype)
            KV_shared = T.alloc_shared([BS, D], dtype)
            KV_tail_shared = T.alloc_shared([BS, D_tail], dtype)
            dO_shared = T.alloc_shared([block_H, D], dtype)
            mask = T.alloc_fragment([BS], "bool")

            P_shared_cast = T.alloc_shared([block_H, BS], dtype)
            dP_shared_cast = T.alloc_shared([block_H, BS], dtype)
            dQ_shared = T.alloc_shared([block_H, D], dtype)
            dQ_tail_shared = T.alloc_shared([block_H, D_tail], dtype)

            acc_p = T.alloc_fragment([block_H, BS], accum_dtype)
            acc_dp = T.alloc_fragment([block_H, BS], accum_dtype)
            acc_dq = T.alloc_fragment([block_H, D], accum_dtype)
            acc_dq_tail = T.alloc_fragment([block_H, D_tail], accum_dtype)
            acc_dkv = T.alloc_fragment([BS, D], accum_dtype)
            acc_dkv_tail = T.alloc_fragment([BS, D_tail], accum_dtype)
            acc_dkv_shared = T.alloc_shared([BS // split_store, D], accum_dtype)
            acc_dkv_tail_shared = T.alloc_shared([BS // split_store, D_tail], accum_dtype)

            # See sparse_mla_fwd: sentinel is at index S_kv - 1 (zero KV), valid indices
            # live in [0, S_kv - 1). Using this single bound makes the kernel work for
            # both full and CP-sharded Q without needing a global Q offset.
            max_kv_i = S_kv - 2

            T.copy(Q[by, s_i, bz * block_H : (bz + 1) * block_H, :D], Q_shared)
            T.copy(Q[by, s_i, bz * block_H : (bz + 1) * block_H, D:], Q_tail_shared)
            T.copy(dO[by, s_i, bz * block_H : (bz + 1) * block_H, :D], dO_shared)

            T.clear(acc_dq)
            T.clear(acc_dq_tail)

            for i_i in T.Pipelined(NS, num_stages=num_stages):
                for bi_i in T.Parallel(BS):
                    mask[bi_i] = Indices[by, s_i, bz // NH, i_i * BS + bi_i] <= max_kv_i

                for h_i, bi_i in T.Parallel(block_H, BS):
                    acc_p[h_i, bi_i] = T.if_then_else(mask[bi_i], 0, -T.infinity(acc_p.dtype))

                for bi_i, d_i in T.Parallel(BS, D):
                    KV_shared[bi_i, d_i] = KV[by, Indices[by, s_i, bz // NH, i_i * BS + bi_i], bz // NH, d_i]

                T.gemm(Q_shared, KV_shared, acc_p, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)

                for bi_i, d_i in T.Parallel(BS, D_tail):
                    KV_tail_shared[bi_i, d_i] = KV[by, Indices[by, s_i, bz // NH, i_i * BS + bi_i], bz // NH, D + d_i]
                T.gemm(Q_tail_shared, KV_tail_shared, acc_p, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)

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
                T.gemm(dP_shared_cast, KV_tail_shared, acc_dq_tail, policy=T.GemmWarpPolicy.FullCol)

                T.gemm(
                    dP_shared_cast,
                    Q_shared,
                    acc_dkv,
                    transpose_A=True,
                    policy=T.GemmWarpPolicy.FullCol,
                    clear_accum=True,
                )
                T.gemm(P_shared_cast, dO_shared, acc_dkv, transpose_A=True, policy=T.GemmWarpPolicy.FullCol)

                T.clear(acc_dkv_tail)
                T.gemm(dP_shared_cast, Q_tail_shared, acc_dkv_tail, transpose_A=True, policy=T.GemmWarpPolicy.FullCol)

                for s in range(split_store):
                    for bi_i, d_i in T.Parallel(BS, D):
                        if bi_i < BS // split_store:
                            acc_dkv_shared[bi_i, d_i] = acc_dkv[bi_i + s * (BS // split_store), d_i]

                    for bi_i, d_i in T.Parallel(BS, D_tail):
                        if bi_i < BS // split_store:
                            acc_dkv_tail_shared[bi_i, d_i] = acc_dkv_tail[bi_i + s * (BS // split_store), d_i]

                    for bi_i, d_i in T.Parallel(BS // split_store, D // 4):
                        T.atomic_addx4(
                            dKV[
                                by,
                                Indices[by, s_i, bz // NH, i_i * BS + bi_i + s * (BS // split_store)],
                                bz // NH,
                                d_i * 4,
                            ],
                            acc_dkv_shared[bi_i, d_i * 4],
                        )

                    for bi_i, d_i in T.Parallel(BS // split_store, D_tail // 4):
                        T.atomic_addx4(
                            dKV[
                                by,
                                Indices[by, s_i, bz // NH, i_i * BS + bi_i + s * (BS // split_store)],
                                bz // NH,
                                D + d_i * 4,
                            ],
                            acc_dkv_tail_shared[bi_i, d_i * 4],
                        )

            T.copy(acc_dq, dQ_shared)
            T.copy(acc_dq_tail, dQ_tail_shared)

            T.copy(dQ_shared, dQ[by, s_i, bz * block_H : (bz + 1) * block_H, :D])
            T.copy(dQ_tail_shared, dQ[by, s_i, bz * block_H : (bz + 1) * block_H, D:])

    return sparse_mla_bwd_kernel


DSA_BACKENDS = ("tilelang", "cudnn_flashmla", "tilelang_legacy")

_CUDNN_SM100_CAPABILITIES = ((10, 0), (10, 3))


def cudnn_backward_arch(q: torch.Tensor, kv: torch.Tensor) -> str | None:
    """The cuDNN DSA backward variant for this device ("sm90" / "sm100"), or None if unsupported."""
    if kv.shape[2] != 1:
        return None
    capability = torch.cuda.get_device_capability(q.device)
    if capability[0] == 9:
        return "sm90"
    if capability in _CUDNN_SM100_CAPABILITIES:
        return "sm100"
    return None


def flat_kv_indices(indices: torch.Tensor, S_kv: int) -> torch.Tensor:
    """Global indices into the batch-flattened KV, with sentinel / out-of-range indices as -1.

    `indices` is [B, S, kv_group, topk]; valid entries live in [0, S_kv - 1) (see sparse_mla_fwd).
    """
    B = indices.shape[0]
    batch_offset = torch.arange(B, device=indices.device, dtype=indices.dtype).view(B, 1, 1, 1) * S_kv
    return torch.where(indices <= S_kv - 2, indices + batch_offset, -1)


def tilelang_sparse_mla_backward(
    q: torch.Tensor,
    kv: torch.Tensor,
    out: torch.Tensor,
    grad_out: torch.Tensor,
    indices: torch.Tensor,
    lse: torch.Tensor,
    sm_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """TileLang backward; takes a natural-log LSE."""
    grad_out = grad_out.contiguous()
    H, dim_plus_tail_dim = q.shape[2:]
    kv_group = kv.shape[2]
    D = 512
    D_tail = dim_plus_tail_dim - D
    topk = indices.shape[-1]

    preprocess_kernel = preprocess(H, D)
    bwd_kernel = bwd(H, D, D_tail, topk, kv_group, sm_scale, True)
    postprocess_kernel = postprocess(D, D_tail, kv_group)

    delta = preprocess_kernel(out, grad_out)
    dkv = torch.zeros_like(kv, dtype=torch.float32)
    dq = bwd_kernel(q, kv, grad_out, indices, lse * math.log2(math.e), delta, dkv)
    dkv = postprocess_kernel(dkv)
    return dq, dkv


def cudnn_sparse_mla_backward(
    q: torch.Tensor,
    kv: torch.Tensor,
    out: torch.Tensor,
    grad_out: torch.Tensor,
    indices: torch.Tensor,
    lse: torch.Tensor,
    sm_scale: float | None,
    arch: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """cuDNN-frontend CuTe-DSL DSA backward (single KV head with K=V).

    The cuDNN kernels take flat (tokens, ...) tensors with global indices, skip
    negative indices (no KV load, no dKV atomics) and expect a natural-log LSE.
    """
    B, S, H, D_qk = q.shape
    S_kv = kv.shape[1]
    topk = indices.shape[-1]
    args = (
        q.view(B * S, H, D_qk),
        kv.view(B * S_kv, D_qk),
        out.view(B * S, H, -1),
        grad_out.contiguous().view(B * S, H, -1),
        lse.view(B * S, H),
    )
    flat_indices = flat_kv_indices(indices, S_kv).view(B * S, topk)
    if arch == "sm90":
        from cudnn.deepseek_sparse_attention.sparse_attention_backward._interface_sm90 import flash_attn_bwd_sm90

        dq, dkv = flash_attn_bwd_sm90(*args, softmax_scale=sm_scale, topk_idxs=flat_indices)
    else:
        # Untested: no SM100 hardware has run this path. A -inf sink leaves the softmax unchanged.
        from cudnn.deepseek_sparse_attention.sparse_attention_backward._interface_sm100 import flash_attn_bwd_sm100

        no_sink = torch.full((H,), float("-inf"), dtype=torch.float32, device=q.device)
        dq, dkv, _ = flash_attn_bwd_sm100(*args, no_sink, flat_indices, softmax_scale=sm_scale)
    return dq.view_as(q), dkv.view_as(kv)


@torch.library.custom_op("prime_rl::sparse_mla_backward", mutates_args=())
def sparse_mla_backward(
    q: torch.Tensor,
    kv: torch.Tensor,
    out: torch.Tensor,
    grad_out: torch.Tensor,
    indices: torch.Tensor,
    lse: torch.Tensor,
    sm_scale: float | None = None,
    backend: str = "tilelang",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sparse MLA backward for the `backend` route (see `sparse_mla`); `lse` is natural-log."""
    assert q.is_contiguous()
    assert kv.is_contiguous()
    assert indices.is_contiguous()
    assert lse.is_contiguous()
    B, S, H, dim_plus_tail_dim = q.shape
    _, S_kv, kv_group, _ = kv.shape
    assert kv.shape[-1] == dim_plus_tail_dim
    assert kv.shape[0] == B
    topk = indices.shape[-1]
    assert indices.shape == (B, S, kv_group, topk)
    assert lse.shape == (B, S, H)
    assert backend in DSA_BACKENDS, f"Unknown DSA backend: {backend}"

    arch = cudnn_backward_arch(q, kv)
    if backend == "cudnn_flashmla":
        assert arch is not None, "dsa_backend='cudnn_flashmla' requires an SM90/SM100/SM103 GPU and a single KV head"
        return cudnn_sparse_mla_backward(q, kv, out, grad_out, indices, lse, sm_scale, arch)
    if backend == "tilelang" and arch == "sm90":
        return cudnn_sparse_mla_backward(q, kv, out, grad_out, indices, lse, sm_scale, arch)
    return tilelang_sparse_mla_backward(q, kv, out, grad_out, indices, lse, sm_scale)


@sparse_mla_backward.register_fake
def _sparse_mla_backward_fake(
    q: torch.Tensor,
    kv: torch.Tensor,
    out: torch.Tensor,
    grad_out: torch.Tensor,
    indices: torch.Tensor,
    lse: torch.Tensor,
    sm_scale: float | None = None,
    backend: str = "tilelang",
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(q), torch.empty_like(kv)
