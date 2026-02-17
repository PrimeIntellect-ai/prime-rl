"""Benchmark: Triton BF16 / Triton FP8 / TileLang FP8 indexers.

All implement the correct DSA scoring: I_{t,s} = Σ_j w_{t,j} · ReLU(q_{t,j} · k_s)

Triton BF16:  fused 2D-grid kernel, BF16 tensor cores, causal mask built-in
Triton FP8:   same 2D-grid structure, FP8 tensor cores, k_scales applied after ReLU
TileLang FP8: 1D-grid kernel with FP8 GEMM + separate clean_logits pass

Run:    uv run python benchmarks/bench_fp8_indexer.py
Output: benchmarks/results/bench_fp8_indexer.md
"""

import gc
import platform
import time
from pathlib import Path

import tilelang
import torch
import triton
import triton.language as tl
from tilelang import language as T
from tilelang.profiler import do_bench

# ---------------------------------------------------------------------------
# Config defaults (from configuration_glm_moe_dsa.py)
# ---------------------------------------------------------------------------
H = 32  # index_n_heads
D = 128  # index_head_dim
TOPK = 2048
WARMUP = 5
REP = 20

SEQ_LENS = [4096, 8192, 16384, 32768, 65536, 131072]
PROFILES = {
    "single_seq": lambda S: [0, S],
    "uniform_2048": lambda S: list(range(0, S + 1, 2048)),
    "uniform_4096": lambda S: list(range(0, S + 1, 4096)),
    "uniform_8192": lambda S: list(range(0, S + 1, 8192)),
    "uniform_16384": lambda S: list(range(0, S + 1, 16384)),
}

DEVICE = "cuda"
RESULTS_DIR = Path(__file__).parent / "results"


# ===========================================================================
# Helpers
# ===========================================================================


def make_cu_seqlens(profile_fn, S):
    boundaries = profile_fn(S)
    return torch.tensor(boundaries, dtype=torch.int32, device=DEVICE)


def cu_seqlens_to_per_token(cu_seqlens, S):
    """Convert cumulative seqlens to per-token (ks, ke).

    ks[p] = sequence start for token p
    ke[p] = p + 1  (causal: each token attends up to itself)
    """
    ks = torch.empty(S, dtype=torch.int32, device=DEVICE)
    ke = torch.arange(1, S + 1, dtype=torch.int32, device=DEVICE)
    num_seqs = cu_seqlens.shape[0] - 1
    for i in range(num_seqs):
        start = cu_seqlens[i].item()
        end = cu_seqlens[i + 1].item()
        ks[start:end] = start
    return ks, ke


def generate_inputs(S):
    q = torch.randn(S, H, D, dtype=torch.bfloat16, device=DEVICE)
    k = torch.randn(S, D, dtype=torch.bfloat16, device=DEVICE)
    w = torch.randn(S, H, dtype=torch.bfloat16, device=DEVICE)
    return q, k, w


# ---------------------------------------------------------------------------
# FP8 quantization (from tilelang examples/deepseek_v32/utils.py)
# ---------------------------------------------------------------------------
FP8_MAX = 448.0


def per_custom_dims_cast_to_fp8(x, dims, use_ue8m0=False):
    excluded_dims = tuple(i for i in range(x.dim()) if i not in set(dims))
    x_amax = x.abs().float().amax(dim=excluded_dims, keepdim=True).clamp(1e-4)
    sf = x_amax / FP8_MAX
    if use_ue8m0:
        sf = torch.pow(2.0, torch.ceil(torch.log2(sf.abs())))
    x_scaled = (x * (1.0 / sf)).to(torch.float8_e4m3fn)
    return x_scaled, sf.squeeze()


# ===========================================================================
# Triton BF16 indexer
# ===========================================================================


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=3),
    ],
    key=["S"],
)
@triton.jit
def _triton_mqa_indexer_kernel(
    Q,
    K,
    W,
    Out,
    KS,
    KE,
    S,
    stride_qs,
    stride_qh,
    stride_ks,
    stride_ws,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)

    mask_m = offs_m < S
    mask_n = offs_n < S

    # Causal bounds for early exit
    ks_vals = tl.load(KS + offs_m, mask=mask_m, other=S)
    ke_vals = tl.load(KE + offs_m, mask=mask_m, other=0)

    # Use int64 for output pointers to avoid overflow when S >= 32768
    offs_m_64 = offs_m.to(tl.int64)
    offs_n_64 = offs_n.to(tl.int64)
    out_ptrs = Out + offs_m_64[:, None] * S + offs_n_64[None, :]
    out_mask = mask_m[:, None] & mask_n[None, :]

    ks_min = tl.min(ks_vals)
    ke_max = tl.max(ke_vals)
    n_lo = pid_n * BLOCK_N

    # Skip blocks entirely outside the causal window
    if n_lo >= ke_max or n_lo + BLOCK_N <= ks_min:
        tl.store(out_ptrs, tl.full([BLOCK_M, BLOCK_N], float("-inf"), tl.float32), mask=out_mask)
        return

    # Load K block once: [BLOCK_N, D]
    k_block = tl.load(
        K + offs_n[:, None] * stride_ks + offs_d[None, :],
        mask=mask_n[:, None],
        other=0.0,
    )

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for h in range(H):
        # Q[m, h, :]: [BLOCK_M, D]
        q_h = tl.load(
            Q + offs_m[:, None] * stride_qs + h * stride_qh + offs_d[None, :],
            mask=mask_m[:, None],
            other=0.0,
        )
        # W[m, h]: [BLOCK_M]
        w_h = tl.load(W + offs_m * stride_ws + h, mask=mask_m, other=0.0).to(tl.float32)

        # [BLOCK_M, D] @ [D, BLOCK_N] -> [BLOCK_M, BLOCK_N]
        scores = tl.dot(q_h, tl.trans(k_block))
        scores = tl.maximum(scores, 0.0)
        acc += scores * w_h[:, None]

    # Causal + sequence boundary mask
    valid = (offs_n[None, :] >= ks_vals[:, None]) & (offs_n[None, :] < ke_vals[:, None])
    acc = tl.where(valid, acc, float("-inf"))

    tl.store(out_ptrs, acc, mask=out_mask)


def triton_indexer(q, k, w, ks, ke, topk):
    """Triton BF16 indexer: fused MQA scoring + causal mask → topk."""
    S = q.shape[0]
    logits = torch.empty(S, S, dtype=torch.float32, device=DEVICE)

    grid = lambda meta: (
        triton.cdiv(S, meta["BLOCK_M"]),
        triton.cdiv(S, meta["BLOCK_N"]),
    )
    _triton_mqa_indexer_kernel[grid](
        q,
        k,
        w,
        logits,
        ks,
        ke,
        S,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        w.stride(0),
        H=H,
        D=D,
    )

    actual_topk = min(topk, S)
    _, indices = torch.topk(logits, actual_topk, dim=-1)
    if actual_topk < topk:
        padding = torch.full((S, topk - actual_topk), S, dtype=indices.dtype, device=indices.device)
        indices = torch.cat([indices, padding], dim=-1)
    return indices.to(torch.int32)


# ===========================================================================
# Triton FP8 indexer
# ===========================================================================


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=3),
    ],
    key=["S"],
)
@triton.jit
def _triton_fp8_indexer_kernel(
    Q_fp8,
    K_fp8,
    K_scales,
    W,
    Out,
    KS,
    KE,
    S,
    stride_qh,
    stride_qs,  # Q_fp8 is [H, S, D]: stride_qh = S*D, stride_qs = D
    stride_ks,
    stride_ws,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)

    mask_m = offs_m < S
    mask_n = offs_n < S

    ks_vals = tl.load(KS + offs_m, mask=mask_m, other=S)
    ke_vals = tl.load(KE + offs_m, mask=mask_m, other=0)

    offs_m_64 = offs_m.to(tl.int64)
    offs_n_64 = offs_n.to(tl.int64)
    out_ptrs = Out + offs_m_64[:, None] * S + offs_n_64[None, :]
    out_mask = mask_m[:, None] & mask_n[None, :]

    ks_min = tl.min(ks_vals)
    ke_max = tl.max(ke_vals)
    n_lo = pid_n * BLOCK_N

    if n_lo >= ke_max or n_lo + BLOCK_N <= ks_min:
        tl.store(out_ptrs, tl.full([BLOCK_M, BLOCK_N], float("-inf"), tl.float32), mask=out_mask)
        return

    # Clamp indices for FP8 loads (can't use mask+other with fp8 dtype).
    # Out-of-bounds positions are masked out by the causal check below.
    offs_n_safe = tl.minimum(offs_n, S - 1)
    offs_m_safe = tl.minimum(offs_m, S - 1)

    # Load K_fp8 block: [BLOCK_N, D]
    k_block = tl.load(K_fp8 + offs_n_safe[:, None] * stride_ks + offs_d[None, :])

    # Load K_scales: [BLOCK_N]
    k_sc = tl.load(K_scales + offs_n_safe, mask=mask_n, other=0.0).to(tl.float32)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for h in range(H):
        # Q_fp8[h, m, :]: [BLOCK_M, D]
        q_h = tl.load(Q_fp8 + h * stride_qh + offs_m_safe[:, None] * stride_qs + offs_d[None, :])
        w_h = tl.load(W + offs_m * stride_ws + h, mask=mask_m, other=0.0).to(tl.float32)

        # FP8 matmul: [BLOCK_M, D] @ [D, BLOCK_N] -> [BLOCK_M, BLOCK_N]
        scores = tl.dot(q_h, tl.trans(k_block)).to(tl.float32)
        # ReLU then apply weight and k_scale (matching TileLang order)
        scores = tl.maximum(scores, 0.0) * k_sc[None, :]
        acc += scores * w_h[:, None]

    valid = (offs_n[None, :] >= ks_vals[:, None]) & (offs_n[None, :] < ke_vals[:, None])
    acc = tl.where(valid, acc, float("-inf"))

    tl.store(out_ptrs, acc, mask=out_mask)


def triton_fp8_indexer(q, k, w, ks, ke, topk):
    """Triton FP8 indexer: FP8 cast + fused kernel + topk."""
    S = q.shape[0]

    # FP8 quantize (same as TileLang): per-row absmax scaling
    q_flat = q.reshape(S * H, D).contiguous()
    q_fp8, _ = per_custom_dims_cast_to_fp8(q_flat, dims=(0,))
    k_fp8, k_scales = per_custom_dims_cast_to_fp8(k, dims=(0,))

    # Rearrange Q_fp8 from [S*H, D] to [H, S, D] for contiguous per-head access
    q_fp8 = q_fp8.view(S, H, D).permute(1, 0, 2).contiguous()

    logits = torch.empty(S, S, dtype=torch.float32, device=DEVICE)

    grid = lambda meta: (
        triton.cdiv(S, meta["BLOCK_M"]),
        triton.cdiv(S, meta["BLOCK_N"]),
    )
    _triton_fp8_indexer_kernel[grid](
        q_fp8,
        k_fp8,
        k_scales,
        w,
        logits,
        ks,
        ke,
        S,
        q_fp8.stride(0),
        q_fp8.stride(1),
        k_fp8.stride(0),
        w.stride(0),
        H=H,
        D=D,
    )

    actual_topk = min(topk, S)
    _, indices = torch.topk(logits, actual_topk, dim=-1)
    if actual_topk < topk:
        padding = torch.full((S, topk - actual_topk), S, dtype=indices.dtype, device=indices.device)
        indices = torch.cat([indices, padding], dim=-1)
    return indices.to(torch.int32)


# ===========================================================================
# TileLang FP8 lightning indexer kernels
# (vendored from tilelang examples/deepseek_v32/fp8_lighting_indexer.py)
# ===========================================================================


@tilelang.jit(
    pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
)
def mqa_attn_return_logits(
    heads,
    index_dim,
    block_N=256,
    num_stages=3,
    threads=512,
    block_Q=None,
):
    if block_Q is None:
        block_Q = 128 // heads
    dtype = T.float8_e4m3fn
    accum_dtype = T.float32
    index_dtype = T.int32

    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")

    @T.prim_func
    def mqa_attn_return_logits_kernel(
        IndexQ: T.Tensor([seq_len * heads, index_dim], dtype),
        IndexK: T.Tensor([seq_len_kv, index_dim], dtype),
        IndexKScale: T.Tensor([seq_len_kv], accum_dtype),
        Logits: T.Tensor([seq_len, seq_len_kv], accum_dtype),
        Weights: T.Tensor([seq_len, heads], accum_dtype),
        CuSeqLenKS: T.Tensor([seq_len], index_dtype),
        CuSeqLenKE: T.Tensor([seq_len], index_dtype),
    ):
        with T.Kernel(T.ceildiv(seq_len, block_Q), threads=threads) as bx:
            index_q_shared = T.alloc_shared([block_Q * heads, index_dim], dtype)
            index_k_shared = T.alloc_shared([block_N, index_dim], dtype)
            index_k_scale_fragment = T.alloc_fragment([block_N], accum_dtype)
            s = T.alloc_fragment([block_N, block_Q * heads], accum_dtype)
            s_reshaped = T.reshape(s, (block_N, block_Q, heads))
            logits_frag = T.alloc_fragment([block_N, block_Q], accum_dtype)
            weights = T.alloc_fragment([block_Q, heads], accum_dtype)

            seq_len_i = bx * block_Q
            cu_k_s_min = T.alloc_var(index_dtype)
            cu_k_e_max = T.alloc_var(index_dtype)
            cu_k_s_min = 2147483647
            cu_k_e_max = -2147483648

            for bq_i in T.serial(block_Q):
                cu_k_s_min = T.min(cu_k_s_min, T.min(CuSeqLenKS[seq_len_i + bq_i], seq_len_kv))
            for bq_i in T.serial(block_Q):
                cu_k_e_max = T.max(cu_k_e_max, T.min(CuSeqLenKE[seq_len_i + bq_i], seq_len_kv))

            T.copy(IndexQ[seq_len_i * heads, 0], index_q_shared)
            T.copy(Weights[seq_len_i, 0], weights)

            for nbn_i in T.Pipelined(T.ceildiv(cu_k_e_max - cu_k_s_min, block_N), num_stages=num_stages):
                T.copy(IndexK[cu_k_s_min + nbn_i * block_N, 0], index_k_shared)
                T.copy(IndexKScale[cu_k_s_min + nbn_i * block_N], index_k_scale_fragment)

                T.gemm(
                    index_k_shared,
                    index_q_shared,
                    s,
                    transpose_B=True,
                    clear_accum=True,
                    policy=T.GemmWarpPolicy.FullCol,
                )

                for bn_i, bq_i, h_i in T.Parallel(block_N, block_Q, heads):
                    s_reshaped[bn_i, bq_i, h_i] = (
                        T.max(s_reshaped[bn_i, bq_i, h_i], 0) * weights[bq_i, h_i]
                    ) * index_k_scale_fragment[bn_i]

                T.reduce_sum(s_reshaped, logits_frag, dim=-1, clear=True)

                for bq_i, bn_i in T.Parallel(block_Q, block_N):
                    Logits[seq_len_i + bq_i, cu_k_s_min + nbn_i * block_N + bn_i] = logits_frag[bn_i, bq_i]

    return mqa_attn_return_logits_kernel


@tilelang.jit
def clean_logits_(threads=512, block_K=4096):
    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")
    dtype = T.float32
    indices_dtype = T.int32

    @T.prim_func
    def clean_logits_kernel(
        Logits: T.Tensor([seq_len, seq_len_kv], dtype),
        CuSeqLenKS: T.Tensor([seq_len], indices_dtype),
        CuSeqLenKE: T.Tensor([seq_len], indices_dtype),
    ):
        with T.Kernel(seq_len, threads=threads) as bx:
            tx = T.thread_binding(0, threads, thread="threadIdx.x")
            cu_k_s = CuSeqLenKS[bx]
            cu_k_e = CuSeqLenKE[bx]
            for n_i in T.Pipelined(T.ceildiv(seq_len_kv, block_K)):
                for k_i in T.serial(block_K // threads):
                    idx = n_i * block_K + k_i * threads + tx
                    if idx < cu_k_s or idx >= cu_k_e:
                        Logits[bx, idx] = -T.infinity(dtype)

    return clean_logits_kernel


def _build_tilelang_kernels():
    kernel_logits = mqa_attn_return_logits(heads=H, index_dim=D)
    kernel_clean = clean_logits_()
    return kernel_logits, kernel_clean


def tilelang_indexer(q, k, w, ks, ke, topk, kernel_logits, kernel_clean):
    """TileLang FP8 indexer: FP8 cast + fused kernel + clean + topk."""
    S = q.shape[0]
    q_flat = q.reshape(S * H, D).contiguous()
    q_fp8, _ = per_custom_dims_cast_to_fp8(q_flat, dims=(0,))
    k_fp8, k_scales = per_custom_dims_cast_to_fp8(k, dims=(0,))

    logits = torch.empty(S, S, dtype=torch.float32, device=DEVICE)
    kernel_logits(q_fp8, k_fp8, k_scales, logits, w.float(), ks, ke)
    kernel_clean(logits, ks, ke)

    actual_topk = min(topk, S)
    _, indices = torch.topk(logits, actual_topk, dim=-1)
    if actual_topk < topk:
        padding = torch.full((S, topk - actual_topk), S, dtype=indices.dtype, device=indices.device)
        indices = torch.cat([indices, padding], dim=-1)
    return indices.to(torch.int32)


# ===========================================================================
# Correctness metrics
# ===========================================================================


def _compute_logits_triton_bf16(q, k, w, ks, ke):
    S = q.shape[0]
    logits = torch.empty(S, S, dtype=torch.float32, device=DEVICE)
    grid = lambda meta: (
        triton.cdiv(S, meta["BLOCK_M"]),
        triton.cdiv(S, meta["BLOCK_N"]),
    )
    _triton_mqa_indexer_kernel[grid](
        q,
        k,
        w,
        logits,
        ks,
        ke,
        S,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        w.stride(0),
        H=H,
        D=D,
    )
    return logits


def _compute_logits_triton_fp8(q, k, w, ks, ke):
    S = q.shape[0]
    q_flat = q.reshape(S * H, D).contiguous()
    q_fp8, _ = per_custom_dims_cast_to_fp8(q_flat, dims=(0,))
    k_fp8, k_scales = per_custom_dims_cast_to_fp8(k, dims=(0,))
    q_fp8 = q_fp8.view(S, H, D).permute(1, 0, 2).contiguous()
    logits = torch.empty(S, S, dtype=torch.float32, device=DEVICE)
    grid = lambda meta: (
        triton.cdiv(S, meta["BLOCK_M"]),
        triton.cdiv(S, meta["BLOCK_N"]),
    )
    _triton_fp8_indexer_kernel[grid](
        q_fp8,
        k_fp8,
        k_scales,
        w,
        logits,
        ks,
        ke,
        S,
        q_fp8.stride(0),
        q_fp8.stride(1),
        k_fp8.stride(0),
        w.stride(0),
        H=H,
        D=D,
    )
    return logits


def _compute_logits_tilelang(q, k, w, ks, ke, kernel_logits, kernel_clean):
    S = q.shape[0]
    q_flat = q.reshape(S * H, D).contiguous()
    q_fp8, _ = per_custom_dims_cast_to_fp8(q_flat, dims=(0,))
    k_fp8, k_scales = per_custom_dims_cast_to_fp8(k, dims=(0,))
    logits = torch.empty(S, S, dtype=torch.float32, device=DEVICE)
    kernel_logits(q_fp8, k_fp8, k_scales, logits, w.float(), ks, ke)
    kernel_clean(logits, ks, ke)
    return logits


def cosine_sim_pair(logits_a, logits_b):
    """Cosine similarity between two logits tensors on finite values."""
    a = logits_a.flatten().float()
    b = logits_b.flatten().float()
    mask = torch.isfinite(a) & torch.isfinite(b)
    a_m, b_m = a[mask], b[mask]
    if a_m.numel() == 0:
        return float("nan")
    return torch.nn.functional.cosine_similarity(a_m.unsqueeze(0), b_m.unsqueeze(0)).item()


def jaccard_topk_fast(idx_a, idx_b):
    """Vectorized Jaccard similarity between two [S, topk] index tensors."""
    S, K = idx_a.shape
    if S * K > 5_000_000:
        sample = min(S, 1024)
        rows = torch.randperm(S, device=idx_a.device)[:sample]
        idx_a, idx_b = idx_a[rows], idx_b[rows]
        S = sample

    a_sorted = idx_a.sort(dim=-1).values
    b_sorted = idx_b.sort(dim=-1).values

    matches = torch.zeros(S, device=idx_a.device)
    chunk = 256
    for j in range(0, K, chunk):
        a_chunk = a_sorted[:, j : j + chunk].unsqueeze(2)
        b_exp = b_sorted.unsqueeze(1)
        matches += (a_chunk == b_exp).any(dim=2).sum(dim=1).float()

    jaccard = matches / (2 * K - matches)
    return jaccard.mean().item()


# ===========================================================================
# Memory helpers
# ===========================================================================


def estimate_logits_memory_mb(S):
    return S * S * 4 / (1024 * 1024)


def check_memory_available_mb():
    free, _ = torch.cuda.mem_get_info()
    return free / (1024 * 1024)


# ===========================================================================
# Benchmark harness
# ===========================================================================


def run_single_benchmark(S, profile_name, profile_fn, kernel_logits, kernel_clean):
    cu_seqlens = make_cu_seqlens(profile_fn, S)
    q, k, w = generate_inputs(S)
    ks, ke = cu_seqlens_to_per_token(cu_seqlens, S)

    result = {
        "S": S,
        "profile": profile_name,
        "triton_bf16_ms": None,
        "triton_fp8_ms": None,
        "tilelang_ms": None,
        "triton_bf16_mem_mb": None,
        "triton_fp8_mem_mb": None,
        "tilelang_mem_mb": None,
        "cos_fp8_tl": None,
        "jac_fp8_tl": None,
        "cos_bf16_tl": None,
        "jac_bf16_tl": None,
    }

    logits_mem = estimate_logits_memory_mb(S)
    avail = check_memory_available_mb()
    if logits_mem * 2.5 > avail:
        print(f"  SKIP S={S} {profile_name}: need ~{logits_mem * 2.5:.0f}MB, have {avail:.0f}MB")
        return result

    # --- Triton BF16 indexer ---
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    triton_bf16_ms = do_bench(lambda: triton_indexer(q, k, w, ks, ke, TOPK), warmup=WARMUP, rep=REP)
    bf16_indices = triton_indexer(q, k, w, ks, ke, TOPK)
    result["triton_bf16_ms"] = triton_bf16_ms
    result["triton_bf16_mem_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024)

    # --- Triton FP8 indexer ---
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    triton_fp8_ms = do_bench(lambda: triton_fp8_indexer(q, k, w, ks, ke, TOPK), warmup=WARMUP, rep=REP)
    fp8_indices = triton_fp8_indexer(q, k, w, ks, ke, TOPK)
    result["triton_fp8_ms"] = triton_fp8_ms
    result["triton_fp8_mem_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024)

    # --- TileLang FP8 indexer ---
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    tilelang_ms = do_bench(
        lambda: tilelang_indexer(q, k, w, ks, ke, TOPK, kernel_logits, kernel_clean), warmup=WARMUP, rep=REP
    )
    tl_indices = tilelang_indexer(q, k, w, ks, ke, TOPK, kernel_logits, kernel_clean)
    result["tilelang_ms"] = tilelang_ms
    result["tilelang_mem_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024)

    # --- Correctness ---
    result["jac_fp8_tl"] = jaccard_topk_fast(fp8_indices, tl_indices)
    result["jac_bf16_tl"] = jaccard_topk_fast(bf16_indices, tl_indices)

    avail_corr = check_memory_available_mb()
    if logits_mem * 4 <= avail_corr:
        tl_logits = _compute_logits_tilelang(q, k, w, ks, ke, kernel_logits, kernel_clean)
        fp8_logits = _compute_logits_triton_fp8(q, k, w, ks, ke)
        result["cos_fp8_tl"] = cosine_sim_pair(fp8_logits, tl_logits)
        del fp8_logits
        bf16_logits = _compute_logits_triton_bf16(q, k, w, ks, ke)
        result["cos_bf16_tl"] = cosine_sim_pair(bf16_logits, tl_logits)
        del bf16_logits, tl_logits

    del bf16_indices, fp8_indices, tl_indices
    torch.cuda.empty_cache()
    return result


def format_val(v, fmt=".2f"):
    return "OOM" if v is None else f"{v:{fmt}}"


def generate_report(results, system_info):
    lines = [
        "# FP8 Lightning Indexer Benchmark",
        "",
        "## System Info",
    ]
    for key, val in system_info.items():
        lines.append(f"- **{key}:** {val}")
    lines += [
        "",
        "## Parameters",
        f"- H={H}, D={D}, topk={TOPK}",
        f"- warmup={WARMUP}, rep={REP}",
        "",
        "## Latency (ms)",
        "",
        "| S | Profile | Triton BF16 (ms) | Triton FP8 (ms) | TileLang FP8 (ms) | BF16 vs TL | FP8 vs TL |",
        "|---|---------|-----------------|----------------|-------------------|-----------|----------|",
    ]
    for r in results:
        bf16 = r["triton_bf16_ms"]
        fp8 = r["triton_fp8_ms"]
        tl_v = r["tilelang_ms"]
        sp_bf16 = f"{tl_v / bf16:.2f}x" if bf16 and tl_v else "N/A"
        sp_fp8 = f"{tl_v / fp8:.2f}x" if fp8 and tl_v else "N/A"
        lines.append(
            f"| {r['S']} | {r['profile']} | {format_val(bf16)} | {format_val(fp8)} | {format_val(tl_v)} | {sp_bf16} | {sp_fp8} |"
        )
    lines += [
        "",
        "## Peak GPU Memory (MB)",
        "",
        "| S | Profile | Triton BF16 (MB) | Triton FP8 (MB) | TileLang (MB) |",
        "|---|---------|-----------------|----------------|--------------|",
    ]
    for r in results:
        lines.append(
            f"| {r['S']} | {r['profile']} | {format_val(r['triton_bf16_mem_mb'], '.0f')} | {format_val(r['triton_fp8_mem_mb'], '.0f')} | {format_val(r['tilelang_mem_mb'], '.0f')} |"
        )
    lines += [
        "",
        "## Correctness vs TileLang FP8",
        "",
        "| S | Profile | Triton FP8 Cosine | Triton FP8 Jaccard | Triton BF16 Cosine | Triton BF16 Jaccard |",
        "|---|---------|------------------|-------------------|-------------------|-------------------|",
    ]
    for r in results:
        lines.append(
            f"| {r['S']} | {r['profile']} | {format_val(r['cos_fp8_tl'], '.4f')} | {format_val(r['jac_fp8_tl'], '.4f')} | {format_val(r['cos_bf16_tl'], '.4f')} | {format_val(r['jac_bf16_tl'], '.4f')} |"
        )
    lines.append("")
    return "\n".join(lines)


def main():
    print("=" * 60)
    print("FP8 Lightning Indexer Benchmark")
    print("=" * 60)

    system_info = {
        "GPU": torch.cuda.get_device_name(0),
        "GPU Memory": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB",
        "PyTorch": torch.__version__,
        "Triton": triton.__version__,
        "TileLang": tilelang.__version__,
        "CUDA": torch.version.cuda or "N/A",
        "Python": platform.python_version(),
    }
    for key, val in system_info.items():
        print(f"  {key}: {val}")
    print()

    print("Compiling TileLang kernels...")
    t0 = time.time()
    kernel_logits, kernel_clean = _build_tilelang_kernels()
    print(f"  Compiled in {time.time() - t0:.1f}s")

    # Warm up Triton autotuning on a small input
    print("Warming up Triton kernels...")
    t0 = time.time()
    _q, _k, _w = generate_inputs(4096)
    _ks, _ke = cu_seqlens_to_per_token(torch.tensor([0, 4096], dtype=torch.int32, device=DEVICE), 4096)
    triton_indexer(_q, _k, _w, _ks, _ke, TOPK)
    triton_fp8_indexer(_q, _k, _w, _ks, _ke, TOPK)
    del _q, _k, _w, _ks, _ke
    torch.cuda.empty_cache()
    print(f"  Triton warm in {time.time() - t0:.1f}s")
    print()

    results = []
    for S in SEQ_LENS:
        for profile_name, profile_fn in PROFILES.items():
            boundaries = profile_fn(S)
            if len(boundaries) < 2:
                continue
            chunk_sizes = [boundaries[i + 1] - boundaries[i] for i in range(len(boundaries) - 1)]
            if any(c <= 0 for c in chunk_sizes):
                continue

            print(f"S={S:>6}, profile={profile_name:>15} ... ", end="", flush=True)
            gc.collect()
            torch.cuda.empty_cache()

            r = run_single_benchmark(S, profile_name, profile_fn, kernel_logits, kernel_clean)
            results.append(r)

            bf16_str = format_val(r["triton_bf16_ms"])
            fp8_str = format_val(r["triton_fp8_ms"])
            tl_str = format_val(r["tilelang_ms"])
            cos_str = format_val(r["cos_fp8_tl"], ".4f")
            jac_str = format_val(r["jac_fp8_tl"], ".4f")
            print(f"bf16={bf16_str}ms  fp8={fp8_str}ms  tl={tl_str}ms  cos_fp8={cos_str}  jac_fp8={jac_str}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report = generate_report(results, system_info)
    out_path = RESULTS_DIR / "bench_fp8_indexer.md"
    out_path.write_text(report)
    print(f"\nReport written to {out_path}")


if __name__ == "__main__":
    main()
