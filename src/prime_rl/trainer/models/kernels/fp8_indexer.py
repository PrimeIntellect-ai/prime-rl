"""Triton FP8 indexer kernel for DSA sparse token selection.

Implements the correct DeepSeek V3.2 scoring formula:
    I_{t,s} = Σ_j w_{t,j} · ReLU(q_{t,j} · k_s)

Uses FP8 tensor cores with per-row absmax scaling for Q and K.
"""

import torch
import triton
import triton.language as tl

FP8_MAX = 448.0


def per_custom_dims_cast_to_fp8(x, dims):
    excluded_dims = tuple(i for i in range(x.dim()) if i not in set(dims))
    x_amax = x.abs().float().amax(dim=excluded_dims, keepdim=True).clamp(1e-4)
    sf = x_amax / FP8_MAX
    x_scaled = (x * (1.0 / sf)).to(torch.float8_e4m3fn)
    return x_scaled, sf.squeeze()


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
    stride_qs,
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

    # Clamp indices for FP8 loads (can't use mask+other with fp8 dtype)
    offs_n_safe = tl.minimum(offs_n, S - 1)
    offs_m_safe = tl.minimum(offs_m, S - 1)

    k_block = tl.load(K_fp8 + offs_n_safe[:, None] * stride_ks + offs_d[None, :])
    k_sc = tl.load(K_scales + offs_n_safe, mask=mask_n, other=0.0).to(tl.float32)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for h in range(H):
        q_h = tl.load(Q_fp8 + h * stride_qh + offs_m_safe[:, None] * stride_qs + offs_d[None, :])
        w_h = tl.load(W + offs_m * stride_ws + h, mask=mask_m, other=0.0).to(tl.float32)

        scores = tl.dot(q_h, tl.trans(k_block)).to(tl.float32)
        scores = tl.maximum(scores, 0.0) * k_sc[None, :]
        acc += scores * w_h[:, None]

    valid = (offs_n[None, :] >= ks_vals[:, None]) & (offs_n[None, :] < ke_vals[:, None])
    acc = tl.where(valid, acc, float("-inf"))

    tl.store(out_ptrs, acc, mask=out_mask)


def fp8_indexer(q, k, w, ks, ke, topk, weight_scale=1.0):
    """Triton FP8 indexer: FP8 cast + fused kernel + topk.

    Args:
        q: [S, H, D] bf16 — query vectors per head
        k: [S, D] bf16 — key vectors (shared across heads)
        w: [S, H] bf16 — per-head weights
        ks: [S] int32 — sequence start per token
        ke: [S] int32 — causal end per token (= position + 1)
        topk: int — number of top indices to return
        weight_scale: float — constant scaling factor for weights (applied in float32)

    Returns:
        [S, topk] int32 — selected token indices per query
    """
    S, H, D = q.shape
    device = q.device

    q_flat = q.reshape(S * H, D).contiguous()
    q_fp8, q_scales = per_custom_dims_cast_to_fp8(q_flat, dims=(0,))
    k_fp8, k_scales = per_custom_dims_cast_to_fp8(k, dims=(0,))

    q_fp8 = q_fp8.view(S, H, D).permute(1, 0, 2).contiguous()

    # The FP8 dot product gives (q · k) / (q_scale * k_scale). The kernel only
    # compensates k_scale, so fold q_scale into the weights to keep the correct
    # per-head contribution: w_j * q_scale_j * ReLU(fp8_dot * k_scale).
    # weight_scale (head_dim^-0.5 * n_head^-0.5) is applied here in float32
    # to match vLLM/slime convention.
    q_scales = q_scales.view(S, H)
    w = w * q_scales * weight_scale

    logits = torch.empty(S, S, dtype=torch.float32, device=device)

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
        padding = torch.full((S, topk - actual_topk), S, dtype=indices.dtype, device=device)
        indices = torch.cat([indices, padding], dim=-1)

    # Replace out-of-range indices with sentinel index S.
    # The indexer marks cross-sequence tokens as -inf, but topk still returns
    # their indices. The sparse MLA kernel's causal mask (Indices <= query_pos)
    # doesn't respect sequence boundaries, so we must replace these indices
    # with S which maps to the zero-valued sentinel KV and always fails the
    # kernel's causal check (S > any query position).
    ks_exp = ks.unsqueeze(1).expand_as(indices)
    ke_exp = ke.unsqueeze(1).expand_as(indices)
    out_of_range = (indices < ks_exp) | (indices >= ke_exp)
    indices = indices.masked_fill(out_of_range, S)

    return indices.to(torch.int32)
