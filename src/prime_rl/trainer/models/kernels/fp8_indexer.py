"""Triton FP8 indexer kernel for DSA sparse token selection.

Scoring formula: I_{t,s} = Σ_j w_{t,j} · ReLU(q_{t,j} · k_s)

FP8 quantization vendored from vLLM (Apache 2.0).
"""

import torch
import triton
import triton.language as tl

FP8_MAX = 448.0
FP8_MIN = -448.0
FP8_EPS = 1e-10


@triton.jit
def _per_token_group_quant_fp8(
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    group_size,
    y_num_columns,
    y_row_stride,
    eps,
    fp8_min,
    fp8_max,
    use_ue8m0: tl.constexpr,
    BLOCK: tl.constexpr,
):
    groups_per_row = y_num_columns // group_size

    g_id = tl.program_id(0)
    row = g_id // groups_per_row
    row_g_id = g_id % groups_per_row

    y_ptr_offset = (row.to(tl.int64) * y_row_stride) + (row_g_id.to(tl.int64) * group_size)
    y_ptr += y_ptr_offset

    y_q_ptr_offset = g_id.to(tl.int64) * group_size
    y_q_ptr += y_q_ptr_offset
    y_s_ptr += g_id

    cols = tl.arange(0, BLOCK)
    mask = cols < group_size

    y = tl.load(y_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
    scale_raw = _absmax / fp8_max
    y_s = tl.math.exp2(tl.ceil(tl.log2(scale_raw))) if use_ue8m0 else scale_raw
    y_q = tl.clamp(y / y_s, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)

    tl.store(y_q_ptr + cols, y_q, mask=mask)
    tl.store(y_s_ptr, y_s)


def per_token_group_quant_fp8(
    x: torch.Tensor, group_size: int, use_ue8m0: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.shape[-1] % group_size == 0
    assert x.stride(-1) == 1

    x_q = torch.empty(x.shape, device=x.device, dtype=torch.float8_e4m3fn)
    shape = x.shape[:-1] + (x.shape[-1] // group_size,)
    x_s = torch.empty(shape, device=x.device, dtype=torch.float32)

    M = x.numel() // group_size
    N = group_size
    BLOCK = triton.next_power_of_2(N)
    num_warps = min(max(BLOCK // 256, 1), 8)

    _per_token_group_quant_fp8[(M,)](
        x,
        x_q,
        x_s,
        group_size,
        x.shape[-1],
        x.stride(-2),
        FP8_EPS,
        FP8_MIN,
        FP8_MAX,
        use_ue8m0=use_ue8m0,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=1,
    )
    return x_q, x_s


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=3),
    ],
    key=["S_K", "N_OUT"],
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
    S_K,
    N_OUT,
    K_START,
    stride_qh,
    stride_qs,
    stride_ks,
    stride_ws,
    stride_out_m,
    stride_out_n,
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

    offs_n_global = K_START + offs_n
    mask_m = offs_m < S_K
    mask_n = (offs_n < N_OUT) & (offs_n_global < S_K)

    ks_vals = tl.load(KS + offs_m, mask=mask_m, other=S_K)
    ke_vals = tl.load(KE + offs_m, mask=mask_m, other=0)

    offs_m_64 = offs_m.to(tl.int64)
    offs_n_local_64 = offs_n.to(tl.int64)
    out_ptrs = Out + offs_m_64[:, None] * stride_out_m + offs_n_local_64[None, :] * stride_out_n
    out_mask = mask_m[:, None] & mask_n[None, :]

    ks_min = tl.min(ks_vals)
    ke_max = tl.max(ke_vals)
    n_lo = K_START + pid_n * BLOCK_N

    if n_lo >= ke_max or n_lo + BLOCK_N <= ks_min:
        tl.store(out_ptrs, tl.full([BLOCK_M, BLOCK_N], float("-inf"), tl.float32), mask=out_mask)
        return

    # Clamp indices for FP8 loads (can't use mask+other with fp8 dtype)
    offs_n_safe = tl.minimum(offs_n_global, S_K - 1)
    offs_m_safe = tl.minimum(offs_m, S_K - 1)

    k_block = tl.load(K_fp8 + offs_n_safe[:, None] * stride_ks + offs_d[None, :])
    k_sc = tl.load(K_scales + offs_n_safe, mask=mask_n, other=0.0).to(tl.float32)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for h in range(H):
        q_h = tl.load(Q_fp8 + h * stride_qh + offs_m_safe[:, None] * stride_qs + offs_d[None, :])
        w_h = tl.load(W + offs_m * stride_ws + h, mask=mask_m, other=0.0).to(tl.float32)

        scores = tl.dot(q_h, tl.trans(k_block)).to(tl.float32)
        scores = tl.maximum(scores, 0.0) * k_sc[None, :]
        acc += scores * w_h[:, None]

    valid = (offs_n_global[None, :] >= ks_vals[:, None]) & (offs_n_global[None, :] < ke_vals[:, None])
    acc = tl.where(valid, acc, float("-inf"))

    tl.store(out_ptrs, acc, mask=out_mask)


FP8_INDEXER_DEFAULT_CHUNK_SIZE = 1024
FP8_INDEXER_FULL_FALLBACK_MULTIPLIER = 12


def _prepare_fp8_inputs(q, k, w, weight_scale):
    # NOTE: We don't use weight scale in this kernel as it produces higher KL mismatch for some reason.
    # This is not a problem result-wise, as it is a constant multiplier.
    _weight_scale = weight_scale
    S, H, D = q.shape

    q_flat = q.reshape(S * H, D).contiguous()
    q_fp8, q_scales = per_token_group_quant_fp8(q_flat, group_size=D, use_ue8m0=True)
    k_fp8, k_scales = per_token_group_quant_fp8(k.contiguous(), group_size=D, use_ue8m0=True)

    q_fp8 = q_fp8.view(S, H, D).permute(1, 0, 2).contiguous()

    # Fold q_scale into weights
    q_scales = q_scales.view(S, H)
    w = w * q_scales
    return q_fp8, k_fp8, k_scales, w


def _run_fp8_indexer_kernel(q_fp8, k_fp8, k_scales, w, ks, ke, out, k_start):
    S_K = q_fp8.shape[1]
    n_out = out.shape[1]

    grid = lambda meta: (  # noqa: E731
        triton.cdiv(S_K, meta["BLOCK_M"]),
        triton.cdiv(n_out, meta["BLOCK_N"]),
    )
    _triton_fp8_indexer_kernel[grid](
        q_fp8,
        k_fp8,
        k_scales,
        w,
        out,
        ks,
        ke,
        S_K,
        n_out,
        k_start,
        q_fp8.stride(0),
        q_fp8.stride(1),
        k_fp8.stride(0),
        w.stride(0),
        out.stride(0),
        out.stride(1),
        H=q_fp8.shape[0],
        D=q_fp8.shape[2],
    )


def _mask_out_of_range_indices(indices, ks, ke, sentinel):
    ks_exp = ks.unsqueeze(1).expand_as(indices)
    ke_exp = ke.unsqueeze(1).expand_as(indices)
    out_of_range = (indices < ks_exp) | (indices >= ke_exp)
    return indices.masked_fill(out_of_range, sentinel)


def _pad_indices(indices, topk, sentinel):
    if indices.shape[1] >= topk:
        return indices
    padding = torch.full(
        (indices.shape[0], topk - indices.shape[1]),
        sentinel,
        dtype=indices.dtype,
        device=indices.device,
    )
    return torch.cat([indices, padding], dim=-1)


def fp8_indexer_full(q, k, w, ks, ke, topk, weight_scale=1.0):
    """Baseline FP8 indexer that materializes the full [S, S] logits matrix."""
    S = q.shape[0]
    device = q.device
    q_fp8, k_fp8, k_scales, w = _prepare_fp8_inputs(q, k, w, weight_scale)

    logits = torch.empty(S, S, dtype=torch.float32, device=device)
    _run_fp8_indexer_kernel(q_fp8, k_fp8, k_scales, w, ks, ke, logits, k_start=0)

    actual_topk = min(topk, S)
    _, indices = torch.topk(logits, actual_topk, dim=-1)
    indices = _pad_indices(indices, topk, S)
    indices = _mask_out_of_range_indices(indices, ks, ke, S)
    return indices.to(torch.int32)


def fp8_indexer(q, k, w, ks, ke, topk, weight_scale=1.0, chunk_size=None):
    """Chunked Triton FP8 indexer that avoids allocating a full [S, S] score matrix.

    Args:
        q: [S, H, D] bf16 query vectors per head
        k: [S, D] bf16 key vectors (shared across heads)
        w: [S, H] bf16 per-head weights
        ks: [S] int32 sequence start per token
        ke: [S] int32 causal end per token (= position + 1)
        topk: number of top indices to return
        weight_scale: constant scaling factor for weights
        chunk_size: key-axis chunk size. If None, uses a memory-safe default.

    Returns:
        [S, topk] int32 selected token indices per query
    """
    S = q.shape[0]
    device = q.device
    actual_topk = min(topk, S)

    if actual_topk == 0:
        return torch.empty((S, 0), dtype=torch.int32, device=device)

    if chunk_size is None:
        if S <= actual_topk * FP8_INDEXER_FULL_FALLBACK_MULTIPLIER:
            return fp8_indexer_full(q, k, w, ks, ke, topk, weight_scale)
        chunk_size = min(S, max(actual_topk, FP8_INDEXER_DEFAULT_CHUNK_SIZE))
    elif chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
    else:
        chunk_size = min(chunk_size, S)

    if chunk_size >= S:
        return fp8_indexer_full(q, k, w, ks, ke, topk, weight_scale)

    q_fp8, k_fp8, k_scales, w = _prepare_fp8_inputs(q, k, w, weight_scale)

    running_scores = torch.full((S, actual_topk), float("-inf"), dtype=torch.float32, device=device)
    running_indices = torch.full((S, actual_topk), S, dtype=torch.int32, device=device)

    for chunk_start in range(0, S, chunk_size):
        curr_chunk = min(chunk_size, S - chunk_start)
        logits_chunk = torch.empty((S, curr_chunk), dtype=torch.float32, device=device)
        _run_fp8_indexer_kernel(q_fp8, k_fp8, k_scales, w, ks, ke, logits_chunk, k_start=chunk_start)

        chunk_topk = min(actual_topk, curr_chunk)
        chunk_scores, chunk_local_indices = torch.topk(logits_chunk, chunk_topk, dim=-1)
        chunk_indices = (chunk_local_indices + chunk_start).to(torch.int32)

        merged_scores = torch.cat([running_scores, chunk_scores], dim=-1)
        merged_indices = torch.cat([running_indices, chunk_indices], dim=-1)

        running_scores, selected = torch.topk(merged_scores, actual_topk, dim=-1)
        running_indices = torch.gather(merged_indices, dim=-1, index=selected)

    indices = _pad_indices(running_indices, topk, S)
    indices = _mask_out_of_range_indices(indices, ks, ke, S)
    return indices.to(torch.int32)
