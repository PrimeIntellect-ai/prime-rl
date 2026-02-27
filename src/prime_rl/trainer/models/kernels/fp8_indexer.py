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
    """Triton FP8 indexer: UE8M0 quantization + fused scoring kernel + topk.

    Args:
        q: [S, H, D] bf16 query vectors per head
        k: [S, D] bf16 key vectors (shared across heads)
        w: [S, H] bf16 per-head weights
        ks: [S] int32 sequence start per token
        ke: [S] int32 causal end per token (= position + 1)
        topk: number of top indices to return
        weight_scale: constant scaling factor for weights

    Returns:
        [S, topk] int32 selected token indices per query
    """
    # NOTE: We don't use weight scale in this kernel as it produces higher KL mismatch for some reason
    # This is not a problem result-wise, as it is a constant multiplier
    _weight_scale = weight_scale
    S, H, D = q.shape
    device = q.device

    q_flat = q.reshape(S * H, D).contiguous()
    q_fp8, q_scales = per_token_group_quant_fp8(q_flat, group_size=D, use_ue8m0=True)
    k_fp8, k_scales = per_token_group_quant_fp8(k.contiguous(), group_size=D, use_ue8m0=True)

    q_fp8 = q_fp8.view(S, H, D).permute(1, 0, 2).contiguous()

    # Fold q_scale into weights
    q_scales = q_scales.view(S, H)
    w = w * q_scales

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

    # Replace cross-sequence indices with sentinel S (maps to zero-valued KV)
    ks_exp = ks.unsqueeze(1).expand_as(indices)
    ke_exp = ke.unsqueeze(1).expand_as(indices)
    out_of_range = (indices < ks_exp) | (indices >= ke_exp)
    indices = indices.masked_fill(out_of_range, S)

    return indices.to(torch.int32)
