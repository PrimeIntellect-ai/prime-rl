"""Fused Hopper (SM90) Triton kernels for MoE activations, zero-copy permute, and weighted combine.

All ops are registered via ``torch.library.custom_op`` with fake tensor and autograd support so they
compose cleanly with FSDP2, activation checkpointing, and ``torch.compile``.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# =============================================================================
# 1. Fused ClampedSilu(limit) — both packed [M, 2*N] gate_up and separate (gate, up)
# =============================================================================


@triton.jit
def _clamped_silu_gate_up_fwd_kernel(
    GateUp_ptr,
    Out_ptr,
    N: tl.constexpr,
    stride_gu_m,
    stride_out_m,
    LIMIT: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    col_blk = tl.program_id(1)
    offs = col_blk * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N

    gu_base = GateUp_ptr + row * stride_gu_m
    g = tl.load(gu_base + offs, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(gu_base + N + offs, mask=mask, other=0.0).to(tl.float32)

    g_c = tl.minimum(g, LIMIT)
    u_c = tl.maximum(tl.minimum(u, LIMIT), -LIMIT)
    out = (g_c * tl.sigmoid(g_c)) * u_c

    tl.store(Out_ptr + row * stride_out_m + offs, out.to(Out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _clamped_silu_gate_up_bwd_kernel(
    DOut_ptr,
    GateUp_ptr,
    DGateUp_ptr,
    N: tl.constexpr,
    stride_do_m,
    stride_gu_m,
    stride_dgu_m,
    LIMIT: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    col_blk = tl.program_id(1)
    offs = col_blk * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N

    do = tl.load(DOut_ptr + row * stride_do_m + offs, mask=mask, other=0.0).to(tl.float32)
    gu_base = GateUp_ptr + row * stride_gu_m
    g = tl.load(gu_base + offs, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(gu_base + N + offs, mask=mask, other=0.0).to(tl.float32)

    g_c = tl.minimum(g, LIMIT)
    u_c = tl.maximum(tl.minimum(u, LIMIT), -LIMIT)
    sig = tl.sigmoid(g_c)
    silu_g = g_c * sig
    dsilu = sig * (1.0 + g_c * (1.0 - sig))

    # Match PyTorch clamp backward: strict inequality at boundaries.
    dg = tl.where(g < LIMIT, do * u_c * dsilu, 0.0)
    du = tl.where((u > -LIMIT) & (u < LIMIT), do * silu_g, 0.0)

    dgu_base = DGateUp_ptr + row * stride_dgu_m
    tl.store(dgu_base + offs, dg.to(DGateUp_ptr.dtype.element_ty), mask=mask)
    tl.store(dgu_base + N + offs, du.to(DGateUp_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _clamped_silu_pair_fwd_kernel(
    Gate_ptr,
    Up_ptr,
    Out_ptr,
    n_elements,
    LIMIT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements

    g = tl.load(Gate_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(Up_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    g_c = tl.minimum(g, LIMIT)
    u_c = tl.maximum(tl.minimum(u, LIMIT), -LIMIT)
    out = (g_c * tl.sigmoid(g_c)) * u_c
    tl.store(Out_ptr + offs, out.to(Out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _clamped_silu_pair_bwd_kernel(
    DOut_ptr,
    Gate_ptr,
    Up_ptr,
    DGate_ptr,
    DUp_ptr,
    n_elements,
    LIMIT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements

    do = tl.load(DOut_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(Gate_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(Up_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    g_c = tl.minimum(g, LIMIT)
    u_c = tl.maximum(tl.minimum(u, LIMIT), -LIMIT)
    sig = tl.sigmoid(g_c)
    silu_g = g_c * sig
    dsilu = sig * (1.0 + g_c * (1.0 - sig))

    dg = tl.where(g < LIMIT, do * u_c * dsilu, 0.0)
    du = tl.where((u > -LIMIT) & (u < LIMIT), do * silu_g, 0.0)
    tl.store(DGate_ptr + offs, dg.to(DGate_ptr.dtype.element_ty), mask=mask)
    tl.store(DUp_ptr + offs, du.to(DUp_ptr.dtype.element_ty), mask=mask)


@torch.library.custom_op("prime_rl::hopper_clamped_silu_gate_up", mutates_args=())
def _clamped_silu_gate_up_op(gate_up: torch.Tensor, limit: float) -> torch.Tensor:
    gate_up_c = gate_up.contiguous()
    M, two_N = gate_up_c.shape
    N = two_N // 2
    out = torch.empty((M, N), device=gate_up_c.device, dtype=gate_up_c.dtype)
    if M == 0:
        return out
    block_n = 512 if N % 512 == 0 else 256
    _clamped_silu_gate_up_fwd_kernel[(M, triton.cdiv(N, block_n))](
        gate_up_c, out, N, gate_up_c.stride(0), out.stride(0), float(limit), BLOCK_N=block_n, num_warps=4
    )
    return out


@_clamped_silu_gate_up_op.register_fake
def _(gate_up: torch.Tensor, limit: float) -> torch.Tensor:
    return gate_up.new_empty((gate_up.shape[0], gate_up.shape[1] // 2))


@torch.library.custom_op("prime_rl::hopper_clamped_silu_gate_up_bwd", mutates_args=())
def _clamped_silu_gate_up_bwd_op(grad_out: torch.Tensor, gate_up: torch.Tensor, limit: float) -> torch.Tensor:
    do_c = grad_out.contiguous()
    gu_c = gate_up.contiguous()
    M, two_N = gu_c.shape
    N = two_N // 2
    d_gate_up = torch.empty_like(gu_c)
    if M == 0:
        return d_gate_up
    block_n = 512 if N % 512 == 0 else 256
    _clamped_silu_gate_up_bwd_kernel[(M, triton.cdiv(N, block_n))](
        do_c,
        gu_c,
        d_gate_up,
        N,
        do_c.stride(0),
        gu_c.stride(0),
        d_gate_up.stride(0),
        float(limit),
        BLOCK_N=block_n,
        num_warps=4,
    )
    return d_gate_up


@_clamped_silu_gate_up_bwd_op.register_fake
def _(grad_out: torch.Tensor, gate_up: torch.Tensor, limit: float) -> torch.Tensor:
    return torch.empty_like(gate_up)


def _clamped_silu_gu_setup(ctx, inputs, output) -> None:
    gate_up, limit = inputs
    ctx.save_for_backward(gate_up)
    ctx.limit = limit


def _clamped_silu_gu_bwd(ctx, grad_out: torch.Tensor):
    (gate_up,) = ctx.saved_tensors
    return _clamped_silu_gate_up_bwd_op(grad_out, gate_up, ctx.limit), None


_clamped_silu_gate_up_op.register_autograd(_clamped_silu_gu_bwd, setup_context=_clamped_silu_gu_setup)


def fused_clamped_silu(gate_up: torch.Tensor, limit: float) -> torch.Tensor:
    """Fused ClampedSilu on packed `[M, 2*N]` gate_up tensor."""
    return _clamped_silu_gate_up_op(gate_up, float(limit))


@torch.library.custom_op("prime_rl::hopper_clamped_silu_pair", mutates_args=())
def _clamped_silu_pair_op(gate: torch.Tensor, up: torch.Tensor, limit: float) -> torch.Tensor:
    g_c = gate.contiguous()
    u_c = up.contiguous()
    out = torch.empty_like(g_c)
    n = g_c.numel()
    if n == 0:
        return out
    block = 1024
    _clamped_silu_pair_fwd_kernel[(triton.cdiv(n, block),)](g_c, u_c, out, n, float(limit), BLOCK=block, num_warps=4)
    return out


@_clamped_silu_pair_op.register_fake
def _(gate: torch.Tensor, up: torch.Tensor, limit: float) -> torch.Tensor:
    return torch.empty_like(gate)


@torch.library.custom_op("prime_rl::hopper_clamped_silu_pair_bwd", mutates_args=())
def _clamped_silu_pair_bwd_op(
    grad_out: torch.Tensor, gate: torch.Tensor, up: torch.Tensor, limit: float
) -> tuple[torch.Tensor, torch.Tensor]:
    do_c = grad_out.contiguous()
    g_c = gate.contiguous()
    u_c = up.contiguous()
    dg = torch.empty_like(g_c)
    du = torch.empty_like(u_c)
    n = g_c.numel()
    if n == 0:
        return dg, du
    block = 1024
    _clamped_silu_pair_bwd_kernel[(triton.cdiv(n, block),)](
        do_c, g_c, u_c, dg, du, n, float(limit), BLOCK=block, num_warps=4
    )
    return dg, du


@_clamped_silu_pair_bwd_op.register_fake
def _(grad_out: torch.Tensor, gate: torch.Tensor, up: torch.Tensor, limit: float) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(gate), torch.empty_like(up)


def _clamped_silu_pair_setup(ctx, inputs, output) -> None:
    gate, up, limit = inputs
    ctx.save_for_backward(gate, up)
    ctx.limit = limit


def _clamped_silu_pair_bwd(ctx, grad_out: torch.Tensor):
    gate, up = ctx.saved_tensors
    dg, du = _clamped_silu_pair_bwd_op(grad_out, gate, up, ctx.limit)
    return dg, du, None


_clamped_silu_pair_op.register_autograd(_clamped_silu_pair_bwd, setup_context=_clamped_silu_pair_setup)


def fused_clamped_silu_pair(gate: torch.Tensor, up: torch.Tensor, limit: float) -> torch.Tensor:
    """Fused ClampedSilu on separate `(gate, up)` tensors of matching shape."""
    return _clamped_silu_pair_op(gate, up, float(limit))


# =============================================================================
# 2. Zero-Copy Padded Gather / Scatter Permute (eliminates 536 MB torch.vstack)
# =============================================================================


@triton.jit
def _zero_pad_gather_kernel(
    X_ptr,
    Idx_ptr,
    Out_ptr,
    N_real,
    D: tl.constexpr,
    stride_x,
    stride_o,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    d_blk = tl.program_id(1)
    offs = d_blk * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = offs < D

    src_row = tl.load(Idx_ptr + row).to(tl.int64)
    valid = (src_row >= 0) & (src_row < N_real)
    safe_row = tl.where(valid, src_row, 0)
    val = tl.load(X_ptr + safe_row * stride_x + offs, mask=d_mask & valid, other=0.0)
    tl.store(Out_ptr + row * stride_o + offs, val, mask=d_mask)


@triton.jit
def _scatter_unpermute_kernel(
    X_ptr,
    Idx_ptr,
    Out_ptr,
    N_real,
    D: tl.constexpr,
    stride_x,
    stride_o,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    d_blk = tl.program_id(1)
    offs = d_blk * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = offs < D

    dst_row = tl.load(Idx_ptr + row).to(tl.int64)
    if (dst_row >= 0) & (dst_row < N_real):
        val = tl.load(X_ptr + row * stride_x + offs, mask=d_mask, other=0.0)
        tl.store(Out_ptr + dst_row * stride_o + offs, val, mask=d_mask)


@torch.library.custom_op("prime_rl::hopper_zero_pad_gather", mutates_args=())
def _zero_pad_gather_op(x: torch.Tensor, permuted_indices: torch.Tensor) -> torch.Tensor:
    x_c = x.contiguous()
    idx_c = permuted_indices.contiguous()
    n_real, D = x_c.shape
    m_pad = idx_c.numel()
    out = torch.empty((m_pad, D), device=x_c.device, dtype=x_c.dtype)
    if m_pad == 0:
        return out
    block_d = 512 if D % 512 == 0 else 256
    _zero_pad_gather_kernel[(m_pad, triton.cdiv(D, block_d))](
        x_c, idx_c, out, n_real, D, x_c.stride(0), out.stride(0), BLOCK_D=block_d, num_warps=4
    )
    return out


@_zero_pad_gather_op.register_fake
def _(x: torch.Tensor, permuted_indices: torch.Tensor) -> torch.Tensor:
    return x.new_empty((permuted_indices.numel(), x.shape[1]))


@torch.library.custom_op("prime_rl::hopper_scatter_unpermute", mutates_args=())
def _scatter_unpermute_op(x: torch.Tensor, permuted_indices: torch.Tensor, n_real: int) -> torch.Tensor:
    x_c = x.contiguous()
    idx_c = permuted_indices.contiguous()
    m_pad, D = x_c.shape
    out = torch.empty((n_real, D), device=x_c.device, dtype=x_c.dtype)
    if m_pad == 0 or n_real == 0:
        return out
    block_d = 512 if D % 512 == 0 else 256
    _scatter_unpermute_kernel[(m_pad, triton.cdiv(D, block_d))](
        x_c, idx_c, out, n_real, D, x_c.stride(0), out.stride(0), BLOCK_D=block_d, num_warps=4
    )
    return out


@_scatter_unpermute_op.register_fake
def _(x: torch.Tensor, permuted_indices: torch.Tensor, n_real: int) -> torch.Tensor:
    return x.new_empty((n_real, x.shape[1]))


def _gather_setup(ctx, inputs, output) -> None:
    x, permuted_indices = inputs
    ctx.save_for_backward(permuted_indices)
    ctx.n_real = x.shape[0]


def _gather_bwd(ctx, grad_out: torch.Tensor):
    (permuted_indices,) = ctx.saved_tensors
    return _scatter_unpermute_op(grad_out, permuted_indices, ctx.n_real), None


def _unpermute_setup(ctx, inputs, output) -> None:
    _, permuted_indices, _ = inputs
    ctx.save_for_backward(permuted_indices)


def _unpermute_bwd(ctx, grad_out: torch.Tensor):
    (permuted_indices,) = ctx.saved_tensors
    return _zero_pad_gather_op(grad_out, permuted_indices), None, None


_zero_pad_gather_op.register_autograd(_gather_bwd, setup_context=_gather_setup)
_scatter_unpermute_op.register_autograd(_unpermute_bwd, setup_context=_unpermute_setup)


def zero_pad_permute(x: torch.Tensor, permuted_indices: torch.Tensor) -> torch.Tensor:
    return _zero_pad_gather_op(x, permuted_indices)


def zero_pad_unpermute(x: torch.Tensor, permuted_indices: torch.Tensor, n_real: int) -> torch.Tensor:
    return _scatter_unpermute_op(x, permuted_indices, n_real)


# =============================================================================
# 3. Fused Weighted Unpermute + Top-K Reduce (Torch EP / LocalTokenDispatcher)
# =============================================================================


@triton.jit
def _fused_weighted_combine_fwd_kernel(
    Routed_ptr,
    InvRoute_ptr,
    Scores_ptr,
    Out_ptr,
    D: tl.constexpr,
    TOP_K: tl.constexpr,
    HAS_SCORES: tl.constexpr,
    stride_r,
    stride_o,
    BLOCK_D: tl.constexpr,
):
    tok = tl.program_id(0).to(tl.int64)
    d_blk = tl.program_id(1)
    offs = d_blk * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = offs < D

    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    inv_base = tok * TOP_K
    for k in tl.static_range(0, TOP_K):
        row = tl.load(InvRoute_ptr + inv_base + k).to(tl.int64)
        val = tl.load(Routed_ptr + row * stride_r + offs, mask=d_mask, other=0.0).to(tl.float32)
        if HAS_SCORES:
            w = tl.load(Scores_ptr + row).to(tl.float32)
            acc += val * w
        else:
            acc += val

    tl.store(Out_ptr + tok * stride_o + offs, acc.to(Out_ptr.dtype.element_ty), mask=d_mask)


@triton.jit
def _fused_weighted_combine_bwd_kernel(
    GradOut_ptr,
    Routed_ptr,
    RouteIdx_ptr,
    Scores_ptr,
    GradRouted_ptr,
    GradScores_ptr,
    D: tl.constexpr,
    TOP_K: tl.constexpr,
    HAS_SCORES: tl.constexpr,
    NEEDS_SCORE_GRAD: tl.constexpr,
    stride_go,
    stride_r,
    stride_gr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    slot = tl.load(RouteIdx_ptr + row).to(tl.int64)
    tok = slot // TOP_K

    w = tl.load(Scores_ptr + row).to(tl.float32) if HAS_SCORES else 1.0
    score_grad_acc = 0.0

    for d_start in tl.static_range(0, D, BLOCK_D):
        offs = d_start + tl.arange(0, BLOCK_D)
        d_mask = offs < D
        go = tl.load(GradOut_ptr + tok * stride_go + offs, mask=d_mask, other=0.0).to(tl.float32)
        gr = go * w if HAS_SCORES else go
        tl.store(GradRouted_ptr + row * stride_gr + offs, gr.to(GradRouted_ptr.dtype.element_ty), mask=d_mask)
        if NEEDS_SCORE_GRAD:
            rv = tl.load(Routed_ptr + row * stride_r + offs, mask=d_mask, other=0.0).to(tl.float32)
            score_grad_acc += tl.sum(go * rv, axis=0)

    if NEEDS_SCORE_GRAD:
        tl.store(GradScores_ptr + row, score_grad_acc)


@torch.library.custom_op("prime_rl::hopper_weighted_combine", mutates_args=())
def _weighted_combine_op(
    routed_output: torch.Tensor,
    route_indices_experts_sorted: torch.Tensor,
    scores_after_experts: torch.Tensor | None,
    num_tokens: int,
) -> torch.Tensor:
    routed_c = routed_output.contiguous()
    route_c = route_indices_experts_sorted.contiguous()
    M, D = routed_c.shape
    out = torch.empty((num_tokens, D), device=routed_c.device, dtype=routed_c.dtype)
    if num_tokens == 0:
        return out
    top_k = M // num_tokens
    inv_route = torch.empty_like(route_c)
    inv_route[route_c] = torch.arange(M, device=route_c.device, dtype=route_c.dtype)

    has_scores = scores_after_experts is not None
    scores_f32 = scores_after_experts.float().contiguous() if has_scores else route_c
    block_d = 512 if D % 512 == 0 else 256
    _fused_weighted_combine_fwd_kernel[(num_tokens, triton.cdiv(D, block_d))](
        routed_c,
        inv_route,
        scores_f32,
        out,
        D=D,
        TOP_K=top_k,
        HAS_SCORES=has_scores,
        stride_r=routed_c.stride(0),
        stride_o=out.stride(0),
        BLOCK_D=block_d,
        num_warps=4,
    )
    return out


@_weighted_combine_op.register_fake
def _(
    routed_output: torch.Tensor,
    route_indices_experts_sorted: torch.Tensor,
    scores_after_experts: torch.Tensor | None,
    num_tokens: int,
) -> torch.Tensor:
    return routed_output.new_empty((num_tokens, routed_output.shape[1]))


@torch.library.custom_op("prime_rl::hopper_weighted_combine_bwd", mutates_args=())
def _weighted_combine_bwd_op(
    grad_out: torch.Tensor,
    routed_output: torch.Tensor,
    route_indices_experts_sorted: torch.Tensor,
    scores_after_experts: torch.Tensor | None,
    needs_score_grad: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    go_c = grad_out.contiguous()
    routed_c = routed_output.contiguous()
    route_c = route_indices_experts_sorted.contiguous()
    M = route_c.numel()
    num_tokens, D = go_c.shape
    grad_routed = torch.empty((M, D), device=go_c.device, dtype=go_c.dtype)
    has_scores = scores_after_experts is not None
    scores_f32 = scores_after_experts.float().contiguous() if has_scores else route_c
    grad_scores_f32 = (
        torch.empty((M,), device=go_c.device, dtype=torch.float32) if (has_scores and needs_score_grad) else route_c
    )
    if M > 0 and num_tokens > 0:
        top_k = M // num_tokens
        block_d = 512 if D % 512 == 0 else 256
        _fused_weighted_combine_bwd_kernel[(M,)](
            go_c,
            routed_c,
            route_c,
            scores_f32,
            grad_routed,
            grad_scores_f32,
            D=D,
            TOP_K=top_k,
            HAS_SCORES=has_scores,
            NEEDS_SCORE_GRAD=bool(has_scores and needs_score_grad),
            stride_go=go_c.stride(0),
            stride_r=routed_c.stride(0),
            stride_gr=grad_routed.stride(0),
            BLOCK_D=block_d,
            num_warps=4,
        )
    if has_scores and needs_score_grad:
        return grad_routed, grad_scores_f32.to(scores_after_experts.dtype)
    return grad_routed, None


@_weighted_combine_bwd_op.register_fake
def _(
    grad_out: torch.Tensor,
    routed_output: torch.Tensor,
    route_indices_experts_sorted: torch.Tensor,
    scores_after_experts: torch.Tensor | None,
    needs_score_grad: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    grad_routed = grad_out.new_empty((route_indices_experts_sorted.numel(), grad_out.shape[1]))
    grad_scores = torch.empty_like(scores_after_experts) if (scores_after_experts is not None and needs_score_grad) else None
    return grad_routed, grad_scores


def _weighted_combine_setup(ctx, inputs, output) -> None:
    routed_output, route_indices, scores_after_experts, _ = inputs
    ctx.save_for_backward(routed_output, route_indices, scores_after_experts)


def _weighted_combine_bwd(ctx, grad_out: torch.Tensor):
    routed_output, route_indices, scores_after_experts = ctx.saved_tensors
    needs_score_grad = bool(scores_after_experts is not None and ctx.needs_input_grad[2])
    grad_routed, grad_scores = _weighted_combine_bwd_op(
        grad_out, routed_output, route_indices, scores_after_experts, needs_score_grad
    )
    return grad_routed, None, grad_scores, None


_weighted_combine_op.register_autograd(_weighted_combine_bwd, setup_context=_weighted_combine_setup)


def fused_weighted_combine(
    routed_output: torch.Tensor,
    *,
    num_tokens: int,
    route_indices_experts_sorted: torch.Tensor,
    scores_after_experts: torch.Tensor | None,
) -> torch.Tensor:
    return _weighted_combine_op(routed_output, route_indices_experts_sorted, scores_after_experts, num_tokens)


# =============================================================================
# 4. Fused DeepEP Weighted SumRoutes (kills .float() + mul + F.pad + 8x index_select)
# =============================================================================


@triton.jit
def _deepep_weighted_sum_routes_fwd_kernel(
    Routed_ptr,
    RouteRows_ptr,
    Scores_ptr,
    Out_ptr,
    M,
    D: tl.constexpr,
    TOP_K: tl.constexpr,
    HAS_SCORES: tl.constexpr,
    stride_r,
    stride_o,
    BLOCK_D: tl.constexpr,
):
    tok = tl.program_id(0).to(tl.int64)
    d_blk = tl.program_id(1)
    offs = d_blk * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = offs < D

    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    base = tok * TOP_K
    for k in tl.static_range(0, TOP_K):
        row = tl.load(RouteRows_ptr + base + k).to(tl.int64)
        valid = (row >= 0) & (row < M)
        safe_row = tl.where(valid, row, 0)
        val = tl.load(Routed_ptr + safe_row * stride_r + offs, mask=d_mask & valid, other=0.0).to(tl.float32)
        if HAS_SCORES:
            w = tl.load(Scores_ptr + safe_row, mask=valid, other=0.0).to(tl.float32)
            acc += val * w
        else:
            acc += val

    tl.store(Out_ptr + tok * stride_o + offs, acc.to(Out_ptr.dtype.element_ty), mask=d_mask)


@triton.jit
def _deepep_weighted_sum_routes_bwd_kernel(
    GradOut_ptr,
    Routed_ptr,
    TokIdx_ptr,
    Scores_ptr,
    GradRouted_ptr,
    GradScores_ptr,
    D: tl.constexpr,
    HAS_SCORES: tl.constexpr,
    NEEDS_SCORE_GRAD: tl.constexpr,
    stride_go,
    stride_r,
    stride_gr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    tok = tl.load(TokIdx_ptr + row).to(tl.int64)
    w = tl.load(Scores_ptr + row).to(tl.float32) if HAS_SCORES else 1.0
    score_grad_acc = 0.0

    for d_start in tl.static_range(0, D, BLOCK_D):
        offs = d_start + tl.arange(0, BLOCK_D)
        d_mask = offs < D
        go = tl.load(GradOut_ptr + tok * stride_go + offs, mask=d_mask, other=0.0).to(tl.float32)
        gr = go * w if HAS_SCORES else go
        tl.store(GradRouted_ptr + row * stride_gr + offs, gr.to(GradRouted_ptr.dtype.element_ty), mask=d_mask)
        if NEEDS_SCORE_GRAD:
            rv = tl.load(Routed_ptr + row * stride_r + offs, mask=d_mask, other=0.0).to(tl.float32)
            score_grad_acc += tl.sum(go * rv, axis=0)

    if NEEDS_SCORE_GRAD:
        tl.store(GradScores_ptr + row, score_grad_acc)


@torch.library.custom_op("prime_rl::hopper_deepep_weighted_sum_routes", mutates_args=())
def _deepep_sum_routes_op(
    routed_output: torch.Tensor,
    route_rows: torch.Tensor,
    token_indices: torch.Tensor,
    scores_after_experts: torch.Tensor | None,
) -> torch.Tensor:
    routed_c = routed_output.contiguous()
    rr_c = route_rows.contiguous()
    M, D = routed_c.shape
    n_recv, top_k = rr_c.shape
    out = torch.empty((n_recv, D), device=routed_c.device, dtype=routed_c.dtype)
    if n_recv == 0:
        return out
    has_scores = scores_after_experts is not None
    scores_f32 = scores_after_experts.float().contiguous() if has_scores else rr_c
    block_d = 512 if D % 512 == 0 else 256
    _deepep_weighted_sum_routes_fwd_kernel[(n_recv, triton.cdiv(D, block_d))](
        routed_c,
        rr_c,
        scores_f32,
        out,
        M=M,
        D=D,
        TOP_K=top_k,
        HAS_SCORES=has_scores,
        stride_r=routed_c.stride(0),
        stride_o=out.stride(0),
        BLOCK_D=block_d,
        num_warps=4,
    )
    return out


@_deepep_sum_routes_op.register_fake
def _(
    routed_output: torch.Tensor,
    route_rows: torch.Tensor,
    token_indices: torch.Tensor,
    scores_after_experts: torch.Tensor | None,
) -> torch.Tensor:
    return routed_output.new_empty((route_rows.shape[0], routed_output.shape[1]))


@torch.library.custom_op("prime_rl::hopper_deepep_weighted_sum_routes_bwd", mutates_args=())
def _deepep_sum_routes_bwd_op(
    grad_out: torch.Tensor,
    routed_output: torch.Tensor,
    token_indices: torch.Tensor,
    scores_after_experts: torch.Tensor | None,
    needs_score_grad: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    go_c = grad_out.contiguous()
    routed_c = routed_output.contiguous()
    ti_c = token_indices.contiguous()
    M = ti_c.numel()
    D = go_c.shape[1]
    grad_routed = torch.empty((M, D), device=go_c.device, dtype=go_c.dtype)
    has_scores = scores_after_experts is not None
    scores_f32 = scores_after_experts.float().contiguous() if has_scores else ti_c
    grad_scores_f32 = (
        torch.empty((M,), device=go_c.device, dtype=torch.float32) if (has_scores and needs_score_grad) else ti_c
    )
    if M > 0:
        block_d = 512 if D % 512 == 0 else 256
        _deepep_weighted_sum_routes_bwd_kernel[(M,)](
            go_c,
            routed_c,
            ti_c,
            scores_f32,
            grad_routed,
            grad_scores_f32,
            D=D,
            HAS_SCORES=has_scores,
            NEEDS_SCORE_GRAD=bool(has_scores and needs_score_grad),
            stride_go=go_c.stride(0),
            stride_r=routed_c.stride(0),
            stride_gr=grad_routed.stride(0),
            BLOCK_D=block_d,
            num_warps=4,
        )
    if has_scores and needs_score_grad:
        return grad_routed, grad_scores_f32.to(scores_after_experts.dtype)
    return grad_routed, None


@_deepep_sum_routes_bwd_op.register_fake
def _(
    grad_out: torch.Tensor,
    routed_output: torch.Tensor,
    token_indices: torch.Tensor,
    scores_after_experts: torch.Tensor | None,
    needs_score_grad: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    grad_routed = grad_out.new_empty((token_indices.numel(), grad_out.shape[1]))
    grad_scores = torch.empty_like(scores_after_experts) if (scores_after_experts is not None and needs_score_grad) else None
    return grad_routed, grad_scores


def _deepep_sum_routes_setup(ctx, inputs, output) -> None:
    routed_output, _, token_indices, scores_after_experts = inputs
    ctx.save_for_backward(routed_output, token_indices, scores_after_experts)


def _deepep_sum_routes_bwd(ctx, grad_out: torch.Tensor):
    routed_output, token_indices, scores_after_experts = ctx.saved_tensors
    needs_score_grad = bool(scores_after_experts is not None and ctx.needs_input_grad[3])
    grad_routed, grad_scores = _deepep_sum_routes_bwd_op(
        grad_out, routed_output, token_indices, scores_after_experts, needs_score_grad
    )
    return grad_routed, None, None, grad_scores


_deepep_sum_routes_op.register_autograd(_deepep_sum_routes_bwd, setup_context=_deepep_sum_routes_setup)


def fused_deepep_weighted_sum_routes(
    routed_output: torch.Tensor,
    route_rows: torch.Tensor,
    token_indices: torch.Tensor,
    scores_after_experts: torch.Tensor | None,
) -> torch.Tensor:
    return _deepep_sum_routes_op(routed_output, route_rows, token_indices, scores_after_experts)


__all__ = [
    "fused_clamped_silu",
    "fused_clamped_silu_pair",
    "fused_deepep_weighted_sum_routes",
    "fused_weighted_combine",
    "zero_pad_permute",
    "zero_pad_unpermute",
]
