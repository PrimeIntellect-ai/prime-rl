"""Experimental deterministic MoE combine without a reordered activation buffer."""

import torch
import triton
import triton.language as tl


@triton.jit
def _invert_routes(ORDER, INVERSE, rows: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    slot = tl.load(ORDER + row, row < rows, other=0)
    # The route permutation assigns each destination exactly once.
    tl.store(INVERSE + slot, row, row < rows)


@triton.jit
def _combine_forward(
    X,
    INVERSE,
    SCORES,
    Y,
    width: tl.constexpr,
    top_k: tl.constexpr,
    stride_row: tl.constexpr,
    stride_col: tl.constexpr,
    stride_score: tl.constexpr,
    HAS_SCORES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    total = tl.full((BLOCK,), 0, tl.float32)
    for slot in tl.static_range(top_k):
        row = tl.load(INVERSE + token * top_k + slot).to(tl.int64)
        value = tl.load(X + row * stride_row + col * stride_col, col < width, other=0).to(tl.float32)
        if HAS_SCORES:
            score = tl.load(SCORES + row * stride_score).to(tl.float32)
            # Preserve the contribution rounding of the unfused multiply/cast.
            value = (value * score).to(X.dtype.element_ty).to(tl.float32)
        total = total + value
    tl.store(Y + token.to(tl.int64) * width + col, total, col < width)


@triton.jit
def _combine_backward(
    X,
    ORDER,
    SCORES,
    GRAD,
    DX,
    DS,
    width: tl.constexpr,
    top_k: tl.constexpr,
    stride_row: tl.constexpr,
    stride_col: tl.constexpr,
    stride_score: tl.constexpr,
    stride_grad_row: tl.constexpr,
    stride_grad_col: tl.constexpr,
    HAS_SCORES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    token = tl.load(ORDER + row) // top_k
    col = tl.arange(0, BLOCK)
    grad = tl.load(GRAD + token.to(tl.int64) * stride_grad_row + col * stride_grad_col, col < width, other=0).to(
        tl.float32
    )
    if HAS_SCORES:
        value = tl.load(X + row.to(tl.int64) * stride_row + col * stride_col, col < width, other=0).to(tl.float32)
        score = tl.load(SCORES + row * stride_score).to(tl.float32)
        tl.store(DS + row, tl.sum(value * grad, axis=0))
        grad = grad * score
    tl.store(DX + row.to(tl.int64) * width + col, grad, col < width)


@torch.library.custom_op("prime_rl::fused_moe_combine", mutates_args=())
def fused_moe_combine(
    values: torch.Tensor,
    order: torch.Tensor,
    scores: torch.Tensor | None,
    num_tokens: int,
) -> torch.Tensor:
    """Gather unique route slots and reduce FP16/BF16 contributions in FP32.

    ``order`` must be the dispatcher's permutation of token/route slots.
    Only first-order autograd is supported by the experimental backward.
    """
    if values.device.type != "cuda" or values.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("Fused MoE combine requires CUDA FP16 or BF16 values")
    if values.ndim != 2 or order.ndim != 1 or order.numel() != values.shape[0]:
        raise ValueError("Expected values [routes, hidden] and one permutation entry per route")
    if order.dtype != torch.int64 or not order.is_contiguous() or order.device != values.device:
        raise ValueError("Route order must be contiguous int64 on the values device")
    if num_tokens <= 0 or values.shape[0] % num_tokens or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("Expected nonempty values with an integral number of routes per token")
    if scores is not None and (
        scores.ndim != 1
        or scores.numel() != values.shape[0]
        or scores.device != values.device
        or scores.dtype not in (torch.float32, torch.float16, torch.bfloat16)
    ):
        raise ValueError("Scores must contain one FP32/FP16/BF16 weight per route on the values device")
    rows, width = values.shape
    top_k = rows // num_tokens
    inverse = torch.empty_like(order)
    output = values.new_empty((num_tokens, width))
    _invert_routes[(triton.cdiv(rows, 256),)](order, inverse, rows, BLOCK=256)
    _combine_forward[(num_tokens, triton.cdiv(width, 256))](
        values,
        inverse,
        scores,
        output,
        width,
        top_k,
        values.stride(0),
        values.stride(1),
        scores.stride(0) if scores is not None else 0,
        HAS_SCORES=scores is not None,
        BLOCK=256,
        enable_fp_fusion=False,
    )
    return output


@fused_moe_combine.register_fake
def _combine_fake(values, order, scores, num_tokens):
    return values.new_empty((num_tokens, values.shape[1]))


@torch.library.custom_op("prime_rl::fused_moe_combine_backward", mutates_args=())
def _backward_op(
    values: torch.Tensor | None,
    order: torch.Tensor,
    scores: torch.Tensor | None,
    grad: torch.Tensor,
    num_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, width = order.numel(), grad.shape[1]
    dx = grad.new_empty((rows, width))
    ds = torch.empty_like(scores, memory_format=torch.contiguous_format) if scores is not None else grad.new_empty((0,))
    _combine_backward[(rows,)](
        values,
        order,
        scores,
        grad,
        dx,
        ds,
        width,
        rows // num_tokens,
        values.stride(0) if values is not None else 0,
        values.stride(1) if values is not None else 0,
        scores.stride(0) if scores is not None else 0,
        grad.stride(0),
        grad.stride(1),
        HAS_SCORES=scores is not None,
        BLOCK=triton.next_power_of_2(width),
        num_warps=8 if width >= 2048 else 4,
        enable_fp_fusion=False,
    )
    return dx, ds


@_backward_op.register_fake
def _backward_fake(values, order, scores, grad, num_tokens):
    dx = grad.new_empty((order.numel(), grad.shape[1]))
    ds = torch.empty_like(scores, memory_format=torch.contiguous_format) if scores is not None else grad.new_empty((0,))
    return dx, ds


def _setup_context(ctx, inputs, output):
    values, order, scores, num_tokens = inputs
    ctx.has_scores = scores is not None
    ctx.num_tokens = num_tokens
    ctx.save_for_backward(order, *(() if scores is None else (values, scores)))


def _backward(ctx, grad):
    order, *weighted_inputs = ctx.saved_tensors
    values, scores = weighted_inputs if ctx.has_scores else (None, None)
    dx, ds = _backward_op(values, order, scores, grad, ctx.num_tokens)
    return dx, None, ds if ctx.has_scores else None, None


fused_moe_combine.register_autograd(_backward, setup_context=_setup_context)
