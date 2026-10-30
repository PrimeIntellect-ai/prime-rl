# Adapted from https://github.com/NVIDIA/Megatron-LM/blob/c16d981ca/megatron/core/fusions/fused_mla_yarn_rope_apply.py (BSD-3, Copyright NVIDIA)
"""Fused interleaved RoPE for DeepSeek-V4 attention, matching vLLM's numerics.

DeepSeek-V4 rotates only the trailing `rope_dim` channels of each head, with each rotary pair stored
adjacently. Eagerly, every rotation widens the cos/sin tables and rebuilds the whole head with
`torch.cat`, a large share of attention time. `dsv4_rope` rotates in one kernel pass, reading
fp32 cos and sin from a `[cos | sin]` cache by position, bit for bit with vLLM's `rotary_embedding`.
`dsv4_q_norm_rope` puts the query's RMSNorm in the same op so the query is rounded to bf16 once, and at the
shapes vLLM serves it runs vLLM's own fused prefill kernel.
"""

import torch
import triton
import triton.language as tl

from prime_rl.trainer.models.kernels.vllm_ops import vllm_op
from prime_rl.trainer.models.layers.norms import get_quack_rmsnorm

VLLM_Q_NORM_ROPE_OP = "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert"
VLLM_Q_NORM_ROPE_SIGNATURE = (
    "(Tensor q_in, Tensor kv, Tensor k_cache, Tensor slot_mapping, Tensor position_ids, Tensor cos_sin_cache, "
    "int q_head_padded, float eps, int cache_block_size, bool apply_q_norm, bool kv_mxfp8, bool apply_q_rope, "
    "bool is_q_interleaved) -> (Tensor)"
)
# The head counts vLLM compiles the op for.
VLLM_Q_NORM_ROPE_HEADS = (8, 16, 32, 64, 128)


@triton.autotune(
    configs=[triton.Config({"BLOCK_H": block_h}, num_warps=warps) for block_h in (1, 2, 4, 8, 16) for warps in (4, 8)],
    key=["num_heads", "head_dim"],
)
@triton.jit
def _triton_rope_kernel(
    X,
    OUT,
    COS_SIN,
    POS,
    num_heads,
    head_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    INVERSE: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Rotate the trailing `rope_dim` channels of `BLOCK_H` heads of one token and copy the rest."""
    token = tl.program_id(0).to(tl.int64)
    heads = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    rope_pairs = tl.arange(0, head_dim // 2) - (head_dim - rope_dim) // 2
    is_rope = (rope_pairs >= 0)[None, :]

    cos_sin = COS_SIN + tl.load(POS + token).to(tl.int64) * rope_dim
    cos = tl.load(cos_sin + rope_pairs, mask=rope_pairs >= 0, other=1.0)
    sin = tl.load(cos_sin + rope_dim // 2 + rope_pairs, mask=rope_pairs >= 0, other=0.0)
    if INVERSE:
        sin = -sin

    offsets = (token * num_heads + heads[:, None]) * head_dim + tl.arange(0, head_dim)[None, :]
    mask = heads[:, None] < num_heads
    x = tl.load(X + offsets, mask=mask).to(tl.float32)
    x1, x2 = tl.split(tl.reshape(x, (BLOCK_H, head_dim // 2, 2)))
    # vLLM's contraction: letting the compiler pick flips a bf16 ULP on ~1e-5 of the outputs.
    y1 = tl.where(is_rope, tl.fma(x1, cos, -(x2 * sin)), x1)
    y2 = tl.where(is_rope, tl.fma(x1, sin, x2 * cos), x2)
    y = tl.reshape(tl.join(y1, y2), (BLOCK_H, head_dim))
    tl.store(OUT + offsets, y.to(OUT.dtype.element_ty), mask=mask)


def _triton_rope(
    x: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    position_ids: torch.Tensor,
    inverse: bool,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Rotated copy of `x`, bypassing autograd, rounded once to `out_dtype` (default: `x`'s dtype)."""
    x = x.contiguous()
    out = x.new_empty(x.shape, dtype=out_dtype)
    if out.numel() == 0:
        return out
    num_heads, head_dim = x.shape[-2:]
    grid = lambda meta: (position_ids.numel(), triton.cdiv(num_heads, meta["BLOCK_H"]))
    _triton_rope_kernel[grid](
        x,
        out,
        cos_sin_cache,
        position_ids.contiguous(),
        num_heads,
        head_dim,
        cos_sin_cache.shape[-1],
        INVERSE=inverse,
    )
    return out


@torch.library.custom_op("prime_rl::dsv4_rope", mutates_args=())
def dsv4_rope(
    x: torch.Tensor, cos_sin_cache: torch.Tensor, position_ids: torch.Tensor, *, inverse: bool = False
) -> torch.Tensor:
    """Interleaved RoPE on the trailing `rope_dim` channels of each head, bit for bit with vLLM's `rotary_embedding`.

    With `x1 = x[..., 2i]` and `x2 = x[..., 2i + 1]` of the rotary slice, the output holds
    `x1 * cos - x2 * sin` and `x2 * cos + x1 * sin`, computed in fp32 with `cos` and `sin` read from row
    `position_ids[t]` of `cos_sin_cache` for token `t`; the leading channels are copied. `inverse=True`
    rotates by the opposite angle, which undoes the rotation.

    Args:
        x: `(..., heads, head_dim)`, one entry of `position_ids` per token; `head_dim` a power of two.
        cos_sin_cache: `(max_position, rope_dim)` fp32 and contiguous, cos then sin, one entry per pair each.
        position_ids: int32 or int64, one per token, each in `[0, max_position)`.
        inverse: rotate by the opposite angle.
    """
    return _triton_rope(x, cos_sin_cache, position_ids, inverse)


@dsv4_rope.register_fake
def _dsv4_rope_fake(
    x: torch.Tensor, cos_sin_cache: torch.Tensor, position_ids: torch.Tensor, *, inverse: bool = False
) -> torch.Tensor:
    return x.new_empty(x.shape)


def _dsv4_rope_setup_context(ctx, inputs, keyword_only_inputs, output) -> None:
    _, cos_sin_cache, position_ids = inputs
    ctx.save_for_backward(cos_sin_cache, position_ids)
    ctx.inverse = keyword_only_inputs["inverse"]


def _dsv4_rope_autograd_backward(ctx, grad: torch.Tensor):
    cos_sin_cache, position_ids = ctx.saved_tensors
    return dsv4_rope(grad, cos_sin_cache, position_ids, inverse=not ctx.inverse), None, None


dsv4_rope.register_autograd(_dsv4_rope_autograd_backward, setup_context=_dsv4_rope_setup_context)


@triton.jit
def _q_norm_rope_backward_kernel(
    GRAD,
    X,
    GRAD_X,
    COS_SIN,
    POS,
    num_heads,
    eps,
    head_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Rotate the gradient back in fp32, then the RMSNorm backward, with `rstd` recomputed from `x`."""
    token = tl.program_id(0).to(tl.int64)
    heads = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    rope_pairs = tl.arange(0, head_dim // 2) - (head_dim - rope_dim) // 2
    is_rope = (rope_pairs >= 0)[None, :]
    cos_sin = COS_SIN + tl.load(POS + token).to(tl.int64) * rope_dim
    cos = tl.load(cos_sin + rope_pairs, mask=rope_pairs >= 0, other=1.0)
    sin = tl.load(cos_sin + rope_dim // 2 + rope_pairs, mask=rope_pairs >= 0, other=0.0)

    offsets = (token * num_heads + heads[:, None]) * head_dim + tl.arange(0, head_dim)[None, :]
    mask = heads[:, None] < num_heads
    x = tl.load(X + offsets, mask=mask, other=0.0).to(tl.float32)
    grad = tl.load(GRAD + offsets, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(x * x, 1) * (1.0 / head_dim) + eps)[:, None]
    normed = x * rstd

    grad1, grad2 = tl.split(tl.reshape(grad, (BLOCK_H, head_dim // 2, 2)))
    grad_normed1 = tl.where(is_rope, tl.fma(grad1, cos, grad2 * sin), grad1)
    grad_normed2 = tl.where(is_rope, tl.fma(grad2, cos, -(grad1 * sin)), grad2)
    grad_normed = tl.reshape(tl.join(grad_normed1, grad_normed2), (BLOCK_H, head_dim))
    mean_dot = tl.sum(grad_normed * normed, 1)[:, None] * (1.0 / head_dim)
    grad_x = (grad_normed - normed * mean_dot) * rstd
    tl.store(GRAD_X + offsets, grad_x.to(GRAD_X.dtype.element_ty), mask=mask)


def _vllm_q_norm_rope(
    q: torch.Tensor, cos_sin_cache: torch.Tensor, position_ids: torch.Tensor, eps: float
) -> torch.Tensor | None:
    """vLLM's fused prefill q norm + RoPE of `q`, or `None` outside the op's contract or without the op."""
    num_heads, head_dim = q.shape[-2:]
    if not (
        q.is_cuda
        and q.dtype == torch.bfloat16
        and head_dim == 512
        and cos_sin_cache.shape[-1] == 64
        and num_heads in VLLM_Q_NORM_ROPE_HEADS
    ):
        return None
    op = vllm_op(VLLM_Q_NORM_ROPE_OP, VLLM_Q_NORM_ROPE_SIGNATURE)
    if op is None:
        return None
    rows = q.reshape(-1, num_heads, head_dim).contiguous()
    if rows.numel() == 0:
        return q.new_empty(q.shape)
    # The kv input is only read for the cache insert, which an empty `slot_mapping` skips.
    kv = rows.view(-1)[: rows.shape[0] * head_dim].view(-1, head_dim)
    out = op(
        rows,
        kv,
        q.new_empty(1, 1, dtype=torch.uint8),
        q.new_empty(0, dtype=torch.int64),
        position_ids.reshape(-1).long(),
        cos_sin_cache,
        num_heads,
        eps,
        1,
        True,
        False,
        True,
        False,
    )
    return out.view(q.shape)


@torch.library.custom_op("prime_rl::dsv4_q_norm_rope", mutates_args=())
def dsv4_q_norm_rope(
    q: torch.Tensor, cos_sin_cache: torch.Tensor, position_ids: torch.Tensor, eps: float
) -> torch.Tensor:
    """Unweighted RMSNorm over `head_dim`, then interleaved RoPE, in fp32 and rounded once to `q`'s dtype.

    At the shapes vLLM serves (bf16, `head_dim` 512, `rope_dim` 64, 8 to 128 heads), the forward is vLLM's
    own fused prefill kernel, so it matches serving bit for bit. Elsewhere, or without vLLM's op, an fp32
    RMSNorm feeds the rotation, which differs from vLLM's on a few elements per million by one ulp. The
    backward is one kernel that reads `q` and the gradient once. Each direction is one op that autograd
    and Inductor cannot see into, so the backward needs no fp32 intermediate and Inductor cannot reorder
    the norm's reduction.

    Args:
        q: `(..., heads, head_dim)`, one entry of `position_ids` per token; `head_dim` a power of two.
        cos_sin_cache: `(max_position, rope_dim)` fp32 and contiguous, cos then sin, one entry per pair each.
        position_ids: int32 or int64, one per token, each in `[0, max_position)`.
        eps: added to the mean square before the reciprocal square root.
    """
    out = _vllm_q_norm_rope(q, cos_sin_cache, position_ids, eps)
    if out is not None:
        return out
    rows = q.reshape(-1, q.shape[-1])
    if get_quack_rmsnorm() is not None:
        from quack.rmsnorm import rmsnorm_fwd

        normed = rmsnorm_fwd(rows, out_dtype=torch.float32, eps=eps)[0]
    else:
        normed = rows.float() * torch.rsqrt(rows.float().square().mean(-1, keepdim=True) + eps)
    return _triton_rope(normed.view(q.shape), cos_sin_cache, position_ids, False, out_dtype=q.dtype)


@dsv4_q_norm_rope.register_fake
def _dsv4_q_norm_rope_fake(
    q: torch.Tensor, cos_sin_cache: torch.Tensor, position_ids: torch.Tensor, eps: float
) -> torch.Tensor:
    return q.new_empty(q.shape)


@torch.library.custom_op("prime_rl::dsv4_q_norm_rope_backward", mutates_args=())
def _dsv4_q_norm_rope_backward(
    grad: torch.Tensor, q: torch.Tensor, cos_sin_cache: torch.Tensor, position_ids: torch.Tensor, eps: float
) -> torch.Tensor:
    grad = grad.contiguous()
    q = q.contiguous()
    grad_q = torch.empty_like(q)
    if grad_q.numel() == 0:
        return grad_q
    num_heads, head_dim = q.shape[-2:]
    block_h = 4
    _q_norm_rope_backward_kernel[(position_ids.numel(), triton.cdiv(num_heads, block_h))](
        grad,
        q,
        grad_q,
        cos_sin_cache,
        position_ids.contiguous(),
        num_heads,
        eps,
        head_dim,
        cos_sin_cache.shape[-1],
        BLOCK_H=block_h,
        num_warps=2,
    )
    return grad_q


@_dsv4_q_norm_rope_backward.register_fake
def _dsv4_q_norm_rope_backward_fake(
    grad: torch.Tensor, q: torch.Tensor, cos_sin_cache: torch.Tensor, position_ids: torch.Tensor, eps: float
) -> torch.Tensor:
    return q.new_empty(q.shape)


def _dsv4_q_norm_rope_setup_context(ctx, inputs, output) -> None:
    q, cos_sin_cache, position_ids, eps = inputs
    ctx.save_for_backward(q, cos_sin_cache, position_ids)
    ctx.eps = eps


def _dsv4_q_norm_rope_autograd_backward(ctx, grad: torch.Tensor):
    q, cos_sin_cache, position_ids = ctx.saved_tensors
    return _dsv4_q_norm_rope_backward(grad, q, cos_sin_cache, position_ids, ctx.eps), None, None, None


dsv4_q_norm_rope.register_autograd(_dsv4_q_norm_rope_autograd_backward, setup_context=_dsv4_q_norm_rope_setup_context)
