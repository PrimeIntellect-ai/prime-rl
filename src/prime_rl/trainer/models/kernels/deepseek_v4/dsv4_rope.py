# Adapted from https://github.com/NVIDIA/Megatron-LM/blob/c16d981ca/megatron/core/fusions/fused_mla_yarn_rope_apply.py (BSD-3, Copyright NVIDIA)
"""Fused interleaved RoPE for DeepSeek-V4 attention, matching vLLM's numerics.

DeepSeek-V4 rotates only the trailing `rope_dim` channels of each head, with each rotary pair stored
adjacently. Eagerly, every rotation widens the cos/sin tables and rebuilds the whole head with
`torch.cat`, a large share of attention time. `dsv4_rope` rotates in one kernel pass, reading
fp32 cos and sin from a `[cos | sin]` cache by position, bit for bit with vLLM's `rotary_embedding`.
`dsv4_q_norm_rope` puts the query's RMSNorm in the same op so the query is rounded to bf16 once, as vLLM's
fused prefill kernel does.
"""

import torch
import triton
import triton.language as tl

from prime_rl.trainer.models.layers.norms import get_quack_rmsnorm


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


@torch.library.custom_op("prime_rl::dsv4_q_norm_rope", mutates_args=())
def _dsv4_q_norm_rope(
    q: torch.Tensor, cos_sin_cache: torch.Tensor, position_ids: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """The normed and rotated `q` in `q`'s dtype, plus the fp32 per-row `rstd` that backward needs."""
    rows = q.reshape(-1, q.shape[-1])
    if get_quack_rmsnorm() is not None:
        from quack.rmsnorm import rmsnorm_fwd

        normed, _, rstd = rmsnorm_fwd(rows, out_dtype=torch.float32, eps=eps, store_rstd=True)
    else:
        rstd = torch.rsqrt(rows.float().square().mean(-1) + eps)
        normed = rows.float() * rstd[:, None]
    return _triton_rope(normed.view(q.shape), cos_sin_cache, position_ids, False, out_dtype=q.dtype), rstd


@_dsv4_q_norm_rope.register_fake
def _dsv4_q_norm_rope_fake(
    q: torch.Tensor, cos_sin_cache: torch.Tensor, position_ids: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    return q.new_empty(q.shape), q.new_empty(q.numel() // q.shape[-1], dtype=torch.float32)


@torch.library.custom_op("prime_rl::dsv4_q_norm_rope_backward", mutates_args=())
def _dsv4_q_norm_rope_backward(
    grad: torch.Tensor, q: torch.Tensor, cos_sin_cache: torch.Tensor, position_ids: torch.Tensor, rstd: torch.Tensor
) -> torch.Tensor:
    rows = q.reshape(-1, q.shape[-1])
    grad_normed = _triton_rope(grad, cos_sin_cache, position_ids, True).view(rows.shape)
    if get_quack_rmsnorm() is not None:
        from quack.rmsnorm import rmsnorm_bwd

        return rmsnorm_bwd(rows, None, grad_normed, rstd)[0].view(q.shape)
    normed = rows.float() * rstd[:, None]
    grad_normed = grad_normed.float()
    grad_rows = (grad_normed - normed * (grad_normed * normed).mean(-1, keepdim=True)) * rstd[:, None]
    return grad_rows.to(q.dtype).view(q.shape)


@_dsv4_q_norm_rope_backward.register_fake
def _dsv4_q_norm_rope_backward_fake(
    grad: torch.Tensor, q: torch.Tensor, cos_sin_cache: torch.Tensor, position_ids: torch.Tensor, rstd: torch.Tensor
) -> torch.Tensor:
    return q.new_empty(q.shape)


def _dsv4_q_norm_rope_setup_context(ctx, inputs, output) -> None:
    q, cos_sin_cache, position_ids, _ = inputs
    ctx.save_for_backward(q, cos_sin_cache, position_ids, output[1])


def _dsv4_q_norm_rope_autograd_backward(ctx, grad: torch.Tensor, grad_rstd: torch.Tensor):
    q, cos_sin_cache, position_ids, rstd = ctx.saved_tensors
    return _dsv4_q_norm_rope_backward(grad, q, cos_sin_cache, position_ids, rstd), None, None, None


_dsv4_q_norm_rope.register_autograd(_dsv4_q_norm_rope_autograd_backward, setup_context=_dsv4_q_norm_rope_setup_context)


def dsv4_q_norm_rope(
    q: torch.Tensor, cos_sin_cache: torch.Tensor, position_ids: torch.Tensor, eps: float
) -> torch.Tensor:
    """Unweighted RMSNorm over `head_dim`, then interleaved RoPE, both in fp32, rounded once to `q`'s dtype.

    The norm and rotation form one op that autograd and Inductor cannot see into:

    - Autograd: as separate nodes, backward would pass an fp32 gradient of the normalized query between
      them, twice the size of `q` (4 GiB per layer for 32k tokens of 64 heads of 512). Here the
      rotation's backward runs on the bf16 gradient and feeds the norm's backward directly.
    - Inductor: without quack, the norm falls back to torch ops, which Inductor would fuse and reorder,
      flipping a bf16 ULP on a few hundred outputs so compiled results drift from eager and from vLLM.

    `q` is `(..., heads, head_dim)`; `cos_sin_cache` and `position_ids` are as in `dsv4_rope`.
    """
    out, _ = _dsv4_q_norm_rope(q, cos_sin_cache, position_ids, eps)
    return out
