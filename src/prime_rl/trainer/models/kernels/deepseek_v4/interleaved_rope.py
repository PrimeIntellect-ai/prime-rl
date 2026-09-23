# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Triton kernels vendored from NVIDIA/Megatron-LM under the BSD-3-clause license
# reproduced above, megatron/core/fusions/fused_mla_yarn_rope_apply.py at commit
# c16d981ca (dev branch), trimmed to the in-place interleaved THD path and modified
# to read half-width fp32 cos and sin rows, one row per token.

import torch
import triton
import triton.language as tl

_BLOCK_H_CONFIGS = [triton.Config({"BLOCK_H": block_h}) for block_h in (1, 2, 4, 8, 16, 32, 64, 128)]


@triton.autotune(configs=_BLOCK_H_CONFIGS, key=["emb_dim", "head_num"], restore_value=["Q"])
@triton.jit
def _mla_rope_fwd_inplace_kernel(
    Q,
    COS,
    SIN,
    nope_dim,
    emb_dim: tl.constexpr,
    head_num: tl.constexpr,
    stride_x_seq,
    stride_x_nheads,
    stride_cos_seq,
    stride_cos_pair,
    stride_sin_seq,
    stride_sin_pair,
    INVERSE: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """
    Forward pass: apply RoPE inplace to the trailing emb_dim elements.
    Reads from interleaved layout, writes back to interleaved layout.

    Input:
        Q: [total_seq_len, head_num, nope_dim + emb_dim]
        COS: [total_seq_len, emb_dim // 2] cos for each token, one entry per interleaved pair
        SIN: [total_seq_len, emb_dim // 2] sin for each token, one entry per interleaved pair
    """
    pid_m = tl.program_id(axis=0).to(tl.int64)
    pid_head = tl.program_id(axis=1)

    cos = tl.load(COS + pid_m * stride_cos_seq + tl.arange(0, emb_dim // 2) * stride_cos_pair)
    sin = tl.load(SIN + pid_m * stride_sin_seq + tl.arange(0, emb_dim // 2) * stride_sin_pair)
    if INVERSE:
        sin = -sin
    cos = cos.expand_dims(0).broadcast_to(BLOCK_H, emb_dim // 2)
    sin = sin.expand_dims(0).broadcast_to(BLOCK_H, emb_dim // 2)

    Q = Q + pid_m * stride_x_seq + pid_head * BLOCK_H * stride_x_nheads

    head_idx = pid_head * BLOCK_H + tl.arange(0, BLOCK_H)[:, None]
    mask = head_idx < head_num
    x_off = tl.arange(0, BLOCK_H)[:, None] * stride_x_nheads + nope_dim
    x_1_off = x_off + tl.arange(0, emb_dim // 2)[None, :] * 2
    x_2_off = x_1_off + 1
    x_1 = tl.load(Q + x_1_off, mask=mask).to(tl.float32)
    x_2 = tl.load(Q + x_2_off, mask=mask).to(tl.float32)

    x_left = x_1 * cos - x_2 * sin
    x_right = x_2 * cos + x_1 * sin

    tl.store(Q + x_1_off, x_left.to(Q.dtype.element_ty), mask=mask)
    tl.store(Q + x_2_off, x_right.to(Q.dtype.element_ty), mask=mask)


@triton.autotune(configs=_BLOCK_H_CONFIGS, key=["emb_dim", "head_num"], restore_value=["DO"])
@triton.jit
def _mla_rope_bwd_kernel(
    DO,
    COS,
    SIN,
    nope_dim,
    emb_dim: tl.constexpr,
    head_num: tl.constexpr,
    stride_x_seq,
    stride_x_nheads,
    stride_cos_seq,
    stride_cos_pair,
    stride_sin_seq,
    stride_sin_pair,
    INVERSE: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """
    Backward pass: inverse RoPE inplace on the trailing emb_dim elements.
    Reads from interleaved layout, writes to interleaved layout.

    Input:
        DO: [total_seq_len, head_num, nope_dim + emb_dim]
        COS and SIN are the same as in the forward pass
    """
    pid_m = tl.program_id(axis=0).to(tl.int64)
    pid_head = tl.program_id(axis=1)

    cos = tl.load(COS + pid_m * stride_cos_seq + tl.arange(0, emb_dim // 2) * stride_cos_pair)
    sin = tl.load(SIN + pid_m * stride_sin_seq + tl.arange(0, emb_dim // 2) * stride_sin_pair)
    if INVERSE:
        sin = -sin
    cos = cos.expand_dims(0).broadcast_to(BLOCK_H, emb_dim // 2)
    sin = sin.expand_dims(0).broadcast_to(BLOCK_H, emb_dim // 2)

    DO = DO + pid_m * stride_x_seq + pid_head * BLOCK_H * stride_x_nheads

    head_idx = pid_head * BLOCK_H + tl.arange(0, BLOCK_H)[:, None]
    mask = head_idx < head_num
    x_off = tl.arange(0, BLOCK_H)[:, None] * stride_x_nheads + nope_dim
    x_1_off = x_off + tl.arange(0, emb_dim // 2)[None, :] * 2
    x_2_off = x_1_off + 1
    x_left = tl.load(DO + x_1_off, mask=mask).to(tl.float32)
    x_right = tl.load(DO + x_2_off, mask=mask).to(tl.float32)

    x_1 = x_left * cos + x_right * sin
    x_2 = -x_left * sin + x_right * cos

    tl.store(DO + x_1_off, x_1.to(DO.dtype.element_ty), mask=mask)
    tl.store(DO + x_2_off, x_2.to(DO.dtype.element_ty), mask=mask)


def _flatten_rope_input(x: torch.Tensor) -> torch.Tensor:
    """Normalize a ``(t, h, d)`` or ``(1, t, h, d)`` RoPE input to a ``(t, h, d)`` view."""
    if x.dim() == 4 and x.shape[0] == 1:
        return x[0]
    if x.dim() == 3:
        return x
    raise ValueError(
        f"RoPE input must be (tokens, heads, head_dim) or (1, tokens, heads, head_dim), got {tuple(x.shape)}"
    )


def _flatten_rope_table(table: torch.Tensor) -> torch.Tensor:
    """Normalize a ``(t, rope_dim / 2)`` or ``(1, t, rope_dim / 2)`` cos or sin to a ``(t, rope_dim / 2)`` view."""
    return table[0] if table.dim() == 3 and table.shape[0] == 1 else table


def _check_rope_layout(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> None:
    """Validate the shape, dtype and contiguity invariants both RoPE kernels rely on."""
    if x.stride(-1) != 1:
        raise ValueError(f"RoPE input must be contiguous in its last dim, got strides {x.stride()}")
    for name, table in (("cos", cos), ("sin", sin)):
        if table.shape != cos.shape or table.dim() != 2 or table.shape[0] != x.shape[0]:
            raise ValueError(
                f"cos and sin must both be (tokens, rope_dim / 2) with {x.shape[0]} tokens, "
                f"got cos {tuple(cos.shape)} and sin {tuple(sin.shape)}"
            )
        if table.dtype != torch.float32:
            raise ValueError(f"{name} must be float32, got {table.dtype}")
    emb_dim = 2 * cos.shape[-1]
    if emb_dim < 2 or 2 * triton.next_power_of_2(emb_dim // 2) != emb_dim:
        raise ValueError(f"rope_dim must be twice a power of two, got {emb_dim}")
    if emb_dim > x.shape[-1]:
        raise ValueError(f"rope_dim {emb_dim} exceeds the head dim {x.shape[-1]}")
    if not x.device == cos.device == sin.device:
        raise ValueError(
            f"RoPE tensors must share a device, got x on {x.device}, cos on {cos.device} and sin on {sin.device}"
        )


def _launch(kernel, t: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, inverse: bool) -> None:
    x = _flatten_rope_input(t)
    cos, sin = _flatten_rope_table(cos), _flatten_rope_table(sin)
    _check_rope_layout(x, cos, sin)
    total_seqlen, nheads, head_dim = x.shape
    if total_seqlen == 0 or nheads == 0:
        return
    emb_dim = 2 * cos.shape[-1]

    grid = lambda META: (total_seqlen, triton.cdiv(nheads, META["BLOCK_H"]))
    kernel[grid](
        x,
        cos,
        sin,
        head_dim - emb_dim,
        emb_dim,
        nheads,
        x.stride(0),
        x.stride(1),
        cos.stride(0),
        cos.stride(1),
        sin.stride(0),
        sin.stride(1),
        INVERSE=inverse,
    )


@torch.library.custom_op("prime_rl::dsv4_interleaved_rope_apply", mutates_args=("t",))
def mla_rope_apply_raw_(t: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, inverse: bool) -> None:
    """Apply the interleaved RoPE rotation to ``t`` in place, bypassing autograd.

    Same kernel and semantics as :func:`apply_interleaved_rope_`, but without the
    autograd node.
    """
    _launch(_mla_rope_fwd_inplace_kernel, t, cos, sin, inverse)


@mla_rope_apply_raw_.register_fake
def _mla_rope_apply_raw_fake(t: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, inverse: bool) -> None:
    return None


@torch.library.custom_op("prime_rl::dsv4_interleaved_rope_unapply", mutates_args=("t",))
def mla_rope_unapply_raw(t: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, inverse: bool) -> None:
    """Undo :func:`mla_rope_apply_raw_` in place, bypassing autograd.

    This is the exact transpose of the forward rotation, so for a unit-magnitude
    rotation it is also its exact inverse. Pass the same ``inverse`` flag that was
    used to apply it.
    """
    _launch(_mla_rope_bwd_kernel, t, cos, sin, inverse)


@mla_rope_unapply_raw.register_fake
def _mla_rope_unapply_raw_fake(t: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, inverse: bool) -> None:
    return None


class _FusedMLARoPEInplace(torch.autograd.Function):
    """
    Autograd function for applying RoPE inplace to the trailing channels of a multi-head tensor.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, inverse: bool) -> torch.Tensor:
        mla_rope_apply_raw_(x, cos, sin, inverse)
        ctx.mark_dirty(x)
        ctx.save_for_backward(cos, sin)
        ctx.inverse = inverse
        return x

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        cos, sin = ctx.saved_tensors
        # Rotates `grad` in place: unsafe if autograd hands this same tensor to another consumer.
        grad = grad.contiguous()
        mla_rope_unapply_raw(grad, cos, sin, ctx.inverse)
        return grad, None, None, None


def apply_interleaved_rope_(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, *, inverse: bool = False
) -> torch.Tensor:
    """
    Fused interleaved RoPE applied inplace to the trailing rope_dim elements of each head,
    leaving the leading nope_dim elements unchanged.

    With ``x1 = x[..., 2i]`` and ``x2 = x[..., 2i + 1]`` of the rotary slice, writes
    ``x1 * cos - x2 * sin`` and ``x2 * cos + x1 * sin``, computed in fp32. When
    ``inverse=True`` the rotation is reversed, which is useful for undoing RoPE on
    the attention output.

    Args:
        x: [total_seq_len, head_num, nope_dim + rope_dim] or [1, total_seq_len, head_num, nope_dim + rope_dim],
            any token and head strides, contiguous in the last dim
        cos: [total_seq_len, rope_dim // 2] or [1, total_seq_len, rope_dim // 2] float32, one entry per
            interleaved pair of each token, any strides
        sin: same shape as ``cos``
        inverse: if True, apply the inverse rotation

    Returns:
        x: inplace modified input tensor
    """
    return _FusedMLARoPEInplace.apply(x, cos, sin, inverse)
