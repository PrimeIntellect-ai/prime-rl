import torch
from flash_attn.cute.interface import _flash_attn_bwd, _flash_attn_fwd

# FA4's public wrapper calls torch._guards.active_fake_mode(), which Dynamo refuses to trace, so
# fullgraph compile breaks inside it. Wrapping the kernels in custom ops keeps them opaque to Dynamo.


@torch.library.custom_op("prime_rl::flash_attn_4_varlen", mutates_args=())
def _flash_attn_4_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    causal: bool,
    window_size_left: int | None,
    window_size_right: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    out, lse, _, _ = _flash_attn_fwd(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        causal=causal,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
        return_lse=True,
    )
    return out, lse


@_flash_attn_4_varlen.register_fake
def _flash_attn_4_varlen_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    causal: bool,
    window_size_left: int | None,
    window_size_right: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    total_q, num_heads, _ = q.shape
    out = q.new_empty((total_q, num_heads, v.shape[-1]))
    lse = q.new_empty((num_heads, total_q), dtype=torch.float32)
    return out, lse


@torch.library.custom_op("prime_rl::flash_attn_4_varlen_backward", mutates_args=())
def _flash_attn_4_varlen_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    grad_out: torch.Tensor,
    lse: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    causal: bool,
    window_size_left: int | None,
    window_size_right: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    grad_q, grad_k, grad_v = _flash_attn_bwd(
        q,
        k,
        v,
        out,
        grad_out,
        lse,
        causal=causal,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
    )
    return grad_q, grad_k, grad_v


@_flash_attn_4_varlen_backward.register_fake
def _flash_attn_4_varlen_backward_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    grad_out: torch.Tensor,
    lse: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    causal: bool,
    window_size_left: int | None,
    window_size_right: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


def _flash_attn_4_varlen_setup_context(ctx, inputs, output) -> None:
    q, k, v, cu_seqlens_q, cu_seqlens_k, causal, window_size_left, window_size_right = inputs
    out, lse = output
    ctx.save_for_backward(q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k)
    ctx.causal = causal
    ctx.window_size_left = window_size_left
    ctx.window_size_right = window_size_right


def _flash_attn_4_varlen_autograd_backward(ctx, grad_out: torch.Tensor, grad_lse: torch.Tensor):
    q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k = ctx.saved_tensors
    grad_q, grad_k, grad_v = _flash_attn_4_varlen_backward(
        q,
        k,
        v,
        out,
        grad_out,
        lse,
        cu_seqlens_q,
        cu_seqlens_k,
        ctx.causal,
        ctx.window_size_left,
        ctx.window_size_right,
    )
    return grad_q, grad_k, grad_v, None, None, None, None, None


_flash_attn_4_varlen.register_autograd(
    _flash_attn_4_varlen_autograd_backward,
    setup_context=_flash_attn_4_varlen_setup_context,
)


def flash_attn_4_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    causal: bool = True,
    window_size: tuple[int | None, int | None] = (None, None),
) -> torch.Tensor:
    out, _ = _flash_attn_4_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, causal, window_size[0], window_size[1])
    return out


__all__ = ["flash_attn_4_varlen"]
