"""DeepSeek-V4.1 sparse attention on FlashMLA's sparse prefill forward and cuDNN's DSA backward.

Same contract as `dsv4_sparse_attn` (the TileLang kernels): every query reads an explicit list of
positions of one shared K = V latent buffer, with a per-head attention sink, and `-1` marks an
empty slot. Both kernels take the sink natively: FlashMLA scales the output by
`exp(lse) / (exp(lse) + exp(sink))` and returns the sink-free LSE, which cuDNN's backward consumes
together with the sink to rebuild the sink-aware probabilities and the sink gradient.

The kernels read flat tensors: `q` `(tokens, heads, dim)`, `kv` `(positions, dim)` and indices
`(tokens, slots)`, so the batch axis (always 1 for packed rows) is folded in and out here.
"""

import functools

import torch

try:
    from cudnn.deepseek_sparse_attention.sparse_attention_backward._interface_sm90 import flash_attn_bwd_sm90
    from flash_mla import flash_mla_sparse_fwd
except ImportError:
    flash_mla_sparse_fwd = None  # type: ignore
    flash_attn_bwd_sm90 = None  # type: ignore


# FlashMLA's SM90 sparse prefill walks the slots two 64-slot tiles at a time.
SLOT_TILE = 128


def _pad_slots(indices: torch.Tensor) -> torch.Tensor:
    """Widen the slot axis to a multiple of `SLOT_TILE` with empty (-1) slots, which change nothing."""
    remainder = indices.shape[-1] % SLOT_TILE
    if remainder == 0:
        return indices
    return torch.nn.functional.pad(indices, (0, SLOT_TILE - remainder), value=-1).contiguous()


def flashmla_sparse_attn_available(num_heads: int, head_dim: int) -> bool:
    """FlashMLA's sparse prefill serves 64 or 128 heads of a 512-wide latent, on SM90 here."""
    return (
        num_heads in (64, 128)
        and head_dim == 512
        and flash_mla_sparse_fwd is not None
        and flash_attn_bwd_sm90 is not None
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability() == (9, 0)
    )


@torch.library.custom_op("prime_rl::dsv41_sparse_attn", mutates_args=())
def dsv41_sparse_attn(
    q: torch.Tensor, kv: torch.Tensor, indices: torch.Tensor, sinks: torch.Tensor, sm_scale: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """`q` `(1, t, h, d)`, `kv` `(1, n, 1, d)`, `indices` `(1, t, 1, k)` int32 -> output and sink-free LSE."""
    _, t, h, d = q.shape
    out, _max_logits, lse = flash_mla_sparse_fwd(
        q.view(t, h, d), kv.view(-1, 1, d), _pad_slots(indices).view(t, 1, -1), sm_scale, d, attn_sink=sinks
    )
    return out.view(1, t, h, d), lse.view(1, t, h)


@dsv41_sparse_attn.register_fake
def _dsv41_sparse_attn_fake(q, kv, indices, sinks, sm_scale):
    return torch.empty_like(q), q.new_empty(q.shape[:-1], dtype=torch.float32)


@torch.library.custom_op("prime_rl::dsv41_sparse_attn_backward", mutates_args=())
def dsv41_sparse_attn_backward(
    grad_out: torch.Tensor,
    q: torch.Tensor,
    kv: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    indices: torch.Tensor,
    sinks: torch.Tensor,
    sm_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _, t, h, d = q.shape
    dq, dkv, dsinks = flash_attn_bwd_sm90(
        q.view(t, h, d),
        kv.view(-1, d),
        out.view(t, h, d),
        grad_out.contiguous().view(t, h, d),
        lse.view(t, h),
        attn_sink=sinks,
        softmax_scale=sm_scale,
        topk_idxs=_pad_slots(indices).view(t, -1),
        need_d_sink=True,
    )
    return dq.view_as(q), dkv.view_as(kv).to(kv.dtype), dsinks.to(sinks.dtype)


@dsv41_sparse_attn_backward.register_fake
def _dsv41_sparse_attn_backward_fake(grad_out, q, kv, out, lse, indices, sinks, sm_scale):
    return torch.empty_like(q), torch.empty_like(kv), torch.empty_like(sinks)


def _setup_context(ctx, inputs, output) -> None:
    q, kv, indices, sinks, sm_scale = inputs
    out, lse = output
    ctx.save_for_backward(q, kv, out, lse, indices, sinks)
    ctx.sm_scale = sm_scale


@functools.cache
def _sparse_attn_backward_impl(num_heads: int, head_dim: int):
    """prime-kernels' Hopper backward when it is built for this GPU and shape (~1.2x cuDNN's), else cuDNN's."""
    import prime_kernels

    if "dsa_sparse_attn_bwd" in prime_kernels.KERNELS and prime_kernels.is_available("dsa_sparse_attn_bwd"):
        kernel = prime_kernels.load("dsa_sparse_attn_bwd")
        if kernel.unsupported_shape_reason(num_heads, head_dim) is None:
            return kernel.dsa_sparse_attn_backward
    return dsv41_sparse_attn_backward


def _backward(ctx, grad_out: torch.Tensor, _grad_lse: torch.Tensor | None):
    q, kv, out, lse, indices, sinks = ctx.saved_tensors
    backward = _sparse_attn_backward_impl(q.shape[-2], q.shape[-1])
    dq, dkv, dsinks = backward(grad_out, q, kv, out, lse, indices, sinks, ctx.sm_scale)
    return dq, dkv, None, dsinks, None


dsv41_sparse_attn.register_autograd(_backward, setup_context=_setup_context)


__all__ = ["dsv41_sparse_attn", "flashmla_sparse_attn_available"]
