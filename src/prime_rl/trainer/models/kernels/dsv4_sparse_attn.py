# Torch-facing layer for the DeepSeek V4 sparse attention kernels: the `prime_rl::dsv4_sparse_attn`
# and `prime_rl::dsv4_sparse_attn_backward` custom ops, their fake (meta) implementations, and the
# autograd rule that ties them together and forms the attention-sink gradient in torch. The
# TileLang kernels themselves live in `dsv4_sparse_attn_fwd.py` and `dsv4_sparse_attn_bwd.py`; this
# module is the only place that knows both exist.

# TileLang ships a libcudart stub that proxies to the real CUDA runtime via
# dlsym(RTLD_DEFAULT, ...).  If the stub's own symbols are the first ones found
# (because nothing loaded the real libcudart globally yet), the self-check fails
# and the stub calls abort().  Pre-loading the real library with RTLD_GLOBAL
# ensures dlsym finds it before the stub's own exports.
import ctypes as _ctypes

try:
    _ctypes.CDLL("libcudart.so", mode=_ctypes.RTLD_GLOBAL)
except Exception:
    # This is expected on CPU-only machines
    pass

import tilelang
import torch

from prime_rl.trainer.models.kernels.dsv4_sparse_attn_bwd import bwd, postprocess, preprocess
from prime_rl.trainer.models.kernels.dsv4_sparse_attn_fwd import dsv4_sparse_attn_fwd

_LOG2E = 1.44269504


@torch.library.custom_op("prime_rl::dsv4_sparse_attn", mutates_args=())
def dsv4_sparse_attn(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sinks: torch.Tensor,
    sm_scale: float | None = None,
    block_I: int = 64,
    num_stages: int = 2,
    threads: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert q.is_contiguous(), "q must be contiguous"
    assert kv.is_contiguous(), "kv must be contiguous"
    assert indices.is_contiguous(), "indices must be contiguous"
    batch, seq_len, heads, dim = q.shape
    _, _, kv_group, _ = kv.shape

    assert kv.shape[-1] == dim, "q and kv must share the full channel dim; DS V4 has no score-only tail"
    assert kv.shape[0] == batch
    # The backward's `preprocess` tiles the channel axis at `block_ND = 32` and reads whole
    # tiles, so a `dim` below that (or not a multiple of it) over-reads into the next head and
    # silently corrupts `Delta`, hence every gradient. This subsumes `atomic_addx4`'s own
    # requirement that four channels be contiguous.
    assert dim % 32 == 0, "the backward tiles the channel axis at 32 and reads whole tiles"
    topk = indices.shape[-1]
    assert indices.shape == (batch, seq_len, kv_group, topk)
    assert sinks.shape == (heads,)
    # The kernel tiles the head axis up to a power of two, at least 16, and indexes `Sinks`
    # over that padded block. `Q` and `Output` absorb the over-read into the next token's
    # heads, but `Sinks` is one row with nothing after it, so refuse a head count that pads.
    head_kv = heads // kv_group
    padded_heads = max(tilelang.math.next_power_of_2(head_kv), 16)
    assert padded_heads == head_kv, (
        f"the kernel tiles {padded_heads} heads per group but sinks has {head_kv}; "
        "a head count the tiler pads would read past the end of sinks"
    )
    # The constraint is the backward's, not this kernel's: it runs `block_H = min(64, padded_H)`
    # rows through a GEMM that needs at least 32, so 16 heads compiles and runs here and then
    # dies mid-step inside tilelang with "warp_row_tiles must be greater than 16". Refuse it up
    # front, and do not relax this after testing only a forward pass.
    assert head_kv >= 32, (
        f"the sparse attention backward needs at least 32 heads per group, got {head_kv}; "
        "its GEMM over min(64, heads) rows fails to compile below that"
    )

    kernel = dsv4_sparse_attn_fwd(
        heads,
        dim,
        topk,
        kv_group,
        sm_scale,
        True,
        block_I=block_I,
        num_stages=num_stages,
        threads=threads,
    )
    out, lse = kernel(q, kv, indices, sinks.float().contiguous())
    return out, lse


# A fake must mirror the op's signature exactly, so it takes every argument even though only `q`,
# the one argument that determines the output shapes, is read.
@dsv4_sparse_attn.register_fake
def _dsv4_sparse_attn_fake(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sinks: torch.Tensor,
    sm_scale: float | None = None,
    block_I: int = 64,
    num_stages: int = 2,
    threads: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(q), q.new_empty(q.shape[:-1], dtype=torch.float32)


@torch.library.custom_op("prime_rl::dsv4_sparse_attn_backward", mutates_args=())
def dsv4_sparse_attn_backward(
    q: torch.Tensor,
    kv: torch.Tensor,
    out: torch.Tensor,
    grad_out: torch.Tensor,
    indices: torch.Tensor,
    lse: torch.Tensor,
    sm_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert q.is_contiguous(), "q must be contiguous"
    assert kv.is_contiguous(), "kv must be contiguous"
    assert indices.is_contiguous(), "indices must be contiguous"
    assert lse.is_contiguous(), "lse must be contiguous"
    grad_out = grad_out.contiguous()
    batch, seq_len, heads, dim = q.shape
    _, _, kv_group, _ = kv.shape
    assert kv.shape[-1] == dim, "q and kv must share the full channel dim; DS V4 has no score-only tail"
    assert kv.shape[0] == batch
    # `preprocess` tiles the channel axis at `block_ND = 32` and reads whole tiles, so a `dim`
    # below that (or not a multiple of it) over-reads into the next head and silently corrupts
    # `Delta`, hence `dq`, `dkv` and `dsink`. This subsumes `atomic_addx4`'s own requirement
    # that four channels be contiguous.
    assert dim % 32 == 0, "preprocess tiles the channel axis at 32 and reads whole tiles"
    topk = indices.shape[-1]
    assert indices.shape == (batch, seq_len, kv_group, topk)
    assert lse.shape == (batch, seq_len, heads)
    # This op is public, so it repeats the forward's head checks rather than trusting autograd.
    # The kernel writes whole `block_H` tiles of `dQ` and reads whole tiles of `Q`, `dO` and
    # `Lse`, so a head count the tiler pads runs off the end of all four. Below 32 heads the
    # `block_H = min(64, padded_H)` GEMM fails to compile at all.
    head_kv = heads // kv_group
    padded_heads = max(tilelang.math.next_power_of_2(head_kv), 16)
    assert padded_heads == head_kv, (
        f"the kernel tiles {padded_heads} heads per group but q has {head_kv}; "
        "a head count the tiler pads would read and write past the end of the head axis"
    )
    assert head_kv >= 32, (
        f"the sparse attention backward needs at least 32 heads per group, got {head_kv}; "
        "its GEMM over min(64, heads) rows fails to compile below that"
    )

    preprocess_kernel = preprocess(heads, dim)
    bwd_kernel = bwd(heads, dim, topk, kv_group, sm_scale, True)
    postprocess_kernel = postprocess(dim, kv_group)

    delta = preprocess_kernel(out, grad_out)
    dkv = torch.zeros_like(kv, dtype=torch.float32)
    dq = bwd_kernel(q, kv, grad_out, indices, lse, delta, dkv)
    dkv = postprocess_kernel(dkv)

    return dq, dkv, delta


# As above: the fake mirrors the op's signature, and only the shape-determining arguments are read.
@dsv4_sparse_attn_backward.register_fake
def _dsv4_sparse_attn_backward_fake(
    q: torch.Tensor,
    kv: torch.Tensor,
    out: torch.Tensor,
    grad_out: torch.Tensor,
    indices: torch.Tensor,
    lse: torch.Tensor,
    sm_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.empty_like(q), torch.empty_like(kv), torch.empty_like(lse)


def _dsv4_sparse_attn_setup_context(ctx, inputs, output) -> None:
    q, kv, indices, sinks, sm_scale, _block_I, _num_stages, _threads = inputs
    out, lse = output
    ctx.save_for_backward(q, kv, out, indices, lse, sinks)
    ctx.sm_scale = sm_scale
    ctx.mark_non_differentiable(lse)


def _dsv4_sparse_attn_autograd_backward(ctx, grad_out: torch.Tensor, _grad_lse: torch.Tensor | None):
    q, kv, out, indices, lse, sinks = ctx.saved_tensors
    dq, dkv, delta = dsv4_sparse_attn_backward(
        q.detach(),
        kv.detach(),
        out.detach(),
        grad_out,
        indices,
        lse.detach(),
        ctx.sm_scale,
    )
    # dp_k/dsink = -p_k * p_sink, so do[d]/dsink = -p_sink * o[d] and the head's sink gradient
    # contracts to -p_sink * Delta. The sink logit is unscaled, hence no sm_scale factor.
    p_sink = torch.exp2(sinks.float().view(1, 1, -1) * _LOG2E - lse)
    dsink = -(p_sink * delta).sum(dim=(0, 1)).to(sinks.dtype)
    return dq, dkv, None, dsink, None, None, None, None


dsv4_sparse_attn.register_autograd(_dsv4_sparse_attn_autograd_backward, setup_context=_dsv4_sparse_attn_setup_context)
