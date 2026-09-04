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

from prime_rl.trainer.models.kernels.deepseek_v4.dsv4_sparse_attn_bwd import bwd, postprocess, preprocess
from prime_rl.trainer.models.kernels.deepseek_v4.dsv4_sparse_attn_fwd import dsv4_sparse_attn_fwd

_LOG2E = 1.44269504


def sparse_attn_shape_error(heads: int, kv_group: int, dim: int) -> str | None:
    """The reason these kernels cannot serve this shape, or ``None`` if they can.

    The forward, the backward and `auto` implementation selection in
    `deepseek_v4/attention.py` all need the same answer, so the constraints live here only.
    """
    # The backward's `preprocess` tiles the channel axis at `block_ND = 32` and reads whole
    # tiles, so a `dim` below that (or not a multiple of it) over-reads into the next head and
    # silently corrupts `Delta`, hence every gradient. This subsumes `atomic_addx4`'s own
    # requirement that four channels be contiguous.
    if dim % 32 != 0:
        return f"the kernels tile the channel axis at 32 and read whole tiles, but head_dim is {dim}"
    # The kernels tile the head axis up to a power of two, at least 16, and index `Sinks`, `Q`,
    # `dO`, `Lse` and `dQ` over that padded block. A head count the tiler pads runs off the end
    # of all of them; `Q` and `Output` absorb the over-read into the next token's heads, but
    # `Sinks` is one row with nothing after it.
    head_kv = heads // kv_group
    padded_heads = max(tilelang.math.next_power_of_2(head_kv), 16)
    if padded_heads != head_kv:
        return (
            f"the kernels tile {padded_heads} heads per group but this shape has {head_kv}; "
            "a head count the tiler pads would read and write past the end of the head axis"
        )
    # The backward runs `block_H = min(64, padded_H)` rows through a GEMM that needs at least 32.
    # At 16 heads the forward compiles and runs, then the backward dies mid-step inside tilelang
    # with "warp_row_tiles must be greater than 16", so a forward-only test will not catch this.
    if head_kv < 32:
        return (
            f"the sparse attention backward needs at least 32 heads per group, got {head_kv}; "
            "its GEMM over min(64, heads) rows fails to compile below that"
        )
    return None


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
    n_sentinel: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert q.is_contiguous(), "q must be contiguous"
    assert kv.is_contiguous(), "kv must be contiguous"
    assert indices.is_contiguous(), "indices must be contiguous"
    batch, seq_len, heads, dim = q.shape
    _, _, kv_group, _ = kv.shape

    assert kv.shape[-1] == dim, "q and kv must share the full channel dim; DS V4 has no score-only tail"
    assert kv.shape[0] == batch
    assert q.dtype == torch.bfloat16, (
        f"the sparse attention kernel runs in bfloat16 only, but the queries are {q.dtype}"
    )
    shape_error = sparse_attn_shape_error(heads, kv_group, dim)
    assert shape_error is None, shape_error
    topk = indices.shape[-1]
    assert indices.shape == (batch, seq_len, kv_group, topk)
    assert sinks.shape == (heads,)

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
        n_sentinel=n_sentinel,
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
    n_sentinel: int = 1,
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
    n_sentinel: int = 1,
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
    # This op is public, so it repeats the forward's shape checks rather than trusting autograd.
    shape_error = sparse_attn_shape_error(heads, kv_group, dim)
    assert shape_error is None, shape_error
    topk = indices.shape[-1]
    assert indices.shape == (batch, seq_len, kv_group, topk)
    assert lse.shape == (batch, seq_len, heads)

    preprocess_kernel = preprocess(heads, dim)
    bwd_kernel = bwd(heads, dim, topk, kv_group, sm_scale, True, n_sentinel=n_sentinel)
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
    n_sentinel: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.empty_like(q), torch.empty_like(kv), torch.empty_like(lse)


def _dsv4_sparse_attn_setup_context(ctx, inputs, output) -> None:
    q, kv, indices, sinks, sm_scale, _block_I, _num_stages, _threads, n_sentinel = inputs
    out, lse = output
    ctx.save_for_backward(q, kv, out, indices, lse, sinks)
    ctx.sm_scale = sm_scale
    ctx.n_sentinel = n_sentinel
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
        n_sentinel=ctx.n_sentinel,
    )
    # dp_k/dsink = -p_k * p_sink, so do[d]/dsink = -p_sink * o[d] and the head's sink gradient
    # contracts to -p_sink * Delta. The sink logit is unscaled, hence no sm_scale factor.
    p_sink = torch.exp2(sinks.float().view(1, 1, -1) * _LOG2E - lse)
    dsink = -(p_sink * delta).sum(dim=(0, 1)).to(sinks.dtype)
    return dq, dkv, None, dsink, None, None, None, None, None


dsv4_sparse_attn.register_autograd(_dsv4_sparse_attn_autograd_backward, setup_context=_dsv4_sparse_attn_setup_context)
