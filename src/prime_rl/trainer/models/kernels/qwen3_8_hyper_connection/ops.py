"""Custom ops of the hyper-connection read and write + read with their fake implementations and autograd:
`fused_hc_mix` and `fused_hc_combine_mix`."""

import torch
from torch import Tensor

from prime_rl.trainer.models.kernels.qwen3_8_hyper_connection.launch import read_backward, read_forward, write_backward


@torch.library.custom_op("prime_rl::qwen3_8_hc_read", mutates_args=())
def _read(
    streams: Tensor, norm_weight: Tensor, down_weight: Tensor, up_weight: Tensor, stream_count: int, eps: float
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return read_forward(streams, norm_weight, down_weight, up_weight, stream_count, eps)[:4]


def _read_outputs(streams: Tensor, down_weight: Tensor, up_weight: Tensor, stream_count: int) -> tuple[Tensor, ...]:
    """Empty tensors shaped like a read's outputs: block input, write gates, rstd, projection."""
    M, E = streams.shape
    P, R = down_weight.shape[0], up_weight.shape[1]
    return (
        streams.new_empty(M, E // stream_count),
        streams.new_empty(M, P - R),
        streams.new_empty(M, stream_count, dtype=torch.float32),
        streams.new_empty(M, P, dtype=torch.float32),
    )


@_read.register_fake
def _read_fake(
    streams: Tensor, norm_weight: Tensor, down_weight: Tensor, up_weight: Tensor, stream_count: int, eps: float
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return _read_outputs(streams, down_weight, up_weight, stream_count)


@torch.library.custom_op("prime_rl::qwen3_8_hc_read_backward", mutates_args=())
def _read_backward(
    grad_out: Tensor,
    grad_inject: Tensor | None,
    streams: Tensor,
    norm_weight: Tensor,
    down_weight: Tensor,
    up_weight: Tensor,
    rstd: Tensor,
    projected: Tensor,
    stream_count: int,
    eps: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return read_backward(
        grad_out, grad_inject, streams, norm_weight, down_weight, up_weight, rstd, projected, stream_count, eps
    )


@_read_backward.register_fake
def _read_backward_fake(
    grad_out: Tensor,
    grad_inject: Tensor | None,
    streams: Tensor,
    norm_weight: Tensor,
    down_weight: Tensor,
    up_weight: Tensor,
    rstd: Tensor,
    projected: Tensor,
    stream_count: int,
    eps: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return (
        torch.empty_like(streams),
        torch.empty_like(norm_weight),
        torch.empty_like(down_weight),
        torch.empty_like(up_weight),
    )


def _read_setup_context(ctx, inputs, output) -> None:
    streams, norm_weight, down_weight, up_weight, stream_count, eps = inputs
    _, _, rstd, projected = output
    ctx.mark_non_differentiable(rstd, projected)
    ctx.save_for_backward(streams, norm_weight, down_weight, up_weight, rstd, projected)
    ctx.stream_count = stream_count
    ctx.eps = eps


def _read_autograd_backward(ctx, grad_out: Tensor, grad_inject: Tensor | None, *_):
    streams, norm_weight, down_weight, up_weight, rstd, projected = ctx.saved_tensors
    if grad_inject is not None:
        grad_inject = grad_inject.contiguous()
    grads = _read_backward(
        grad_out.contiguous(),
        grad_inject,
        streams,
        norm_weight,
        down_weight,
        up_weight,
        rstd,
        projected,
        ctx.stream_count,
        ctx.eps,
    )
    return *grads, None, None


_read.register_autograd(_read_autograd_backward, setup_context=_read_setup_context)


def _projection_weight(down_weight: Tensor, inject_weight: Tensor | None) -> Tensor:
    """The read's down projection stacked on the write-gate projection: both project the same normalized streams."""
    return down_weight if inject_weight is None else torch.cat([down_weight, inject_weight])


def fused_hc_mix(
    x: Tensor,
    norm_weight: Tensor,
    down_weight: Tensor,
    up_weight: Tensor,
    inject_weight: Tensor | None,
    stream_count: int,
    eps: float,
) -> tuple[Tensor, Tensor | None]:
    """The block input of a read of ``x`` (``[..., stream_count * hidden]``) and its injection weights (None without
    ``inject_weight``)."""
    batch_shape = x.shape[:-1]
    out, inject, _, _ = _read(
        x.reshape(-1, x.shape[-1]),
        norm_weight,
        _projection_weight(down_weight, inject_weight),
        up_weight,
        stream_count,
        eps,
    )
    return out.view(*batch_shape, -1), None if inject_weight is None else inject.view(*batch_shape, -1)


@torch.library.custom_op("prime_rl::qwen3_8_hc_write_read", mutates_args=())
def _write_read(
    streams: Tensor,
    block_output: Tensor,
    inject: Tensor,
    norm_weight: Tensor,
    down_weight: Tensor,
    up_weight: Tensor,
    stream_count: int,
    eps: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    out, new_inject, rstd, projected, new_streams = read_forward(
        streams, norm_weight, down_weight, up_weight, stream_count, eps, (block_output, inject)
    )
    return new_streams, out, new_inject, rstd, projected


@_write_read.register_fake
def _write_read_fake(
    streams: Tensor,
    block_output: Tensor,
    inject: Tensor,
    norm_weight: Tensor,
    down_weight: Tensor,
    up_weight: Tensor,
    stream_count: int,
    eps: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    return (torch.empty_like(streams), *_read_outputs(streams, down_weight, up_weight, stream_count))


@torch.library.custom_op("prime_rl::qwen3_8_hc_write_read_backward", mutates_args=())
def _write_read_backward(
    grad_new_streams: Tensor | None,
    grad_out: Tensor,
    grad_new_inject: Tensor | None,
    new_streams: Tensor,
    block_output: Tensor,
    inject: Tensor,
    norm_weight: Tensor,
    down_weight: Tensor,
    up_weight: Tensor,
    rstd: Tensor,
    projected: Tensor,
    stream_count: int,
    eps: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    grad_streams, grad_norm, grad_down, grad_up = read_backward(
        grad_out,
        grad_new_inject,
        new_streams,
        norm_weight,
        down_weight,
        up_weight,
        rstd,
        projected,
        stream_count,
        eps,
        grad_new_streams,
    )
    # The written streams are the read streams plus the write: the streams get the same gradient.
    grad_block_output, grad_inject = write_backward(grad_streams, block_output, inject)
    return grad_streams, grad_block_output, grad_inject, grad_norm, grad_down, grad_up


@_write_read_backward.register_fake
def _write_read_backward_fake(
    grad_new_streams: Tensor | None,
    grad_out: Tensor,
    grad_new_inject: Tensor | None,
    new_streams: Tensor,
    block_output: Tensor,
    inject: Tensor,
    norm_weight: Tensor,
    down_weight: Tensor,
    up_weight: Tensor,
    rstd: Tensor,
    projected: Tensor,
    stream_count: int,
    eps: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    return (
        torch.empty_like(new_streams),
        torch.empty_like(block_output),
        torch.empty_like(inject),
        torch.empty_like(norm_weight),
        torch.empty_like(down_weight),
        torch.empty_like(up_weight),
    )


def _write_read_setup_context(ctx, inputs, output) -> None:
    _, block_output, inject, norm_weight, down_weight, up_weight, stream_count, eps = inputs
    new_streams, _, _, rstd, projected = output
    ctx.mark_non_differentiable(rstd, projected)
    ctx.save_for_backward(new_streams, block_output, inject, norm_weight, down_weight, up_weight, rstd, projected)
    ctx.stream_count = stream_count
    ctx.eps = eps


def _write_read_autograd_backward(ctx, grad_new_streams: Tensor | None, grad_out: Tensor, grad_new_inject, *_):
    grads = _write_read_backward(
        None if grad_new_streams is None else grad_new_streams.contiguous(),
        grad_out.contiguous(),
        None if grad_new_inject is None else grad_new_inject.contiguous(),
        *ctx.saved_tensors,
        ctx.stream_count,
        ctx.eps,
    )
    return *grads, None, None


_write_read.register_autograd(_write_read_autograd_backward, setup_context=_write_read_setup_context)


def fused_hc_combine_mix(
    x: Tensor,
    block_output: Tensor,
    injection: Tensor,
    norm_weight: Tensor,
    down_weight: Tensor,
    up_weight: Tensor,
    inject_weight: Tensor | None,
    stream_count: int,
    eps: float,
) -> tuple[Tensor, Tensor, Tensor | None]:
    """The previous sublayer's write (``combine``: x + block_output * injection per stream) fused with the next read:
    the written streams, and the read's block input and injection weights (see ``fused_hc_mix``)."""
    batch_shape = x.shape[:-1]
    new_x, out, new_inject, _, _ = _write_read(
        x.reshape(-1, x.shape[-1]),
        block_output.reshape(-1, block_output.shape[-1]),
        injection.reshape(-1, stream_count),
        norm_weight,
        _projection_weight(down_weight, inject_weight),
        up_weight,
        stream_count,
        eps,
    )
    new_inject = None if inject_weight is None else new_inject.view(*batch_shape, -1)
    return new_x.view(x.shape), out.view(*batch_shape, -1), new_inject
