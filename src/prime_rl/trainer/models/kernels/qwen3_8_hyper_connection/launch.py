"""Launch configurations and host wrappers of the hyper-connection Triton kernels: the read forward and backward and the
write backward."""

import torch
import triton
from torch import Tensor

from prime_rl.trainer.models.kernels.qwen3_8_hyper_connection.kernels import (
    _gate_backward_kernel,
    _gates_kernel,
    _mix_kernel,
    _projection_backward_kernel,
    _rstd_kernel,
    _stream_backward_kernel,
    _write_backward_kernel,
)

CHUNK_TOKENS = 16384


def _rank_block(rank: int) -> int:
    return max(16, triton.next_power_of_2(rank))


def _rank_class(rank: int) -> int:
    """The rank block the launch configurations below were tuned for: 128 (ranks up to 128), 256, or 512 (larger)."""
    return min(512, max(128, _rank_block(rank)))


# Launch configurations of the kernels with the gate GEMM, measured on H100 for ranks 128, 160 and 320: rows and
# channels per tile, ranks per step of the gate GEMM (ranks up to 256 take one step), warps, stages, and for the
# backward kernels programs per SM.
_MIX_CONFIGS = {128: (64, 128, 128, 4, 2), 256: (128, 64, 256, 4, 1), 512: (64, 128, 64, 4, 3)}
_GATE_BACKWARD_CONFIGS = {128: (128, 64, 128, 4, 1, 2), 256: (64, 128, 256, 4, 1, 4), 512: (64, 128, 64, 4, 3, 8)}
_STREAM_BACKWARD_CONFIGS = {128: (128, 64, 128, 4, 2, 2), 256: (64, 128, 256, 4, 1, 4), 512: (128, 64, 64, 4, 3, 8)}


def _gate_config(configs: dict[int, tuple[int, ...]], rank: int) -> tuple[int, ...]:
    block_m, block_d, block_r, *rest = configs[_rank_class(rank)]
    return block_m, block_d, min(block_r, _rank_block(rank)), *rest


def _project(streams: Tensor, norm_weight: Tensor, down_weight: Tensor, stream_count: int) -> Tensor:
    """Each stream's projection, [stream_count, M, P padded to a multiple of 8 for aligned cuBLAS kernels] in fp32:
    the unnormalized stream times the down weight with the norm's gain folded in, so each stream's rstd still has to
    scale its rows."""
    M, E = streams.shape
    D = E // stream_count
    P = down_weight.shape[0]
    p_pad = triton.cdiv(P, 8) * 8
    weight = streams.new_zeros(p_pad, E)
    torch.addcmul(down_weight, down_weight, norm_weight, out=weight[:P])
    return torch.bmm(
        streams.view(M, stream_count, D).transpose(0, 1),
        weight.view(p_pad, stream_count, D).permute(1, 2, 0),
        out_dtype=torch.float32,
    )


def _row_splits(streams: Tensor, column_blocks: int, block_m: int, waves: int) -> int:
    """Programs per column block, each looping over a share of the row blocks, for about `waves` programs per SM."""
    sms = torch.cuda.get_device_properties(streams.device).multi_processor_count
    return max(1, min(triton.cdiv(streams.shape[0], block_m), waves * sms // column_blocks))


def read_forward(
    streams: Tensor,
    norm_weight: Tensor,
    down_weight: Tensor,
    up_weight: Tensor,
    stream_count: int,
    eps: float,
    write: tuple[Tensor, Tensor] | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """The read's block input, write gates, rstd, projection and the streams it read: ``streams``, or with ``write`` =
    (block_output, inject) the streams after the previous sublayer's write."""
    M, E = streams.shape
    # The kernels index the streams with 32-bit offsets.
    assert streams.numel() < 2**31
    D = E // stream_count
    R = up_weight.shape[1]
    P = down_weight.shape[0]
    NI = P - R
    rstd = streams.new_empty(M, stream_count, dtype=torch.float32)
    block_output, inject, new_streams = (
        (*write, torch.empty_like(streams)) if write is not None else (streams, rstd, streams)
    )
    _rstd_kernel[(stream_count, triton.cdiv(M, 8))](
        streams,
        block_output,
        inject,
        new_streams,
        rstd,
        M,
        D,
        eps,
        NS=stream_count,
        WRITE=write is not None,
        BLOCK_M=8,
        BLOCK_D=min(512, triton.next_power_of_2(D)),
        num_warps=4,
    )
    streams = new_streams
    own = _project(streams, norm_weight, down_weight, stream_count)
    projected = own.new_empty(M, P)
    hidden = streams.new_empty(M, R, dtype=up_weight.dtype)
    inject = streams.new_empty(M, NI)
    _gates_kernel[(triton.cdiv(M, 2),)](
        own,
        rstd,
        projected,
        hidden,
        inject,
        M,
        R,
        own.shape[2],
        NS=stream_count,
        NI=NI,
        BLOCK_M=2,
        BLOCK_P=_rank_block(P),
        num_warps=4,
    )
    out = streams.new_empty(M, D)
    block_m, block_d, block_r, warps, stages = _gate_config(_MIX_CONFIGS, R)
    _mix_kernel[(triton.cdiv(D, block_d), triton.cdiv(M, block_m))](
        streams,
        norm_weight,
        up_weight,
        rstd,
        hidden,
        out,
        M,
        D,
        R,
        NS=stream_count,
        ONE_STEP=R <= block_r,
        BLOCK_M=block_m,
        BLOCK_D=block_d,
        BLOCK_R=block_r,
        num_warps=warps,
        num_stages=stages,
    )
    return out, inject, rstd, projected, streams


def _read_backward_rows(
    grad_out: Tensor,
    grad_inject: Tensor | None,
    streams: Tensor,
    norm_weight: Tensor,
    down_weight: Tensor,
    up_weight: Tensor,
    rstd: Tensor,
    projected: Tensor,
    stream_count: int,
    grad_streams_in: Tensor | None,
    grad_streams: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """``read_backward`` for a block of rows: writes the streams' gradient to ``grad_streams`` and returns this block's
    share of the fp32 gradients of the norm weight, the down projection (without the gain) and the up projection."""
    M, E = streams.shape
    D = E // stream_count
    R = up_weight.shape[1]
    P = down_weight.shape[0]
    NI = P - R
    if grad_inject is None:
        grad_inject = projected.new_zeros(M, NI)
    hidden = torch.nn.functional.silu(projected[:, :R]).to(up_weight.dtype)
    grad_logits = streams.new_empty(M, E, dtype=up_weight.dtype)
    block_m, block_d, block_r, warps, stages, waves = _gate_config(_GATE_BACKWARD_CONFIGS, R)
    channel_blocks = triton.cdiv(D, block_d)
    row_dot_partial = rstd.new_empty(channel_blocks, M, stream_count)
    splits = _row_splits(streams, stream_count * channel_blocks, block_m, waves)
    _gate_backward_kernel[(stream_count * channel_blocks, splits)](
        streams,
        norm_weight,
        up_weight,
        rstd,
        hidden,
        grad_out,
        grad_logits,
        row_dot_partial,
        M,
        D,
        R,
        NS=stream_count,
        ONE_STEP=R <= block_r,
        BLOCK_M=block_m,
        BLOCK_D=block_d,
        BLOCK_R=block_r,
        num_warps=warps,
        num_stages=stages,
    )
    grad_hidden = grad_logits @ up_weight
    grad_up = torch.mm(grad_logits.t(), hidden, out_dtype=torch.float32)
    # The streams' own projections again, for the projection's share of the norm backward's row dot product.
    own = _project(streams, norm_weight, down_weight, stream_count)
    p_pad = own.shape[2]
    grad_projected = projected.new_empty(M, p_pad, dtype=down_weight.dtype)
    scaled = projected.new_empty(stream_count, M, p_pad, dtype=down_weight.dtype)
    row_dot = row_dot_partial.sum(0)
    _projection_backward_kernel[(triton.cdiv(M, 2),)](
        projected,
        grad_hidden,
        grad_inject,
        own,
        rstd,
        row_dot,
        grad_projected,
        scaled,
        M,
        R,
        p_pad,
        NS=stream_count,
        NI=NI,
        BLOCK_M=2,
        BLOCK_P=_rank_block(p_pad),
        num_warps=2,
    )
    grad_projection = grad_projected @ torch.nn.functional.pad(down_weight, (0, 0, 0, p_pad - P))
    block_m, block_d, block_r, warps, stages, waves = _gate_config(_STREAM_BACKWARD_CONFIGS, R)
    channel_blocks = triton.cdiv(D, block_d)
    splits = _row_splits(streams, stream_count * channel_blocks, block_m, waves)
    grad_norm = projected.new_empty(splits, E)
    _stream_backward_kernel[(stream_count * channel_blocks, splits)](
        streams,
        norm_weight,
        up_weight,
        rstd,
        hidden,
        grad_out,
        grad_projection,
        row_dot,
        grad_streams if grad_streams_in is None else grad_streams_in,
        grad_streams,
        grad_norm,
        M,
        D,
        R,
        NS=stream_count,
        HAS_GRAD_STREAMS_IN=grad_streams_in is not None,
        ONE_STEP=R <= block_r,
        BLOCK_M=block_m,
        BLOCK_D=block_d,
        BLOCK_R=block_r,
        num_warps=warps,
        num_stages=stages,
    )
    # The down projection's input is the normalized streams: the projection gradient scaled by each stream's rstd
    # (above) times the streams; the caller applies the gain.
    grad_down = torch.bmm(
        scaled.transpose(1, 2), streams.view(M, stream_count, D).transpose(0, 1), out_dtype=torch.float32
    )
    return grad_norm.sum(0), grad_down[:, :P].transpose(0, 1).reshape(P, E), grad_up


def read_backward(
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
    grad_streams_in: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """The gradients of the streams, the norm weight, the down and the up projection. ``grad_streams_in`` is the
    gradient the streams get from elsewhere, added to theirs. The rows go in blocks of CHUNK_TOKENS, so the
    ``(tokens, stream_count * hidden)`` intermediates exist for one block at a time."""
    grad_streams = torch.empty_like(streams)
    grad_norm, grad_down, grad_up = 0, 0, 0
    for start in range(0, streams.shape[0], CHUNK_TOKENS):
        rows = slice(start, start + CHUNK_TOKENS)
        norm, down, up = _read_backward_rows(
            grad_out[rows],
            None if grad_inject is None else grad_inject[rows],
            streams[rows],
            norm_weight,
            down_weight,
            up_weight,
            rstd[rows],
            projected[rows],
            stream_count,
            None if grad_streams_in is None else grad_streams_in[rows],
            grad_streams[rows],
        )
        grad_norm, grad_down, grad_up = grad_norm + norm, grad_down + down, grad_up + up
    grad_down = grad_down * (1.0 + norm_weight.float())
    return grad_streams, grad_norm.to(norm_weight.dtype), grad_down.to(down_weight.dtype), grad_up.to(up_weight.dtype)


def write_backward(grad_streams: Tensor, block_output: Tensor, inject: Tensor) -> tuple[Tensor, Tensor]:
    """The gradients of a write's block output and its write gates, given the gradient of the written streams."""
    M, D = block_output.shape
    stream_count = inject.shape[1]
    grad_block_output = torch.empty_like(block_output)
    grad_inject = torch.empty_like(inject)
    _write_backward_kernel[(triton.cdiv(M, 32),)](
        grad_streams,
        block_output,
        inject,
        grad_block_output,
        grad_inject,
        M,
        D,
        NS=stream_count,
        NS_POW2=triton.next_power_of_2(stream_count),
        BLOCK_M=32,
        BLOCK_D=128,
        num_warps=4,
    )
    return grad_block_output, grad_inject
