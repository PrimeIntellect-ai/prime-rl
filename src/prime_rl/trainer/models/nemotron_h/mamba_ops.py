import torch
from mamba_ssm.ops.triton.ssd_combined import _mamba_chunk_scan_combined_bwd, _mamba_chunk_scan_combined_fwd
from torch import Tensor


@torch.library.custom_op("prime_rl::mamba_chunk_scan", mutates_args=())
def _mamba_chunk_scan(
    x: Tensor,
    dt: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    D: Tensor,
    dt_bias: Tensor,
    seq_idx: Tensor,
    chunk_size: int,
    dt_min: float,
    dt_max: float,
) -> Tensor:
    out, *_ = _mamba_chunk_scan_combined_fwd(
        x,
        dt,
        A,
        B,
        C,
        chunk_size,
        D=D,
        dt_bias=dt_bias,
        seq_idx=seq_idx,
        dt_softplus=True,
        dt_limit=(dt_min, dt_max),
    )
    return out


@_mamba_chunk_scan.register_fake
def _(x, *args):
    return torch.empty_like(x)


@torch.library.custom_op("prime_rl::mamba_chunk_scan_backward", mutates_args=())
def _mamba_chunk_scan_backward(
    dout: Tensor,
    out: Tensor,
    x: Tensor,
    dt: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    D: Tensor,
    dt_bias: Tensor,
    seq_idx: Tensor,
    chunk_size: int,
    dt_min: float,
    dt_max: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    dx, ddt, dA, dB, dC, dD, _, ddt_bias, _ = _mamba_chunk_scan_combined_bwd(
        dout,
        x,
        dt,
        A,
        B,
        C,
        out,
        chunk_size,
        D=D,
        dt_bias=dt_bias,
        seq_idx=seq_idx,
        dt_softplus=True,
        dt_limit=(dt_min, dt_max),
    )
    return dx.to(x), ddt.to(dt), dA.to(A), dB.to(B), dC.to(C), dD.to(D), ddt_bias.to(dt_bias)


@_mamba_chunk_scan_backward.register_fake
def _(dout, out, x, dt, A, B, C, D, dt_bias, *args):
    return tuple(torch.empty_like(tensor) for tensor in (x, dt, A, B, C, D, dt_bias))


def _mamba_chunk_scan_setup_context(ctx, inputs, output) -> None:
    x, dt, A, B, C, D, dt_bias, seq_idx, chunk_size, dt_min, dt_max = inputs
    ctx.save_for_backward(output, x, dt, A, B, C, D, dt_bias, seq_idx)
    ctx.chunk_size = chunk_size
    ctx.dt_limit = (dt_min, dt_max)


def _mamba_chunk_scan_autograd(ctx, dout: Tensor):
    grads = _mamba_chunk_scan_backward(dout.contiguous(), *ctx.saved_tensors, ctx.chunk_size, *ctx.dt_limit)
    return *grads, None, None, None, None


_mamba_chunk_scan.register_autograd(_mamba_chunk_scan_autograd, setup_context=_mamba_chunk_scan_setup_context)


def mamba_chunk_scan(
    x: Tensor,
    dt: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    D: Tensor,
    dt_bias: Tensor,
    seq_idx: Tensor,
    chunk_size: int,
    dt_limit: tuple[float, float],
) -> Tensor:
    return _mamba_chunk_scan(
        x.contiguous(),
        dt.contiguous(),
        A.contiguous(),
        B.contiguous(),
        C.contiguous(),
        D.contiguous(),
        dt_bias.contiguous(),
        seq_idx,
        chunk_size,
        *dt_limit,
    )
