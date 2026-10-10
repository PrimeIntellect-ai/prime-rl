"""Fused single-stream gated residual update for DeepSeek-V4.1's `residual_type = "gated" | "layerscale"`.

`y = x + g * (o + o2)`, one Triton program per token row, with the gate either

- per token (`gated`): `g = 2 * sigmoid(scale * rstd(x) * (x . w) + base)`, computed in the same pass from the
  stream entering the sublayer, or
- per channel (`layerscale`): `g = lambda`.

The forward reads `x`, `o`, `o2` and writes `y` (the traffic of a plain residual add). The backward reads them
and `dy` once more, writes `dx` (gated only; for layerscale `dx = dy`) and the shared gradient of `o` and `o2`,
and reduces the parameter gradients over per-program partial sums.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _gated_residual_fwd_kernel(
    x_ptr, o_ptr, o2_ptr, w_ptr, base_ptr, scale_ptr, y_ptr, D, eps,
    TOKEN_GATE: tl.constexpr, HAS_O2: tl.constexpr, BLOCK_D: tl.constexpr,
):  # fmt: skip
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D
    x = tl.load(x_ptr + row * D + offs, mask=mask, other=0.0).to(tl.float32)
    o = tl.load(o_ptr + row * D + offs, mask=mask, other=0.0).to(tl.float32)
    if HAS_O2:
        o += tl.load(o2_ptr + row * D + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    if TOKEN_GATE:
        rstd = tl.rsqrt(tl.sum(x * x, axis=0) / D + eps)
        z = tl.load(scale_ptr).to(tl.float32) * rstd * tl.sum(x * w, axis=0) + tl.load(base_ptr).to(tl.float32)
        y = x + 2 * tl.sigmoid(z) * o
    else:
        y = x + w * o
    tl.store(y_ptr + row * D + offs, y.to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _gated_residual_bwd_kernel(
    dy_ptr, x_ptr, o_ptr, o2_ptr, w_ptr, base_ptr, scale_ptr, dx_ptr, do_ptr, dw_part_ptr, dbs_part_ptr,
    N, D, eps,
    TOKEN_GATE: tl.constexpr, HAS_O2: tl.constexpr, BLOCK_D: tl.constexpr,
):  # fmt: skip
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    dw_acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    dbase_acc = 0.0
    dscale_acc = 0.0
    if TOKEN_GATE:
        base = tl.load(base_ptr).to(tl.float32)
        scale = tl.load(scale_ptr).to(tl.float32)
    for row in range(pid, N, num_programs):
        row = row.to(tl.int64)
        dy = tl.load(dy_ptr + row * D + offs, mask=mask, other=0.0).to(tl.float32)
        o = tl.load(o_ptr + row * D + offs, mask=mask, other=0.0).to(tl.float32)
        if HAS_O2:
            o += tl.load(o2_ptr + row * D + offs, mask=mask, other=0.0).to(tl.float32)
        if TOKEN_GATE:
            x = tl.load(x_ptr + row * D + offs, mask=mask, other=0.0).to(tl.float32)
            rstd = tl.rsqrt(tl.sum(x * x, axis=0) / D + eps)
            dot = tl.sum(x * w, axis=0)
            g = 2 * tl.sigmoid(scale * rstd * dot + base)
            # dg/dz of g = 2 sigmoid(z) is g (1 - g / 2); n = rstd * (x . w) has dn/dx = rstd w - rstd^3 (x . w) x / D.
            dz = tl.sum(dy * o, axis=0) * g * (1 - 0.5 * g)
            dbase_acc += dz
            dscale_acc += dz * rstd * dot
            dn = dz * scale
            dx = dy + dn * rstd * (w - (rstd * rstd * dot / D) * x)
            dw_acc += (dn * rstd) * x
            tl.store(dx_ptr + row * D + offs, dx.to(dx_ptr.dtype.element_ty), mask=mask)
            tl.store(do_ptr + row * D + offs, (g * dy).to(do_ptr.dtype.element_ty), mask=mask)
        else:
            dw_acc += dy * o
            tl.store(do_ptr + row * D + offs, (w * dy).to(do_ptr.dtype.element_ty), mask=mask)
    tl.store(dw_part_ptr + pid * D + offs, dw_acc, mask=mask)
    if TOKEN_GATE:
        tl.store(dbs_part_ptr + 2 * pid, dbase_acc)
        tl.store(dbs_part_ptr + 2 * pid + 1, dscale_acc)


def _launch_meta(D: int) -> dict:
    block = triton.next_power_of_2(D)
    return dict(BLOCK_D=block, num_warps=16 if block >= 8192 else 8)


def _fwd(x, o, o2, w, base, scale, eps):
    D = x.shape[-1]
    x2d, o2d = x.reshape(-1, D), o.contiguous().view(-1, D)
    o22d = o2.contiguous().view(-1, D) if o2 is not None else o2d
    token = base is not None
    y = torch.empty_like(x2d)
    _gated_residual_fwd_kernel[(x2d.shape[0],)](
        x2d, o2d, o22d, w, base if token else w, scale if token else w, y, D, eps,
        TOKEN_GATE=token, HAS_O2=o2 is not None, **_launch_meta(D),
    )  # fmt: skip
    return y.view_as(x)


def _bwd(dy, x, o, o2, w, base, scale, eps):
    D = x.shape[-1]
    dy2d, x2d, o2d = dy.contiguous().view(-1, D), x.reshape(-1, D), o.contiguous().view(-1, D)
    o22d = o2.contiguous().view(-1, D) if o2 is not None else o2d
    token = base is not None
    N = x2d.shape[0]
    P = min(N, 4 * torch.cuda.get_device_properties(x.device).multi_processor_count)
    dx = torch.empty_like(x2d) if token else x2d.new_empty(0)
    do = torch.empty_like(o2d)
    dw_part = torch.empty(P, D, dtype=torch.float32, device=x.device)
    dbs_part = torch.empty(P, 2, dtype=torch.float32, device=x.device)
    _gated_residual_bwd_kernel[(P,)](
        dy2d, x2d, o2d, o22d, w, base if token else w, scale if token else w, dx, do, dw_part, dbs_part, N, D, eps,
        TOKEN_GATE=token, HAS_O2=o2 is not None, **_launch_meta(D),
    )  # fmt: skip
    dw = dw_part.sum(0).to(w.dtype)
    if not token:
        return dx, do.view_as(o), dw, None, None
    dbase, dscale = (dbs_part[:, i].sum().reshape(1).to(base.dtype) for i in range(2))
    return dx.view_as(x), do.view_as(o), dw, dbase, dscale


@torch.library.custom_op("prime_rl::dsv41_gated_residual", mutates_args=())
def _gated_residual(
    x: torch.Tensor,
    o: torch.Tensor,
    o2: torch.Tensor | None,
    w: torch.Tensor,
    base: torch.Tensor | None,
    scale: torch.Tensor | None,
    eps: float,
) -> torch.Tensor:
    return _fwd(x, o, o2, w, base, scale, eps)


@_gated_residual.register_fake
def _gated_residual_fake(x, o, o2, w, base, scale, eps):
    return torch.empty_like(x)


@torch.library.custom_op("prime_rl::dsv41_gated_residual_backward", mutates_args=())
def _gated_residual_backward(
    dy: torch.Tensor,
    x: torch.Tensor,
    o: torch.Tensor,
    o2: torch.Tensor | None,
    w: torch.Tensor,
    base: torch.Tensor | None,
    scale: torch.Tensor | None,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dx, do, dw, dbase, dscale = _bwd(dy, x, o, o2, w, base, scale, eps)
    if base is None:  # custom ops cannot return None; the autograd wrapper drops these
        dbase, dscale = w.new_empty(0), w.new_empty(0)
    return dx, do, dw, dbase, dscale


@_gated_residual_backward.register_fake
def _gated_residual_backward_fake(dy, x, o, o2, w, base, scale, eps):
    token = base is not None
    dx = torch.empty_like(x) if token else x.new_empty(0)
    small = (lambda t: torch.empty_like(t)) if token else (lambda t: w.new_empty(0))
    return dx, torch.empty_like(o), torch.empty_like(w), small(base), small(scale)


def _setup_context(ctx, inputs, output) -> None:
    x, o, o2, w, base, scale, eps = inputs
    ctx.eps = eps
    ctx.has_o2 = o2 is not None
    ctx.token = base is not None
    ctx.save_for_backward(x, o, o2, w, base, scale)


def _autograd_backward(ctx, dy: torch.Tensor):
    x, o, o2, w, base, scale = ctx.saved_tensors
    dx, do, dw, dbase, dscale = _gated_residual_backward(dy, x, o, o2, w, base, scale, ctx.eps)
    if not ctx.token:
        dx, dbase, dscale = dy, None, None
    # The two parts of a split sublayer output share its gradient, as an add's inputs do.
    return dx, do, do if ctx.has_o2 else None, dw, dbase, dscale, None


_gated_residual.register_autograd(_autograd_backward, setup_context=_setup_context)


def token_gated_residual(
    x: torch.Tensor,
    o: torch.Tensor,
    o2: torch.Tensor | None,
    w: torch.Tensor,
    base: torch.Tensor,
    scale: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """`x + 2 sigmoid(scale * rstd(x) (x . w) + base) * (o + o2)` per row; `x` is `(..., 1, D)`, `o` `(..., D)`."""
    return _gated_residual(x, o, o2, w, base, scale, eps)


def channel_scaled_residual(
    x: torch.Tensor, o: torch.Tensor, o2: torch.Tensor | None, scale: torch.Tensor
) -> torch.Tensor:
    """`x + scale * (o + o2)` with one scale per channel."""
    return _gated_residual(x, o, o2, scale, None, None, 0.0)


__all__ = ["channel_scaled_residual", "token_gated_residual"]
