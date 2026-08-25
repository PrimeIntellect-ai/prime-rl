"""Low-level varlen flash-attention forward/backward wrappers, one pair per FA version.

Each pair exposes the same signature regardless of the kernel underneath, so callers that need
raw `out`/`lse` access (ring CP in `ring_attn.py`, ulysses CP in `ulysses_attn.py`) can select a
kernel by version number instead of branching on it.
"""

from __future__ import annotations

# ruff: noqa: I001 (`prime_rl._compat` must run before the `ring_flash_attn` imports below)
import prime_rl._compat  # noqa: F401

import torch
from ring_flash_attn.utils import get_default_args


def _set_causal_and_window_params(params: dict, causal: bool, window_size: tuple[int, int]) -> None:
    """Fill in the causal/window arguments, whose names differ across flash-attention builds."""
    if "is_causal" in params:
        params["is_causal"] = causal
    else:
        params["causal"] = causal

    if "window_size" in params:
        params["window_size"] = window_size
    else:
        params["window_size_left"] = window_size[0]
        params["window_size_right"] = window_size[1]


def _fa2_varlen_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    causal: bool,
    window_size: tuple[int, int] = (-1, -1),
) -> tuple[torch.Tensor, torch.Tensor]:
    from flash_attn.flash_attn_interface import _flash_attn_varlen_forward

    params = get_default_args(_flash_attn_varlen_forward).copy()
    params.update(
        {
            "q": q,
            "k": k,
            "v": v,
            "cu_seqlens_q": cu_seqlens_q,
            "cu_seqlens_k": cu_seqlens_k,
            "max_seqlen_q": max_seqlen_q,
            "max_seqlen_k": max_seqlen_k,
            "dropout_p": 0.0,
            "softmax_scale": softmax_scale,
        }
    )
    _set_causal_and_window_params(params, causal, window_size)
    out, lse, _, _ = _flash_attn_varlen_forward(**params)
    return out, lse


def _fa2_varlen_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    softmax_scale: float,
    causal: bool,
    window_size: tuple[int, int] = (-1, -1),
) -> None:
    from flash_attn.flash_attn_interface import _flash_attn_varlen_backward

    params = get_default_args(_flash_attn_varlen_backward).copy()
    params.update(
        {
            "dout": dout,
            "q": q,
            "k": k,
            "v": v,
            "out": out,
            "softmax_lse": softmax_lse,
            "cu_seqlens_q": cu_seqlens_q,
            "cu_seqlens_k": cu_seqlens_k,
            "max_seqlen_q": max_seqlen_q,
            "max_seqlen_k": max_seqlen_k,
            "dq": dq,
            "dk": dk,
            "dv": dv,
            "dropout_p": 0.0,
            "softmax_scale": softmax_scale,
        }
    )
    _set_causal_and_window_params(params, causal, window_size)
    _flash_attn_varlen_backward(**params)


def _fa3_varlen_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    causal: bool,
    window_size: tuple[int, int] = (-1, -1),
) -> tuple[torch.Tensor, torch.Tensor]:
    from flash_attn_interface import _flash_attn_forward

    params = get_default_args(_flash_attn_forward).copy()
    params.update(
        {
            "q": q,
            "k": k,
            "v": v,
            "cu_seqlens_q": cu_seqlens_q,
            "cu_seqlens_k": cu_seqlens_k,
            "max_seqlen_q": max_seqlen_q,
            "max_seqlen_k": max_seqlen_k,
            "softmax_scale": softmax_scale,
        }
    )
    _set_causal_and_window_params(params, causal, window_size)
    out, lse, _, _ = _flash_attn_forward(**params)
    return out, lse


def _fa3_varlen_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    softmax_scale: float,
    causal: bool,
    window_size: tuple[int, int] = (-1, -1),
) -> None:
    from flash_attn_interface import _flash_attn_backward

    params = get_default_args(_flash_attn_backward).copy()
    params.update(
        {
            "dout": dout,
            "q": q,
            "k": k,
            "v": v,
            "out": out,
            "softmax_lse": softmax_lse,
            "cu_seqlens_q": cu_seqlens_q,
            "cu_seqlens_k": cu_seqlens_k,
            "max_seqlen_q": max_seqlen_q,
            "max_seqlen_k": max_seqlen_k,
            "dq": dq,
            "dk": dk,
            "dv": dv,
            "softmax_scale": softmax_scale,
        }
    )
    _set_causal_and_window_params(params, causal, window_size)
    _flash_attn_backward(**params)


def _fa4_varlen_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    causal: bool,
    window_size: tuple[int, int] = (-1, -1),
) -> tuple[torch.Tensor, torch.Tensor]:
    from flash_attn.cute.interface import _flash_attn_fwd

    wl = window_size[0] if window_size[0] != -1 else None
    wr = window_size[1] if window_size[1] != -1 else None
    out, lse = _flash_attn_fwd(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size_left=wl,
        window_size_right=wr,
        return_lse=True,
    )
    return out, lse


def _fa4_varlen_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    softmax_scale: float,
    causal: bool,
    window_size: tuple[int, int] = (-1, -1),
) -> None:
    from flash_attn.cute.interface import _flash_attn_bwd

    wl = window_size[0] if window_size[0] != -1 else None
    wr = window_size[1] if window_size[1] != -1 else None
    _flash_attn_bwd(
        q,
        k,
        v,
        out,
        dout,
        softmax_lse,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size_left=wl,
        window_size_right=wr,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        dq=dq,
        dk=dk,
        dv=dv,
    )


VARLEN_FORWARD = {2: _fa2_varlen_forward, 3: _fa3_varlen_forward, 4: _fa4_varlen_forward}
VARLEN_BACKWARD = {2: _fa2_varlen_backward, 3: _fa3_varlen_backward, 4: _fa4_varlen_backward}


def apply_sink(out: torch.Tensor, lse: torch.Tensor, sink: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Fold a per-head additive sink logit into a sink-unaware kernel's output and LSE.

    A sink is a virtual softmax column whose value vector is zero, so it contributes nothing to
    the numerator and only enlarges the denominator. Writing `Z = exp(lse)` for the sink-free
    denominator, the sink-inclusive output is `out * Z / (Z + exp(sink))`, i.e.
    `out * sigmoid(lse - sink)`, and the new LSE is `logaddexp(lse, sink)`.

    `out` is `[total_q, nheads, head_dim]`, `lse` is `[nheads, total_q]`, `sink` is `[nheads]`.
    """
    sink = sink.to(lse.dtype).unsqueeze(-1)
    scale = torch.sigmoid(lse - sink)
    out = (out.float() * scale.transpose(0, 1).unsqueeze(-1)).to(out.dtype)
    return out, torch.logaddexp(lse, sink.expand_as(lse))


def sink_grad(out: torch.Tensor, lse: torch.Tensor, dout: torch.Tensor, sink: torch.Tensor) -> torch.Tensor:
    """Gradient of the loss with respect to the per-head sink logit.

    The sink column's softmax weight is `p = exp(sink - lse)` and its value vector is zero, so its
    contribution to the usual `dscore = p * (dot(dout, v) - D)` reduces to `-p * D`, where
    `D[q,h] = sum_d out[q,h,d] * dout[q,h,d]`. Summing over queries gives the whole gradient.

    `out` and `lse` must both already include the sink (see `apply_sink`).
    """
    D = (out.float() * dout.float()).sum(-1).transpose(0, 1)
    p = torch.exp(sink.to(lse.dtype).unsqueeze(-1) - lse)
    return -(p * D).sum(-1).to(sink.dtype)


class _SinkVarlen(torch.autograd.Function):
    """Varlen flash attention with a per-head additive sink logit, on one device.

    No flash-attention backward kernel differentiates its own sink argument, so the sink is
    applied outside the kernel instead (`apply_sink`), which also makes this work with FA2, whose
    forward has no sink argument at all. `dq`/`dk`/`dv` still come straight from the ordinary
    backward kernel: fed the sink-inclusive `out` and `lse`, its `D = rowsum(dout * out)` term is
    already the sink-inclusive one, since the sink column contributes zero to it.
    """

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        sink: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        causal: bool,
        flash_attn_version: int,
        softmax_scale: float | None = None,
        window_size_left: int = -1,
        window_size_right: int = -1,
    ) -> torch.Tensor:
        window_size = (window_size_left, window_size_right)
        softmax_scale = q.shape[-1] ** (-0.5) if softmax_scale is None else softmax_scale
        out, lse = VARLEN_FORWARD[flash_attn_version](
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
        )
        out, lse = apply_sink(out, lse, sink)

        ctx.save_for_backward(q, k, v, sink, out, lse, cu_seqlens_q, cu_seqlens_k)
        ctx.softmax_scale = softmax_scale
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.causal = causal
        ctx.flash_attn_version = flash_attn_version
        ctx.window_size = window_size
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        q, k, v, sink, out, lse, cu_seqlens_q, cu_seqlens_k = ctx.saved_tensors
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        VARLEN_BACKWARD[ctx.flash_attn_version](
            dout=dout,
            q=q,
            k=k,
            v=v,
            out=out,
            softmax_lse=lse,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=ctx.max_seqlen_q,
            max_seqlen_k=ctx.max_seqlen_k,
            dq=dq,
            dk=dk,
            dv=dv,
            softmax_scale=ctx.softmax_scale,
            causal=ctx.causal,
            window_size=ctx.window_size,
        )
        dsink = sink_grad(out, lse, dout, sink)
        # Grads for: q, k, v, sink, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
        #            causal, flash_attn_version, softmax_scale, window_size_left, window_size_right
        return dq, dk, dv, dsink, None, None, None, None, None, None, None, None, None


def sink_flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sink: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    causal: bool,
    flash_attn_version: int,
    softmax_scale: float | None = None,
    window_size: tuple[int, int] = (-1, -1),
) -> torch.Tensor:
    return _SinkVarlen.apply(
        q,
        k,
        v,
        sink,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        causal,
        flash_attn_version,
        softmax_scale,
        window_size[0],
        window_size[1],
    )
