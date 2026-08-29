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
