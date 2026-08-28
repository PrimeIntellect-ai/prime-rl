from __future__ import annotations

# ruff: noqa: I001 — `prime_rl._compat` must run before `ring_flash_attn` imports below.
import prime_rl._compat  # noqa: F401

import torch
import torch.distributed as dist
from ring_flash_attn.utils import AllGatherComm, get_default_args


def _set_fa3_signature_params(params: dict, causal: bool, window_size: tuple[int, int]) -> None:
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
            "causal": causal,
            "alibi_slopes": None,
            "return_softmax": False,
        }
    )
    if "window_size" in params:
        params["window_size"] = window_size
    else:
        params["window_size_left"] = window_size[0]
        params["window_size_right"] = window_size[1]

    outputs = _flash_attn_varlen_forward(**params)
    if len(outputs) == 8:
        out, _, _, _, _, lse, _, _ = outputs
    else:
        out, lse, _, _ = outputs
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
            "dq": dq,
            "dk": dk,
            "dv": dv,
            "cu_seqlens_q": cu_seqlens_q,
            "cu_seqlens_k": cu_seqlens_k,
            "max_seqlen_q": max_seqlen_q,
            "max_seqlen_k": max_seqlen_k,
            "dropout_p": 0.0,
            "softmax_scale": softmax_scale,
            "causal": causal,
            "alibi_slopes": None,
            "deterministic": False,
        }
    )
    if "window_size" in params:
        params["window_size"] = window_size
    else:
        params["window_size_left"] = window_size[0]
        params["window_size_right"] = window_size[1]
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
    _set_fa3_signature_params(params, causal, window_size)
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
    _set_fa3_signature_params(params, causal, window_size)
    _flash_attn_backward(**params)


def _ring_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    local_k_slice_start: int,
    local_k_slice_stop: int,
    heads_k_stride: int,
    causal: bool,
    group_name: str,
    window_size_left: int,
    window_size_right: int,
    flash_forward,
) -> tuple[torch.Tensor, torch.Tensor]:
    group = dist.distributed_c10d._resolve_process_group(group_name)
    local_k_slice = slice(local_k_slice_start, local_k_slice_stop)
    window_size = (window_size_left, window_size_right)
    softmax_scale = q.shape[-1] ** (-0.5)

    nheads = q.shape[1]
    total_k, nheads_k, head_dim = k.shape
    world_size = group.size()
    kv_buffer = torch.empty((2, total_k * world_size, heads_k_stride, head_dim), dtype=k.dtype, device=k.device)
    kv_buffer_copy = torch.empty_like(kv_buffer)
    comm = AllGatherComm(group)
    comm.all_gather(kv_buffer_copy[0], k[:, :heads_k_stride].contiguous())
    comm.all_gather(kv_buffer_copy[1], v[:, :heads_k_stride].contiguous())

    out_list = []
    lse_list = []
    for i in range(0, nheads_k, heads_k_stride):
        comm.wait()
        kv_buffer, kv_buffer_copy = kv_buffer_copy, kv_buffer
        if i < nheads_k - heads_k_stride:
            left = i + heads_k_stride
            right = left + heads_k_stride
            comm.all_gather(kv_buffer_copy[0], k[:, left:right].contiguous())
            comm.all_gather(kv_buffer_copy[1], v[:, left:right].contiguous())

        q_i = q[:, i * nheads // nheads_k : (i + heads_k_stride) * nheads // nheads_k]
        out_i, lse_i = flash_forward(
            q=q_i,
            k=kv_buffer[0][local_k_slice],
            v=kv_buffer[1][local_k_slice],
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
        )
        out_list.append(out_i)
        lse_list.append(lse_i)

    return torch.cat(out_list, dim=1), torch.cat(lse_list, dim=-2)


def _ring_backward(
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
    local_k_slice_start: int,
    local_k_slice_stop: int,
    heads_k_stride: int,
    causal: bool,
    group_name: str,
    window_size_left: int,
    window_size_right: int,
    flash_backward,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    group = dist.distributed_c10d._resolve_process_group(group_name)
    local_k_slice = slice(local_k_slice_start, local_k_slice_stop)
    window_size = (window_size_left, window_size_right)
    softmax_scale = q.shape[-1] ** (-0.5)

    nheads = q.shape[1]
    total_k, nheads_k, head_dim = k.shape
    world_size = group.size()
    kv_buffer = torch.empty((2, total_k * world_size, heads_k_stride, head_dim), dtype=k.dtype, device=k.device)
    kv_buffer_copy = torch.empty_like(kv_buffer)
    dkv_buffer = torch.empty((2, total_k * world_size, heads_k_stride, head_dim), dtype=k.dtype, device=k.device)
    kv_contiguous_buffer = (
        torch.empty((2, total_k, heads_k_stride, head_dim), dtype=k.dtype, device=k.device)
        if heads_k_stride != nheads_k
        else None
    )
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)

    comm = AllGatherComm(group)
    comm.all_gather(kv_buffer_copy[0], k[:, :heads_k_stride].contiguous())
    comm.all_gather(kv_buffer_copy[1], v[:, :heads_k_stride].contiguous())
    for i in range(0, nheads_k, heads_k_stride):
        dkv_buffer.zero_()
        q_slice = slice(i * nheads // nheads_k, (i + heads_k_stride) * nheads // nheads_k)
        comm.wait()
        kv_buffer, kv_buffer_copy = kv_buffer_copy, kv_buffer
        if i < nheads_k - heads_k_stride:
            left = i + heads_k_stride
            right = left + heads_k_stride
            comm.all_gather(kv_buffer_copy[0], k[:, left:right].contiguous())
            comm.all_gather(kv_buffer_copy[1], v[:, left:right].contiguous())

        # Varlen FA2, FA3, and FA4 all return LSE as [heads, total_tokens].
        lse_i = softmax_lse[q_slice].contiguous()
        flash_backward(
            dout=dout[:, q_slice],
            q=q[:, q_slice],
            k=kv_buffer[0][local_k_slice],
            v=kv_buffer[1][local_k_slice],
            out=out[:, q_slice],
            softmax_lse=lse_i,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            dq=dq[:, q_slice],
            dk=dkv_buffer[0][local_k_slice],
            dv=dkv_buffer[1][local_k_slice],
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
        )

        if kv_contiguous_buffer is None:
            dk_i = dk
            dv_i = dv
        else:
            dk_i = kv_contiguous_buffer[0]
            dv_i = kv_contiguous_buffer[1]
        dist.reduce_scatter_tensor(dk_i, dkv_buffer[0], group=group)
        dist.reduce_scatter_tensor(dv_i, dkv_buffer[1], group=group)
        if kv_contiguous_buffer is not None:
            dk[:, i : i + heads_k_stride] = dk_i
            dv[:, i : i + heads_k_stride] = dv_i

    return dq, dk, dv


def _register_ring_op(name: str, flash_forward, flash_backward):
    @torch.library.custom_op(f"prime_rl_ring::{name}", mutates_args=())
    def ring_forward_op(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        local_k_slice_start: int,
        local_k_slice_stop: int,
        heads_k_stride: int,
        causal: bool,
        group_name: str,
        window_size_left: int,
        window_size_right: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _ring_forward(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            local_k_slice_start,
            local_k_slice_stop,
            heads_k_stride,
            causal,
            group_name,
            window_size_left,
            window_size_right,
            flash_forward,
        )

    @ring_forward_op.register_fake
    def ring_forward_fake(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        local_k_slice_start: int,
        local_k_slice_stop: int,
        heads_k_stride: int,
        causal: bool,
        group_name: str,
        window_size_left: int,
        window_size_right: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.empty_like(q), q.new_empty((q.shape[1], q.shape[0]), dtype=torch.float32)

    @torch.library.custom_op(f"prime_rl_ring::{name}_backward", mutates_args=())
    def ring_backward_op(
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
        local_k_slice_start: int,
        local_k_slice_stop: int,
        heads_k_stride: int,
        causal: bool,
        group_name: str,
        window_size_left: int,
        window_size_right: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _ring_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            local_k_slice_start,
            local_k_slice_stop,
            heads_k_stride,
            causal,
            group_name,
            window_size_left,
            window_size_right,
            flash_backward,
        )

    @ring_backward_op.register_fake
    def ring_backward_fake(
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
        local_k_slice_start: int,
        local_k_slice_stop: int,
        heads_k_stride: int,
        causal: bool,
        group_name: str,
        window_size_left: int,
        window_size_right: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)

    def setup_context(ctx, inputs, output) -> None:
        q, k, v, cu_seqlens_q, cu_seqlens_k, *options = inputs
        out, softmax_lse = output
        ctx.save_for_backward(q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k)
        ctx.options = options
        ctx.mark_non_differentiable(softmax_lse)

    def backward(ctx, dout: torch.Tensor, _dlse: torch.Tensor | None):
        q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k = ctx.saved_tensors
        dq, dk, dv = ring_backward_op(
            dout.contiguous(),
            q.detach(),
            k.detach(),
            v.detach(),
            out.detach(),
            softmax_lse.detach(),
            cu_seqlens_q,
            cu_seqlens_k,
            *ctx.options,
        )
        return (dq, dk, dv) + (None,) * 11

    ring_forward_op.register_autograd(backward, setup_context=setup_context)
    return ring_forward_op


_RING_FA2_OP = _register_ring_op("fa2", _fa2_varlen_forward, _fa2_varlen_backward)
_RING_FA3_OP = _register_ring_op("fa3", _fa3_varlen_forward, _fa3_varlen_backward)


def _call_ring_op(
    op,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    local_k_slice: slice,
    causal: bool,
    heads_k_stride: int,
    group: dist.ProcessGroup,
    window_size: tuple[int, int],
) -> torch.Tensor:
    out, _ = op(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        local_k_slice.start,
        local_k_slice.stop,
        heads_k_stride,
        causal,
        group.group_name,
        window_size[0],
        window_size[1],
    )
    return out


def ring_fa2_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    local_k_slice: slice,
    causal: bool,
    heads_k_stride: int,
    group: dist.ProcessGroup,
    window_size: tuple[int, int] = (-1, -1),
) -> torch.Tensor:
    return _call_ring_op(
        _RING_FA2_OP,
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        local_k_slice,
        causal,
        heads_k_stride,
        group,
        window_size,
    )


def ring_fa3_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    local_k_slice: slice,
    causal: bool,
    heads_k_stride: int,
    group: dist.ProcessGroup,
    window_size: tuple[int, int] = (-1, -1),
) -> torch.Tensor:
    return _call_ring_op(
        _RING_FA3_OP,
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        local_k_slice,
        causal,
        heads_k_stride,
        group,
        window_size,
    )


# ---------------------------------------------------------------------------
# FA4 (flash_attn.cute) ring attention
# ---------------------------------------------------------------------------


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


_RING_FA4_OP = _register_ring_op("fa4", _fa4_varlen_forward, _fa4_varlen_backward)


def ring_fa4_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    local_k_slice: slice,
    causal: bool,
    heads_k_stride: int,
    group: dist.ProcessGroup,
    window_size: tuple[int, int] = (-1, -1),
) -> torch.Tensor:
    return _call_ring_op(
        _RING_FA4_OP,
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        local_k_slice,
        causal,
        heads_k_stride,
        group,
        window_size,
    )
