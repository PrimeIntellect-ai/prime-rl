from __future__ import annotations

import torch
import torch.distributed as dist
from ring_flash_attn.utils import AllGatherComm, get_default_args


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
            "causal": causal,
        }
    )
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
            "causal": causal,
        }
    )
    _flash_attn_backward(**params)


class _RingFA3Varlen(torch.autograd.Function):
    """Ring attention using FA3 kernels with all-gather communication.

    Mirrors the llama3_flash_attn_varlen pattern from ring-flash-attn but
    calls flash-attention-3 low-level forward/backward instead of FA2.
    """

    @staticmethod
    def forward(
        ctx,
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
    ) -> torch.Tensor:
        group = dist.group.WORLD
        for pg in dist.distributed_c10d._world.pg_map:
            if pg.group_name == group_name:
                group = pg
                break

        local_k_slice = slice(local_k_slice_start, local_k_slice_stop)
        softmax_scale = q.shape[-1] ** (-0.5)
        out_list = []
        lse_list = []

        nheads = q.shape[1]
        total_k, nheads_k, head_dim = k.shape
        world_size = dist.get_world_size(group)

        kv_buffer = torch.empty((2, total_k * world_size, heads_k_stride, head_dim), dtype=k.dtype, device=k.device)
        kv_buffer_copy = torch.empty_like(kv_buffer)
        comm = AllGatherComm(group)

        comm.all_gather(kv_buffer_copy[0], k[:, :heads_k_stride].contiguous())
        comm.all_gather(kv_buffer_copy[1], v[:, :heads_k_stride].contiguous())

        for i in range(0, nheads_k, heads_k_stride):
            comm.wait()
            kv_buffer, kv_buffer_copy = kv_buffer_copy, kv_buffer

            if i < nheads_k - heads_k_stride:
                left = i + heads_k_stride
                right = left + heads_k_stride
                comm.all_gather(kv_buffer_copy[0], k[:, left:right].contiguous())
                comm.all_gather(kv_buffer_copy[1], v[:, left:right].contiguous())

            q_i = q[:, i * nheads // nheads_k : (i + heads_k_stride) * nheads // nheads_k]
            k_i = kv_buffer[0][local_k_slice]
            v_i = kv_buffer[1][local_k_slice]
            out_i, lse_i = _fa3_varlen_forward(
                q=q_i,
                k=k_i,
                v=v_i,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                softmax_scale=softmax_scale,
                causal=causal,
            )
            out_list.append(out_i)
            lse_list.append(lse_i)

        out = torch.cat(out_list, dim=1)
        lse = torch.cat(lse_list, dim=-2)

        ctx.save_for_backward(q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k)
        ctx.softmax_scale = softmax_scale
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.local_k_slice = local_k_slice
        ctx.heads_k_stride = heads_k_stride
        ctx.causal = causal
        ctx.group_name = group_name
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k = ctx.saved_tensors
        heads_k_stride = ctx.heads_k_stride
        local_k_slice = ctx.local_k_slice
        causal = ctx.causal

        group = dist.group.WORLD
        for pg in dist.distributed_c10d._world.pg_map:
            if pg.group_name == ctx.group_name:
                group = pg
                break

        nheads = q.shape[1]
        total_k, nheads_k, head_dim = k.shape
        world_size = dist.get_world_size(group)

        kv_buffer = torch.empty((2, total_k * world_size, heads_k_stride, head_dim), dtype=k.dtype, device=k.device)
        kv_buffer_copy = torch.empty_like(kv_buffer)
        dkv_buffer = torch.empty((2, total_k * world_size, heads_k_stride, head_dim), dtype=k.dtype, device=k.device)

        kv_contiguous_buffer = None
        if heads_k_stride != nheads_k:
            kv_contiguous_buffer = torch.empty((2, total_k, heads_k_stride, head_dim), dtype=k.dtype, device=k.device)

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)

        comm = AllGatherComm(group)
        comm.all_gather(kv_buffer_copy[0], k[:, :heads_k_stride].contiguous())
        comm.all_gather(kv_buffer_copy[1], v[:, :heads_k_stride].contiguous())

        for i in range(0, nheads_k, heads_k_stride):
            dkv_buffer.zero_()
            q_slice = slice(i * nheads // nheads_k, (i + heads_k_stride) * nheads // nheads_k)
            q_i = q[:, q_slice]
            dout_i = dout[:, q_slice]
            out_i = out[:, q_slice]
            dq_i = dq[:, q_slice]
            lse_i = softmax_lse[q_slice].contiguous()

            comm.wait()
            kv_buffer, kv_buffer_copy = kv_buffer_copy, kv_buffer
            if i < nheads_k - heads_k_stride:
                left = i + heads_k_stride
                right = left + heads_k_stride
                comm.all_gather(kv_buffer_copy[0], k[:, left:right].contiguous())
                comm.all_gather(kv_buffer_copy[1], v[:, left:right].contiguous())

            k_i = kv_buffer[0][local_k_slice]
            v_i = kv_buffer[1][local_k_slice]
            dk_i = dkv_buffer[0][local_k_slice]
            dv_i = dkv_buffer[1][local_k_slice]

            _fa3_varlen_backward(
                dout=dout_i,
                q=q_i,
                k=k_i,
                v=v_i,
                out=out_i,
                softmax_lse=lse_i,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=ctx.max_seqlen_q,
                max_seqlen_k=ctx.max_seqlen_k,
                dq=dq_i,
                dk=dk_i,
                dv=dv_i,
                softmax_scale=ctx.softmax_scale,
                causal=causal,
            )

            if heads_k_stride != nheads_k:
                dk_i = kv_contiguous_buffer[0]
                dv_i = kv_contiguous_buffer[1]
            else:
                dk_i = dk
                dv_i = dv

            dist.reduce_scatter_tensor(dk_i, dkv_buffer[0], group=group)
            dist.reduce_scatter_tensor(dv_i, dkv_buffer[1], group=group)
            if heads_k_stride != nheads_k:
                dk[:, i : i + heads_k_stride] = dk_i
                dv[:, i : i + heads_k_stride] = dv_i

        # Grads for: q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
        #            local_k_slice_start, local_k_slice_stop, heads_k_stride, causal, group_name
        return dq, dk, dv, None, None, None, None, None, None, None, None, None


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
) -> torch.Tensor:
    return _RingFA3Varlen.apply(
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
    )
