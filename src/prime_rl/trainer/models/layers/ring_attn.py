from __future__ import annotations

# ruff: noqa: I001 (`prime_rl._compat` must run before the `ring_flash_attn` imports below)
import prime_rl._compat  # noqa: F401

import torch
import torch.distributed as dist
from ring_flash_attn.utils import AllGatherComm

from .flash_varlen import VARLEN_BACKWARD, VARLEN_FORWARD


def _resolve_group(group_name: str) -> dist.ProcessGroup:
    for pg in dist.distributed_c10d._world.pg_map:
        if pg.group_name == group_name:
            return pg
    return dist.group.WORLD


class _RingVarlen(torch.autograd.Function):
    """Ring attention with all-gather communication, for any flash-attention version.

    Mirrors ring-flash-attn's `llama3_flash_attn_varlen` pattern: all-gather the whole K/V across
    the CP group once, then make one ordinary varlen flash call per head group over this rank's
    key range. Only the kernel pair (`VARLEN_FORWARD` / `VARLEN_BACKWARD`) varies by version; the
    communication is identical, so all versions share this one implementation.

    Non-tensor arguments are ints and strings rather than objects (`local_k_slice` as start/stop,
    the process group by name) so the Function stays traceable.
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
        flash_attn_version: int,
        window_size_left: int = -1,
        window_size_right: int = -1,
    ) -> torch.Tensor:
        group = _resolve_group(group_name)
        flash_forward = VARLEN_FORWARD[flash_attn_version]

        local_k_slice = slice(local_k_slice_start, local_k_slice_stop)
        window_size = (window_size_left, window_size_right)
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
            out_i, lse_i = flash_forward(
                q=q_i,
                k=k_i,
                v=v_i,
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
        ctx.flash_attn_version = flash_attn_version
        ctx.window_size = window_size
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k = ctx.saved_tensors
        heads_k_stride = ctx.heads_k_stride
        local_k_slice = ctx.local_k_slice
        causal = ctx.causal

        group = _resolve_group(ctx.group_name)
        flash_backward = VARLEN_BACKWARD[ctx.flash_attn_version]

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

            flash_backward(
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
                window_size=ctx.window_size,
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
        #            local_k_slice_start, local_k_slice_stop, heads_k_stride, causal, group_name,
        #            flash_attn_version, window_size_left, window_size_right
        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None, None, None


def ring_flash_attn_varlen_func(
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
    flash_attn_version: int,
    window_size: tuple[int, int] = (-1, -1),
) -> torch.Tensor:
    return _RingVarlen.apply(
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
        flash_attn_version,
        window_size[0],
        window_size[1],
    )
