from __future__ import annotations

# ruff: noqa: I001 — `prime_rl._compat` must run before `ring_flash_attn` imports below.
import prime_rl._compat  # noqa: F401

import torch
import torch.distributed as dist
from ring_flash_attn.utils import AllGatherComm


def _flash_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    causal: bool,
    window_size: tuple[int, int],
    attention_backend: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if attention_backend == "flash_attention_2":
        from flash_attn.flash_attn_interface import _flash_attn_varlen_forward

        out, softmax_lse, _, _ = _flash_attn_varlen_forward(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            dropout_p=0.0,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            softcap=0.0,
            alibi_slopes=None,
            return_softmax=False,
        )
        return out, softmax_lse

    if attention_backend == "flash_attention_3":
        from flash_attn_interface import _flash_attn_forward

        out, softmax_lse, _, _ = _flash_attn_forward(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
        )
        return out, softmax_lse

    if attention_backend == "flash_attention_4":
        from flash_attn.cute.interface import _flash_attn_fwd

        window_size_left = window_size[0] if window_size[0] != -1 else None
        window_size_right = window_size[1] if window_size[1] != -1 else None
        out, softmax_lse, _, _ = _flash_attn_fwd(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            return_lse=True,
        )
        return out, softmax_lse

    raise ValueError(f"Unsupported ring attention backend: {attention_backend}")


def _flash_attention_backward(
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
    window_size: tuple[int, int],
    attention_backend: str,
) -> None:
    if attention_backend == "flash_attention_2":
        from flash_attn.flash_attn_interface import _flash_attn_varlen_backward

        _flash_attn_varlen_backward(
            dout=dout,
            q=q,
            k=k,
            v=v,
            out=out,
            softmax_lse=softmax_lse,
            dq=dq,
            dk=dk,
            dv=dv,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            dropout_p=0.0,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            softcap=0.0,
            alibi_slopes=None,
            deterministic=False,
        )
        return

    if attention_backend == "flash_attention_3":
        from flash_attn_interface import _flash_attn_backward

        _flash_attn_backward(
            dout=dout,
            q=q,
            k=k,
            v=v,
            out=out,
            softmax_lse=softmax_lse,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            dq=dq,
            dk=dk,
            dv=dv,
            softmax_scale=softmax_scale,
            is_causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
        )
        return

    if attention_backend == "flash_attention_4":
        from flash_attn.cute.interface import _flash_attn_bwd

        window_size_left = window_size[0] if window_size[0] != -1 else None
        window_size_right = window_size[1] if window_size[1] != -1 else None
        _flash_attn_bwd(
            q,
            k,
            v,
            out,
            dout,
            softmax_lse,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            dq=dq,
            dk=dk,
            dv=dv,
        )
        return

    raise ValueError(f"Unsupported ring attention backend: {attention_backend}")


@torch.library.custom_op("prime_rl_ring::attention", mutates_args=())
def ring_attention_forward(
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
    attention_backend: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    group = dist.distributed_c10d._resolve_process_group(group_name)
    local_k_slice = slice(local_k_slice_start, local_k_slice_stop)
    window_size = (window_size_left, window_size_right)
    softmax_scale = q.shape[-1] ** -0.5

    _, num_query_heads, _ = q.shape
    local_k_tokens, num_kv_heads, head_dim = k.shape
    world_size = group.size()
    gathered_kv_shape = (2, local_k_tokens * world_size, heads_k_stride, head_dim)
    gathered_kv = torch.empty(gathered_kv_shape, dtype=k.dtype, device=k.device)
    next_gathered_kv = torch.empty_like(gathered_kv)

    communication = AllGatherComm(group)
    communication.all_gather(next_gathered_kv[0], k[:, :heads_k_stride].contiguous())
    communication.all_gather(next_gathered_kv[1], v[:, :heads_k_stride].contiguous())

    outputs = []
    softmax_lses = []
    query_heads_per_kv_head = num_query_heads // num_kv_heads
    for kv_head_start in range(0, num_kv_heads, heads_k_stride):
        communication.wait()
        gathered_kv, next_gathered_kv = next_gathered_kv, gathered_kv

        next_kv_head_start = kv_head_start + heads_k_stride
        if next_kv_head_start < num_kv_heads:
            next_kv_head_stop = next_kv_head_start + heads_k_stride
            communication.all_gather(next_gathered_kv[0], k[:, next_kv_head_start:next_kv_head_stop].contiguous())
            communication.all_gather(next_gathered_kv[1], v[:, next_kv_head_start:next_kv_head_stop].contiguous())

        query_head_start = kv_head_start * query_heads_per_kv_head
        query_head_stop = (kv_head_start + heads_k_stride) * query_heads_per_kv_head
        query_head_slice = slice(query_head_start, query_head_stop)
        out, softmax_lse = _flash_attention_forward(
            q=q[:, query_head_slice],
            k=gathered_kv[0][local_k_slice],
            v=gathered_kv[1][local_k_slice],
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            attention_backend=attention_backend,
        )
        outputs.append(out)
        softmax_lses.append(softmax_lse)

    return torch.cat(outputs, dim=1), torch.cat(softmax_lses, dim=-2)


@ring_attention_forward.register_fake
def _ring_attention_forward_fake(
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
    attention_backend: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(q), q.new_empty((q.shape[1], q.shape[0]), dtype=torch.float32)


@torch.library.custom_op("prime_rl_ring::attention_backward", mutates_args=())
def ring_attention_backward(
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
    attention_backend: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    group = dist.distributed_c10d._resolve_process_group(group_name)
    local_k_slice = slice(local_k_slice_start, local_k_slice_stop)
    window_size = (window_size_left, window_size_right)
    softmax_scale = q.shape[-1] ** -0.5

    _, num_query_heads, _ = q.shape
    local_k_tokens, num_kv_heads, head_dim = k.shape
    world_size = group.size()
    gathered_kv_shape = (2, local_k_tokens * world_size, heads_k_stride, head_dim)
    gathered_kv = torch.empty(gathered_kv_shape, dtype=k.dtype, device=k.device)
    next_gathered_kv = torch.empty_like(gathered_kv)
    gathered_kv_grad = torch.empty_like(gathered_kv)
    local_kv_grad = (
        torch.empty((2, local_k_tokens, heads_k_stride, head_dim), dtype=k.dtype, device=k.device)
        if heads_k_stride != num_kv_heads
        else None
    )
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)

    communication = AllGatherComm(group)
    communication.all_gather(next_gathered_kv[0], k[:, :heads_k_stride].contiguous())
    communication.all_gather(next_gathered_kv[1], v[:, :heads_k_stride].contiguous())

    query_heads_per_kv_head = num_query_heads // num_kv_heads
    for kv_head_start in range(0, num_kv_heads, heads_k_stride):
        gathered_kv_grad.zero_()
        communication.wait()
        gathered_kv, next_gathered_kv = next_gathered_kv, gathered_kv

        next_kv_head_start = kv_head_start + heads_k_stride
        if next_kv_head_start < num_kv_heads:
            next_kv_head_stop = next_kv_head_start + heads_k_stride
            communication.all_gather(next_gathered_kv[0], k[:, next_kv_head_start:next_kv_head_stop].contiguous())
            communication.all_gather(next_gathered_kv[1], v[:, next_kv_head_start:next_kv_head_stop].contiguous())

        query_head_start = kv_head_start * query_heads_per_kv_head
        query_head_stop = (kv_head_start + heads_k_stride) * query_heads_per_kv_head
        query_head_slice = slice(query_head_start, query_head_stop)
        # FA2, FA3, and FA4 varlen kernels all return LSE as [heads, total_tokens].
        head_softmax_lse = softmax_lse[query_head_slice].contiguous()
        _flash_attention_backward(
            dout=dout[:, query_head_slice],
            q=q[:, query_head_slice],
            k=gathered_kv[0][local_k_slice],
            v=gathered_kv[1][local_k_slice],
            out=out[:, query_head_slice],
            softmax_lse=head_softmax_lse,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            dq=dq[:, query_head_slice],
            dk=gathered_kv_grad[0][local_k_slice],
            dv=gathered_kv_grad[1][local_k_slice],
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            attention_backend=attention_backend,
        )

        if local_kv_grad is None:
            reduced_dk = dk
            reduced_dv = dv
        else:
            reduced_dk = local_kv_grad[0]
            reduced_dv = local_kv_grad[1]
        dist.reduce_scatter_tensor(reduced_dk, gathered_kv_grad[0], group=group)
        dist.reduce_scatter_tensor(reduced_dv, gathered_kv_grad[1], group=group)
        if local_kv_grad is not None:
            kv_head_stop = kv_head_start + heads_k_stride
            dk[:, kv_head_start:kv_head_stop] = reduced_dk
            dv[:, kv_head_start:kv_head_stop] = reduced_dv

    return dq, dk, dv


@ring_attention_backward.register_fake
def _ring_attention_backward_fake(
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
    attention_backend: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


def _ring_attention_setup_context(ctx, inputs, output) -> None:
    (
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
        attention_backend,
    ) = inputs
    out, softmax_lse = output
    ctx.save_for_backward(q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k)
    ctx.max_seqlen_q = max_seqlen_q
    ctx.max_seqlen_k = max_seqlen_k
    ctx.local_k_slice_start = local_k_slice_start
    ctx.local_k_slice_stop = local_k_slice_stop
    ctx.heads_k_stride = heads_k_stride
    ctx.causal = causal
    ctx.group_name = group_name
    ctx.window_size_left = window_size_left
    ctx.window_size_right = window_size_right
    ctx.attention_backend = attention_backend
    ctx.mark_non_differentiable(softmax_lse)


def _ring_attention_autograd_backward(ctx, dout: torch.Tensor, _dsoftmax_lse: torch.Tensor | None):
    q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k = ctx.saved_tensors
    dq, dk, dv = ring_attention_backward(
        dout=dout.contiguous(),
        q=q.detach(),
        k=k.detach(),
        v=v.detach(),
        out=out.detach(),
        softmax_lse=softmax_lse.detach(),
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=ctx.max_seqlen_q,
        max_seqlen_k=ctx.max_seqlen_k,
        local_k_slice_start=ctx.local_k_slice_start,
        local_k_slice_stop=ctx.local_k_slice_stop,
        heads_k_stride=ctx.heads_k_stride,
        causal=ctx.causal,
        group_name=ctx.group_name,
        window_size_left=ctx.window_size_left,
        window_size_right=ctx.window_size_right,
        attention_backend=ctx.attention_backend,
    )
    return (dq, dk, dv) + (None,) * 12


ring_attention_forward.register_autograd(
    _ring_attention_autograd_backward,
    setup_context=_ring_attention_setup_context,
)


def ring_varlen_attention(
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
    attention_backend: str,
    window_size: tuple[int, int] = (-1, -1),
) -> torch.Tensor:
    out, _ = ring_attention_forward(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        local_k_slice_start=local_k_slice.start,
        local_k_slice_stop=local_k_slice.stop,
        heads_k_stride=heads_k_stride,
        causal=causal,
        group_name=group.group_name,
        window_size_left=window_size[0],
        window_size_right=window_size[1],
        attention_backend=attention_backend,
    )
    return out
