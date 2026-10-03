"""Context-parallel attention over an explicit, balanced token partition."""

import torch

from prime_rl.trainer.distributed.collectives import all_gather_cp
from prime_rl.trainer.models.layers.ulysses_attn import ulysses_flash_attn_varlen_func
from prime_rl.utils.cp import CPContext
from prime_rl.utils.sequence import CPPartition


def context_parallel_attention(
    flash_fn,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    total_tokens: int,
    context: CPContext,
    flash_attn_version: int,
    window_size: tuple[int, int] = (-1, -1),
    softmax_scale: float | None = None,
    learnable_sink: torch.Tensor | None = None,
) -> torch.Tensor:
    if context.cp_style == "ulysses":
        return ulysses_flash_attn_varlen_func(
            flash_fn,
            q,
            k,
            v,
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            True,
            context.cp_group,
            context.cp_world_size,
            flash_attn_version=flash_attn_version,
            window_size=window_size,
            softmax_scale=softmax_scale,
            learnable_sink=learnable_sink,
            total_tokens=total_tokens,
        )

    partition = CPPartition(total_tokens, context.cp_world_size)
    start, end = partition.offsets[context.cp_rank : context.cp_rank + 2]
    cu_q = (cu_seqlens - start).clamp(min=0, max=end - start)
    cu_k = cu_seqlens.clamp(max=end)
    kv = all_gather_cp(torch.cat((k, v), dim=-1), 0, total_tokens, context.cp_group)[:end]
    key, value = kv.split((k.shape[-1], v.shape[-1]), dim=-1)
    if q.shape[0] == 0:
        output = q + kv.sum().to(q.dtype)
        if learnable_sink is not None:
            output = output + learnable_sink.sum().to(output.dtype)
        return output

    kwargs = {"causal": True}
    if window_size != (-1, -1):
        kwargs["window_size"] = window_size
    if softmax_scale is not None:
        kwargs["softmax_scale"] = softmax_scale
    if learnable_sink is not None:
        kwargs["learnable_sink"] = learnable_sink
    if flash_attn_version == 4:
        output, _ = flash_fn(
            q,
            key,
            value,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=min(max_seqlen, end - start),
            max_seqlen_k=max_seqlen,
            **kwargs,
        )
    else:
        output = flash_fn(q, key, value, cu_q, cu_k, min(max_seqlen, end - start), max_seqlen, **kwargs)
    return output
