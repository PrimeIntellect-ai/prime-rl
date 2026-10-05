"""Serve DeepSeek-V4.1 on Hopper: paged indexer logits for 32-row cache pages.

On SM90 vLLM 0.30 gives every V4.1 attention cache 64-token kernel blocks, so the indexer cache
of a 2:1 compressed layer holds 32 rows per page. DeepGEMM's paged MQA-logits kernels only accept
32-row pages on Blackwell and assert on Hopper, at the first decode (already in the startup
profile run). This routes exactly that case, an SM90 paged indexer cache with 32-row pages, to a
batched torch implementation of the same kernel; every other cache keeps DeepGEMM.

The math follows DeepGEMM's reference (`tests/test_attention.py`): with the FP8 cache page laid
out as all of its rows' fp8 values followed by their fp32 scales,

    logits[b * next_n + i, t] = scale[t] * sum_h weights[b * next_n + i, h] * relu(q[b, i, h] . k[t])

for `t < context_lens[b, i]` and `-inf` elsewhere. It is synchronization-free, so it also runs
inside captured CUDA graphs.
"""

import torch

HOPPER_FALLBACK_PAGE_ROWS = 32


def _needs_fallback(page_rows: int) -> bool:
    from vllm.platforms import current_platform

    return page_rows == HOPPER_FALLBACK_PAGE_ROWS and current_platform.is_device_capability_family(90)


def paged_mqa_logits_torch(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
) -> torch.Tensor:
    """`q` `(B, next_n, H, D)` fp8; `kv_cache` `(num_blocks, page_rows, 1, D + 4)` uint8 -> `(B * next_n, max_model_len)` fp32."""
    batch, next_n, heads, dim = q.shape
    page_rows = kv_cache.shape[1]
    pages = kv_cache.view(kv_cache.shape[0], -1)[block_tables]  # (B, max_blocks, page_rows * (D + 4))
    num_positions = pages.shape[1] * page_rows
    keys = pages[..., : page_rows * dim].contiguous().view(torch.float8_e4m3fn).view(batch, num_positions, dim)
    scales = pages[..., page_rows * dim :].contiguous().view(torch.float32).view(batch, num_positions)

    scores = torch.einsum("bnhd,btd->bnht", q.float(), keys.float()).relu_()
    scores = (scores * weights.view(batch, next_n, heads, 1)).sum(dim=2) * scales[:, None, :]
    context = context_lens.view(batch, -1).expand(batch, next_n)
    positions = torch.arange(num_positions, device=q.device)
    scores = scores.masked_fill(positions >= context[..., None], float("-inf"))

    logits = scores.new_full((batch * next_n, max_model_len), float("-inf"))
    width = min(num_positions, max_model_len)
    logits[:, :width] = scores.view(batch * next_n, num_positions)[:, :width]
    return logits


def patch_deepseek_v41_hopper_indexer() -> None:
    import vllm.model_executor.layers.sparse_attn_indexer as indexer_ops
    import vllm.v1.attention.backends.mla.indexer as indexer_backend

    get_metadata = indexer_backend.get_paged_mqa_logits_metadata
    paged_logits = indexer_ops.fp8_fp4_paged_mqa_logits

    def get_paged_mqa_logits_metadata(context_lens, block_kv, num_sms, *args, **kwargs):
        if _needs_fallback(block_kv):
            # The fallback schedules itself; the buffer it would fill is never read.
            return torch.zeros((num_sms + 1, 2), dtype=torch.int32, device=context_lens.device)
        return get_metadata(context_lens, block_kv, num_sms, *args, **kwargs)

    def fp8_fp4_paged_mqa_logits(
        q, kv_cache, weights, context_lens, block_tables, schedule_metadata, max_model_len, **kwargs
    ):
        values, q_scale = q
        if q_scale is None and _needs_fallback(kv_cache.shape[1]):
            indices = kwargs.get("indices")
            if indices is not None:
                # Varlen rows: each row reads its own request's pages.
                block_tables = block_tables[indices.long()]
            return paged_mqa_logits_torch(values, kv_cache, weights, context_lens, block_tables, max_model_len)
        return paged_logits(
            q, kv_cache, weights, context_lens, block_tables, schedule_metadata, max_model_len, **kwargs
        )

    indexer_backend.get_paged_mqa_logits_metadata = get_paged_mqa_logits_metadata
    indexer_ops.fp8_fp4_paged_mqa_logits = fp8_fp4_paged_mqa_logits
