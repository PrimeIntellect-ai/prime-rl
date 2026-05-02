import torch

from prime_rl.trainer.models.kernels.sparse_mla_bwd import bwd, postprocess, preprocess
from prime_rl.trainer.models.kernels.sparse_mla_fwd import sparse_mla_fwd_interface

DEFAULT_BUCKET_SIZES = (256, 512, 1024, 1536, 2048)


def _query_buckets(
    indices: torch.Tensor,
    kv_seq_len: int,
    bucket_sizes: tuple[int, ...],
) -> list[tuple[int, torch.Tensor]]:
    assert indices.shape[0] == 1, "bucketed sparse MLA currently supports batch_size=1"
    max_valid_index = kv_seq_len - 2
    valid_counts = (indices <= max_valid_index).sum(dim=-1).flatten()

    topk = indices.shape[-1]
    sizes = tuple(size for size in bucket_sizes if size < topk) + (topk,)
    buckets = []
    previous = 0
    for size in sizes:
        query_indices = torch.nonzero((valid_counts > previous) & (valid_counts <= size), as_tuple=False).flatten()
        if query_indices.numel() > 0:
            buckets.append((size, query_indices.contiguous()))
        previous = size
    return buckets


def sparse_mla_fwd_bucketed(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale=None,
    d_v: int = 512,
    bucket_sizes: tuple[int, ...] = DEFAULT_BUCKET_SIZES,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert q.is_contiguous() and kv.is_contiguous() and indices.is_contiguous()
    assert q.shape[0] == 1, "bucketed sparse MLA currently supports batch_size=1"

    _, seq_len, heads, dim_plus_tail_dim = q.shape
    out = torch.empty((1, seq_len, heads, d_v), device=q.device, dtype=q.dtype)
    lse = torch.empty((1, seq_len, heads), device=q.device, dtype=torch.float32)

    for bucket_topk, query_indices in _query_buckets(indices, kv.shape[1], bucket_sizes):
        q_bucket = q.index_select(1, query_indices).contiguous()
        indices_bucket = indices.index_select(1, query_indices)[..., :bucket_topk].contiguous()
        out_bucket, lse_bucket = sparse_mla_fwd_interface(
            q_bucket,
            kv,
            indices_bucket,
            sm_scale=sm_scale,
            d_v=d_v,
        )
        out.index_copy_(1, query_indices, out_bucket)
        lse.index_copy_(1, query_indices, lse_bucket)

    return out, lse


def sparse_mla_bwd_bucketed(
    q: torch.Tensor,
    kv: torch.Tensor,
    o: torch.Tensor,
    do: torch.Tensor,
    indices: torch.Tensor,
    lse: torch.Tensor,
    sm_scale=None,
    bucket_sizes: tuple[int, ...] = DEFAULT_BUCKET_SIZES,
    sort_bwd_indices: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert q.is_contiguous()
    assert kv.is_contiguous()
    assert o.is_contiguous()
    assert do.is_contiguous()
    assert indices.is_contiguous()
    assert lse.is_contiguous()
    assert q.shape[0] == 1, "bucketed sparse MLA currently supports batch_size=1"

    _, _, heads, dim_plus_tail_dim = q.shape
    _, _, kv_group, _ = kv.shape
    d_v = 512
    d_tail = dim_plus_tail_dim - d_v
    assert kv_group == 1, "bucketed sparse MLA currently supports kv_group=1"

    preprocess_kernel = preprocess(heads, d_v)
    postprocess_kernel = postprocess(d_v, d_tail, kv_group)
    dkv_acc = torch.zeros_like(kv, dtype=torch.float32)
    dq = torch.empty_like(q)

    for bucket_topk, query_indices in _query_buckets(indices, kv.shape[1], bucket_sizes):
        q_bucket = q.index_select(1, query_indices).contiguous()
        o_bucket = o.index_select(1, query_indices).contiguous()
        do_bucket = do.index_select(1, query_indices).contiguous()
        lse_bucket = lse.index_select(1, query_indices).contiguous()
        indices_bucket = indices.index_select(1, query_indices)[..., :bucket_topk].contiguous()
        if sort_bwd_indices:
            indices_bucket = torch.sort(indices_bucket, dim=-1).values.contiguous()

        delta = preprocess_kernel(o_bucket, do_bucket)
        bwd_kernel = bwd(
            heads,
            d_v,
            d_tail,
            bucket_topk,
            kv_group,
            sm_scale,
            True,
            block_size=32,
            split_store=4,
            use_gemm_v1=True,
        )
        dq_bucket = bwd_kernel(q_bucket, kv, do_bucket, indices_bucket, lse_bucket, delta, dkv_acc)
        dq.index_copy_(1, query_indices, dq_bucket)

    dkv = postprocess_kernel(dkv_acc)
    return dq, dkv


def sparse_mla_bwd_bucketed_sorted(
    q: torch.Tensor,
    kv: torch.Tensor,
    o: torch.Tensor,
    do: torch.Tensor,
    indices: torch.Tensor,
    lse: torch.Tensor,
    sm_scale=None,
    bucket_sizes: tuple[int, ...] = DEFAULT_BUCKET_SIZES,
) -> tuple[torch.Tensor, torch.Tensor]:
    return sparse_mla_bwd_bucketed(
        q,
        kv,
        o,
        do,
        indices,
        lse,
        sm_scale=sm_scale,
        bucket_sizes=bucket_sizes,
        sort_bwd_indices=True,
    )
