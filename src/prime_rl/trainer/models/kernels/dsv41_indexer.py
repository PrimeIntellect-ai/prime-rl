"""DeepSeek-V4.1 Lightning Indexer selection: fp8 scoring, optional two-level candidate filter, top-k.

V4.1's indexer scores every compressed entry a query may read, like V4's, and some layers then
narrow that in two levels. The candidate source layer ranks fixed-size blocks of entries by their
best score and keeps the top `candidate_topk_blocks` per query; later indexers score with their own
weights but only inside those blocks. Blocks are counted from the start of the query's document,
so a packed row gives each document what running it alone would.

Candidates travel between layers as doc-local block indices, so a consumer only gathers and ranks
its `candidate_topk_blocks * block_size` candidate entries rather than every entry of the row.

The score matrix is `(queries, entries)` fp32, so the queries are processed in chunks that keep
it near `CHUNK_BYTES`. The scoring kernel and the fp8 quantization are `fp8_indexer`'s.
"""

import torch
import triton

from prime_rl.trainer.models.kernels.fp8_indexer import _triton_fp8_indexer_kernel, per_token_group_quant_fp8

CHUNK_BYTES = 1 << 30
NEG_INF = float("-inf")


def _score_chunk(q_fp8, k_fp8, k_scales, w, ks, ke) -> torch.Tensor:
    """`(rows, S_k)` fp32 indexer scores, `-inf` outside each row's `[ks, ke)`."""
    H, S_q, D = q_fp8.shape
    S_k = k_fp8.shape[0]
    logits = torch.empty(S_q, S_k, dtype=torch.float32, device=q_fp8.device)
    grid = lambda meta: (triton.cdiv(S_q, meta["BLOCK_M"]), triton.cdiv(S_k, meta["BLOCK_N"]))
    _triton_fp8_indexer_kernel[grid](
        q_fp8,
        k_fp8,
        k_scales,
        w,
        logits,
        ks,
        ke,
        S_q,
        S_k,
        q_fp8.stride(0),
        q_fp8.stride(1),
        k_fp8.stride(0),
        w.stride(0),
        H=H,
        D=D,
        S_K_BUCKET=triton.next_power_of_2(S_k),
    )
    return logits


def _doc_local(logits: torch.Tensor, ks: torch.Tensor, ke: torch.Tensor, width: int) -> torch.Tensor:
    """Each row shifted so its document's first entry sits in column 0, `-inf` past its readable end."""
    S_k = logits.shape[-1]
    if bool((ks == ks[0]).all()):
        # One document start for the whole chunk (the common long-document case): a slice, no copy.
        start = int(ks[0])
        local = logits[:, start : start + width]
        local = torch.nn.functional.pad(local, (0, width - local.shape[-1]), value=NEG_INF)
    else:
        cols = ks[:, None].long() + torch.arange(width, device=logits.device)
        local = logits.gather(1, cols.clamp_max(S_k - 1))
    cols = torch.arange(width, device=logits.device)
    return local.masked_fill(cols >= (ke - ks)[:, None], NEG_INF)


def _select_candidate_blocks(local: torch.Tensor, n_valid: torch.Tensor, block_size: int, topk_blocks: int):
    """Doc-local `(rows, topk_blocks)` indices of each row's best blocks, -1 for unused picks.

    The block holding the row's newest entry is only partly filled, so it is always kept: it holds
    the most recent tokens but could otherwise be outscored by an older, full block.
    """
    width = local.shape[-1]
    scores = torch.nn.functional.pad(local, (0, -width % block_size), value=NEG_INF)
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    n_blocks = scores.shape[-1]
    last = ((n_valid - 1).clamp_min(0) // block_size)[:, None]
    scores = scores.masked_fill(torch.arange(n_blocks, device=local.device) == last, float("inf"))
    top = scores.topk(min(topk_blocks, n_blocks), dim=-1, sorted=False)
    blocks = torch.where(top.values > NEG_INF, top.indices, -1)
    return torch.nn.functional.pad(blocks, (0, topk_blocks - blocks.shape[-1]), value=-1).int()


@torch.library.custom_op("prime_rl::dsv41_index_topk", mutates_args=())
def dsv41_index_topk(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    ks: torch.Tensor,
    ke: torch.Tensor,
    topk: int,
    candidates: torch.Tensor | None,
    emit_candidates: bool,
    max_entries_per_doc: int,
    block_size: int,
    topk_blocks: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k entries per query, plus the candidate blocks when `emit_candidates`.

    Args:
        q: `(S_q, H, D)` bf16 index queries. k: `(S_k, D)` bf16 index keys, shared across heads.
        w: `(S_q, H)` per-head weights.
        ks, ke: `(S_q,)` int32, each query's readable entries `[ks, ke)`; `ks` is its document's
            first entry.
        candidates: `(S_q, topk_blocks)` int32 doc-local block indices (-1 unused) restricting the
            scores, or None.
        emit_candidates: also return this layer's own candidate blocks.
        max_entries_per_doc: width of the doc-local view the candidate source ranks blocks over.

    Returns `(S_q, topk)` int64 entry indices, -1 where a query has fewer readable entries, and
    the candidate blocks (an empty tensor unless `emit_candidates`).
    """
    S_q, H, D = q.shape
    S_k = k.shape[0]
    q_fp8, q_scales = per_token_group_quant_fp8(q.reshape(S_q * H, D).contiguous(), group_size=D, use_ue8m0=True)
    k_fp8, k_scales = per_token_group_quant_fp8(k.contiguous(), group_size=D, use_ue8m0=True)
    q_fp8 = q_fp8.view(S_q, H, D).permute(1, 0, 2).contiguous()
    w = (w.float() * q_scales.view(S_q, H)).contiguous()

    out = torch.full((S_q, topk), -1, dtype=torch.int64, device=q.device)
    cand_out = torch.full((S_q, topk_blocks) if emit_candidates else (0, 0), -1, dtype=torch.int32, device=q.device)
    block_offsets = torch.arange(block_size, device=q.device)
    chunk = max(1, CHUNK_BYTES // (4 * max(S_k, 1)))
    for r0 in range(0, S_q, chunk):
        r1 = min(S_q, r0 + chunk)
        rows_ks, rows_ke = ks[r0:r1].long(), ke[r0:r1].long()
        logits = _score_chunk(q_fp8[:, r0:r1], k_fp8, k_scales, w[r0:r1], ks[r0:r1], ke[r0:r1])
        if emit_candidates:
            local = _doc_local(logits, rows_ks, rows_ke, max_entries_per_doc)
            cand_out[r0:r1] = _select_candidate_blocks(local, rows_ke - rows_ks, block_size, topk_blocks)
        if candidates is not None:
            # Rank only the candidate entries, gathered straight out of the row's scores.
            blocks = candidates[r0:r1].long()
            local_pos = (blocks[:, :, None] * block_size + block_offsets).flatten(1)
            valid = (blocks[:, :, None] >= 0).expand(-1, -1, block_size).flatten(1)
            pos = rows_ks[:, None] + local_pos
            valid &= pos < rows_ke[:, None]
            scores = logits.gather(1, pos.clamp(0, S_k - 1)).masked_fill(~valid, NEG_INF)
            k_eff = min(topk, scores.shape[-1])
            values, picks = scores.topk(k_eff, dim=-1, sorted=False)
            out[r0:r1, :k_eff] = torch.where(values > NEG_INF, pos.gather(1, picks), -1)
        else:
            k_eff = min(topk, S_k)
            values, indices = logits.topk(k_eff, dim=-1, sorted=False)
            out[r0:r1, :k_eff] = torch.where(values > NEG_INF, indices, -1)
    return out, cand_out


@dsv41_index_topk.register_fake
def _dsv41_index_topk_fake(
    q, k, w, ks, ke, topk, candidates, emit_candidates, max_entries_per_doc, block_size, topk_blocks
):
    cand_shape = (q.shape[0], topk_blocks) if emit_candidates else (0, 0)
    return q.new_empty((q.shape[0], topk), dtype=torch.int64), q.new_empty(cand_shape, dtype=torch.int32)


__all__ = ["dsv41_index_topk"]
