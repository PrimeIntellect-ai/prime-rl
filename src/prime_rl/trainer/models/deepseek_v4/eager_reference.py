"""Naive DeepSeek V4 attention reference.

Nothing in the production path calls these. They exist for the tests, where a dense, obviously
correct implementation is the standard the fused kernel is measured against. They take plain
tensors, so this module imports nothing from `attention.py` and the dependency runs one way only.
"""

import torch
import torch.nn.functional as F
from torch import Tensor


def eager_attention_with_sinks(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sinks: torch.Tensor,
    attention_mask: torch.Tensor,
    scaling: float,
    dropout: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    attn_weights = attn_weights + attention_mask

    sink_logits = sinks.reshape(1, -1, 1, 1).expand(query.shape[0], -1, query.shape[-2], -1)
    combined_logits = torch.cat([attn_weights, sink_logits], dim=-1)
    # Row-max subtraction is not free here: without it the exponentials overflow in bf16.
    combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
    probs = F.softmax(combined_logits, dim=-1, dtype=combined_logits.dtype)

    scores = F.dropout(probs[..., :-1], p=dropout, training=training).to(value.dtype)
    attn_output = torch.matmul(scores, value)
    return attn_output.transpose(1, 2).contiguous()


def build_sliding_window_mask(*, tok_doc_idx: Tensor, sliding_window: int, dtype: torch.dtype) -> Tensor:
    """Additive `(1, 1, seq_len, seq_len)` mask over query rows and key columns.

    A key is readable when it lies in the query's own document and within the `sliding_window`
    tokens up to and including the query.

    A padded micro-batch folds its padding into the last document, so the padding is masked as a
    continuation of the last document. Causality already keeps it away from every real token, and it
    is loss-masked.
    """
    seq_len = tok_doc_idx.shape[0]
    device = tok_doc_idx.device
    tok_idx = torch.arange(seq_len, device=device)

    distance = tok_idx[:, None] - tok_idx[None, :]
    in_causal_window = (distance >= 0) & (distance < sliding_window)
    same_document = tok_doc_idx[:, None] == tok_doc_idx[None, :]
    readable = in_causal_window & same_document

    mask = torch.zeros(seq_len, seq_len, dtype=dtype, device=device)
    return mask.masked_fill_(~readable, torch.finfo(dtype).min)[None, None]


def token_entry_causal_mask(
    *, tok_doc_idx: Tensor, entry_doc_idx: Tensor, entry_local_idx: Tensor, threshold: Tensor
) -> Tensor:
    """`(1, seq_len, n_entries)` bool: which compressed entries each query token may read.

    Element `[0, t, e]` is true when query token `t` may read entry `e`. Both of these have to
    hold:

    - `e` belongs to `t`'s own document, so no query reads another document's history;
    - `e` closed before `t` arrived, i.e. its index within that document is below
      `threshold[0, t]`, the count of entries the query's position has completed.

    One `seq_lens` describes one packed row, so the leading axis is 1 and broadcasts over the
    batch, as `threshold` does.

    `threshold` counts per document, so it is compared against `entry_local_idx` and not against
    the sequence-global entry number; those two coordinate systems disagree for every document
    after the first. That is the whole reason this exists: the indexer hands its kernel one
    contiguous `[ks, ke)` range per query instead, and this is the dense statement of the same
    rule to measure that range against.
    """
    same_document = tok_doc_idx[None, :, None] == entry_doc_idx[None, None, :]
    return same_document & (threshold.unsqueeze(-1) > entry_local_idx[None, None, :])


def block_bias_from_indices(top_k_indices: Tensor, n_entries: int, dtype: torch.dtype) -> Tensor:
    """Render the indexer's picks as the dense additive `(batch, 1, seq_len, n_entries)` bias.

    `0` on the selected entries, `-inf` everywhere else. The dense and sparse attention paths
    both start from the same index tensor, so they cannot disagree about which entries a query
    reads.
    """
    batch, seq_len, _ = top_k_indices.shape
    # The `IGNORE_SLOT` (-1) sentinels are scattered into one throwaway column that is sliced back off.
    safe_indices = torch.where(top_k_indices >= 0, top_k_indices, torch.full_like(top_k_indices, n_entries))
    block_bias = torch.full((batch, 1, seq_len, n_entries + 1), float("-inf"), dtype=dtype, device=top_k_indices.device)
    block_bias.scatter_(-1, safe_indices.unsqueeze(1), 0.0)
    return block_bias[..., :n_entries]


def dense_mask_from_indices(indices: Tensor, n_positions: int, dtype: torch.dtype) -> Tensor:
    """Render a gather-index tensor as the dense additive `(batch, 1, seq_len, n_positions)` mask.

    `indices` is the `(batch, seq_len, 1, n_slots)` int32 tensor addressing the position axis of a
    `kv_buf` with `n_positions` positions. The mask is `0` on every position at least one of a
    query's slots names and `-inf` everywhere else. A slot holding `IGNORE_SLOT` (-1) marks an absent
    key and names no position, so it admits nothing.

    This is the fused kernel's oracle. Rendering the index tensor dense and running naive eager
    attention over the whole `kv_buf` exercises the index contract and the attention math together.
    """
    batch, seq_len, _, _ = indices.shape
    slots = indices[:, :, 0, :].to(torch.int64).unsqueeze(1)
    # `scatter_` has no negative indexing, so `IGNORE_SLOT` (-1) goes into one throwaway column that
    # is sliced back off. Clamping it to a real position instead would admit a key the query cannot read.
    safe = torch.where(slots >= 0, slots, n_positions)
    mask = torch.full((batch, 1, seq_len, n_positions + 1), float("-inf"), dtype=dtype, device=indices.device)
    mask.scatter_(-1, safe, 0.0)
    return mask[..., :n_positions].contiguous()


def indexer_scores(q: Tensor, compressed_kv: Tensor, weights: Tensor) -> Tensor:
    """Lightning-Indexer score `score[b,t,e] = sum_h w[b,t,h] * relu(sum_d q[b,t,h,d] k[b,e,d])`.

    `q` is `(batch, seq_len, heads, dim)`, `compressed_kv` is `(batch, n_entries, dim)`, `weights`
    is `(batch, seq_len, heads)`, and the result is `(batch, seq_len, n_entries)`, in float32.

    This is the standard `fp8_indexer` is measured against: the same formula, in float32 and
    through the `(batch, seq_len, heads, n_entries)` intermediate the kernel exists to avoid, so
    the only difference between the two is the kernel's FP8 quantization. Both constant scales the
    model applies, `index_head_dim ** -0.5` and `index_n_heads ** -0.5`, are dropped here because
    the kernel drops them too, and being positive they cannot change which entries a top-k selects.
    """
    scores = q.float() @ compressed_kv.transpose(-1, -2).float().unsqueeze(1)
    scores = F.relu(scores) * weights.float().unsqueeze(-1)
    return scores.sum(dim=2)
