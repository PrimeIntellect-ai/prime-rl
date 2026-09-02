"""
[DeepSeek V4 Attention Layers]

The attention layers in this architecture generally begin with a few sliding-window attention layers
(just the first two layers in V4 Flash) followed by interleaved complex compressed attention
variants involving either Compressed Sparse Attention (CSA) or Heavily Compressed Attention (HCA).
CSA compresses with a smaller window (~4 toks) and adds additional sparsity on top via a "Lightning
Indexer", while HCA uses a more aggressive window (~128 toks) with no additional sparsity. A sketch
of the compressed variants is below, tensors flowing downwards:

                          hidden_states
                                │
                 ┌──────────────┴──────────────┐
                 │                             │
             local KV                   long-range KV
        sliding_window ~ 128        compress hidden states
                 │                   into compact entries
                 │                             │
                 │              ┌──────────────┴──────────────┐
                 │              │        (choose one)         │
                 │              │                             │
                 │             CSA                           HCA
                 │        compress_rate ~ 4          compress_rate ~ 128
                 │     index_topk sparsity via                │
                 │       Lightning Indexer                    │
                 │              │                             │
                 │              └──────────────┬──────────────┘
                 │                             │
               RoPE                          RoPE
         at token positions             at each entry's
                 │                     first token position
                 │                             │
                 └────────── concatenate ──────┘
                                │
                               QKᵀ (with {compression,position}-aware masking)
                                │
                         softmax + sink
                                │
                        values (= keys) softmax weighting
                                │
                          de-rotate output (undo RoPE on values = keys)
                                │
                    grouped output projection

[Packing Details]

We describe our abstractions and nomenclature for DeepSeek V4 packed sequences below, which are
useful due to the complexities introduced by the compressed attention variants. Ultimately, all
necessary attention data is organized into a `PackedContext` object (directly consumed by attention
layers), built from one `seq_lens` and carrying:

  - `attention_mask`: causal, local-window, clipped at document boundaries.
  - `position_ids`: each token's position within its own document.
  - `tok_doc_idx`: which document each token belongs to.
  - `position_embeddings`: the RoPE tables, one per rope type, evaluated at `position_ids`.
  - `window_indices`: for each query, the indices of the tokens its local window covers. Same
    information as `attention_mask`, but more efficient and consumed by the sparse attn kernel.
  - `compression_layouts`: one `CompressionLayout` per compress rate in the architecture.

The last of those characterizes the token-compression mechanism of DeepSeek V4. We start with it
below.

We pack several documents end to end in a flat token stream. Each compressed attention variant
defines a `compress_rate`: that variant compresses each group of `compress_rate` consecutive tokens
into an individual `entry`. For packed sequence and each `compress_rate` in the architecture, we
build one `CompressionLayout` object whose responsibility is to handle the document-aware bookkeeping
for such packed-document compression.

Take the following illustrative example of two packed documents and `compress_rate = 4`:

  token             0  1  2  3  4  5  6  7  8 │   9 10 11 12 13
                  └───────── doc 0 ─────────┘   └─── doc 1 ───┘
  entry           └─── e0 ───┘└─── e1 ───┘  x   └─── e2 ───┘  x

We've indicated which tokens get pooled into which entries (tokens marked `x` belong to no entry) .
A complete, generic description of the packed and compressed state requires four pieces of data:

  - Which tokens belong to which entries: `entry_tok_idx`.
  - Which document each entry belongs to (for causality): `entry_doc_idx`.
  - Where an entry sits within its own document (useful for RoPE + causality): `entry_local_idx`.
  - Which document each token belongs to (causality, again): `tok_doc_idx`.

For the above example:

  token             0  1  2  3  4  5  6  7  8 │   9 10 11 12 13
                  └───────── doc 0 ─────────┘   └─── doc 1 ───┘
  entry           └─── e0 ───┘└─── e1 ───┘  x   └─── e2 ───┘  x

  entry_tok_idx    [0  1  2  3][4  5  6  7]      [9 10 11 12]
  entry_doc_idx        0           0                 1
  entry_local_idx      0           1                 0
  tok_doc_idx       0  0  0  0  0  0  0  0  0     1  1  1  1  1

The first three depend on the compress rate and are stored on the `CompressionLayout` abstraction
used below. The fourth does not: `tok_doc_idx` describes the token stream alone, so `PackedContext`
holds it once and shares it across rates.

Compression is only part of the story: every attention layer also reads a local sliding window of
the most recent tokens directly, and the compressed entries are how it reaches anything older.
That window needs a causal sliding-window mask applied per document, `attention_mask`, and every
rotation in the block needs `position_ids` together with the RoPE tables evaluated at them,
`position_embeddings`. None of those belongs to any single compress rate.

`PackedContext.build` takes `seq_lens` and derives every one of its fields from it. Nothing else is
an input, so a position that disagrees with a document boundary, a mask that spans one, or a RoPE
table evaluated at positions other than the ones the causal thresholds count in, cannot be
constructed. It runs once per model forward, since none of this depends on depth.
"""

import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from prime_rl.trainer.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from prime_rl.trainer.models.deepseek_v4.hyperconnections import DeepseekV4UnweightedRMSNorm
from prime_rl.trainer.models.deepseek_v4.rotary import DeepseekV4RotaryEmbedding, apply_rotary_pos_emb_interleaved
from prime_rl.trainer.models.layers.norms import RMSNorm, RMSNormConfig
from prime_rl.utils.sequence import get_cu_seqlens_from_seq_lens

_ATTN_IMPLS = frozenset({"eager", "gather"})

# Temporary scaffolding for the sparse-attention rollout, deliberately not a config field.
DSV4_ATTN_IMPL = os.environ.get("PRIME_RL_DSV4_ATTN", "eager")
if DSV4_ATTN_IMPL not in _ATTN_IMPLS:
    raise ValueError(f"PRIME_RL_DSV4_ATTN must be one of {sorted(_ATTN_IMPLS)}, got {DSV4_ATTN_IMPL!r}")

# The forward kernel tiles the gather-slot axis at `block_I = 64` and the backward at
# `block_size = 32`, so the slot count must be a multiple of `lcm(64, 32) = 64`. The production
# config's `sliding_window + index_topk = 128 + 512 = 640` satisfies it for free; a toy config
# does not, and pads with sentinel slots, which are masked and therefore semantically free.
_SLOT_TILE = 64


class DeepseekV4GroupedLinear(nn.Linear):
    """Block-diagonal grouped linear, the first half of the output projection.

    The stacked attention output is `num_attention_heads * head_dim` wide, so a direct
    projection to `hidden_size` would dominate the per-token cost. Instead the heads are split
    into `n_groups` groups, each projected independently to `out_features / n_groups` channels;
    a single follow-up linear (`o_b_proj`) mixes the concatenation back to `hidden_size`.

    Input is `(..., n_groups, in_features_per_group)`, output `(..., n_groups, out_features / n_groups)`.
    """

    def __init__(self, in_features_per_group: int, out_features: int, n_groups: int, bias: bool = False):
        super().__init__(in_features_per_group, out_features, bias=bias)
        self.n_groups = n_groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape[:-2]
        hidden_dim = x.shape[-1]
        w = self.weight.view(self.n_groups, -1, hidden_dim).transpose(1, 2)
        x = x.reshape(-1, self.n_groups, hidden_dim).transpose(0, 1)
        y = torch.bmm(x, w).transpose(0, 1)
        return y.reshape(*input_shape, self.n_groups, -1)


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


def sparse_attention_gather(q: Tensor, kv_buf: Tensor, indices: Tensor, sinks: Tensor, scale: float) -> Tensor:
    """Attention of each query over the `n_slots` KV positions it gathers, in float32.

    `q` is `(batch, seq_len, heads, head_dim)`, `kv_buf` is `(batch, n_positions, 1, head_dim)`,
    `indices` is `(batch, seq_len, 1, n_slots)` int32 addressing `kv_buf`'s position axis, and
    `sinks` is `(heads,)`. The output is `(batch, seq_len, heads, head_dim)` in `q.dtype`, the
    layout `eager_attention_with_sinks` returns.

    A slot holding the sentinel (the trailing zero position of `kv_buf`) is masked out, so a
    query with a short window or fewer picks than slots costs nothing but the loads.

    This materializes the `(batch, seq_len, n_slots, head_dim)` gather: at production shapes
    (640 slots, 512 channels) that is 640 KB per token in bf16, about 42 GB at `seq_len = 65536`.
    It is an oracle for the kernel that replaces it and a small-shape fallback, not a
    long-context path. It runs entirely in float32 so it matches a kernel accumulating in fp32.
    """
    sentinel = kv_buf.shape[1] - 1
    slot_idx = indices[:, :, 0, :].to(torch.int64)  # (b, s, k)
    batch_idx = torch.arange(kv_buf.shape[0], device=kv_buf.device)[:, None, None]
    keys = kv_buf[batch_idx, slot_idx, 0].float()  # (b, s, k, d)

    logits = torch.einsum("bshd,bskd->bshk", q.float(), keys) * scale
    logits = logits.masked_fill((slot_idx == sentinel).unsqueeze(2), float("-inf"))

    # The sink logit is unscaled, as in `eager_attention_with_sinks`, which adds it after the
    # dot products are scaled. Subtracting the row max keeps a fully sentinel row finite.
    sink_logits = sinks.float().reshape(1, 1, -1, 1).expand(*logits.shape[:-1], 1)
    combined_logits = torch.cat([logits, sink_logits], dim=-1)
    probs = F.softmax(combined_logits - combined_logits.max(dim=-1, keepdim=True).values, dim=-1)

    return torch.einsum("bshk,bskd->bshd", probs[..., :-1], keys).to(q.dtype)


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


@dataclass(frozen=True)
class CompressionLayout:
    """Per-document compressed-entry layout for one compress rate.

    An entry is one compressed KV vector: a compressor pools a window of `compress_rate`
    consecutive tokens of the packed input sequence, the entry's source tokens, into a single
    `head_dim` vector, and the attention block reads the resulting series as extra keys and values
    along with its local sliding window.
    """

    entry_tok_idx: Tensor  # (n_entries, compress_rate) int64 - token index in the packed sequence, per entry
    entry_doc_idx: Tensor  # (n_entries,) int64 - which document each entry belongs to
    entry_local_idx: Tensor  # (n_entries,) int64 - entry index within its own document

    @classmethod
    def build(cls, *, cu_seqlens: Tensor, compress_rate: int) -> "CompressionLayout":
        """Lay out the compressed entries of a packed sequence, document by document.

        Document `doc` of length `L_doc` gets `L_doc // compress_rate` entries; its entry `e` covers
        the `compress_rate` source tokens starting at `cu_seqlens[doc] + e * compress_rate`. The
        trailing `L_doc % compress_rate` tokens get no entry, exactly as the unpacked case drops
        its trailing partial window; they stay visible through the local sliding window.

        A packed sequence whose every document is shorter than `compress_rate` yields zero entries,
        which is well-formed: the compressors then contribute nothing beyond their local window.
        """
        device = cu_seqlens.device
        starts = cu_seqlens[:-1].to(torch.int64)
        lengths = cu_seqlens[1:].to(torch.int64) - starts
        counts = lengths // compress_rate

        entry_doc_idx = torch.repeat_interleave(torch.arange(counts.numel(), device=device), counts)
        first_entry_of_doc = counts.cumsum(0) - counts
        entry_local_idx = torch.arange(int(counts.sum()), device=device) - first_entry_of_doc[entry_doc_idx]
        entry_pos = entry_local_idx * compress_rate
        entry_tok_idx = (
            starts[entry_doc_idx, None] + entry_pos[:, None] + torch.arange(compress_rate, device=device)[None, :]
        )

        return cls(
            entry_tok_idx=entry_tok_idx,
            entry_doc_idx=entry_doc_idx,
            entry_local_idx=entry_local_idx,
        )


@dataclass(frozen=True)
class PackedContext:
    """Everything an attention layer needs to know about the packed row it is running on.

    The mask, the positions, the RoPE tables and the layouts all encode the same document
    boundaries and are only correct together. As separate arguments they can contradict each
    other: a mask built without document boundaries spans documents while a layout does not, a
    sequence-global `position_ids` feeds `causal_threshold` a count that a per-document
    `entry_local_idx` cannot be compared against, and a RoPE table evaluated at one set of
    positions rotates queries the thresholds were not counted at. `build` derives every field
    from one `seq_lens`, so none of those is reachable. It runs once per model forward, since
    none of this depends on depth.
    """

    attention_mask: Tensor  # (1, 1, seq_len, seq_len) additive - causal, local window, document-clipped
    position_ids: Tensor  # (1, seq_len) int64 - token position within its own document
    tok_doc_idx: Tensor  # (seq_len,) int64 - which document each packed token belongs to
    position_embeddings: dict[str, tuple[Tensor, Tensor]]  # (cos, sin) keyed by rope type
    window_indices: Tensor | None  # (seq_len, sliding_window) int32 - packed token per window slot, -1 where unused
    compression_layouts: dict[int, CompressionLayout]  # keyed by compress rate

    def __post_init__(self) -> None:
        # Only reachable by constructing the dataclass directly; `build` cannot violate it.
        total_tokens = self.tok_doc_idx.shape[0]
        if self.attention_mask.shape[-2] != total_tokens or self.position_ids.shape[-1] != total_tokens:
            raise ValueError(
                f"attention_mask covers {self.attention_mask.shape[-2]} query rows and position_ids "
                f"{self.position_ids.shape[-1]} tokens, but the row has {total_tokens}"
            )
        if self.window_indices is not None and self.window_indices.shape[0] != total_tokens:
            raise ValueError(
                f"window_indices covers {self.window_indices.shape[0]} query rows, but the row has {total_tokens}"
            )

    @classmethod
    def build(
        cls,
        *,
        rotary_emb: DeepseekV4RotaryEmbedding,
        seq_lens: Tensor,
        dtype: torch.dtype,
        device: torch.device,
    ) -> "PackedContext":
        """Derive every field from one `seq_lens`, ensuring mutual consistency.

        `rotary_emb` supplies the RoPE tables and, through the config it was built from, the
        sliding window and the compress rates in use. Taking the config from it rather than
        alongside it keeps them from naming different architectures. `dtype` must be the dtype
        attention runs at, since the mask is additive. The row is as wide as `seq_lens` says,
        padding included: both packers fold their padding into the last document.
        """
        config = rotary_emb.config
        # Read the width before `seq_lens` moves: on a CPU `seq_lens` that costs no device sync.
        total_tokens = int(seq_lens.sum())
        cu_seqlens, _ = get_cu_seqlens_from_seq_lens(seq_lens.to(device=device))
        tok_idx = torch.arange(total_tokens, device=device)
        tok_doc_idx = torch.searchsorted(cu_seqlens[1:].to(tok_idx.dtype), tok_idx, right=True)
        # Document-local by construction: a token's position is its distance from its own
        # document's start, which is what `causal_threshold` and the entry rotation count in.
        position_ids = (tok_idx - cu_seqlens[tok_doc_idx])[None]
        compress_rates = {
            config.compress_rates[layer_type]
            for layer_type in set(config.layer_types)
            if layer_type in config.compress_rates
        }

        # Only the sparse attention path reads these, so an architecture without a compressed
        # layer builds none.
        window_indices = None
        if compress_rates:
            # Query `s` may read token `n` when `n` lies in `s`'s own document and within the
            # window, `0 <= s - n < sliding_window`. Since `n <= s`, only the lower bound binds,
            # and the document starts at `s - position_ids[s]`.
            base = torch.maximum(tok_idx - position_ids[0], tok_idx - config.sliding_window + 1)
            slots = base[:, None] + torch.arange(config.sliding_window, device=device)[None, :]
            # An unused slot holds `-1` here, not the pad-slot index that the sparse path marks
            # it with downstream: that index is `S + E`, and `E` varies with the layer's compress
            # rate while this context is depth-independent. `SparseAttnInputs.build` translates.
            window_indices = torch.where(slots <= tok_idx[:, None], slots, -1).to(torch.int32)
            # TODO: these slots are `base[s] + arange(W)`, so materializing an (S, W) int32 slab is
            # redundant. A kernel taking (window_base, top_k_indices) separately would avoid it, saving
            # 4*W bytes per token (512 at W=128, 33.5 MB at S=65536 per CSA layer) plus the per-layer
            # concatenation, and dropping W index loads per query. Deferred: computing the window slots
            # arithmetically requires them to fill whole slot-tiles, i.e. W % block_I == 0. Production
            # satisfies that at W=128, but it would constrain test configs to W in {64, 128}.

        return cls(
            attention_mask=build_sliding_window_mask(
                tok_doc_idx=tok_doc_idx, sliding_window=config.sliding_window, dtype=dtype
            ),
            position_ids=position_ids,
            tok_doc_idx=tok_doc_idx,
            position_embeddings={
                rope_type: rotary_emb(position_ids, rope_type, dtype=dtype) for rope_type in rotary_emb.layer_types
            },
            compression_layouts={
                rate: CompressionLayout.build(cu_seqlens=cu_seqlens, compress_rate=rate) for rate in compress_rates
            },
            window_indices=window_indices,
        )

    def check_position_ids(self, position_ids: Tensor) -> None:
        """Raise unless `position_ids` agrees with the document boundaries this context came from.

        A document starts exactly where the derived positions are zero, so the check is that the
        caller's positions vanish there too. A padded micro-batch restarts `position_ids` at 0
        inside its last document, which this permits: padding sits mid-document, never at a start.
        A sequence-global `arange` over a packed row never restarts, and a 1-based one never
        reaches zero at all; both are rejected.
        """
        disagrees = (self.position_ids == 0) & (position_ids != 0)
        if disagrees.any():
            token = int(disagrees.any(dim=0).nonzero()[0])
            raise ValueError(
                f"position_ids must restart at 0 at every document boundary of seq_lens: token "
                f"{token} starts a document but carries {position_ids[:, token].tolist()}. A caller "
                "that passes none of its own gets the 1-based arange the injected LM head "
                "substitutes (see `prime_rl.trainer.models.layers.lm_head`), which this rejects."
            )

    def token_entry_causal_mask(self, compress_rate: int, threshold: Tensor) -> Tensor:
        """`(1, seq_len, n_entries)` bool: which compressed entries each query token may read.

        Element `[0, t, e]` is true when query token `t` may read entry `e` of the rate's layout.
        Both of these have to hold:

        - `e` belongs to `t`'s own document, so no query reads another document's history;
        - `e` closed before `t` arrived, i.e. its index within that document is below
          `threshold[0, t]`, the count of entries the query's position has completed.

        One `seq_lens` describes one packed row, so the leading axis is 1 and broadcasts over the
        batch, as `threshold` does.

        `threshold` counts per document, so it is compared against `entry_local_idx` and not
        against the sequence-global entry number; those two coordinate systems disagree for every
        document after the first.
        """
        layout = self.compression_layouts[compress_rate]
        same_document = self.tok_doc_idx[None, :, None] == layout.entry_doc_idx[None, None, :]
        return same_document & (threshold.unsqueeze(-1) > layout.entry_local_idx[None, None, :])


@dataclass(frozen=True)
class SparseAttnInputs:
    """The KV buffer one CSA layer gathers from, and the gather indices addressing it.

    `build` constructs the two together so they stay mutually consistent and cannot drift apart.

    With `S` tokens in the packed row and `E` compressed entries for this layer's rate:

        kv_buf[b, n, 0, d]:  n in [0, S)     -> local token stream
                             n in [S, S + E) -> compressed entry (n - S)
                             n = S + E       -> zeros; this position is also the sentinel index
    """

    kv_buf: Tensor  # (batch, n_positions, 1, head_dim), trailing position is the zero pad slot
    indices: Tensor  # (batch, seq_len, 1, n_slots) int32 into kv_buf's position axis

    def __post_init__(self) -> None:
        # Shape invariants only. Asserting on index values would read the device, and this runs
        # once per layer per step.
        if self.kv_buf.ndim != 4 or self.kv_buf.shape[2] != 1:
            raise ValueError(f"kv_buf must be (batch, n_positions, 1, head_dim), got {tuple(self.kv_buf.shape)}")
        if self.indices.ndim != 4 or self.indices.shape[2] != 1:
            raise ValueError(f"indices must be (batch, seq_len, 1, n_slots), got {tuple(self.indices.shape)}")
        if self.indices.shape[0] != self.kv_buf.shape[0]:
            raise ValueError(f"kv_buf covers {self.kv_buf.shape[0]} batch entries and indices {self.indices.shape[0]}")
        if self.indices.shape[-1] % _SLOT_TILE != 0:
            raise ValueError(f"n_slots must be a multiple of {_SLOT_TILE}, got {self.indices.shape[-1]}")

    @property
    def sentinel(self) -> int:
        """The pad position, read off the buffer so the two cannot disagree."""
        return self.kv_buf.shape[1] - 1

    @classmethod
    def build(
        cls,
        *,
        kv: Tensor,  # (batch, 1, seq_len, head_dim), the rotated local token stream
        compressed_kv: Tensor,  # (batch, 1, n_entries, head_dim)
        top_k_indices: Tensor,  # (batch, seq_len, min(index_topk, n_entries)) int64, -1 marks a surplus pick
        window_indices: Tensor,  # (seq_len, sliding_window) int32, -1 marks an invalid slot
        sliding_window: int,
        index_topk: int,
    ) -> "SparseAttnInputs":
        """Lay out one CSA layer's gather slots: the local window first, then the indexer's picks."""
        batch, _, seq_len, head_dim = kv.shape
        positions = torch.cat([kv, compressed_kv], dim=2).transpose(1, 2)  # (b, S + E, 1, d)
        # The pad slot is appended here, so the sentinel index cannot disagree with the buffer.
        kv_buf = torch.cat([positions, positions.new_zeros(batch, 1, 1, head_dim)], dim=1).contiguous()
        sentinel = kv_buf.shape[1] - 1

        # The slot count comes from the config and never from `top_k_indices.shape[-1]`: the
        # indexer returns only `min(index_topk, n_entries)` picks, so a shape-derived width would
        # vary with sequence length and force the kernel to recompile per shape.
        n_slots = ((sliding_window + index_topk + _SLOT_TILE - 1) // _SLOT_TILE) * _SLOT_TILE
        # Allocated at full width and prefilled with the sentinel, so the indices cannot be too
        # narrow and everything the two writes below leave untouched is masked. That is what
        # makes `n_entries < index_topk` safe.
        indices = torch.full((batch, seq_len, 1, n_slots), sentinel, dtype=torch.int32, device=kv.device)
        # `-1` is consumed here and never escapes into the buffer's index space.
        indices[..., :sliding_window] = torch.where(window_indices >= 0, window_indices, sentinel).unsqueeze(1)
        n_picks = top_k_indices.shape[-1]
        entry_slots = torch.where(top_k_indices >= 0, top_k_indices + seq_len, sentinel)
        indices[..., sliding_window : sliding_window + n_picks] = entry_slots.unsqueeze(2).to(torch.int32)
        return cls(kv_buf=kv_buf, indices=indices)

    def attend(self, q: Tensor, sinks: Tensor, scale: float) -> Tensor:
        """Attend `q` (batch, seq_len, heads, head_dim) over this layer's gathered slots."""
        return sparse_attention_gather(q, self.kv_buf, self.indices, sinks, scale)


class DeepseekV4Compressor(nn.Module):
    """Softmax-gated pooling of the token stream into one entry per `compress_rate` tokens, per the
    `CompressionLayout` specification. Schematic output:

        `C[e,d] = sum_s softmax_s(gate[e,s,d] + position_bias[s,d]) * kv[e,s,d]`

    `kv` and `gate` are this compressor's own projections of the hidden state, gathered at the
    source tokens of entry `e`'s pooling window, and `d` runs over `head_dim`. Each entry is
    RMSNormed and rotated with the `compress` RoPE at its window's first source position, which
    is what makes it comparable with the attention block's locally rotated KV stream. `forward`
    returns the entries alongside this layer's entry selection, in whatever form the layer's
    attention path consumes: an additive `block_bias` to concatenate onto the local sliding
    window for the dense path, the indexer's picks for the sparse one.

    `n_series` sets the slots `s` the gate ranges over. With `1` a token joins only its own
    window, so windows are disjoint. With `2` the projections emit two `head_dim`-wide series
    `Ca` and `Cb`, and entry `e` pools `Ca` from entry `e - 1`'s tokens together with `Cb` from
    its own, so windows overlap at stride `compress_rate`; a document's first entry has no
    predecessor, so its `Ca` slots are gated with `-inf`.
    """

    rope_layer_type = "compress"

    def __init__(self, config: DeepseekV4Config, head_dim: int, compress_rate: int, n_series: int):
        super().__init__()
        if n_series not in (1, 2):
            raise ValueError(f"n_series must be 1 or 2, got {n_series}")
        self.compress_rate = compress_rate
        self.head_dim = head_dim
        self.n_series = n_series
        self.kv_proj = nn.Linear(config.hidden_size, n_series * head_dim, bias=False)
        self.gate_proj = nn.Linear(config.hidden_size, n_series * head_dim, bias=False)
        self.position_bias = nn.Parameter(torch.zeros(compress_rate, n_series * head_dim))
        self.kv_norm = RMSNorm(RMSNormConfig(hidden_size=head_dim, eps=config.rms_norm_eps))
        self.rotary_emb = DeepseekV4RotaryEmbedding(config)

    def _overlap_with_previous_window(
        self, kv: torch.Tensor, gate: torch.Tensor, layout: CompressionLayout
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Widen each entry from `compress_rate` slots to `2 * compress_rate`, the `n_series == 2` case."""
        n_entries = layout.entry_tok_idx.shape[0]

        # Shift the `Ca` series one entry later so entry `e` sees entry `e - 1`'s. The first
        # entry of every document has no predecessor, and the entry sitting before it in the
        # packed sequence belongs to another document, so both halves are cleared: the gate to
        # `-inf` and the values to zero. Zeroing is not redundant with the gate, because a
        # zero softmax weight against a non-finite value would still yield NaN.
        previous = (torch.arange(n_entries, device=kv.device) - 1).clamp(min=0)
        is_first_entry_in_doc = (layout.entry_local_idx == 0)[None, :, None, None]
        previous_kv = kv[:, previous, :, : self.head_dim].masked_fill(is_first_entry_in_doc, 0.0)
        previous_gate = gate[:, previous, :, : self.head_dim].masked_fill(is_first_entry_in_doc, float("-inf"))
        return (
            torch.cat([previous_kv, kv[..., self.head_dim :]], dim=2),
            torch.cat([previous_gate, gate[..., self.head_dim :]], dim=2),
        )

    def compress(self, hidden_states: torch.Tensor, packed: PackedContext) -> torch.Tensor:
        """Compress `(batch, seq_len, hidden_size)` to `(batch, n_entries, head_dim)`.

        The layout at this compressor's own rate decides which source tokens each entry pools.
        """
        batch = hidden_states.shape[0]
        layout = packed.compression_layouts[self.compress_rate]

        kv = self.kv_proj(hidden_states)[:, layout.entry_tok_idx]
        gate = self.gate_proj(hidden_states)[:, layout.entry_tok_idx] + self.position_bias
        if self.n_series == 2:
            kv, gate = self._overlap_with_previous_window(kv, gate, layout)

        # fp32 softmax: in bf16 the gate logits of a wide window collapse onto each other.
        weights = gate.softmax(dim=2, dtype=torch.float32).to(kv.dtype)
        compressed = self.kv_norm((kv * weights).sum(dim=2))

        entry_first_tok_pos = layout.entry_local_idx * self.compress_rate
        cos, sin = self.rotary_emb(
            entry_first_tok_pos.unsqueeze(0).expand(batch, -1), self.rope_layer_type, dtype=compressed.dtype
        )
        return apply_rotary_pos_emb_interleaved(compressed.unsqueeze(1), cos, sin).squeeze(1)

    def causal_threshold(self, position_ids: torch.Tensor) -> torch.Tensor:
        """Number of compressed entries that query `t` may read, shaped like `position_ids`.

        Entry `e` pools source tokens up to index `(e + 1) * compress_rate - 1`, so it only
        becomes readable once the query has reached that token.
        """
        return (position_ids + 1) // self.compress_rate

    def init_weights(self, init_std: float) -> None:
        # `init_std` is unused: the projections are initialized by the caller and the
        # position bias starts at zero, i.e. a uniform gate over the pooling window.
        nn.init.zeros_(self.position_bias)


class DeepseekV4IndexerScorer(nn.Module):
    """Lightning-Indexer score `score[t,e] = sum_h w[t,h] * ReLU(q[t,h,d] * k[e,d])`.

    Query token `t` against compressed entry `e`, over indexer heads `h` and `index_head_dim`
    channels `d`. The per-head weights `w[t,h]` come off the hidden state directly rather than
    from a query-key interaction, which keeps the scorer one matmul deep. It runs in fp32: the
    scores only feed a top-k, so the width costs little and near-ties are not decided by bf16
    rounding.
    """

    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.softmax_scale = config.index_head_dim**-0.5
        self.weights_scaling = config.index_n_heads**-0.5
        self.weights_proj = nn.Linear(config.hidden_size, config.index_n_heads, bias=False)

    def forward(self, q: torch.Tensor, compressed_kv: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        """Score `q` `(batch, seq, heads, dim)` against `compressed_kv` `(batch, entries, dim)`."""
        scores = torch.matmul(q.float(), compressed_kv.transpose(-1, -2).float().unsqueeze(1))
        scores = F.relu(scores) * self.softmax_scale
        weights = self.weights_proj(hidden_states).float() * self.weights_scaling
        return (scores * weights.unsqueeze(-1)).sum(dim=2)


class DeepseekV4Indexer(nn.Module):
    """Lightning Indexer: picks the `index_topk` compressed entries each query may read.

    It owns a compressor at the narrow `index_head_dim` and scores each query against its
    entries. The indices it returns address the entries of the compressor that owns it: both
    share `compress_rate` and the `compress` RoPE base, so entry `e` in one covers the same
    source tokens as entry `e` in the other, and the scores depend only on the query-key
    distance.

    Each query gets `min(index_topk, entries)` picks. An early query has fewer entries whose
    source tokens all lie at or before it, and its surplus picks come back as `-1`.
    """

    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.head_dim = config.index_head_dim
        self.num_heads = config.index_n_heads
        self.index_topk = config.index_topk
        self.compressor = DeepseekV4Compressor(
            config, self.head_dim, config.compress_rates["compressed_sparse_attention"], n_series=2
        )
        self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.head_dim, bias=False)
        self.scorer = DeepseekV4IndexerScorer(config)

    def forward(self, hidden_states: torch.Tensor, q_residual: torch.Tensor, packed: PackedContext) -> torch.Tensor:
        batch, seq_len, _ = hidden_states.shape
        compressed_kv = self.compressor.compress(hidden_states, packed)
        compressed_len = compressed_kv.shape[1]
        top_k = min(self.index_topk, compressed_len)

        # The token-position table for this rope type is already on `packed`; the compressor's own
        # rotary is only ever evaluated at entry positions.
        cos, sin = packed.position_embeddings[self.compressor.rope_layer_type]
        q = self.q_b_proj(q_residual).view(batch, seq_len, -1, self.head_dim).transpose(1, 2)
        q = apply_rotary_pos_emb_interleaved(q, cos, sin).transpose(1, 2)

        scores = self.scorer(q, compressed_kv, hidden_states)
        if compressed_len == 0:
            return scores.topk(top_k, dim=-1).indices

        threshold = self.compressor.causal_threshold(packed.position_ids)
        readable = packed.token_entry_causal_mask(self.compressor.compress_rate, threshold).expand_as(scores)
        scores = scores.masked_fill(~readable, float("-inf"))
        top_k_indices = scores.topk(top_k, dim=-1).indices
        # An early query has fewer than `top_k` readable entries, so top-k still hands back
        # masked-out ones. Mark those `-1` rather than letting them leak into attention.
        return torch.where(readable.gather(-1, top_k_indices), top_k_indices, torch.full_like(top_k_indices, -1))

    def init_weights(self, init_std: float) -> None:
        self.compressor.init_weights(init_std)


def block_bias_from_indices(top_k_indices: Tensor, n_entries: int, dtype: torch.dtype) -> Tensor:
    """Render the indexer's picks as the dense additive `(batch, 1, seq_len, n_entries)` bias.

    `0` on the selected entries, `-inf` everywhere else. The dense and sparse attention paths
    both start from the same index tensor, so they cannot disagree about which entries a query
    reads.
    """
    batch, seq_len, _ = top_k_indices.shape
    # The `-1` sentinels are scattered into one throwaway column that is sliced back off.
    safe_indices = torch.where(top_k_indices >= 0, top_k_indices, torch.full_like(top_k_indices, n_entries))
    block_bias = torch.full((batch, 1, seq_len, n_entries + 1), float("-inf"), dtype=dtype, device=top_k_indices.device)
    block_bias.scatter_(-1, safe_indices.unsqueeze(1), 0.0)
    return block_bias[..., :n_entries]


class DeepseekV4CSACompressor(DeepseekV4Compressor):
    """Compressed Sparse Attention compressor: the sparse long-range half of a CSA layer.

    Two series at a fine compress rate, with overlapping windows. A Lightning Indexer scores
    the entries and keeps the `index_topk` best per query, and the returned `top_k_indices` is
    that selection, with `-1` marking a surplus pick. It needs no separate causal term, because
    the indexer only selects entries whose source tokens all lie at or before the query.
    """

    def __init__(self, config: DeepseekV4Config):
        super().__init__(config, config.head_dim, config.compress_rates["compressed_sparse_attention"], n_series=2)
        self.indexer = DeepseekV4Indexer(config)

    def forward(
        self, hidden_states: torch.Tensor, q_residual: torch.Tensor, packed: PackedContext
    ) -> tuple[torch.Tensor, torch.Tensor]:
        compressed_kv = self.compress(hidden_states, packed).unsqueeze(1)
        # The indexer reads the same layout: it compresses the same source windows at a narrower
        # head dim, so its entry `e` and this compressor's entry `e` are the same window.
        return compressed_kv, self.indexer(hidden_states, q_residual, packed)

    def init_weights(self, init_std: float) -> None:
        super().init_weights(init_std)
        self.indexer.init_weights(init_std)


class DeepseekV4HCACompressor(DeepseekV4Compressor):
    """Heavily Compressed Attention compressor: the dense long-range half of an HCA layer.

    One series at a coarse compress rate, with disjoint windows. There is no indexer: a query
    reads every entry whose source tokens all lie at or before it, and the returned
    `block_bias` carries that rule.
    """

    def __init__(self, config: DeepseekV4Config):
        super().__init__(config, config.head_dim, config.compress_rates["heavily_compressed_attention"], n_series=1)

    def forward(
        self, hidden_states: torch.Tensor, q_residual: torch.Tensor, packed: PackedContext
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`q_residual` is part of the compressor contract but unused: HCA has no indexer."""
        batch, seq_len, _ = hidden_states.shape
        compressed_kv = self.compress(hidden_states, packed).unsqueeze(1)
        compressed_len = compressed_kv.shape[2]

        threshold = self.causal_threshold(packed.position_ids)
        readable = packed.token_entry_causal_mask(self.compress_rate, threshold).unsqueeze(1)
        block_bias = compressed_kv.new_zeros((batch, 1, seq_len, compressed_len))
        return compressed_kv, block_bias.masked_fill_(~readable, float("-inf"))


COMPRESSOR_CLASSES = {
    "sliding_attention": None,
    "compressed_sparse_attention": DeepseekV4CSACompressor,
    "heavily_compressed_attention": DeepseekV4HCACompressor,
}


class DeepseekV4Attention(nn.Module):
    """DeepSeek-V4 self-attention.

    Four things set it apart from a standard attention block:

    1. Shared-KV multi-query attention. `kv_proj` emits a single `head_dim`-wide vector
       per token that serves as both key and value for every query head.
    2. Partial interleaved RoPE on the trailing `qk_rope_head_dim` channels of each head.
       Because the value carries that rotation too, the conjugate rotation is applied to
       the attention output, which leaves each key's contribution a function of its
       relative distance to the query.
    3. A per-head learnable attention sink.
    4. A grouped low-rank output projection (`o_a_proj` then `o_b_proj`).

    Every layer type runs that same core over its local sliding window. The two compressed
    types additionally own a `compressor` whose output is concatenated onto the local KV,
    which is how a layer sees past the window.
    """

    def __init__(self, config: DeepseekV4Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        # Rope types are labelled `main` / `compress`, independently of `layer_types`:
        # sliding layers take the plain base, the compressed variants share their
        # compressor's base.
        self.rope_layer_type = "main" if self.layer_type == "sliding_attention" else "compress"
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.attention_dropout = config.attention_dropout
        self.scaling = self.head_dim**-0.5

        self.q_a_proj = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_a_norm = RMSNorm(RMSNormConfig(hidden_size=config.q_lora_rank, eps=config.rms_norm_eps))
        self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.head_dim, bias=False)
        self.q_b_norm = DeepseekV4UnweightedRMSNorm(eps=config.rms_norm_eps)
        self.kv_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.kv_norm = RMSNorm(RMSNormConfig(hidden_size=self.head_dim, eps=config.rms_norm_eps))
        self.o_a_proj = DeepseekV4GroupedLinear(
            self.num_heads * self.head_dim // config.o_groups,
            config.o_groups * config.o_lora_rank,
            config.o_groups,
        )
        self.o_b_proj = nn.Linear(config.o_groups * config.o_lora_rank, config.hidden_size, bias=False)
        self.sinks = nn.Parameter(torch.zeros(self.num_heads))
        compressor_class = COMPRESSOR_CLASSES[self.layer_type]
        self.compressor = compressor_class(config) if compressor_class is not None else None

    def _eager(self, q: Tensor, kv: Tensor, attention_mask: Tensor) -> Tensor:
        return eager_attention_with_sinks(
            q,
            kv,
            kv,
            self.sinks,
            attention_mask,
            scaling=self.scaling,
            dropout=self.attention_dropout,
            training=self.training,
        )

    def _attend(
        self, hidden_states: Tensor, q_residual: Tensor, q: Tensor, kv: Tensor, packed: PackedContext
    ) -> Tensor:
        """Attend `q` (b, h, t, d) over the local KV `kv` (b, 1, t, d), plus any compressed entries.

        Returns (b, t, h, d). A sliding layer sees only its window; a compressed layer reaches
        further through its compressor's entries, either by concatenating them onto the dense
        mask or, on the sparse path, by gathering the selected ones per query.
        """
        if self.compressor is None:
            return self._eager(q, kv, packed.attention_mask)

        if self.layer_type == "compressed_sparse_attention":
            compressed_kv, top_k_indices = self.compressor(hidden_states, q_residual, packed)
            if DSV4_ATTN_IMPL != "eager":
                # `eager_attention_with_sinks` drops attention weights, the sparse path does not,
                # so the two only agree at zero. The default is 0.0 but a config may set it.
                assert self.attention_dropout == 0.0, "the sparse attention path implements no dropout"
                inputs = SparseAttnInputs.build(
                    kv=kv,
                    compressed_kv=compressed_kv,
                    top_k_indices=top_k_indices,
                    window_indices=packed.window_indices,
                    sliding_window=self.config.sliding_window,
                    index_topk=self.config.index_topk,
                )
                # The kernel that will replace the gather reference asserts contiguity.
                return inputs.attend(q.transpose(1, 2).contiguous(), self.sinks, self.scaling)
            block_bias = block_bias_from_indices(top_k_indices, compressed_kv.shape[2], packed.attention_mask.dtype)
        else:
            compressed_kv, block_bias = self.compressor(hidden_states, q_residual, packed)

        kv = torch.cat([kv, compressed_kv], dim=2)  # (b, 1, t + e, d)
        # The compressed entries live outside the local window, so the sliding mask says
        # nothing about them; `block_bias` carries their per-query causality and the
        # indexer's selection. Zero-padding instead would let every query read every one.
        attention_mask = torch.cat(
            [packed.attention_mask.expand(*block_bias.shape[:-1], -1), block_bias.to(packed.attention_mask.dtype)],
            dim=-1,
        )  # (b, 1, t, t + e)
        return self._eager(q, kv, attention_mask)

    def forward(self, hidden_states: torch.Tensor, packed: PackedContext) -> tuple[torch.Tensor, None]:
        """`packed` carries the document boundaries every pathway below is clipped at."""
        # Shape keys in the comments below:
        #
        # - `b`: batch
        # - `t`: token in the packed row
        # - `h`: attention head
        # - `d`: head_dim
        # - `e`: compressed entry
        # - `r`: q_lora_rank
        # - `g`: o_groups
        # - `l`: o_lora_rank
        #
        # `hidden_states` is (b, t, hidden_size).

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)  # (b, t, -1, d): the -1 is h for q, 1 for kv
        cos, sin = packed.position_embeddings[self.rope_layer_type]  # (1, t, qk_rope_head_dim // 2) each

        q_residual = self.q_a_norm(self.q_a_proj(hidden_states))  # (b, t, r)
        q = self.q_b_proj(q_residual).view(*hidden_shape).transpose(1, 2)  # (b, h, t, d)
        q = apply_rotary_pos_emb_interleaved(self.q_b_norm(q), cos, sin)

        kv = self.kv_norm(self.kv_proj(hidden_states)).view(*hidden_shape).transpose(1, 2)  # (b, 1, t, d)
        kv = apply_rotary_pos_emb_interleaved(kv, cos, sin)

        attn_output = self._attend(hidden_states, q_residual, q, kv, packed)  # (b, t, h, d)

        # The value stream is the key stream, so it arrived rotated. Rotating the output
        # by the conjugate angle at the query position cancels that out.
        attn_output = apply_rotary_pos_emb_interleaved(attn_output, cos, -sin, unsqueeze_dim=2)

        # (b, t, g, h * d // g) -> (b, t, g, l) -> (b, t, g * l)
        grouped = self.o_a_proj(attn_output.reshape(*input_shape, self.config.o_groups, -1)).flatten(2)
        return self.o_b_proj(grouped), None  # (b, t, hidden_size)

    def init_weights(self, init_std: float) -> None:
        # `init_std` is only passed through: the sinks are the only parameter this owns
        # outright and they start at zero.
        nn.init.zeros_(self.sinks)
        if self.compressor is not None:
            self.compressor.init_weights(init_std)


__all__ = [
    "CompressionLayout",
    "DeepseekV4Attention",
    "DeepseekV4CSACompressor",
    "DeepseekV4GroupedLinear",
    "DeepseekV4HCACompressor",
    "DeepseekV4Indexer",
    "DeepseekV4IndexerScorer",
    "PackedContext",
    "SparseAttnInputs",
    "block_bias_from_indices",
    "build_sliding_window_mask",
    "eager_attention_with_sinks",
    "sparse_attention_gather",
]
