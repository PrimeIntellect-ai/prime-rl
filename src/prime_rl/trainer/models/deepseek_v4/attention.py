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
                     gather each query's slots, then QKᵀ
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

  - `position_ids`: each token's position within its own document.
  - `tok_doc_idx`: which document each token belongs to.
  - `position_embeddings`: the RoPE tables, one per rope type, evaluated at `position_ids`.
  - `window_indices`: for each query, the indices of the tokens its local window covers, causal
    and clipped at document boundaries, with `IGNORE_SLOT` (-1) marking invalid/masked entries.
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
That window is enumerated per document by `window_indices`, and every rotation in the block needs
`position_ids` together with the RoPE tables evaluated at them, `position_embeddings`. None of
those belongs to any single compress rate.

`PackedContext.build` takes `seq_lens` and derives every one of its fields from it. Nothing else is
an input, so a position that disagrees with a document boundary, a window slot that spans one, or a
RoPE table evaluated at positions other than the ones the causal thresholds count in, cannot be
constructed. It runs once per model forward.

Context parallelism splits the queries across ranks and leaves everything else alone: every field
above has one entry per query token and so covers this rank's `n_queries` tokens, while
`compression_layouts` covers all `total_tokens` of the sequence. Token indices always count from
the start of the whole sequence.

[The Index Contract]

All three layer types reach their keys the same way. `SparseAttnInputs` lays out one KV buffer,

    kv_buf[b, n, 0, d]:  the packed token stream, then this layer's compressed entries, and
                         nothing else: an absent key needs no position of its own

and one int32 index tensor addressing that position axis, `sliding_window + picks` slots per
query: the local window first, the picks after. `picks` is the indexer's `index_topk` for CSA,
`max_entries_per_doc` for HCA, and zero for a sliding layer, which reads its window alone. A slot
with nothing to read holds `IGNORE_SLOT` (-1), which the kernel masks on, so a short window and a
surplus pick cost only their loads.
"""

from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import Tensor, nn

from prime_rl.trainer.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from prime_rl.trainer.models.deepseek_v4.hyperconnections import DeepseekV4UnweightedRMSNorm
from prime_rl.trainer.models.deepseek_v4.rotary import DeepseekV4RotaryEmbedding, apply_rotary_pos_emb_interleaved
from prime_rl.trainer.models.kernels.deepseek_v4 import IGNORE_SLOT
from prime_rl.trainer.models.kernels.fp8_indexer import fp8_indexer
from prime_rl.trainer.models.layers.norms import RMSNorm, RMSNormConfig
from prime_rl.utils.cp import gather_for_cp
from prime_rl.utils.sequence import get_cu_seqlens_from_seq_lens

# Guarded because tilelang ships in the linux-gated `gpu` extra, so some installs lack it.
try:
    from prime_rl.trainer.models.kernels.deepseek_v4.dsv4_sparse_attn import dsv4_sparse_attn, sparse_attn_shape_error
except ImportError:
    dsv4_sparse_attn = None  # type: ignore
    sparse_attn_shape_error = None  # type: ignore


def _kernel_blocker(num_heads: int, head_dim: int) -> str | None:
    """Why the fused kernel cannot run at this shape, or ``None`` if it can."""
    if dsv4_sparse_attn is None:
        return "the tilelang sparse-attention kernel failed to import; install the `gpu` extra"
    # CSA gives every query head the same single KV head, so the kernel's `kv_group` is 1. The
    # shape constraints themselves are stated once, next to the kernels they come from.
    return sparse_attn_shape_error(num_heads, 1, head_dim)


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
    first_entry_of_doc: Tensor  # (n_docs,) int64 - sequence-global index of each document's first entry
    max_entries_per_doc: int  # largest entry count any single document contributes

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
            first_entry_of_doc=first_entry_of_doc,
            max_entries_per_doc=int(counts.max()),
        )


@dataclass(frozen=True)
class PackedContext:
    """Everything an attention layer needs to know about the packed row it is running on.

    The window indices, the positions, the RoPE tables and the layouts all encode the same
    document boundaries and are only correct together. As separate arguments they can contradict
    each other: a window enumerated without document boundaries spans documents while a layout
    does not, a sequence-global `position_ids` feeds `causal_threshold` a count that a
    per-document `entry_local_idx` cannot be compared against, and a RoPE table evaluated at one
    set of positions rotates queries the thresholds were not counted at. `build` derives every
    field from one `seq_lens`, so none of those is reachable. It runs once per model forward.

    Context parallelism gives each rank a contiguous run of `n_queries` tokens to use as queries
    and a full copy of the keys, so every field but `compression_layouts` has one entry per query
    token and covers this rank's run alone, while `compression_layouts` covers the whole sequence.
    Token indices always count from the start of the whole sequence, never from this rank's run.
    """

    position_ids: Tensor  # (1, n_queries) int64 - token position within its own document
    tok_doc_idx: Tensor  # (n_queries,) int64 - which document each query token belongs to
    position_embeddings: dict[str, tuple[Tensor, Tensor]]  # (cos, sin) keyed by rope type, at `position_ids`
    window_indices: Tensor  # (n_queries, sliding_window) int32 - global token per window slot, IGNORE_SLOT if unused
    compression_layouts: dict[int, CompressionLayout]  # keyed by compress rate

    @classmethod
    def build(
        cls,
        *,
        rotary_emb: DeepseekV4RotaryEmbedding,
        seq_lens: Tensor,
        dtype: torch.dtype,
        device: torch.device,
        cp_rank: int = 0,
        cp_world_size: int = 1,
    ) -> "PackedContext":
        """Derive every field from one `seq_lens`, ensuring mutual consistency.

        `rotary_emb` supplies the RoPE tables and, through the config it was built from, the
        sliding window and the compress rates in use. Taking the config from it rather than
        alongside it keeps them from naming different architectures. `dtype` must be the dtype
        attention runs at, since it types the RoPE tables. The sequence is as long as `seq_lens` says,
        padding included: both packers fold their padding into the last document.

        `seq_lens` always describes the whole sequence. `cp_rank` and `cp_world_size` say which
        contiguous shard of it this rank holds the queries of; the keys, the entries and the index
        values addressing them stay global, so only the query side narrows.
        """
        config = rotary_emb.config
        # Read the width before `seq_lens` moves: on a CPU `seq_lens` that costs no device sync.
        total_tokens = int(seq_lens.sum())
        assert total_tokens % cp_world_size == 0, (
            f"{total_tokens} tokens do not split evenly across {cp_world_size} CP ranks"
        )
        n_queries = total_tokens // cp_world_size
        q_start = cp_rank * n_queries

        cu_seqlens, _ = get_cu_seqlens_from_seq_lens(seq_lens.to(device=device))
        compress_rates = {
            config.compress_rates[layer_type]
            for layer_type in set(config.layer_types)
            if layer_type in config.compress_rates
        }

        # These fields have one entry per query token, so they cover this rank's tokens only.
        # `cu_seqlens` still spans the whole sequence, so document boundaries stay available.
        tok_idx = torch.arange(q_start, q_start + n_queries, device=device)
        tok_doc_idx = torch.searchsorted(cu_seqlens[1:].to(tok_idx.dtype), tok_idx, right=True)
        # Document-local by construction: a token's position is its distance from its own
        # document's start, which is what `causal_threshold` and the entry rotation count in.
        position_ids = (tok_idx - cu_seqlens[tok_doc_idx])[None]
        position_embeddings = {
            rope_type: rotary_emb(position_ids, rope_type, dtype=dtype) for rope_type in rotary_emb.layer_types
        }

        # A token attends the last `sliding_window` positions (itself included), clipped to its own
        # document.
        window_base = torch.maximum(tok_idx - position_ids[0], tok_idx - config.sliding_window + 1)
        slots = window_base[:, None] + torch.arange(config.sliding_window, device=device)[None, :]
        window_indices = torch.where(slots <= tok_idx[:, None], slots, IGNORE_SLOT).to(torch.int32)

        return cls(
            position_ids=position_ids,
            tok_doc_idx=tok_doc_idx,
            position_embeddings=position_embeddings,
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
        reaches zero at all; both are rejected. Under CP the comparison is against this rank's
        queries alone, which lines up because the trainer shards the caller's `position_ids` the
        same way.
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


@dataclass(frozen=True)
class SparseAttnInputs:
    """The KV buffer one attention layer gathers from, and the gather indices addressing it.

    `build` constructs the two together so they stay mutually consistent and cannot drift apart.

    With `S` tokens in the packed row and `E` compressed entries for this layer's rate, `E` being
    zero for a layer that reads no entries at all:

        kv_buf[b, n, 0, d]:  n in [0, S)     -> local token stream
                             n in [S, S + E) -> compressed entry (n - S)

    Every index must be a real key in `[0, n_positions)` or `IGNORE_SLOT` (-1), which marks an absent
    key, which `build` enforces.

    Under context parallelism the token half of `kv_buf` is still the whole global stream, `S`
    being every token of the packed row, while `indices` has one entry per query token this rank
    holds.
    """

    kv_buf: Tensor  # (batch, n_positions, 1, head_dim)
    indices: Tensor  # (batch, n_queries, 1, n_slots) int32 into kv_buf's position axis

    @classmethod
    def build(
        cls,
        *,
        kv: Tensor,  # (batch, 1, n_tokens, head_dim), the rotated token stream
        compressed_kv: Tensor | None = None,  # (batch, 1, n_entries, head_dim)
        top_k_indices: Tensor | None = None,  # (batch, n_queries, n_picks) int64, IGNORE_SLOT (-1) marks a surplus pick
        window_indices: Tensor,  # (n_queries, sliding_window) int32, IGNORE_SLOT marks an invalid slot
    ) -> "SparseAttnInputs":
        """Lay out one layer's gather slots: the local window first, then any compressed picks.

        A layer with no entries passes neither `compressed_kv` nor `top_k_indices`, receiving only
        the local sliding window.
        """
        assert (compressed_kv is None) == (top_k_indices is None), (
            "compressed_kv and top_k_indices describe the same entries: pass both or neither"
        )
        # The two counts differ under CP: the keys are global and the queries are this rank's.
        batch, _, n_tokens, _ = kv.shape
        n_queries = window_indices.shape[0]
        assert top_k_indices is None or top_k_indices.shape[1] == n_queries, (
            f"top_k_indices covers {top_k_indices.shape[1]} query tokens and window_indices {n_queries}"
        )

        positions = kv if compressed_kv is None else torch.cat([kv, compressed_kv], dim=2)
        kv_buf = positions.transpose(1, 2).contiguous()  # (b, S + E, 1, d)

        window = window_indices[None, :, None, :].expand(batch, n_queries, 1, -1)
        if top_k_indices is None:
            return cls(kv_buf=kv_buf, indices=window.contiguous())

        # A surplus pick is `IGNORE_SLOT` (-1) and stays `IGNORE_SLOT`; a real one names an entry,
        # which sits past the token stream in `kv_buf`, hence the shift by `n_tokens`.
        # NOTE: the attention kernel recompiles for every unique `indices.shape[-1]` value. If
        # recompilation becomes a bottleneck, consider padding to fixed length with `IGNORE_SLOT` values.
        picks = torch.where(top_k_indices >= 0, top_k_indices + n_tokens, IGNORE_SLOT)
        indices = torch.cat([window, picks[:, :, None, :].to(torch.int32)], dim=-1)
        return cls(kv_buf=kv_buf, indices=indices)


class DeepseekV4Compressor(nn.Module):
    """Softmax-gated pooling of the token stream into one entry per `compress_rate` tokens, per the
    `CompressionLayout` specification. Schematic output:

        `C[e,d] = sum_s softmax_s(gate[e,s,d] + position_bias[s,d]) * kv[e,s,d]`

    `kv` and `gate` are this compressor's own projections of the hidden state, gathered at the
    source tokens of entry `e`'s pooling window, and `d` runs over `head_dim`. Each entry is
    RMSNormed and rotated with the `compress` RoPE at its window's first source position, which
    is what makes it comparable with the attention block's locally rotated KV stream. `forward`
    returns the entries alongside this layer's entry selection: the per-query entry indices the
    attention block gathers, with `IGNORE_SLOT` (-1) marking a slot the query has nothing to read into.

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

    def compress(
        self,
        hidden_states: torch.Tensor,
        packed: PackedContext,
        cp_group: dist.ProcessGroup | None = None,
        cp_world_size: int = 1,
    ) -> torch.Tensor:
        """Compress `(batch, seq_len, hidden_size)` to `(batch, n_entries, head_dim)`."""
        batch = hidden_states.shape[0]
        layout = packed.compression_layouts[self.compress_rate]

        width = self.n_series * self.head_dim
        proj = torch.cat([self.kv_proj(hidden_states), self.gate_proj(hidden_states)], dim=-1)
        if cp_world_size > 1:
            proj = gather_for_cp(proj, cp_group)
        kv, gate = proj.split(width, dim=-1)

        kv = kv[:, layout.entry_tok_idx]
        gate = gate[:, layout.entry_tok_idx] + self.position_bias
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


class DeepseekV4Indexer(nn.Module):
    """Lightning Indexer: picks the `index_topk` compressed entries each query may read.

    Every query gets `index_topk` picks, the width the kernel pads to. An early query has fewer
    entries whose source tokens all lie at or before it, and its surplus picks come back as
    `IGNORE_SLOT` (-1).
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
        self.weights_proj = nn.Linear(config.hidden_size, self.num_heads, bias=False)

    @torch.no_grad()  # Returns non-differentiable integer indices.
    def forward(
        self,
        hidden_states: torch.Tensor,
        q_residual: torch.Tensor,
        packed: PackedContext,
        cp_group: dist.ProcessGroup | None = None,
        cp_world_size: int = 1,
    ) -> torch.Tensor:
        batch, seq_len, _ = hidden_states.shape
        assert batch == 1, f"the indexer needs a packed batch of size 1, got {batch}"
        compressed_kv = self.compressor.compress(hidden_states, packed, cp_group=cp_group, cp_world_size=cp_world_size)
        n_entries = compressed_kv.shape[1]

        cos, sin = packed.position_embeddings[self.compressor.rope_layer_type]
        q = self.q_b_proj(q_residual).view(batch, seq_len, -1, self.head_dim).transpose(1, 2)
        q = apply_rotary_pos_emb_interleaved(q, cos, sin).transpose(1, 2)
        w = self.weights_proj(hidden_states)

        layout = packed.compression_layouts[self.compressor.compress_rate]
        entry_start = layout.first_entry_of_doc[packed.tok_doc_idx].int()
        entry_stop = (entry_start + self.compressor.causal_threshold(packed.position_ids)[0]).int()

        # fp8_indexer has no batch axis
        top_k_indices = fp8_indexer(q[0], compressed_kv[0], w[0], entry_start, entry_stop, self.index_topk).unsqueeze(0)

        # Mark indices-to-ignore with IGNORE_SLOT
        in_range = top_k_indices < n_entries
        top_k_indices = torch.where(in_range, top_k_indices, torch.full_like(top_k_indices, IGNORE_SLOT))
        return top_k_indices.long()

    def init_weights(self, init_std: float) -> None:
        self.compressor.init_weights(init_std)


class DeepseekV4CSACompressor(DeepseekV4Compressor):
    """Compressed Sparse Attention compressor: the sparse long-range half of a CSA layer.

    Two series at a fine compress rate, with overlapping windows. A Lightning Indexer scores
    the entries and keeps the `index_topk` best per query, and the returned `top_k_indices` is
    that selection, with `IGNORE_SLOT` (-1) marking a surplus pick. It needs no separate causal term,
    because the indexer only selects entries whose source tokens all lie at or before the query.
    """

    def __init__(self, config: DeepseekV4Config):
        super().__init__(config, config.head_dim, config.compress_rates["compressed_sparse_attention"], n_series=2)
        self.indexer = DeepseekV4Indexer(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_residual: torch.Tensor,
        packed: PackedContext,
        cp_group: dist.ProcessGroup | None = None,
        cp_world_size: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        compressed_kv = self.compress(hidden_states, packed, cp_group=cp_group, cp_world_size=cp_world_size)
        # The indexer reads the same layout: it compresses the same source windows at a narrower
        # head dim, so its entry `e` and this compressor's entry `e` are the same window.
        picks = self.indexer(hidden_states, q_residual, packed, cp_group=cp_group, cp_world_size=cp_world_size)
        return compressed_kv.unsqueeze(1), picks

    def init_weights(self, init_std: float) -> None:
        super().init_weights(init_std)
        self.indexer.init_weights(init_std)


class DeepseekV4HCACompressor(DeepseekV4Compressor):
    """Heavily Compressed Attention compressor: the dense long-range half of an HCA layer.

    One series at a coarse compress rate, with disjoint windows. There is no indexer: a query
    reads every entry whose source tokens all lie at or before it. A document's entries are
    numbered consecutively, so that set is the contiguous range starting at the document's first
    entry, and the picks the layer gathers are arithmetic rather than learned. Every document is
    afforded `max_entries_per_doc` picks; a query that has completed fewer entries than that pads
    the rest with `IGNORE_SLOT` (-1), as the indexer's surplus picks do.
    """

    def __init__(self, config: DeepseekV4Config):
        super().__init__(config, config.head_dim, config.compress_rates["heavily_compressed_attention"], n_series=1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_residual: torch.Tensor,
        packed: PackedContext,
        cp_group: dist.ProcessGroup | None = None,
        cp_world_size: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`q_residual` is part of the compressor contract but unused: HCA has no indexer."""
        batch = hidden_states.shape[0]
        compressed_kv = self.compress(hidden_states, packed, cp_group=cp_group, cp_world_size=cp_world_size)

        layout = packed.compression_layouts[self.compress_rate]
        # `threshold` counts entries within the query's own document, so it selects how far into
        # that document's range to read, and the document's own base turns that into an entry index.
        threshold = self.causal_threshold(packed.position_ids).unsqueeze(-1)  # (1, seq_len, 1)
        base = layout.first_entry_of_doc[packed.tok_doc_idx][None, :, None]  # (1, seq_len, 1)
        offsets = torch.arange(layout.max_entries_per_doc, device=hidden_states.device)
        picks = torch.where(offsets < threshold, base + offsets, IGNORE_SLOT)
        return compressed_kv.unsqueeze(1), picks.expand(batch, -1, -1)


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
        # Raised here rather than from the first forward, where it would surface as an ImportError
        # or a tilelang compile failure a long way from the config that caused it.
        blocker = _kernel_blocker(self.num_heads, self.head_dim)
        if blocker is not None:
            raise ValueError(f"DeepSeek V4 cannot run the fused sparse-attention kernel: {blocker}")
        assert config.attention_dropout == 0.0, "the fused sparse attention kernel implements no dropout"
        compressor_class = COMPRESSOR_CLASSES[self.layer_type]
        self.compressor = compressor_class(config) if compressor_class is not None else None

        self._cp_group: dist.ProcessGroup | None = None
        self._cp_rank: int = 0
        self._cp_world_size: int = 1

    def set_context_parallel_attributes(self, cp_group: dist.ProcessGroup, cp_rank: int, cp_world_size: int) -> None:
        self._cp_group = cp_group
        self._cp_rank = cp_rank
        self._cp_world_size = cp_world_size

    @property
    def cp_enabled(self) -> bool:
        return self._cp_world_size > 1

    def forward(self, hidden_states: torch.Tensor, packed: PackedContext) -> tuple[torch.Tensor, None]:
        """`packed` carries the document boundaries every pathway below is clipped at."""
        # Shape keys in the comments below:
        #
        # - `b`: batch
        # - `t`: token in this rank's query shard
        # - `T`: token in the whole packed row, which is `t` unless CP is on
        # - `h`: attention head
        # - `d`: head_dim
        # - `e`: compressed entry
        # - `r`: q_lora_rank
        # - `g`: o_groups
        # - `l`: o_lora_rank
        #
        # `hidden_states` is (b, t, hidden_size).

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)  # (b, t, h, d), the query view
        cos, sin = packed.position_embeddings[self.rope_layer_type]  # (1, t, qk_rope_head_dim // 2) each

        q_residual = self.q_a_norm(self.q_a_proj(hidden_states))  # (b, t, r)
        q = self.q_b_proj(q_residual).view(*hidden_shape).transpose(1, 2)  # (b, h, t, d)
        q = apply_rotary_pos_emb_interleaved(self.q_b_norm(q), cos, sin)

        kv = self.kv_norm(self.kv_proj(hidden_states))  # (b, t, d)
        kv = kv.view(*kv.shape[:2], 1, self.head_dim)  # (b, t, 1, d)
        kv = apply_rotary_pos_emb_interleaved(kv, cos, sin, unsqueeze_dim=2)
        if self.cp_enabled:
            kv = gather_for_cp(kv, self._cp_group)  # (b, T, 1, d)
        kv = kv.transpose(1, 2)  # (b, 1, T, d)

        compressed = (
            self.compressor(
                hidden_states, q_residual, packed, cp_group=self._cp_group, cp_world_size=self._cp_world_size
            )
            if self.compressor is not None
            else None
        )
        compressed_kv, top_k_indices = compressed if compressed is not None else (None, None)
        inputs = SparseAttnInputs.build(
            kv=kv,
            compressed_kv=compressed_kv,
            top_k_indices=top_k_indices,
            window_indices=packed.window_indices,
        )
        attn_output, _ = dsv4_sparse_attn(
            q.transpose(1, 2).contiguous(),
            inputs.kv_buf,
            inputs.indices,
            self.sinks,
            self.scaling,
        )  # (b, t, h, d)

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
    "PackedContext",
    "SparseAttnInputs",
]
