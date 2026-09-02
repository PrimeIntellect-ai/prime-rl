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
                          de-rotate output
                                │
                    grouped output projection
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from prime_rl.trainer.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from prime_rl.trainer.models.deepseek_v4.hyperconnections import DeepseekV4UnweightedRMSNorm
from prime_rl.trainer.models.deepseek_v4.rotary import DeepseekV4RotaryEmbedding, apply_rotary_pos_emb_interleaved
from prime_rl.trainer.models.layers.norms import RMSNorm, RMSNormConfig


class DeepseekV4GroupedLinear(nn.Linear):
    """Block-diagonal grouped linear, the first half of the output projection.

    The stacked attention output is `num_attention_heads * head_dim` wide (32768 for
    V4-Flash), so a direct projection to `hidden_size` would dominate the per-token cost.
    Instead the heads are split into `n_groups` groups, each projected independently to
    `out_features / n_groups` channels; a single follow-up linear (`o_b_proj`) mixes the
    concatenation back to `hidden_size`.

    Input is `[..., n_groups, in_features_per_group]`, output `[..., n_groups, out_features / n_groups]`.
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


def build_sliding_window_mask(
    seq_len: int,
    sliding_window: int,
    dtype: torch.dtype,
    device: torch.device,
    *,
    cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    """Additive `[1, 1, seq_len, seq_len]` mask: causal, restricted to a local window.

    A padded micro-batch folds its padding into the last document (`batch.py:717-718`
    extends `seq_lens[-1]` while restarting `position_ids`), so the padding is masked as a
    continuation of the last document. Causality already keeps it away from every real
    token, and it is loss-masked.
    """
    positions = torch.arange(seq_len, device=device)
    distance = positions[:, None] - positions[None, :]
    allowed = (distance >= 0) & (distance < sliding_window)
    document_ids = torch.searchsorted(cu_seqlens[1:].to(positions.dtype), positions, right=True)
    allowed &= document_ids[:, None] == document_ids[None, :]
    mask = torch.zeros(seq_len, seq_len, dtype=dtype, device=device)
    return mask.masked_fill_(~allowed, torch.finfo(dtype).min)[None, None]


@dataclass(frozen=True)
class CompressionLayout:
    """Per-document compressed-entry layout for one compress rate.

    An entry is one compressed KV vector: a compressor pools a window of `compress_rate`
    consecutive source tokens into a single `head_dim` vector, and the attention block reads the
    resulting series as extra keys and values beside its local sliding window. Document `d` of
    length `L_d` owns `L_d // compress_rate` entries, and its trailing `L_d % compress_rate` tokens
    are dropped. CSA's compressor also pools the previous entry's window, so its entries overlap
    and its first entry per document has no predecessor to pool; `is_first` marks those.

    Every entry has to belong to exactly one document: one that straddled a boundary would blend
    two independent documents, and one numbered from the start of the packed sequence could not be
    compared against a per-document causal threshold. This carries both, precomputed once per rate
    and shared by every layer that uses it.

    `seq_lens` has no batch dimension, so one layout describes every sequence in the batch.
    """

    src_idx: Tensor  # [n_entries, compress_rate] int64 - source token index in the packed sequence
    entry_doc: Tensor  # [n_entries] int64 - which document each entry belongs to
    entry_local: Tensor  # [n_entries] int64 - entry index within its own document
    is_first: Tensor  # [n_entries] bool - entry 0 of its document
    entry_pos: Tensor  # [n_entries] int64 - == entry_local * compress_rate, the compress-RoPE position
    doc_of_token: Tensor  # [seq_len] int64 - which document each packed token belongs to


def build_compression_layout(cu_seqlens: Tensor, compress_rate: int, total_tokens: int) -> CompressionLayout:
    """Lay out the compressed entries of a packed sequence, document by document.

    Document `d` of length `L_d` gets `L_d // compress_rate` entries; entry `j` covers the
    `compress_rate` source tokens starting at `cu_seqlens[d] + j * compress_rate`. The
    trailing `L_d % compress_rate` tokens get no entry, exactly as the unpacked case drops
    its trailing partial window; they stay visible through the local sliding window.

    A packed sequence whose every document is shorter than `compress_rate` yields zero entries,
    which is well-formed: the compressors then contribute nothing beyond their local window.
    """
    device = cu_seqlens.device
    starts = cu_seqlens[:-1].to(torch.int64)
    lengths = cu_seqlens[1:].to(torch.int64) - starts
    counts = lengths // compress_rate

    entry_doc = torch.repeat_interleave(torch.arange(counts.numel(), device=device), counts)
    first_entry_of_doc = counts.cumsum(0) - counts
    entry_local = torch.arange(int(counts.sum()), device=device) - first_entry_of_doc[entry_doc]
    entry_pos = entry_local * compress_rate
    src_idx = starts[entry_doc, None] + entry_pos[:, None] + torch.arange(compress_rate, device=device)[None, :]

    tokens = torch.arange(total_tokens, device=device)
    doc_of_token = torch.searchsorted(cu_seqlens[1:].to(tokens.dtype), tokens, right=True)

    return CompressionLayout(
        src_idx=src_idx,
        entry_doc=entry_doc,
        entry_local=entry_local,
        is_first=entry_local == 0,
        entry_pos=entry_pos,
        doc_of_token=doc_of_token,
    )


@dataclass(frozen=True)
class PackedContext:
    """Every artifact a layer derives from the packed sequence's document map, carried together.

    The sliding mask, the query positions and the compressed-entry layouts all encode the same
    boundaries, and they are only correct together. Passed as three optional arguments they can
    contradict each other: a mask built without `cu_seqlens` spans documents while a layout does
    not, and a sequence-global `position_ids` feeds `causal_threshold` a count that a per-document
    `entry_local` cannot be compared against. Neither mistake is reachable through `build`, which
    derives all three from one `cu_seqlens`.

    Built once per model forward and shared by every layer, since none of it depends on depth.
    """

    attention_mask: Tensor
    position_ids: Tensor
    compression_layouts: dict[int, CompressionLayout]

    def __post_init__(self) -> None:
        total_tokens = self.position_ids.shape[-1]
        if self.attention_mask.shape[-2] != total_tokens:
            raise ValueError(
                f"attention_mask covers {self.attention_mask.shape[-2]} query rows but "
                f"position_ids has {total_tokens} tokens"
            )
        for rate, layout in self.compression_layouts.items():
            if layout.doc_of_token.shape[0] != total_tokens:
                raise ValueError(
                    f"the rate-{rate} layout maps {layout.doc_of_token.shape[0]} tokens to "
                    f"documents but position_ids has {total_tokens}"
                )

    @classmethod
    def build(
        cls,
        *,
        cu_seqlens: Tensor,
        position_ids: Tensor,
        total_tokens: int,
        compress_rates: set[int],
        sliding_window: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> "PackedContext":
        """Derive all three fields from one `cu_seqlens`, which is what keeps them consistent."""
        return cls(
            attention_mask=build_sliding_window_mask(
                total_tokens, sliding_window, dtype, device, cu_seqlens=cu_seqlens
            ),
            position_ids=position_ids,
            compression_layouts={
                rate: build_compression_layout(cu_seqlens, rate, total_tokens) for rate in compress_rates
            },
        )


def get_token_entry_causal_mask(layout: CompressionLayout, threshold: Tensor) -> Tensor:
    """`[batch, seq_len, n_entries]` bool: which compressed entries each query token may read.

    Element `[b, t, e]` is true when query token `t` may read compressed entry `e`. Both of
    these have to hold:

    - `e` belongs to `t`'s own document, so no query reads another document's history;
    - `e` closed before `t` arrived, i.e. its index within that document is below
      `threshold[b, t]`, the count of entries the query's position has completed.

    `threshold` is `[batch, seq_len]` and counts per document, so it is compared against
    `layout.entry_local` and not against the sequence-global entry number; those two
    coordinate systems disagree for every document after the first.
    """
    same_document = layout.doc_of_token[None, :, None] == layout.entry_doc[None, None, :]
    return same_document & (threshold.unsqueeze(-1) > layout.entry_local[None, None, :])


class DeepseekV4Compressor(nn.Module):
    """Softmax-gated pooling of the token stream into one entry per `compress_rate` tokens.

    `kv_proj` and `gate_proj` emit `2 * head_dim` features per token, read as two
    independent series: `Ca = [..., :head_dim]` and `Cb = [..., head_dim:]`. Compressed
    entry `w` pools window `w - 1`'s `Ca` slice together with window `w`'s `Cb` slice, so
    the pooling window is `2 * compress_rate` wide with stride `compress_rate` and
    consecutive entries overlap. The first entry of a document has no predecessor, so its
    `Ca` half is gated with `-inf` and contributes nothing.

    Every entry is rotated with the `compress` RoPE at the absolute position of its own
    window's first source token, which is what makes it comparable with the attention
    block's locally rotated KV stream once the two are concatenated.

    Both halves of Compressed Sparse Attention are built on this: the CSA compressor runs
    it at `config.head_dim`, the Lightning Indexer runs it over the same windows at the
    much narrower `config.index_head_dim`.
    """

    rope_layer_type = "compress"

    def __init__(self, config: DeepseekV4Config, head_dim: int):
        super().__init__()
        self.compress_rate = config.compress_rates["compressed_sparse_attention"]
        self.head_dim = head_dim
        self.kv_proj = nn.Linear(config.hidden_size, 2 * head_dim, bias=False)
        self.gate_proj = nn.Linear(config.hidden_size, 2 * head_dim, bias=False)
        self.position_bias = nn.Parameter(torch.zeros(self.compress_rate, 2 * head_dim))
        self.kv_norm = RMSNorm(RMSNormConfig(hidden_size=head_dim, eps=config.rms_norm_eps))
        self.rotary_emb = DeepseekV4RotaryEmbedding(config)

    def compress(self, hidden_states: torch.Tensor, layout: CompressionLayout) -> torch.Tensor:
        """Compress `[batch, seq_len, hidden_size]` to `[batch, n_entries, head_dim]`.

        `layout` decides which source tokens each entry pools.
        """
        batch = hidden_states.shape[0]
        n_entries = layout.src_idx.shape[0]

        kv = self.kv_proj(hidden_states)[:, layout.src_idx]
        gate = self.gate_proj(hidden_states)[:, layout.src_idx] + self.position_bias

        # Shift the `Ca` series one entry later so entry `w` sees entry `w - 1`'s. The first
        # entry of every document has no predecessor, and the entry sitting before it in the
        # packed sequence belongs to another document, so both halves are cleared: the gate to
        # `-inf` and the values to zero. Zeroing is not redundant with the gate, because a
        # zero softmax weight against a non-finite value would still yield NaN.
        previous = (torch.arange(n_entries, device=hidden_states.device) - 1).clamp(min=0)
        first_entry = layout.is_first[None, :, None, None]
        previous_kv = kv[:, previous, :, : self.head_dim].masked_fill(first_entry, 0.0)
        previous_gate = gate[:, previous, :, : self.head_dim].masked_fill(first_entry, float("-inf"))
        pooled_kv = torch.cat([previous_kv, kv[..., self.head_dim :]], dim=2)
        pooled_gate = torch.cat([previous_gate, gate[..., self.head_dim :]], dim=2)

        # fp32 softmax: in bf16 the gate logits of a wide window collapse onto each other.
        weights = pooled_gate.softmax(dim=2, dtype=torch.float32).to(pooled_kv.dtype)
        compressed = self.kv_norm((pooled_kv * weights).sum(dim=2))

        cos, sin = self.rotary_emb(compressed, layout.entry_pos.unsqueeze(0).expand(batch, -1), self.rope_layer_type)
        return apply_rotary_pos_emb_interleaved(compressed.unsqueeze(1), cos, sin).squeeze(1)

    def causal_threshold(self, position_ids: torch.Tensor) -> torch.Tensor:
        """Number of compressed entries that query `t` may read, shaped `[batch, seq_len]`.

        Entry `w` pools source tokens up to index `(w + 1) * compress_rate - 1`, so it only
        becomes readable once the query has reached that token.
        """
        return (position_ids + 1) // self.compress_rate

    def init_weights(self, init_std: float) -> None:
        # `init_std` is unused: the projections are initialized by the caller and the
        # position bias starts at zero, i.e. a uniform gate over the pooling window.
        nn.init.zeros_(self.position_bias)


class DeepseekV4IndexerScorer(nn.Module):
    """Lightning-Indexer score `sum_h w_th * ReLU(q_th . k_s)` of query `t` against entry `s`.

    The per-head weights `w_th` are read straight off the hidden state rather than from a
    query/key interaction, which is what makes the whole scorer one matmul deep. It runs
    in fp32: the scores only ever feed a top-k, so the extra width is cheap and it keeps
    near-ties from being decided by bf16 rounding.
    """

    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.softmax_scale = config.index_head_dim**-0.5
        self.weights_scaling = config.index_n_heads**-0.5
        self.weights_proj = nn.Linear(config.hidden_size, config.index_n_heads, bias=False)

    def forward(self, q: torch.Tensor, compressed_kv: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        """Score `q` `[batch, seq, heads, dim]` against `compressed_kv` `[batch, entries, dim]`."""
        scores = torch.matmul(q.float(), compressed_kv.transpose(-1, -2).float().unsqueeze(1))
        scores = F.relu(scores) * self.softmax_scale
        weights = self.weights_proj(hidden_states).float() * self.weights_scaling
        return (scores * weights.unsqueeze(-1)).sum(dim=2)


class DeepseekV4Indexer(DeepseekV4Compressor):
    """Lightning Indexer: picks the `index_topk` compressed entries each query may read.

    It repeats the CSA compressor's compression at the much narrower `index_head_dim`,
    scores the queries against those cheap compressed keys, and returns indices into the
    *outer* compressor's entries. Both compressions share `compress_rate` and the
    `compress` RoPE base, so entry `w` here indexes the same source window as entry `w`
    there, and the scores stay translation invariant in the query-key distance.

    Each query gets `min(index_topk, entries)` picks. A query early in the sequence has
    fewer readable entries than that, and its surplus picks come back as `-1`.
    """

    def __init__(self, config: DeepseekV4Config):
        super().__init__(config, config.index_head_dim)
        self.num_heads = config.index_n_heads
        self.index_topk = config.index_topk
        self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.head_dim, bias=False)
        self.scorer = DeepseekV4IndexerScorer(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_residual: torch.Tensor,
        position_ids: torch.Tensor,
        layout: CompressionLayout,
    ) -> torch.Tensor:
        batch, seq_len, _ = hidden_states.shape
        compressed_kv = self.compress(hidden_states, layout)
        compressed_len = compressed_kv.shape[1]
        top_k = min(self.index_topk, compressed_len)

        cos, sin = self.rotary_emb(hidden_states, position_ids, self.rope_layer_type)
        q = self.q_b_proj(q_residual).view(batch, seq_len, -1, self.head_dim).transpose(1, 2)
        q = apply_rotary_pos_emb_interleaved(q, cos, sin).transpose(1, 2)

        scores = self.scorer(q, compressed_kv, hidden_states)
        if compressed_len == 0:
            return scores.topk(top_k, dim=-1).indices

        readable = get_token_entry_causal_mask(layout, self.causal_threshold(position_ids)).expand_as(scores)
        scores = scores.masked_fill(~readable, float("-inf"))
        top_k_indices = scores.topk(top_k, dim=-1).indices
        # An early query has fewer than `top_k` readable entries, so top-k still hands back
        # masked-out ones. Mark those `-1` rather than letting them leak into attention.
        return torch.where(readable.gather(-1, top_k_indices), top_k_indices, torch.full_like(top_k_indices, -1))


class DeepseekV4CSACompressor(DeepseekV4Compressor):
    """Compressed Sparse Attention compressor: the sparse long-range half of a CSA layer.

    Returns the compressed history as extra KV entries for the attention block to
    concatenate onto its local sliding window, plus the additive `block_bias` that decides
    which query reads which of them: `0` for the entries its indexer selected, `-inf`
    everywhere else.
    """

    def __init__(self, config: DeepseekV4Config):
        super().__init__(config, config.head_dim)
        self.indexer = DeepseekV4Indexer(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_residual: torch.Tensor,
        position_ids: torch.Tensor,
        layout: CompressionLayout,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len, _ = hidden_states.shape
        compressed_kv = self.compress(hidden_states, layout).unsqueeze(1)
        compressed_len = compressed_kv.shape[2]

        # The indexer shares this layout: it compresses the same source windows at a narrower
        # head dim, so its entry `n` and this compressor's entry `n` are the same window.
        top_k_indices = self.indexer(hidden_states, q_residual, position_ids, layout)
        # The `-1` sentinels are scattered into one throwaway column that is sliced back off.
        safe_indices = torch.where(top_k_indices >= 0, top_k_indices, torch.full_like(top_k_indices, compressed_len))
        block_bias = compressed_kv.new_full((batch, 1, seq_len, compressed_len + 1), float("-inf"))
        block_bias.scatter_(-1, safe_indices.unsqueeze(1), 0.0)
        return compressed_kv, block_bias[..., :compressed_len]

    def init_weights(self, init_std: float) -> None:
        super().init_weights(init_std)
        self.indexer.init_weights(init_std)


class DeepseekV4HCACompressor(nn.Module):
    """Heavily Compressed Attention compressor: the dense long-range half of an HCA layer.

    It pools every non-overlapping window of `compress_rate` (128) tokens into a single
    entry, `C_w = sum_j softmax(gate_j + position_bias_j) * kv_j` over the window's tokens
    `j`, then rotates the entry with the `compress` RoPE at its window's first source
    position, which is what makes it comparable with the attention block's locally rotated
    KV stream.

    Both differences from `DeepseekV4CSACompressor` pull in the same direction. The windows
    do not overlap, so `kv_proj` / `gate_proj` / `position_bias` stay `head_dim` wide (CSA
    doubles them to carry two series) and no entry depends on its predecessor. And there is
    no Lightning Indexer, so every query reads every entry its position has made causally
    readable: the returned `block_bias` carries that threshold and nothing else.
    """

    rope_layer_type = "compress"

    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.compress_rate = config.compress_rates["heavily_compressed_attention"]
        self.head_dim = config.head_dim
        self.kv_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.gate_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.position_bias = nn.Parameter(torch.zeros(self.compress_rate, self.head_dim))
        self.kv_norm = RMSNorm(RMSNormConfig(hidden_size=self.head_dim, eps=config.rms_norm_eps))
        self.rotary_emb = DeepseekV4RotaryEmbedding(config)

    def compress(self, hidden_states: torch.Tensor, layout: CompressionLayout) -> torch.Tensor:
        """Compress `[batch, seq_len, hidden_size]` to `[batch, n_entries, head_dim]`.

        `layout` decides which source tokens each entry pools.
        """
        batch = hidden_states.shape[0]

        kv = self.kv_proj(hidden_states)[:, layout.src_idx]
        gate = self.gate_proj(hidden_states)[:, layout.src_idx] + self.position_bias

        # fp32 softmax: in bf16 the gate logits of a wide window collapse onto each other.
        weights = gate.softmax(dim=2, dtype=torch.float32).to(kv.dtype)
        compressed = self.kv_norm((kv * weights).sum(dim=2))

        cos, sin = self.rotary_emb(compressed, layout.entry_pos.unsqueeze(0).expand(batch, -1), self.rope_layer_type)
        return apply_rotary_pos_emb_interleaved(compressed.unsqueeze(1), cos, sin).squeeze(1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_residual: torch.Tensor,
        position_ids: torch.Tensor,
        layout: CompressionLayout,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`q_residual` is part of the compressor contract but unused: HCA has no indexer."""
        batch, seq_len, _ = hidden_states.shape
        compressed_kv = self.compress(hidden_states, layout).unsqueeze(1)
        compressed_len = compressed_kv.shape[2]

        # Entry `j` of a document pools its source tokens up to local index
        # `(j + 1) * compress_rate - 1`, so it only becomes readable once the query has
        # reached that token of its own document.
        threshold = (position_ids + 1) // self.compress_rate
        readable = get_token_entry_causal_mask(layout, threshold).unsqueeze(1)
        block_bias = compressed_kv.new_zeros((batch, 1, seq_len, compressed_len))
        return compressed_kv, block_bias.masked_fill_(~readable, float("-inf"))

    def init_weights(self, init_std: float) -> None:
        # `init_std` is unused: the projections are initialized by the caller and the
        # position bias starts at zero, i.e. a uniform gate over the pooling window.
        nn.init.zeros_(self.position_bias)


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
    which is how a layer sees past the window: CSA reads a sparse top-k of finely
    compressed entries, HCA reads all of its heavily compressed ones.
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: dict[str, tuple[torch.Tensor, torch.Tensor]],
        packed: PackedContext,
    ) -> tuple[torch.Tensor, None]:
        """`packed` carries the document boundaries every pathway below is clipped at."""
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        cos, sin = position_embeddings[self.rope_layer_type]

        q_residual = self.q_a_norm(self.q_a_proj(hidden_states))
        q = self.q_b_proj(q_residual).view(*hidden_shape).transpose(1, 2)
        q = apply_rotary_pos_emb_interleaved(self.q_b_norm(q), cos, sin)

        kv = self.kv_norm(self.kv_proj(hidden_states)).view(*hidden_shape).transpose(1, 2)
        kv = apply_rotary_pos_emb_interleaved(kv, cos, sin)

        attention_mask = packed.attention_mask

        if self.compressor is not None:
            # A missing rate is a wiring bug: the context was built for a different model.
            layout = packed.compression_layouts[self.compressor.compress_rate]
            compressed_kv, block_bias = self.compressor(hidden_states, q_residual, packed.position_ids, layout)
            kv = torch.cat([kv, compressed_kv], dim=2)
            # The compressed entries live outside the local window, so the sliding mask says
            # nothing about them; `block_bias` carries their per-query causality and the
            # indexer's selection. Zero-padding instead would let every query read every one.
            attention_mask = torch.cat(
                [attention_mask.expand(*block_bias.shape[:-1], -1), block_bias.to(attention_mask.dtype)], dim=-1
            )

        attn_output = eager_attention_with_sinks(
            q,
            kv,
            kv,
            self.sinks,
            attention_mask,
            scaling=self.scaling,
            dropout=self.attention_dropout,
            training=self.training,
        )

        # The value stream is the key stream, so it arrived rotated. Rotating the output
        # by the conjugate angle at the query position cancels that out.
        attn_output = apply_rotary_pos_emb_interleaved(attn_output, cos, -sin, unsqueeze_dim=2)

        grouped = self.o_a_proj(attn_output.reshape(*input_shape, self.config.o_groups, -1)).flatten(2)
        return self.o_b_proj(grouped), None

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
    "build_compression_layout",
    "build_sliding_window_mask",
    "eager_attention_with_sinks",
    "get_token_entry_causal_mask",
]
