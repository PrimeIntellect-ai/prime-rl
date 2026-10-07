"""DeepSeek-V4.1 attention.

The per-layer core is V4's: a single shared `head_dim` latent per token serves as key and value
for every head, partial interleaved RoPE on its tail (undone on the output), a per-head sink, and
a grouped low-rank output projection. Every layer reads a 128-token sliding window, and a
compressed layer (`compress_ratio` 1 or 2) also reads `index_topk` compressed entries reaching
further back. Unlike V4, the query gets no per-head RMSNorm.

What is new is that the long-range machinery is shared across layers rather than owned by each:

    KV source layer      compresses the token stream into entries (ratio 1: one per token, ratio 2:
                         a softmax-gated pool of two), and derives the index keys from them
    later layers         read the most recent source's entries
    index source layer   runs a Lightning Indexer (own query and head weights) over the shared
                         index keys and publishes its picks
    later layers         reuse the most recent index source's picks
    candidate source     additionally ranks blocks of entries; later index sources only score
                         inside each query's top blocks

Those hand-offs are carried as explicit tensors in `SharedAttnState`, passed from layer to layer
by the decoder stack. They must not live on a Python object the layers mutate: an activation
checkpointed layer recomputes its forward during backward, long after later layers would have
replaced such an object's contents.

Packed sequences and context parallelism follow V4 (see `deepseek_v4/attention.py`): entries are
laid out per document by `CompressionLayout`, every key-side tensor is global, and only the query
side is this rank's shard.
"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.distributed._functional_collectives import wait_tensor

from prime_rl.trainer.models.deepseek_v4.attention import (
    CompressionLayout,
    DeepseekV4GroupedLinear,
    SparseAttnInputs,
)
from prime_rl.trainer.models.deepseek_v4.rotary import DeepseekV4RotaryEmbedding
from prime_rl.trainer.models.deepseek_v41.configuration_deepseek_v41 import DeepseekV41TextConfig
from prime_rl.trainer.models.kernels.deepseek_v4 import IGNORE_SLOT
from prime_rl.trainer.models.kernels.deepseek_v4.dsv4_rope import dsv4_rope, dsv4_rope_inplace
from prime_rl.trainer.models.kernels.dsv41_indexer import dsv41_index_topk
from prime_rl.trainer.models.kernels.dsv41_sparse_attn import dsv41_sparse_attn, flashmla_sparse_attn_available
from prime_rl.trainer.models.layers.norms import RMSNorm, RMSNormConfig
from prime_rl.utils.cp import CPContext, gather_for_cp
from prime_rl.utils.sequence import get_cu_seqlens_from_seq_lens

try:
    from prime_rl.trainer.models.kernels.deepseek_v4.dsv4_sparse_attn import dsv4_sparse_attn, sparse_attn_shape_error
except ImportError:
    dsv4_sparse_attn = None  # type: ignore
    sparse_attn_shape_error = None  # type: ignore


def _index_topk_op(num_heads: int, head_dim: int, block_size: int):
    """prime-kernels' fused indexer top-k when it is built for this GPU and shape, else prime-rl's chunked one."""
    import prime_kernels

    if "dsa_indexer_topk" in prime_kernels.KERNELS and prime_kernels.is_available("dsa_indexer_topk"):
        kernel = prime_kernels.load("dsa_indexer_topk")
        if kernel.unsupported_shape_reason(num_heads, head_dim, block_size) is None:
            return kernel.dsv41_index_topk
    return dsv41_index_topk


@dataclass(frozen=True)
class PackedContext:
    """Document-aware bookkeeping for one packed row, derived once per forward from `seq_lens`.

    Query-side fields cover this rank's `n_queries` tokens; `compression_layouts` covers the whole
    row. Token indices count from the start of the whole row.
    """

    position_ids: Tensor  # (1, n_queries) int64 - token position within its own document
    tok_doc_idx: Tensor  # (n_queries,) int64 - which document each query token belongs to
    tok_doc_start: Tensor  # (n_queries,) int64 - row index of each query token's document start
    window_indices: Tensor  # (n_queries, sliding_window) int32, IGNORE_SLOT if unused
    compression_layouts: dict[int, CompressionLayout]  # keyed by compress ratio
    use_candidates: bool  # whether some document is long enough for candidate blocks to bind

    @classmethod
    def build(
        cls, *, config: DeepseekV41TextConfig, seq_lens: Tensor, device: torch.device, cp_rank: int, cp_world_size: int
    ) -> "PackedContext":
        total_tokens = int(seq_lens.sum())
        assert total_tokens % cp_world_size == 0, f"{total_tokens} tokens do not split across {cp_world_size} CP ranks"
        assert total_tokens <= config.max_position_embeddings, (
            f"{total_tokens} tokens exceed the {config.max_position_embeddings} positions of the RoPE cache"
        )
        n_queries = total_tokens // cp_world_size
        q_start = cp_rank * n_queries
        cu_seqlens, _ = get_cu_seqlens_from_seq_lens(seq_lens.to(device=device))

        tok_idx = torch.arange(q_start, q_start + n_queries, device=device)
        tok_doc_idx = torch.searchsorted(cu_seqlens[1:].to(tok_idx.dtype), tok_idx, right=True)
        tok_doc_start = cu_seqlens[tok_doc_idx].to(torch.int64)
        position_ids = (tok_idx - tok_doc_start)[None]
        window_base = torch.maximum(tok_doc_start, tok_idx - config.sliding_window + 1)
        slots = window_base[:, None] + torch.arange(config.sliding_window, device=device)[None, :]
        window_indices = torch.where(slots <= tok_idx[:, None], slots, IGNORE_SLOT).to(torch.int32)

        layouts = {
            rate: CompressionLayout.build(cu_seqlens=cu_seqlens, compress_rate=rate) for rate in config.compress_rates
        }
        use_candidates = False
        if 0 <= config.candidate_source_layer_id < config.num_hidden_layers:
            rate = config.compress_ratios[config.candidate_source_layer_id]
            use_candidates = (
                layouts[rate].max_entries_per_doc > config.candidate_topk_blocks * config.candidate_block_size
            )
        return cls(
            position_ids=position_ids,
            tok_doc_idx=tok_doc_idx,
            tok_doc_start=tok_doc_start,
            window_indices=window_indices,
            compression_layouts=layouts,
            use_candidates=use_candidates,
        )

    def check_position_ids(self, position_ids: Tensor) -> None:
        """Raise unless `position_ids` restarts at 0 at every document start of `seq_lens`."""
        disagrees = (self.position_ids == 0) & (position_ids != 0)
        if disagrees.any():
            token = int(disagrees.any(dim=0).nonzero()[0])
            raise ValueError(
                f"position_ids must restart at 0 at every document boundary of seq_lens: token "
                f"{token} starts a document but carries {position_ids[:, token].tolist()}."
            )


@dataclass
class SharedAttnState:
    """What attention layers hand down the stack. Each field is replaced by its source layer."""

    compressed_kv: Tensor | None = None  # (1, 1, n_entries, head_dim) rotated entries
    index_k: Tensor | None = None  # (n_entries, index_head_dim) rotated index keys
    top_k_indices: Tensor | None = None  # (1, n_queries, index_topk) entry index, IGNORE_SLOT if none
    candidates: Tensor | None = None  # (n_queries, candidate_topk_blocks) int32 doc-local block, -1 if unused

    def as_tuple(self) -> tuple[Tensor | None, ...]:
        return (self.compressed_kv, self.index_k, self.top_k_indices, self.candidates)


class DeepseekV41Compressor(nn.Module):
    """A KV source layer's compressor: entries over the whole row, before and after RoPE.

    Ratio 1 projects every token to its own entry. Ratio 2 pools disjoint pairs with a per-channel
    softmax gate, in fp32 as the reference does. Either way the entry is RMSNormed; the unrotated
    form feeds the index keys and the rotated one (at the entry's first source position, with the
    `compress` RoPE) is what attention reads.
    """

    def __init__(self, config: DeepseekV41TextConfig, compress_ratio: int, rotary_emb: DeepseekV4RotaryEmbedding):
        super().__init__()
        self.compress_ratio = compress_ratio
        self.head_dim = config.head_dim
        self.kv_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.gate_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False) if compress_ratio > 1 else None
        self.kv_norm = RMSNorm(RMSNormConfig(hidden_size=self.head_dim, eps=config.rms_norm_eps))
        self.rotary_emb = rotary_emb

    def forward(self, hidden_states: Tensor, packed: PackedContext, cp_context: CPContext) -> tuple[Tensor, Tensor]:
        """`(b, t, hidden)` -> unrotated and rotated `(n_entries, head_dim)` entries."""
        layout = packed.compression_layouts[self.compress_ratio]
        if self.gate_proj is None:
            kv = self.kv_norm(self.kv_proj(hidden_states))
            if cp_context.cp_enabled:
                kv = gather_for_cp(kv, cp_context.cp_group)
            # One entry per token, so the entries are the token stream itself.
            latent = kv[0]
        else:
            x = hidden_states.float()
            proj = torch.cat([x @ self.kv_proj.weight.float().t(), x @ self.gate_proj.weight.float().t()], dim=-1)
            if cp_context.cp_enabled:
                proj = gather_for_cp(proj, cp_context.cp_group)
            kv, gate = proj[0].split(self.head_dim, dim=-1)
            kv, gate = kv[layout.entry_tok_idx], gate[layout.entry_tok_idx]  # (n_entries, ratio, head_dim)
            latent = self.kv_norm((kv * gate.softmax(dim=1)).sum(dim=1).to(hidden_states.dtype))

        entry_pos = layout.entry_local_idx * self.compress_ratio
        rotated = dsv4_rope(latent.unsqueeze(1), self.rotary_emb.cos_sin_cache("compress"), entry_pos).squeeze(1)
        return latent, rotated


class DeepseekV41Indexer(nn.Module):
    """Lightning Indexer of an index source layer: picks `index_topk` entries per query.

    Only an indexer in a KV source layer owns the index-key projection (`wk`, `k_norm`); the rest
    score against the keys that layer published. It returns non-differentiable integer indices and
    is frozen, as for every other prime-rl indexer.
    """

    def __init__(
        self,
        config: DeepseekV41TextConfig,
        layer_idx: int,
        compress_ratio: int,
        rotary_emb: DeepseekV4RotaryEmbedding,
    ):
        super().__init__()
        self.compress_ratio = compress_ratio
        self.num_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.index_topk = config.index_topk
        self.owns_k = layer_idx in config.kv_source_layer_ids
        self.is_candidate_source = layer_idx == config.candidate_source_layer_id
        self.uses_candidates = 0 <= config.candidate_source_layer_id < layer_idx
        self.candidate_block_size = config.candidate_block_size
        self.candidate_topk_blocks = config.candidate_topk_blocks
        self.index_topk_op = _index_topk_op(self.num_heads, self.head_dim, self.candidate_block_size)
        self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.head_dim, bias=False)
        self.weights_proj = nn.Linear(config.hidden_size, self.num_heads, bias=False)
        if self.owns_k:
            self.k_proj = nn.Linear(config.head_dim, self.head_dim, bias=False)
            self.k_norm = RMSNorm(RMSNormConfig(hidden_size=self.head_dim, eps=config.rms_norm_eps))
        self.rotary_emb = rotary_emb

    @torch.no_grad()
    def forward(
        self,
        hidden_states: Tensor,
        q_residual: Tensor,
        latent: Tensor | None,
        packed: PackedContext,
        state: SharedAttnState,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        """Returns `(top_k_indices, index_k, candidates)` for the layers below to share."""
        batch, seq_len, _ = hidden_states.shape
        assert batch == 1, f"the indexer needs a packed batch of size 1, got {batch}"
        cos_sin_cache = self.rotary_emb.cos_sin_cache("compress")
        layout = packed.compression_layouts[self.compress_ratio]
        if self.owns_k:
            index_k = self.k_norm(self.k_proj(latent))
            entry_pos = layout.entry_local_idx * self.compress_ratio
            index_k = dsv4_rope(index_k.unsqueeze(1), cos_sin_cache, entry_pos).squeeze(1)
        else:
            index_k = state.index_k

        q = self.q_b_proj(q_residual).view(batch, seq_len, self.num_heads, self.head_dim)
        q = dsv4_rope(q, cos_sin_cache, packed.position_ids)
        w = self.weights_proj(hidden_states)

        entry_start = layout.first_entry_of_doc[packed.tok_doc_idx]
        entry_stop = entry_start + (packed.position_ids[0] + 1) // self.compress_ratio
        emit = self.is_candidate_source and packed.use_candidates
        candidates_in = state.candidates if self.uses_candidates and packed.use_candidates else None
        top_k, candidates = self.index_topk_op(
            q[0],
            index_k,
            w[0],
            entry_start.int(),
            entry_stop.int(),
            self.index_topk,
            candidates_in,
            emit,
            layout.max_entries_per_doc,
            self.candidate_block_size,
            self.candidate_topk_blocks,
        )
        return top_k.unsqueeze(0), index_k, candidates if emit else state.candidates


class DeepseekV41Attention(nn.Module):
    def __init__(self, config: DeepseekV41TextConfig, layer_idx: int, rotary_emb: DeepseekV4RotaryEmbedding):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.compress_ratio = config.compress_ratios[layer_idx]
        # Sliding-window layers rotate with the plain `main` RoPE, compressed ones with `compress`.
        self.rope_layer_type = "compress" if self.compress_ratio else "main"
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.scaling = self.head_dim**-0.5

        self.q_a_proj = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_a_norm = RMSNorm(RMSNormConfig(hidden_size=config.q_lora_rank, eps=config.rms_norm_eps))
        self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.head_dim, bias=False)
        self.kv_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.kv_norm = RMSNorm(RMSNormConfig(hidden_size=self.head_dim, eps=config.rms_norm_eps))
        self.o_a_proj = DeepseekV4GroupedLinear(
            self.num_heads * self.head_dim // config.o_groups, config.o_groups * config.o_lora_rank, config.o_groups
        )
        self.o_b_proj = nn.Linear(config.o_groups * config.o_lora_rank, config.hidden_size, bias=False)
        self.sinks = nn.Parameter(torch.zeros(self.num_heads))
        self.rotary_emb = rotary_emb

        if dsv4_sparse_attn is None:
            raise ValueError("DeepSeek V4.1 needs the tilelang sparse-attention kernel; install the `gpu` extra")
        blocker = sparse_attn_shape_error(self.num_heads, 1, self.head_dim)
        if blocker is not None:
            raise ValueError(f"DeepSeek V4.1 cannot run the fused sparse-attention kernel: {blocker}")
        assert config.attention_dropout == 0.0, "the fused sparse attention kernel implements no dropout"
        # FlashMLA forward + cuDNN backward on Hopper (about twice as fast); TileLang elsewhere.
        self.sparse_attn = (
            dsv41_sparse_attn if flashmla_sparse_attn_available(self.num_heads, self.head_dim) else dsv4_sparse_attn
        )

        self.is_kv_source = layer_idx in config.kv_source_layer_ids
        self.is_index_source = layer_idx in config.index_source_layer_ids
        self.compressor = DeepseekV41Compressor(config, self.compress_ratio, rotary_emb) if self.is_kv_source else None
        self.indexer = (
            DeepseekV41Indexer(config, layer_idx, self.compress_ratio, rotary_emb) if self.is_index_source else None
        )
        self.cp_context = CPContext()

    def forward(
        self, hidden_states: Tensor, packed: PackedContext, state: SharedAttnState
    ) -> tuple[Tensor, SharedAttnState]:
        """`hidden_states` is `(b, t, hidden)` for this rank's queries; returns the output and the
        shared state as this layer leaves it."""
        input_shape = hidden_states.shape[:-1]
        cos_sin_cache = self.rotary_emb.cos_sin_cache(self.rope_layer_type)
        cp = self.cp_context

        kv = self.kv_norm(self.kv_proj(hidden_states)).unsqueeze(2)  # (b, t, 1, d)
        kv = dsv4_rope(kv, cos_sin_cache, packed.position_ids)
        if cp.cp_enabled:
            # Launch on NCCL's stream; the query and compressor work does not read KV.
            kv = torch.ops._c10d_functional.all_gather_into_tensor(
                kv.movedim(1, 0).contiguous(), cp.cp_world_size, cp.cp_group.group_name
            )

        q_residual = self.q_a_norm(self.q_a_proj(hidden_states))
        q = self.q_b_proj(q_residual).view(*input_shape, self.num_heads, self.head_dim)
        # The projection's output is read by nothing else, and the attention kernel's query gradient is fresh.
        q = dsv4_rope_inplace(q, cos_sin_cache, packed.position_ids)

        if self.compress_ratio:
            latent = None
            if self.compressor is not None:
                latent, entries = self.compressor(hidden_states, packed, cp)
                state = SharedAttnState(entries[None, None], state.index_k, state.top_k_indices, state.candidates)
            if self.indexer is not None:
                top_k, index_k, candidates = self.indexer(hidden_states, q_residual, latent, packed, state)
                state = SharedAttnState(state.compressed_kv, index_k, top_k, candidates)

        if cp.cp_enabled:
            kv = wait_tensor(kv).movedim(0, 1).contiguous()  # (b, T, 1, d)
        inputs = SparseAttnInputs.build(
            kv=kv.transpose(1, 2),
            compressed_kv=state.compressed_kv if self.compress_ratio else None,
            top_k_indices=state.top_k_indices if self.compress_ratio else None,
            window_indices=packed.window_indices,
        )
        attn_output, _ = self.sparse_attn(q, inputs.kv_buf, inputs.indices, self.sinks.float(), self.scaling)
        # Values are the rotated keys; the conjugate rotation at the query position cancels that.
        # The attention backward reads its output, so only the gradient (fresh from `o_a_proj`) rotates in place.
        attn_output = dsv4_rope_inplace(
            attn_output, cos_sin_cache, packed.position_ids, inverse=True, in_place_forward=False
        )
        grouped = self.o_a_proj(attn_output.reshape(*input_shape, self.config.o_groups, -1)).flatten(2)
        return self.o_b_proj(grouped), state

    def init_weights(self, init_std: float) -> None:
        nn.init.zeros_(self.sinks)


__all__ = [
    "DeepseekV41Attention",
    "DeepseekV41Compressor",
    "DeepseekV41Indexer",
    "PackedContext",
    "SharedAttnState",
]
