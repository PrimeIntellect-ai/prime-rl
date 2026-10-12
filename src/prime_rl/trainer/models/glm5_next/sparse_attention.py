import torch
import torch.nn.functional as F
from torch import nn

from prime_rl.configs.trainer import DSABackend
from prime_rl.trainer.models.kernels.fp8_indexer import fp8_indexer
from prime_rl.trainer.models.layers.norms import LayerNorm, RMSNorm, RMSNormConfig
from prime_rl.utils.cp import CPContext, gather_for_cp

try:
    from prime_rl.trainer.models.kernels.sparse_mla_fwd import sparse_mla
except ImportError:
    sparse_mla = None  # type: ignore

# TileLang consumes indices in blocks of 64, FlashMLA's sparse prefill in pairs of them.
INDEX_BLOCK = 128


def hadamard_128(x: torch.Tensor) -> torch.Tensor:
    """Orthonormal Hadamard rotation of the last (128-wide) dim, in vLLM's fp32 butterfly order."""
    shape = x.shape
    x = x.float()
    for stride in (1, 2, 4, 8, 16, 32, 64):
        pairs = x.reshape(*shape[:-1], 128 // (2 * stride), 2, stride)
        a, b = pairs.unbind(-2)
        x = torch.stack((a + b, a - b), dim=-2)
    return x.reshape(shape) * 0.08838834764831845


class KPoolIndexer(nn.Module):
    """GLM-5.3 Flash's DSA indexer over k-pools.

    Each document's keys are grouped into pools of `pool_size` consecutive tokens, starting at the
    document's first token; a pool is the softmax(gate + ape)-weighted mean of its keys. A query
    scores the complete pools that end at or before it, keeps the best `token_budget // pool_size`,
    and also attends to the tokens of its own incomplete pool. Like vLLM, queries and pooled keys
    are Hadamard-rotated before FP8 scoring, and the indexer stays frozen.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        q_lora_rank: int,
        num_heads: int,
        head_dim: int,
        token_budget: int,
        pool_size: int,
    ) -> None:
        super().__init__()
        assert head_dim == 128, "the Hadamard rotation is 128-wide"
        assert token_budget % pool_size == 0
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.token_budget = token_budget
        self.pool_size = pool_size
        self.pool_budget = token_budget // pool_size

        self.wq_b = nn.Linear(q_lora_rank, num_heads * head_dim, bias=False)
        self.wk = nn.Linear(hidden_size, head_dim, bias=False)
        self.k_norm = LayerNorm(dim=head_dim, eps=1e-6)
        self.weights_proj = nn.Linear(hidden_size, num_heads, bias=False)
        self.index_kpool_compress_ape = nn.Parameter(torch.zeros(pool_size, head_dim))
        self.index_kpool_compress_gate = nn.Parameter(torch.zeros(head_dim, hidden_size))
        self.requires_grad_(False)

        self.cp_context = CPContext()

    @property
    def output_width(self) -> int:
        width = self.token_budget + self.pool_size - 1
        return (width + INDEX_BLOCK - 1) // INDEX_BLOCK * INDEX_BLOCK

    def _pool_keys(self, key: torch.Tensor, gate: torch.Tensor, pool_start_tokens: torch.Tensor) -> torch.Tensor:
        # Same accumulation order as vLLM's kpool kernel: max, exp, running sums, then one divide.
        logits = gate[pool_start_tokens[:, None] + torch.arange(self.pool_size, device=key.device)].float()
        logits = logits + self.index_kpool_compress_ape.float()
        weights = (logits - logits.amax(dim=1, keepdim=True)).exp()
        pooled = torch.zeros_like(weights[:, 0])
        normalizer = torch.zeros_like(weights[:, 0])
        for slot in range(self.pool_size):
            pooled = pooled + key[pool_start_tokens + slot].float() * weights[:, slot]
            normalizer = normalizer + weights[:, slot]
        pooled = (pooled / normalizer).to(key.dtype)
        return hadamard_128(pooled).to(key.dtype)

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        q_latent: torch.Tensor,
        cu_seqlens: torch.LongTensor,
    ) -> torch.Tensor:
        """Returns `[1, local_tokens, 1, output_width]` key indices; the sentinel is `total_tokens`."""
        local_tokens = hidden_states.shape[1]
        device = hidden_states.device
        hidden_states = hidden_states[0]

        query = self.wq_b(q_latent[0]).view(local_tokens, self.num_heads, self.head_dim)
        query = hadamard_128(query).to(q_latent.dtype)
        head_weights = F.linear(hidden_states.float(), self.weights_proj.weight.float())
        key_and_gate = torch.cat(
            (self.k_norm(self.wk(hidden_states)), F.linear(hidden_states, self.index_kpool_compress_gate)), dim=-1
        )
        if self.cp_context.cp_enabled:
            key_and_gate = gather_for_cp(key_and_gate.unsqueeze(0), self.cp_context.cp_group)[0]
        key, gate = key_and_gate.chunk(2, dim=-1)
        total_tokens = key.shape[0]

        # Pool slots are laid out document by document; a document's partial last pool gets none.
        token_indices = torch.arange(total_tokens, dtype=cu_seqlens.dtype, device=device)
        token_documents = torch.searchsorted(cu_seqlens[1:], token_indices, right=True)
        token_positions = token_indices - cu_seqlens[:-1][token_documents]
        is_pool_start = (token_positions.remainder(self.pool_size) == 0) & (
            token_indices + self.pool_size <= cu_seqlens[1:][token_documents]
        )
        max_pools = total_tokens // self.pool_size
        pool_slots = (is_pool_start.cumsum(0) - 1).masked_fill(~is_pool_start, max_pools)
        pool_start_tokens = torch.zeros(max_pools + 1, dtype=torch.long, device=device)
        pool_start_tokens.scatter_(0, pool_slots.long(), token_indices.long())
        pool_start_tokens = pool_start_tokens[:-1]
        pooled_key = self._pool_keys(key, gate, pool_start_tokens)

        query_tokens = token_indices[self.cp_context.cp_rank * local_tokens :][:local_tokens]
        query_documents = token_documents[query_tokens]
        query_positions = token_positions[query_tokens]
        pools_per_document = (cu_seqlens[1:] - cu_seqlens[:-1]).div(self.pool_size, rounding_mode="floor")
        first_pool = F.pad(pools_per_document.cumsum(0), (1, 0))[query_documents]
        visible_pools = (query_positions + 1).div(self.pool_size, rounding_mode="floor")
        selected_pools = fp8_indexer(
            query,
            pooled_key,
            head_weights,
            first_pool.to(torch.int32),
            (first_pool + visible_pools).to(torch.int32),
            self.pool_budget,
        )

        # Selected pools fill the front of each row (fp8_indexer ranks invalid slots last); the
        # incomplete pool's tokens follow them, and the rest of the row is the sentinel.
        sentinel = total_tokens
        pool_start_tokens = F.pad(pool_start_tokens, (0, 1), value=sentinel)
        offsets = torch.arange(self.pool_size, device=device)
        selected_tokens = pool_start_tokens[selected_pools.long()][:, :, None] + offsets
        selected_tokens = selected_tokens.masked_fill((selected_pools == max_pools)[:, :, None], sentinel)
        indices = torch.full((local_tokens, self.output_width), sentinel, dtype=torch.int32, device=device)
        indices[:, : self.token_budget] = selected_tokens.flatten(1).to(torch.int32)

        tail_length = (query_positions + 1).remainder(self.pool_size)
        tail_offsets = offsets[:-1]
        tail_tokens = (query_tokens - tail_length + 1)[:, None] + tail_offsets
        tail_tokens = tail_tokens.masked_fill(tail_offsets >= tail_length[:, None], sentinel)
        tail_columns = visible_pools.clamp(max=self.pool_budget)[:, None] * self.pool_size + tail_offsets
        indices.scatter_(1, tail_columns.long(), tail_tokens.to(torch.int32))
        return indices.view(1, local_tokens, 1, self.output_width)


class SparseMlaAttention(nn.Module):
    """Multi-head latent attention without RoPE, over the indexer's selected keys.

    Runs in the absorbed form: queries are projected into the 512-dim KV latent, attention reads
    the latent directly as both key and value, and the output is projected back per head.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        qk_head_dim: int,
        v_head_dim: int,
        norm_eps: float,
        index_num_heads: int,
        index_head_dim: int,
        index_token_budget: int,
        index_pool_size: int,
        dsa_backend: DSABackend,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.kv_lora_rank = kv_lora_rank
        self.qk_head_dim = qk_head_dim
        self.v_head_dim = v_head_dim
        self.scaling = qk_head_dim**-0.5
        self.dsa_backend = dsa_backend

        self.q_a_proj = nn.Linear(hidden_size, q_lora_rank, bias=False)
        self.q_a_layernorm = RMSNorm(RMSNormConfig(hidden_size=q_lora_rank, eps=norm_eps))
        self.q_b_proj = nn.Linear(q_lora_rank, num_heads * qk_head_dim, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(hidden_size, kv_lora_rank, bias=False)
        self.kv_a_layernorm = RMSNorm(RMSNormConfig(hidden_size=kv_lora_rank, eps=norm_eps))
        self.kv_b_proj = nn.Linear(kv_lora_rank, num_heads * (qk_head_dim + v_head_dim), bias=False)
        self.o_proj = nn.Linear(num_heads * v_head_dim, hidden_size, bias=False)
        self.indexer = KPoolIndexer(
            hidden_size=hidden_size,
            q_lora_rank=q_lora_rank,
            num_heads=index_num_heads,
            head_dim=index_head_dim,
            token_budget=index_token_budget,
            pool_size=index_pool_size,
        )

        self.cp_context = CPContext()

    def forward(self, hidden_states: torch.Tensor, cu_seqlens: torch.LongTensor) -> torch.Tensor:
        batch_size, local_tokens, _ = hidden_states.shape
        q_latent = self.q_a_layernorm(self.q_a_proj(hidden_states))
        kv_latent = self.kv_a_layernorm(self.kv_a_proj_with_mqa(hidden_states))
        if self.cp_context.cp_enabled:
            kv_latent = gather_for_cp(kv_latent, self.cp_context.cp_group)

        indices = self.indexer(hidden_states, q_latent, cu_seqlens)

        kv_b_weight = self.kv_b_proj.weight.view(self.num_heads, self.qk_head_dim + self.v_head_dim, self.kv_lora_rank)
        key_weight, value_weight = kv_b_weight.split((self.qk_head_dim, self.v_head_dim), dim=1)
        query = self.q_b_proj(q_latent).view(batch_size, local_tokens, self.num_heads, self.qk_head_dim)
        query = torch.einsum("bshd,hdk->bshk", query, key_weight).contiguous()
        # The zero row past the last token is the indexer's sentinel.
        kv_latent = F.pad(kv_latent, (0, 0, 0, 1)).unsqueeze(2)

        out, _ = sparse_mla(query, kv_latent, indices, self.dsa_backend, self.scaling, self.kv_lora_rank)
        out = torch.einsum("bshk,hdk->bshd", out, value_weight)
        return self.o_proj(out.reshape(batch_size, local_tokens, self.num_heads * self.v_head_dim))


__all__ = ["KPoolIndexer", "SparseMlaAttention", "hadamard_128"]
