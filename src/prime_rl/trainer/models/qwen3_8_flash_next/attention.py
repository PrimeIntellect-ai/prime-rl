import prime_kernels
import torch
from torch import nn
from torch.distributed import ProcessGroup

from prime_rl.trainer.models.qwen3_8_flash_next.indexer import SparseAttentionIndexer
from prime_rl.trainer.models.qwen3_8_flash_next.norm import RMSNorm
from prime_rl.trainer.models.qwen3_8_flash_next.rotary_embedding import apply_rotary_embedding
from prime_rl.utils.cp import gather_for_cp


class IndexedGatedAttention(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        norm_eps: float,
        indexer_num_heads: int,
        indexer_head_dim: int,
        indexer_token_budget: int,
        indexer_compression_ratio: int,
    ) -> None:
        super().__init__()
        self.num_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim * 2, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=False)
        self.q_norm = RMSNorm(head_dim, norm_eps)
        self.k_norm = RMSNorm(head_dim, norm_eps)
        self.indexer = SparseAttentionIndexer(
            hidden_size=hidden_size,
            num_query_heads=indexer_num_heads,
            head_dim=indexer_head_dim,
            token_budget=indexer_token_budget,
            compression_ratio=indexer_compression_ratio,
            norm_eps=norm_eps,
        )

        self.context_parallel_group: ProcessGroup | None = None
        # Resolve once before torch.compile traces the layer; the loader caches the module.
        prime_kernels.load("indexed_attention")

    def set_context_parallel_attributes(self, process_group: ProcessGroup, rank: int, world_size: int) -> None:
        self.context_parallel_group = process_group
        self.indexer.set_context_parallel_attributes(process_group, rank, world_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.LongTensor,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape
        cos, sin = position_embeddings
        if self.context_parallel_group is None:
            full_position_embeddings = position_embeddings
        else:
            full_cos_sin = gather_for_cp(torch.cat((cos, sin), dim=-1), self.context_parallel_group)
            full_position_embeddings = full_cos_sin.chunk(2, dim=-1)

        indices = self.indexer(hidden_states, full_position_embeddings, cu_seqlens)
        query_and_gate = self.q_proj(hidden_states).view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim * 2,
        )
        query_states, output_gate = query_and_gate.chunk(2, dim=-1)
        key_states = self.k_proj(hidden_states).view(
            batch_size,
            sequence_length,
            self.num_key_value_heads,
            self.head_dim,
        )
        value_states = self.v_proj(hidden_states).view(
            batch_size,
            sequence_length,
            self.num_key_value_heads,
            self.head_dim,
        )

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)
        query_states = apply_rotary_embedding(query_states, cos, sin)

        if self.context_parallel_group is not None:
            key_value_states = gather_for_cp(
                torch.cat((key_states, value_states), dim=-1),
                self.context_parallel_group,
            )
            key_states, value_states = key_value_states.chunk(2, dim=-1)
        full_cos, full_sin = full_position_embeddings
        key_states = apply_rotary_embedding(key_states, full_cos, full_sin)

        kernel = prime_kernels.load("indexed_attention")
        attention_output = kernel.indexed_attention(
            query_states,
            key_states,
            value_states,
            indices,
            self.scale,
        ).reshape(batch_size, sequence_length, -1)
        attention_output = attention_output * output_gate.reshape(batch_size, sequence_length, -1).sigmoid()
        return self.o_proj(attention_output)


__all__ = ["IndexedGatedAttention"]
