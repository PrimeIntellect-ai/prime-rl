import prime_kernels
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed import ProcessGroup

from prime_rl.trainer.models.qwen3_8_flash_next.norm import RMSNorm
from prime_rl.trainer.models.qwen3_8_flash_next.rotary_embedding import apply_rotary_embedding
from prime_rl.utils.cp import gather_for_cp


class SparseAttentionIndexer(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_query_heads: int,
        head_dim: int,
        token_budget: int,
        compression_ratio: int,
        norm_eps: float,
    ) -> None:
        super().__init__()
        self.num_query_heads = num_query_heads
        self.head_dim = head_dim
        self.token_budget = token_budget
        self.compression_ratio = compression_ratio
        self.block_budget = token_budget // compression_ratio

        self.index_qk_proj = nn.Linear(
            hidden_size,
            (num_query_heads + 1) * head_dim,
            bias=False,
        )
        self.q_layernorm = RMSNorm(head_dim, norm_eps)
        self.k_layernorm = RMSNorm(head_dim, norm_eps)
        self.requires_grad_(False)

        self.context_parallel_group: ProcessGroup | None = None
        self.context_parallel_rank = 0

    def set_context_parallel_attributes(self, process_group: ProcessGroup, rank: int, world_size: int) -> None:
        self.context_parallel_group = process_group
        self.context_parallel_rank = rank

    @property
    def output_width(self) -> int:
        return self.token_budget + self.compression_ratio - 1

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        full_position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.LongTensor,
    ) -> torch.Tensor:
        local_tokens = hidden_states.shape[1]
        projected_query_key = self.index_qk_proj(hidden_states[0])
        query, local_raw_key = projected_query_key.split(
            (self.num_query_heads * self.head_dim, self.head_dim),
            dim=-1,
        )
        query = self.q_layernorm(query.view(local_tokens, self.num_query_heads, self.head_dim))

        if self.context_parallel_group is None:
            raw_key = local_raw_key
        else:
            raw_key = gather_for_cp(local_raw_key.unsqueeze(0), self.context_parallel_group)[0]
        total_tokens = raw_key.shape[0]

        token_indices = torch.arange(total_tokens, dtype=cu_seqlens.dtype, device=hidden_states.device)
        query_start = self.context_parallel_rank * local_tokens
        query_token_indices = token_indices[query_start : query_start + local_tokens]
        sequence_indices = torch.searchsorted(cu_seqlens[1:], query_token_indices, right=True)
        sequence_starts = cu_seqlens[:-1][sequence_indices]
        sequence_positions = query_token_indices - sequence_starts

        all_sequence_indices = torch.searchsorted(cu_seqlens[1:], token_indices, right=True)
        all_sequence_ends = cu_seqlens[1:][all_sequence_indices]
        all_sequence_positions = token_indices - cu_seqlens[:-1][all_sequence_indices]
        is_block_start = (all_sequence_positions.remainder(self.compression_ratio) == 0) & (
            token_indices + self.compression_ratio <= all_sequence_ends
        )
        max_blocks = total_tokens // self.compression_ratio
        block_slots = is_block_start.cumsum(0) - 1
        block_slots = block_slots.masked_fill(~is_block_start, max_blocks)
        block_start_tokens = torch.zeros(
            max_blocks + 1,
            dtype=torch.long,
            device=hidden_states.device,
        )
        block_start_tokens.scatter_(0, block_slots.long(), token_indices.long())
        block_start_tokens = block_start_tokens[:-1]
        offsets = torch.arange(self.compression_ratio, device=hidden_states.device)
        compressed_key = raw_key[block_start_tokens[:, None] + offsets].float().mean(dim=1).to(raw_key.dtype)
        compressed_key = self.k_layernorm(compressed_key).unsqueeze(1)

        cos, sin = full_position_embeddings
        query = apply_rotary_embedding(
            query.unsqueeze(0),
            cos.index_select(1, query_token_indices.long()),
            sin.index_select(1, query_token_indices.long()),
        )[0]
        compressed_key = apply_rotary_embedding(
            compressed_key.unsqueeze(0),
            cos.index_select(1, block_start_tokens),
            sin.index_select(1, block_start_tokens),
        )[0, :, 0]

        blocks_per_sequence = (cu_seqlens[1:] - cu_seqlens[:-1]).div(
            self.compression_ratio,
            rounding_mode="floor",
        )
        cumulative_blocks = F.pad(blocks_per_sequence.cumsum(0), (1, 0))
        sequence_block_start = cumulative_blocks[sequence_indices]
        causal_block_end = sequence_block_start + (sequence_positions + 1).div(
            self.compression_ratio,
            rounding_mode="floor",
        )
        kernel = prime_kernels.load("indexed_attention")
        selected_blocks = kernel.select_indexed_blocks(
            query,
            compressed_key,
            sequence_block_start.to(torch.int32),
            causal_block_end.to(torch.int32),
            self.block_budget,
        )

        num_blocks = block_start_tokens.shape[0]
        sentinel = total_tokens
        block_start_tokens = F.pad(block_start_tokens, (0, 1), value=sentinel)
        selected_block_starts = block_start_tokens[selected_blocks.long()]
        selected_tokens = selected_block_starts[:, :, None] + offsets
        selected_tokens = selected_tokens.masked_fill(selected_blocks[:, :, None] == num_blocks, sentinel)

        indices = torch.full(
            (local_tokens, self.output_width),
            sentinel,
            dtype=torch.int32,
            device=hidden_states.device,
        )
        indices[:, : self.token_budget] = selected_tokens.flatten(1).to(torch.int32)

        visible_blocks = (causal_block_end - sequence_block_start).clamp(max=self.block_budget)
        tail_length = (sequence_positions + 1).remainder(self.compression_ratio)
        tail_start = query_token_indices - tail_length + 1
        tail_offsets = offsets[:-1]
        tail_tokens = tail_start[:, None] + tail_offsets
        tail_tokens = tail_tokens.masked_fill(tail_offsets >= tail_length[:, None], sentinel)
        tail_columns = visible_blocks[:, None] * self.compression_ratio + tail_offsets
        indices.scatter_(1, tail_columns.long(), tail_tokens.to(torch.int32))
        return indices.unsqueeze(0)


__all__ = ["SparseAttentionIndexer"]
