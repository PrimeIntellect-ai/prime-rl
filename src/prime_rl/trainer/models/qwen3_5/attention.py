from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from prime_rl.trainer.models.layers.attn import AttentionConfig, FlashAttention
from prime_rl.trainer.models.layers.norms import ZeroCenteredRMSNorm
from prime_rl.trainer.models.layers.rotary_emb import apply_rotary_pos_emb
from prime_rl.trainer.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig


@dataclass(frozen=True)
class TwoBlockLayout:
    """A packed row's two-block attention, as one non-causal varlen call of the flash attention kernels.

    A segment is a prefix followed by a second block, and a token attends the tokens at or before it and the tokens
    of its own block. So the prefix attends the prefix, and the second block attends the whole segment: two batch
    elements of the call per segment, or only the first for a segment that is all prefix. Built once per forward,
    outside the compiled decoder layers.
    """

    cu_seqlens_q: torch.Tensor
    """`[elements + 1]` boundaries of each element's queries. The elements' queries are the row's tokens in order."""
    cu_seqlens_k: torch.Tensor
    """`[elements + 1]` boundaries of each element's keys among the gathered keys."""
    key_index: torch.Tensor
    """`[keys]` row token of each gathered key and value. A prefix with a second block after it is listed twice."""
    max_seqlen_q: int
    """Most queries of one element."""
    max_seqlen_k: int
    """Most keys of one element."""


def prepare_two_block_layout(cu_seqlens: torch.Tensor, prefix_lens: torch.Tensor) -> TwoBlockLayout:
    """Build a packed row's ``TwoBlockLayout`` from its ``cu_seqlens`` once, outside the compiled layers, since it
    reads the lengths on the host. ``prefix_lens`` holds one integer per segment: the tokens of its prefix, at least
    one and at most the segment."""
    seq_lens = torch.diff(cu_seqlens).long()
    if prefix_lens.shape != seq_lens.shape:
        raise ValueError(
            f"prefix_lens must have one entry per segment, got shape={tuple(prefix_lens.shape)} "
            f"for {seq_lens.numel()} segments"
        )
    if prefix_lens.is_floating_point() or prefix_lens.dtype == torch.bool:
        raise ValueError(f"prefix_lens must hold integers, got dtype={prefix_lens.dtype}")
    prefix_lens = prefix_lens.long()
    if bool(((prefix_lens <= 0) | (prefix_lens > seq_lens)).any().item()):
        raise ValueError(
            f"prefix_lens must be positive and at most seq_lens: {prefix_lens.tolist()} vs {seq_lens.tolist()}"
        )

    # Per segment: the prefix's queries over the prefix, then the second block's queries over the whole segment
    query_lens = torch.stack((prefix_lens, seq_lens - prefix_lens), dim=1).flatten()
    key_lens = torch.stack((prefix_lens, seq_lens), dim=1).flatten()
    key_starts = cu_seqlens[:-1].long().repeat_interleave(2)
    # A segment that is all prefix has no second element: the kernels are not given an element without queries
    has_queries = query_lens > 0
    query_lens, key_lens, key_starts = query_lens[has_queries], key_lens[has_queries], key_starts[has_queries]

    cu_seqlens_k = F.pad(key_lens.cumsum(dim=0), (1, 0))
    key_offsets = (key_starts - cu_seqlens_k[:-1]).repeat_interleave(key_lens)
    return TwoBlockLayout(
        cu_seqlens_q=F.pad(query_lens.cumsum(dim=0, dtype=torch.int32), (1, 0)),
        cu_seqlens_k=cu_seqlens_k.to(torch.int32),
        key_index=torch.arange(key_offsets.numel(), device=key_offsets.device) + key_offsets,
        max_seqlen_q=int(query_lens.max().item()),
        max_seqlen_k=int(key_lens.max().item()),
    )


class Qwen3_5Attention(FlashAttention):
    def __init__(self, config: Qwen3_5TextConfig, attention_implementation: str) -> None:
        flash_attn_version = {
            "flash_attention_2": 2,
            "flash_attention_3": 3,
            "flash_attention_4": 4,
        }[attention_implementation]
        super().__init__(
            AttentionConfig(
                hidden_size=config.hidden_size,
                head_dim=config.head_dim,
                num_attention_heads=config.num_attention_heads,
                num_key_value_heads=config.num_key_value_heads,
                is_causal=True,
                attention_bias=config.attention_bias,
                output_bias=config.attention_bias,
                use_qk_norm=False,
                rms_norm_eps=config.rms_norm_eps,
            ),
            flash_attn_version=flash_attn_version,
        )
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.attn_output_gate = config.attn_output_gate
        q_output_size = self.num_heads * self.head_dim * (2 if self.attn_output_gate else 1)
        self.qkv_sizes = (q_output_size, self.qkv_sizes[1], self.qkv_sizes[2])
        self.q_proj = nn.Linear(config.hidden_size, q_output_size, bias=config.attention_bias)
        self.q_norm = ZeroCenteredRMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = ZeroCenteredRMSNorm(self.head_dim, config.rms_norm_eps)

    def _compute_two_block_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layout: TwoBlockLayout
    ) -> torch.Tensor:
        """Run the flash attention kernel with ``layout``'s mask instead of the causal one. q/k/v are
        [total_tokens, heads, dim], after RoPE: a gathered key keeps its own position."""
        # index_select's backward sums the two copies of a prefix's keys and values with index_add_
        k = k.index_select(0, layout.key_index)
        v = v.index_select(0, layout.key_index)
        if self._flash_attn_version == 4:
            # As in FlashAttention._compute_attention: cu_seqlens by keyword, and no max lengths
            out, _ = self.func(
                q, k, v, causal=False, cu_seqlens_q=layout.cu_seqlens_q, cu_seqlens_k=layout.cu_seqlens_k
            )
        else:
            out = self.func(
                q,
                k,
                v,
                layout.cu_seqlens_q,
                layout.cu_seqlens_k,
                layout.max_seqlen_q,
                layout.max_seqlen_k,
                causal=False,
            )
        return out

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.LongTensor,
        max_seqlen: int,
        two_block: TwoBlockLayout | None = None,
    ) -> tuple[torch.Tensor, None]:
        """With ``two_block``, attention uses its mask; without it, the causal mask."""
        batch_size, sequence_length, _ = hidden_states.shape

        query_states, key_states, value_states = self.project_qkv(hidden_states)
        if self.attn_output_gate:
            query_states, output_gate = query_states.view(
                batch_size, sequence_length, self.num_heads, self.head_dim * 2
            ).chunk(2, dim=-1)
            output_gate = output_gate.reshape(batch_size, sequence_length, -1)
        else:
            query_states = query_states.view(batch_size, sequence_length, self.num_heads, self.head_dim)
            output_gate = None

        key_states = key_states.view(batch_size, sequence_length, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(batch_size, sequence_length, self.num_key_value_heads, self.head_dim)
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states.transpose(1, 2),
            key_states.transpose(1, 2),
            cos,
            sin,
        )
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)

        if two_block is None:
            attention_output = self._compute_attention(
                query_states[0],
                key_states[0],
                value_states[0],
                cu_seqlens,
                max_seqlen,
            )
        else:
            attention_output = self._compute_two_block_attention(
                query_states[0], key_states[0], value_states[0], two_block
            )
        attention_output = attention_output.reshape(batch_size, sequence_length, -1)
        if output_gate is not None:
            attention_output = attention_output * output_gate.sigmoid()
        return self.o_proj(attention_output), None


__all__ = ["Qwen3_5Attention", "TwoBlockLayout", "prepare_two_block_layout"]
