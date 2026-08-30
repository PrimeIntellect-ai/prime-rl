import torch
from torch import nn

from prime_rl.trainer.models.layers.attn import AttentionConfig, FlashAttention
from prime_rl.trainer.models.layers.rotary_emb import apply_rotary_pos_emb
from prime_rl.trainer.models.qwen3_8_flash_next.norm import RMSNorm


class DenseGatedAttention(FlashAttention):
    """Dense causal form of Qwen's gated attention."""

    def __init__(
        self,
        *,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        norm_eps: float,
        attention_implementation: str,
    ) -> None:
        flash_attn_version = {
            "flash_attention_2": 2,
            "flash_attention_3": 3,
            "flash_attention_4": 4,
        }[attention_implementation]
        super().__init__(
            AttentionConfig(
                hidden_size=hidden_size,
                head_dim=head_dim,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                is_causal=True,
                attention_bias=False,
                output_bias=False,
                use_qk_norm=False,
                rms_norm_eps=norm_eps,
            ),
            flash_attn_version=flash_attn_version,
        )
        self.num_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim * 2, bias=False)
        self.q_norm = RMSNorm(head_dim, norm_eps)
        self.k_norm = RMSNorm(head_dim, norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.LongTensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape

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
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
            unsqueeze_dim=2,
        )

        attention_output = self._compute_attention(
            query_states[0],
            key_states[0],
            value_states[0],
            cu_seqlens,
            max_seqlen,
        ).reshape(batch_size, sequence_length, -1)
        attention_output = attention_output * output_gate.reshape(batch_size, sequence_length, -1).sigmoid()
        return self.o_proj(attention_output)


__all__ = ["DenseGatedAttention"]
