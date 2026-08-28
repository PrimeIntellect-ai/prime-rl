import torch
from torch import nn

from prime_rl.trainer.models.layers.attn import AttentionConfig, FlashAttention
from prime_rl.trainer.models.layers.rotary_emb import apply_rotary_pos_emb
from prime_rl.trainer.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from prime_rl.trainer.models.qwen3_5.norm import Qwen3_5RMSNorm


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
        self.q_proj = nn.Linear(config.hidden_size, q_output_size, bias=config.attention_bias)
        self.q_norm = Qwen3_5RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = Qwen3_5RMSNorm(self.head_dim, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.LongTensor,
        max_seqlen: int,
    ) -> tuple[torch.Tensor, None]:
        batch_size, sequence_length, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        if self.attn_output_gate:
            query_states, output_gate = query_states.view(
                batch_size, sequence_length, self.num_heads, self.head_dim * 2
            ).chunk(2, dim=-1)
            output_gate = output_gate.reshape(batch_size, sequence_length, -1)
        else:
            query_states = query_states.view(batch_size, sequence_length, self.num_heads, self.head_dim)
            output_gate = None

        key_states = self.k_proj(hidden_states).view(
            batch_size, sequence_length, self.num_key_value_heads, self.head_dim
        )
        value_states = self.v_proj(hidden_states).view(
            batch_size, sequence_length, self.num_key_value_heads, self.head_dim
        )
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

        attention_output = self._compute_attention(
            query_states[0],
            key_states[0],
            value_states[0],
            cu_seqlens,
            max_seqlen,
        ).reshape(batch_size, sequence_length, -1)
        if output_gate is not None:
            attention_output = attention_output * output_gate.sigmoid()
        return self.o_proj(attention_output), None


__all__ = ["Qwen3_5Attention"]
