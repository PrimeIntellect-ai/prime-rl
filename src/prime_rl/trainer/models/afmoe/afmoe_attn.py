"""AFMoE-specific attention implementations."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn

from prime_rl.trainer.models.layers.rms_norm import RMSNorm, RMSNormConfig
from prime_rl.trainer.models.layers.rotary_emb import apply_rotary_pos_emb

if TYPE_CHECKING:
    from torch.nn.attention.flex_attention import BlockMask


@dataclass
class AfmoeAttentionConfig:
    """Configuration for AFMoE attention layers."""

    hidden_size: int
    head_dim: int
    num_attention_heads: int
    num_key_value_heads: int
    rms_norm_eps: float
    is_local_attention: bool
    sliding_window: int | None = None
    attention_dropout: float = 0.0


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads to match query heads for GQA."""
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


class AfmoeSDPAAttention(nn.Module):
    """AFMoE attention using PyTorch's scaled_dot_product_attention.
    """

    def __init__(self, config: AfmoeAttentionConfig):
        super().__init__()
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.is_local_attention = config.is_local_attention
        self.sliding_window = config.sliding_window if config.is_local_attention else None
        self.attention_dropout = config.attention_dropout

        # Projections
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

        # Output gating
        self.gate_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)

        # QK normalization
        self.q_norm = RMSNorm(RMSNormConfig(hidden_size=self.head_dim, eps=config.rms_norm_eps))
        self.k_norm = RMSNorm(RMSNormConfig(hidden_size=self.head_dim, eps=config.rms_norm_eps))

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        block_mask: "BlockMask | None" = None,
    ) -> tuple[torch.Tensor, None]:
        del block_mask

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # Project Q, K, V and gate
        query_states = self.q_proj(hidden_states).view(hidden_shape)
        key_states = self.k_proj(hidden_states).view(hidden_shape)
        value_states = self.v_proj(hidden_states).view(hidden_shape)
        gate_states = self.gate_proj(hidden_states)

        # QK normalization
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        # Transpose for attention: (batch, heads, seq_len, head_dim)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # Apply RoPE only to local (sliding window) attention layers
        if self.is_local_attention:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Expand KV for GQA
        key_states = _repeat_kv(key_states, self.num_key_value_groups)
        value_states = _repeat_kv(value_states, self.num_key_value_groups)

        dropout_p = self.attention_dropout if self.training else 0.0
        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=dropout_p,
            is_causal=attention_mask is None,  # Use causal if no explicit mask
            scale=self.scaling,
        )

        # Reshape output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(*input_shape, -1)

        attn_output = attn_output * F.sigmoid(gate_states)

        # Output projection
        attn_output = self.o_proj(attn_output)

        return attn_output, None


class AfmoeFlexAttention(nn.Module):
    """AFMoE attention using PyTorch's flex_attention.
    """

    def __init__(self, config: AfmoeAttentionConfig):
        super().__init__()
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.is_local_attention = config.is_local_attention
        self.sliding_window = config.sliding_window if config.is_local_attention else None
        self.attention_dropout = config.attention_dropout

        # Projections
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

        # Output gating
        self.gate_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)

        # QK normalization
        self.q_norm = RMSNorm(RMSNormConfig(hidden_size=self.head_dim, eps=config.rms_norm_eps))
        self.k_norm = RMSNorm(RMSNormConfig(hidden_size=self.head_dim, eps=config.rms_norm_eps))

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        block_mask: "BlockMask | None" = None,
    ) -> tuple[torch.Tensor, None]:
        from transformers.integrations.flex_attention import compile_friendly_flex_attention

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # Project Q, K, V and gate
        query_states = self.q_proj(hidden_states).view(hidden_shape)
        key_states = self.k_proj(hidden_states).view(hidden_shape)
        value_states = self.v_proj(hidden_states).view(hidden_shape)
        gate_states = self.gate_proj(hidden_states)

        # QK normalization
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        # Transpose for attention: (batch, heads, seq_len, head_dim)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # Apply RoPE only to local (sliding window) attention layers
        if self.is_local_attention:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Expand KV for GQA
        key_states = _repeat_kv(key_states, self.num_key_value_groups)
        value_states = _repeat_kv(value_states, self.num_key_value_groups)

        if block_mask is not None:
            attn_output = compile_friendly_flex_attention(
                query_states,
                key_states,
                value_states,
                training=self.training,
                block_mask=block_mask,
                scale=self.scaling,
            )
        elif attention_mask is not None:
            dropout_p = self.attention_dropout if self.training else 0.0
            attn_output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=attention_mask,
                dropout_p=dropout_p,
                scale=self.scaling,
            )
        else:
            attn_output = compile_friendly_flex_attention(
                query_states,
                key_states,
                value_states,
                training=self.training,
                block_mask=None,
                scale=self.scaling,
            )

        # Reshape output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(*input_shape, -1)

        attn_output = attn_output * F.sigmoid(gate_states)

        # Output projection
        attn_output = self.o_proj(attn_output)

        return attn_output, None


AFMOE_ATTN_IMPL2CLASS = {
    "sdpa": AfmoeSDPAAttention,
    "flex_attention": AfmoeFlexAttention,
}
