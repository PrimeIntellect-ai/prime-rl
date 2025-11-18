from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .rms_norm import RMSNorm, RMSNormConfig
from .rotary_emb import apply_rotary_pos_emb

try:
    from flash_attn import flash_attn_varlen_func
except ImportError:
    flash_attn_varlen_func = None


@dataclass
class AttentionConfig:
    hidden_size: int
    head_dim: int
    num_attention_heads: int
    num_key_value_heads: int
    is_causal: bool
    attention_bias: bool
    use_qk_norm: bool
    rms_norm_eps: float


# TODO: Does torch compile support config._attn_implementation forking?
# If so, we can combine FlashAttention and SDPAAttention into one class
# Otherwise, do ABC or something to make the signatures match


class FlashAttention(nn.Module):
    """Flash Attention"""

    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.head_dim = config.head_dim
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.is_causal = config.is_causal

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        self.use_qk_norm = config.use_qk_norm
        if self.use_qk_norm:
            self.q_norm = RMSNorm(RMSNormConfig(hidden_size=self.head_dim, eps=config.rms_norm_eps))
            self.k_norm = RMSNorm(RMSNormConfig(hidden_size=self.head_dim, eps=config.rms_norm_eps))

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.LongTensor | None = None,
        max_seqlen: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        key_states = self.k_proj(hidden_states).view(hidden_shape)
        value_states = self.v_proj(hidden_states).view(hidden_shape)

        if self.use_qk_norm:  # main diff from Llama
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # TODO: Can we optimize the rotary applicaiton instead of double transpose?
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        out = flash_attn_varlen_func(
            query_states[0],
            key_states[0],
            value_states[0],
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            causal=True,
        )
        out = out.contiguous()
        attn_output = out.view(1, out.shape[0], -1)
        attn_weights = None

        # attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class SDPAAttention(nn.Module):
    """SDPA Attention"""

    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.head_dim = config.head_dim
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.is_causal = config.is_causal

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        self.use_qk_norm = config.use_qk_norm
        if self.use_qk_norm:
            self.q_norm = RMSNorm(RMSNormConfig(hidden_size=self.head_dim, eps=config.rms_norm_eps))
            self.k_norm = RMSNorm(RMSNormConfig(hidden_size=self.head_dim, eps=config.rms_norm_eps))

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.LongTensor | None = None,
        max_seqlen: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states: torch.Tensor = self.q_proj(hidden_states).view(hidden_shape)
        key_states: torch.Tensor = self.k_proj(hidden_states).view(hidden_shape)
        value_states: torch.Tensor = self.v_proj(hidden_states).view(hidden_shape)

        if self.use_qk_norm:  # main diff from Llama
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # TODO: Can we optimize the rotary applicaiton instead of double transpose?
        key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
        value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)
        out = F.scaled_dot_product_attention(query_states, key_states, value_states, is_causal=True)
        out = out.transpose(1, 2).contiguous()  # .view(out.shape[0], out.shape[1], -1)
        attn_output = out.view(out.shape[0], out.shape[1], -1)
        attn_weights = None

        # attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


ATTN_IMPL2CLASS = {
    "flash_attention_2": FlashAttention,
    "sdpa": SDPAAttention,
}
