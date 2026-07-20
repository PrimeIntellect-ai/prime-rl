"""Plain (dense, non-absorbed) Multi-head Latent Attention (MLA) — DeepSeek-V3's attention
mechanism, shared by any family built on it (Kimi K2 reuses it verbatim; DeepSeek-V3 itself
would too). No sparsity/indexer awareness — see `layers/dsa.py` for the DSA-capable sibling
used to convert a checkpoint built on this module into DeepSeek Sparse Attention.
"""

import functools

import torch
import torch.nn.functional as F
from torch import nn

from .rms_norm import RMSNorm, RMSNormConfig
from .rotary_emb import apply_rotary_pos_emb, rotate_half

try:
    from flash_attn import flash_attn_varlen_func
except ImportError:
    flash_attn_varlen_func = None  # type: ignore

try:
    from flash_attn_interface import flash_attn_varlen_func as flash_attn_3_varlen_func
except ImportError:
    flash_attn_3_varlen_func = None  # type: ignore

try:
    from flash_attn.cute import flash_attn_varlen_func as flash_attn_4_varlen_func
except ImportError:
    flash_attn_4_varlen_func = None  # type: ignore


class MLAConfig:
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        attention_bias: bool,
        rms_norm_eps: float,
        softmax_scale: float,
        rope_interleave: bool,
    ):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.attention_bias = attention_bias
        self.rms_norm_eps = rms_norm_eps
        self.softmax_scale = softmax_scale
        self.rope_interleave = rope_interleave


def apply_rotary_pos_emb_interleaved(q, k, cos, sin, unsqueeze_dim=1):
    """DeepSeek-V3-style interleaved RoPE: view as (d//2, 2) -> transpose -> standard rotate_half."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    b, h, s, d = q.shape
    q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    b, h, s, d = k.shape
    k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class _MLAProjections(nn.Module):
    """Shared parameter set for the MLA attention variants below (q/kv LoRA bottleneck)."""

    def __init__(self, config: MLAConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.softmax_scale = config.softmax_scale
        self.rope_interleave = config.rope_interleave

        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.qk_head_dim, bias=False)
        else:
            self.q_a_proj = nn.Linear(config.hidden_size, config.q_lora_rank, bias=config.attention_bias)
            self.q_a_layernorm = RMSNorm(RMSNormConfig(hidden_size=config.q_lora_rank, eps=config.rms_norm_eps))
            self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False)

        self.kv_a_proj_with_mqa = nn.Linear(
            config.hidden_size, config.kv_lora_rank + config.qk_rope_head_dim, bias=config.attention_bias
        )
        self.kv_a_layernorm = RMSNorm(RMSNormConfig(hidden_size=config.kv_lora_rank, eps=config.rms_norm_eps))
        self.kv_b_proj = nn.Linear(
            config.kv_lora_rank, self.num_heads * (config.qk_nope_head_dim + config.v_head_dim), bias=False
        )

        self.o_proj = nn.Linear(self.num_heads * config.v_head_dim, config.hidden_size, bias=config.attention_bias)

    def _project_qkv(self, hidden_states: torch.Tensor):
        batch_size, seq_length = hidden_states.shape[:2]

        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(batch_size, seq_length, self.num_heads, self.qk_head_dim).transpose(1, 2)
        q_nope, q_rope = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        kv_compressed, k_rope = compressed_kv.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

        kv = self.kv_b_proj(self.kv_a_layernorm(kv_compressed))
        kv = kv.view(batch_size, seq_length, self.num_heads, self.qk_nope_head_dim + self.v_head_dim).transpose(1, 2)
        k_nope, value_states = kv.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        k_rope = k_rope.view(batch_size, 1, seq_length, self.qk_rope_head_dim)
        return q_nope, q_rope, k_nope, k_rope, value_states

    def _apply_rope(self, q_rope, k_rope, position_embeddings):
        cos, sin = position_embeddings
        if self.rope_interleave:
            return apply_rotary_pos_emb_interleaved(q_rope, k_rope, cos, sin)
        return apply_rotary_pos_emb(q_rope, k_rope, cos, sin)


class MLAFlashAttention(_MLAProjections):
    """MLA via flash-attn's varlen kernels."""

    _funcs = {
        2: flash_attn_varlen_func,
        3: flash_attn_3_varlen_func,
        4: flash_attn_4_varlen_func,
    }

    def __init__(self, config: MLAConfig, flash_attn_version: int = 2):
        super().__init__(config)
        self._flash_attn_version = flash_attn_version
        self.func = self._funcs[flash_attn_version]
        self._flash_attn_call = self.func
        if self._flash_attn_version == 4:
            self._flash_attn_call = torch._dynamo.disable(self.func)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.LongTensor | None = None,
        max_seqlen: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        q_nope, q_rope, k_nope, k_rope, value_states = self._project_qkv(hidden_states)
        q_rope, k_rope = self._apply_rope(q_rope, k_rope, position_embeddings)

        k_rope = k_rope.expand(*k_nope.shape[:-1], -1)
        query_states = torch.cat([q_nope, q_rope], dim=-1)
        key_states = torch.cat([k_nope, k_rope], dim=-1)

        # Flash attention expects (batch, seq, heads, dim); we take [0] below for varlen.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # Pad V to qk_head_dim (flash attention requires the same head dim for Q/K/V).
        if self.qk_head_dim != self.v_head_dim:
            value_states = F.pad(value_states, [0, self.qk_head_dim - self.v_head_dim])

        args = [query_states[0], key_states[0], value_states[0], cu_seqlens, cu_seqlens]
        if self._flash_attn_version != 4:
            args.extend([max_seqlen, max_seqlen])

        out = self._flash_attn_call(*args, causal=True, softmax_scale=self.softmax_scale)
        if isinstance(out, tuple):
            out = out[0]

        if self.qk_head_dim != self.v_head_dim:
            out = out[..., : self.v_head_dim]  # truncate padding back to v_head_dim

        out = out.contiguous()
        attn_output = out.view(1, out.shape[0], -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, None


class MLASDPAAttention(_MLAProjections):
    """MLA via `F.scaled_dot_product_attention`."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.LongTensor | None = None,
        max_seqlen: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        q_nope, q_rope, k_nope, k_rope, value_states = self._project_qkv(hidden_states)
        q_rope, k_rope = self._apply_rope(q_rope, k_rope, position_embeddings)

        k_rope = k_rope.expand(*k_nope.shape[:-1], -1)
        query_states = torch.cat([q_nope, q_rope], dim=-1)
        key_states = torch.cat([k_nope, k_rope], dim=-1)

        # SDPA supports a different V head dim natively — no padding needed.
        out = F.scaled_dot_product_attention(
            query_states, key_states, value_states, is_causal=True, scale=self.softmax_scale
        )
        out = out.transpose(1, 2).contiguous()
        attn_output = out.view(out.shape[0], out.shape[1], -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, None


MLA_IMPL2CLASS = {
    "flash_attention_2": functools.partial(MLAFlashAttention, flash_attn_version=2),
    "sdpa": MLASDPAAttention,
    "flash_attention_3": functools.partial(MLAFlashAttention, flash_attn_version=3),
    "flash_attention_4": functools.partial(MLAFlashAttention, flash_attn_version=4),
}

__all__ = ["MLAConfig", "MLA_IMPL2CLASS", "MLAFlashAttention", "MLASDPAAttention", "apply_rotary_pos_emb_interleaved"]
