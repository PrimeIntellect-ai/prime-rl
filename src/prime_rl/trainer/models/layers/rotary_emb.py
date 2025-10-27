from dataclasses import dataclass

import torch
from torch import nn
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update


@dataclass
class RotaryEmbeddingConfig:
    max_position_embeddings: int
    rope_type: str
    model_config: any


class RotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: RotaryEmbeddingConfig, device=None):
        super().__init__()
        self.rope_type = config.rope_type
        self.max_position_embeddings = config.max_position_embeddings
        self.config = config.model_config
        partial_rotary_factor = getattr(self.config, "partial_rotary_factor", 1.0)
        head_dim = getattr(self.config, "head_dim", None) or self.config.hidden_size // self.config.num_attention_heads
        self.rotary_dim = int(head_dim * partial_rotary_factor)

        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        cos_sin_cache = self._compute_cos_sin_cache()
        self.register_buffer("cos_sin_cache", cos_sin_cache, persistent=False)
    
    @torch.no_grad()
    def _compute_cos_sin_cache(self) -> torch.Tensor:
        """Compute the cos and sin cache."""
        inv_freq, attention_scaling = self.rope_init_fn(self.config)
        t = torch.arange(self.max_position_embeddings, dtype=torch.float)

        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos() * attention_scaling
        sin = freqs.sin() * attention_scaling
        cache = torch.cat((cos, sin), dim=-1)
        return cache

def apply_rotary_emb_torch(x, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim).to(x.dtype)
    sin = sin.unsqueeze(unsqueeze_dim).to(x.dtype)
    x1, x2 = torch.chunk(x, 2, dim=-1)
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return torch.cat((o1, o2), dim=-1)

def apply_rotary_pos_emb(q, k, cos_sin_cache, position_ids=None, unsqueeze_dim=1):
    position_ids = position_ids.flatten()
    num_tokens = position_ids.shape[0]
    cos_sin = cos_sin_cache.index_select(0, position_ids)
    cos, sin = cos_sin.chunk(2, dim=-1)
    rotary_dim = cos.shape[-1] * 2
    q_shape, k_shape = q.shape, k.shape
    q = q.reshape(num_tokens, -1, q_shape[-1])
    k = k.reshape(num_tokens, -1, k_shape[-1])
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_embed = apply_rotary_emb_torch(q_rot, cos, sin, unsqueeze_dim)
    k_embed = apply_rotary_emb_torch(k_rot, cos, sin, unsqueeze_dim)
    q = torch.cat([q_embed, q_pass], dim=-1).reshape(q_shape)
    k = torch.cat([k_embed, k_pass], dim=-1).reshape(k_shape)
    return q, k
