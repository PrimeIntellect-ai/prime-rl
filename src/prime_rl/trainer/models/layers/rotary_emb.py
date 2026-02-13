from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update


def _compute_default_rope_parameters(
    config: Optional["PreTrainedConfig"] = None,
    device: Optional["torch.device"] = None,
    seq_len: int | None = None,
    layer_type: str | None = None,
) -> tuple["torch.Tensor", float]:
    """Computes standard RoPE inverse frequencies without any scaling (factor=1.0)."""
    config.standardize_rope_params()
    rope_parameters_dict = config.rope_parameters[layer_type] if layer_type is not None else config.rope_parameters

    base = rope_parameters_dict["rope_theta"]
    partial_rotary_factor = rope_parameters_dict.get("partial_rotary_factor", 1.0)
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    dim = int(head_dim * partial_rotary_factor)

    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
    return inv_freq, 1.0


if "default" not in ROPE_INIT_FUNCTIONS:
    ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters


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
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config.model_config

        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def compute_default_rope_parameters(self, config=None, device=None, seq_len=None, layer_type=None):
        """Required by transformers 5.0.0 for weight initialization when rope_type is 'default'."""
        return _compute_default_rope_parameters(config or self.config, device, seq_len, layer_type)

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    # Keep half or full tensor for later concatenation
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    # Apply rotary embeddings on the first half or full tensor
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

    # Concatenate back to full shape
    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed


def apply_rotary_pos_emb_interleave(q, k, cos, sin, unsqueeze_dim=1):
    """Applies interleaved Rotary Position Embedding to the query and key tensors.

    This is the non-neox style RoPE used by GLM-5 (glm_moe_dsa) where real/imaginary
    components are interleaved rather than split in half. The input is reshaped from
    (..., d) to (..., d//2, 2) and transposed before applying the standard rotation.

    Args:
        q (`torch.Tensor`): The query tensor with shape (batch, heads, seq_len, head_dim).
        k (`torch.Tensor`): The key tensor with shape (batch, heads, seq_len, head_dim).
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The dimension along which to unsqueeze cos and sin for broadcasting.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the
        interleaved Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    # Interleave: reshape (..., d) -> (..., d//2, 2) -> transpose -> reshape back
    # This converts from interleaved [r0, i0, r1, i1, ...] to half [r0, r1, ..., i0, i1, ...]
    b, h, s, d = q.shape
    q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
    b, h, s, d = k.shape
    k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
