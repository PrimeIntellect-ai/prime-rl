import math
from typing import Any, Literal

import torch
from pydantic import BaseModel, ConfigDict
from torch import nn

RopeType = Literal["default", "linear", "llama3", "yarn"]


class RopeParameters(BaseModel):
    """RoPE hyperparameters, as stored under ``rope_parameters`` in a checkpoint's ``config.json``."""

    model_config = ConfigDict(extra="ignore")

    rope_type: RopeType = "default"
    rope_theta: float
    partial_rotary_factor: float = 1.0
    factor: float | None = None
    # Pretraining context length for llama3/yarn; defaults to the model's max_position_embeddings.
    original_max_position_embeddings: int | None = None
    # llama3
    low_freq_factor: float | None = None
    high_freq_factor: float | None = None
    # yarn
    attention_factor: float | None = None
    beta_fast: float | None = None
    beta_slow: float | None = None
    mscale: float | None = None
    mscale_all_dim: float | None = None
    truncate: bool = True


def standardize_rope_dict(
    rope: dict[str, Any],
    *,
    rope_theta: float | None,
    partial_rotary_factor: float | None = None,
    original_max_position_embeddings: int | None = None,
) -> dict[str, Any]:
    """Fill one RoPE dict from the legacy top-level keys, matching transformers' ``standardize_rope_params``."""
    rope = dict(rope)
    rope.setdefault("rope_type", rope.pop("type", "default"))
    if rope_theta is not None:
        rope.setdefault("rope_theta", rope_theta)
    if partial_rotary_factor is not None:
        rope["partial_rotary_factor"] = partial_rotary_factor
    if original_max_position_embeddings is not None and rope["rope_type"] in ("llama3", "yarn"):
        rope["original_max_position_embeddings"] = original_max_position_embeddings
    return rope


def standardize_rope_parameters(data: dict[str, Any], default_rope_theta: float) -> dict[str, Any]:
    """Build ``rope_parameters`` from a raw config dict that may use the legacy ``rope_theta``/``rope_scaling`` keys.

    Meant for a config's ``mode="before"`` validator; returns a new dict.
    """
    rope = data.get("rope_parameters") or data.get("rope_scaling") or {}
    rope_parameters = standardize_rope_dict(
        rope,
        rope_theta=data.get("rope_theta", default_rope_theta),
        partial_rotary_factor=data.get("partial_rotary_factor"),
        original_max_position_embeddings=data.get("original_max_position_embeddings"),
    )
    return {**data, "rope_parameters": rope_parameters}


def compute_rope_inv_freq(
    rope: RopeParameters,
    head_dim: int,
    max_position_embeddings: int,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, float]:
    """Return the RoPE inverse frequencies and the scale applied to cos/sin."""
    base = rope.rope_theta
    dim = int(head_dim * rope.partial_rotary_factor)
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))

    if rope.rope_type == "default":
        return inv_freq, 1.0

    if rope.rope_type == "linear":
        return inv_freq / rope.factor, 1.0

    if rope.rope_type == "llama3":
        factor = rope.factor
        old_context_len = rope.original_max_position_embeddings or max_position_embeddings
        low_freq_wavelen = old_context_len / rope.low_freq_factor
        high_freq_wavelen = old_context_len / rope.high_freq_factor

        wavelen = 2 * math.pi / inv_freq
        inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
        smooth_factor = (old_context_len / wavelen - rope.low_freq_factor) / (
            rope.high_freq_factor - rope.low_freq_factor
        )
        smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
        is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
        return torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama), 1.0

    if rope.rope_type == "yarn":
        original_max = rope.original_max_position_embeddings or max_position_embeddings
        # DeepSeek-V3 style configs leave `factor` unset and derive it from the context extension.
        factor = rope.factor if rope.factor is not None else max_position_embeddings / original_max

        def get_mscale(scale: float, mscale: float = 1) -> float:
            return 1.0 if scale <= 1 else 0.1 * mscale * math.log(scale) + 1.0

        attention_factor = rope.attention_factor
        if attention_factor is None:
            if rope.mscale and rope.mscale_all_dim:
                attention_factor = float(get_mscale(factor, rope.mscale) / get_mscale(factor, rope.mscale_all_dim))
            else:
                attention_factor = get_mscale(factor)

        beta_fast = rope.beta_fast or 32
        beta_slow = rope.beta_slow or 1

        def find_correction_dim(num_rotations: float) -> float:
            return (dim * math.log(original_max / (num_rotations * 2 * math.pi))) / (2 * math.log(base))

        low, high = find_correction_dim(beta_fast), find_correction_dim(beta_slow)
        if rope.truncate:
            low, high = math.floor(low), math.ceil(high)
        low, high = max(low, 0), min(high, dim - 1)
        if low == high:
            high += 0.001  # Prevent singularity

        pos_freqs = base ** (torch.arange(0, dim, 2).to(device=device, dtype=torch.float) / dim)
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (factor * pos_freqs)
        ramp = torch.clamp((torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low), 0, 1)
        extrapolation_factor = 1 - ramp.to(device=device, dtype=torch.float)
        inv_freq = inv_freq_interpolation * (1 - extrapolation_factor) + inv_freq_extrapolation * extrapolation_factor
        return inv_freq, attention_factor

    raise ValueError(f"Unsupported rope_type: {rope.rope_type!r}")


class RotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, rope: RopeParameters, head_dim: int, max_position_embeddings: int):
        super().__init__()
        self.rope = rope
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        inv_freq, self.attention_scaling = compute_rope_inv_freq(rope, head_dim, max_position_embeddings)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def reset_parameters(self) -> None:
        """Recompute ``inv_freq``, which is not in the state dict, after materializing from the meta device."""
        inv_freq, self.attention_scaling = compute_rope_inv_freq(
            self.rope, self.head_dim, self.max_position_embeddings, self.inv_freq.device
        )
        self.inv_freq.copy_(inv_freq)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
