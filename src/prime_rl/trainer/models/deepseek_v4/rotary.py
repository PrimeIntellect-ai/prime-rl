from typing import Literal

import torch
from torch import nn

from prime_rl.trainer.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from prime_rl.trainer.models.layers.rotary_emb import compute_rope_inv_freq

RopeLabel = Literal["main", "compress"]
ROPE_LABELS: tuple[RopeLabel, ...] = ("main", "compress")


class DeepseekV4RotaryEmbedding(nn.Module):
    """Rotary embedding carrying one inverse-frequency table per RoPE type.

    DeepSeek-V4 keys its RoPE parameters by rope type (`main` / `compress`), which is
    independent of `config.layer_types`: sliding-window layers read `main`, the two
    compressed attention variants share `compress` with their compressor. Each type gets
    its own `<type>_inv_freq` buffer and `<type>_attention_scaling` scalar.

    Because the rotation is interleaved, `forward` returns `cos` / `sin` at half the
    rotary width (one entry per pair).

    Each type also keeps a static fp32 `<type>_cos_sin_cache`, `[cos | sin]` at every position in
    vLLM's layout, which the fused RoPE kernels index by position.
    """

    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.config = config
        for layer_type in ROPE_LABELS:
            inv_freq, attention_scaling = self._compute_inv_freq(layer_type, device=None)
            self.register_buffer(f"{layer_type}_inv_freq", inv_freq, persistent=False)
            setattr(self, f"{layer_type}_attention_scaling", attention_scaling)
            self.register_buffer(
                f"{layer_type}_cos_sin_cache", self._compute_cos_sin_cache(layer_type), persistent=False
            )

    def _compute_inv_freq(self, layer_type: RopeLabel, device: torch.device | None) -> tuple[torch.Tensor, float]:
        return compute_rope_inv_freq(
            getattr(self.config.rope_parameters, layer_type),
            self.config.head_dim,
            self.config.max_position_embeddings,
            device,
        )

    def init_buffers_post_meta(self) -> None:
        """Re-derive the per-rope-type inverse frequencies and cos/sin caches in place.

        The tables are computed eagerly in `__init__` and registered non-persistently, so they
        survive neither meta-device construction nor a `load_state_dict`. Re-deriving them is
        cheap and idempotent.
        """
        for layer_type in ROPE_LABELS:
            inv_freq_buffer = getattr(self, f"{layer_type}_inv_freq")
            inv_freq, attention_scaling = self._compute_inv_freq(layer_type, inv_freq_buffer.device)
            inv_freq_buffer.copy_(inv_freq)
            setattr(self, f"{layer_type}_attention_scaling", attention_scaling)
            getattr(self, f"{layer_type}_cos_sin_cache").copy_(self._compute_cos_sin_cache(layer_type))

    @torch.no_grad()
    def forward(
        self, position_ids: torch.Tensor, layer_type: RopeLabel, *, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = position_ids.device
        inv_freq = getattr(self, f"{layer_type}_inv_freq")
        attention_scaling = getattr(self, f"{layer_type}_attention_scaling")
        inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = device.type if isinstance(device.type, str) and device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            # No `cat([freqs, freqs])`: interleaved RoPE needs one theta per pair.
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            cos = freqs.cos() * attention_scaling
            sin = freqs.sin() * attention_scaling

        return cos.to(dtype=dtype), sin.to(dtype=dtype)

    def _compute_cos_sin_cache(self, layer_type: RopeLabel) -> torch.Tensor:
        """`(max_position_embeddings, rope_dim)` fp32 `[cos | sin]` of `layer_type` at every position."""
        inv_freq = getattr(self, f"{layer_type}_inv_freq")
        if inv_freq.is_meta:
            return torch.empty(
                self.config.max_position_embeddings, 2 * inv_freq.shape[0], device="meta", dtype=torch.float32
            )
        # TODO: size to the run's seq_len; 1M positions cost 256 MiB per rope type.
        positions = torch.arange(self.config.max_position_embeddings, device=inv_freq.device)
        cos, sin = self(positions[None], layer_type, dtype=torch.float32)
        return torch.cat([cos[0], sin[0]], dim=-1)

    def cos_sin_cache(self, layer_type: RopeLabel) -> torch.Tensor:
        """The fp32 `[cos | sin]` cache of `layer_type`, one row per position."""
        return getattr(self, f"{layer_type}_cos_sin_cache")


__all__ = ["DeepseekV4RotaryEmbedding"]
