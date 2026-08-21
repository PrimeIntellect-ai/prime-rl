import torch
from torch import nn
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update

from prime_rl.trainer.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config


def rotate_half_interleaved(x: torch.Tensor) -> torch.Tensor:
    """Rotate consecutive channel pairs: `(x0, x1, x2, x3, ...) -> (-x1, x0, -x3, x2, ...)`.

    Not the same as `prime_rl.trainer.models.layers.rotary_emb.rotate_half`, which pairs
    channel `i` with channel `i + dim / 2` (the GPT-NeoX layout). DeepSeek-V4 stores each
    rotary pair adjacently, so the two are not interchangeable.
    """
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rotary_pos_emb_interleaved(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1
) -> torch.Tensor:
    """Apply interleaved RoPE to the trailing rotary slice of `x`.

    `cos` and `sin` arrive at half width (one entry per interleaved pair) and are widened
    with `repeat_interleave`. Each head is laid out as `[nope | rope]`, so only the last
    `2 * cos.shape[-1]` channels rotate and the leading ones pass through untouched.
    The rotation itself runs in fp32 and is cast back to `x`'s dtype.

    Args:
        x: Tensor whose last dimension is the head dimension.
        cos: Half-width cosines, shape `[batch, seq, rope_dim / 2]`.
        sin: Half-width sines, same shape as `cos`.
        unsqueeze_dim: Axis of `x` that `cos` / `sin` must broadcast over. Use `1` for a
            `[batch, heads, seq, head_dim]` layout and `2` for `[batch, seq, heads, head_dim]`.
    """
    cos = cos.repeat_interleave(2, dim=-1).unsqueeze(unsqueeze_dim)
    sin = sin.repeat_interleave(2, dim=-1).unsqueeze(unsqueeze_dim)
    rope_dim = cos.shape[-1]
    nope, rope = x[..., :-rope_dim], x[..., -rope_dim:]
    rotated = ((rope.float() * cos) + (rotate_half_interleaved(rope).float() * sin)).to(x.dtype)
    return torch.cat([nope, rotated], dim=-1)


class DeepseekV4RotaryEmbedding(nn.Module):
    """Rotary embedding carrying one inverse-frequency table per RoPE type.

    DeepSeek-V4 keys its RoPE parameters by rope type (`main` / `compress`), which is
    independent of `config.layer_types`: sliding-window layers read `main`, the two
    compressed attention variants share `compress` with their compressor. Each type gets
    its own `<type>_inv_freq` / `<type>_original_inv_freq` buffers and
    `<type>_attention_scaling` scalar.

    Because the rotation is interleaved, `forward` returns `cos` / `sin` at half the
    rotary width (one entry per pair). `apply_rotary_pos_emb_interleaved` widens them.
    """

    def __init__(self, config: DeepseekV4Config, device: torch.device | None = None):
        super().__init__()
        self.config = config
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        # Only the nested per-rope-type sub-dicts are real rope types; a flat leftover
        # `rope_type` key at the top level is not one.
        self.layer_types = [name for name, params in config.rope_parameters.items() if isinstance(params, dict)]
        self.rope_type: dict[str, str] = {}
        for layer_type in self.layer_types:
            rope_type = config.rope_parameters[layer_type]["rope_type"]
            self.rope_type[layer_type] = rope_type
            rope_init_fn = self.compute_default_rope_parameters
            if rope_type != "default":
                rope_init_fn = ROPE_INIT_FUNCTIONS[rope_type]
            inv_freq, attention_scaling = rope_init_fn(config, device, layer_type=layer_type)
            self.register_buffer(f"{layer_type}_inv_freq", inv_freq, persistent=False)
            self.register_buffer(f"{layer_type}_original_inv_freq", inv_freq.clone(), persistent=False)
            setattr(self, f"{layer_type}_attention_scaling", attention_scaling)

    @staticmethod
    def compute_default_rope_parameters(
        config: DeepseekV4Config,
        device: torch.device | None = None,
        seq_len: int | None = None,
        layer_type: str | None = None,
    ) -> tuple[torch.Tensor, float]:
        """Unscaled inverse frequencies for one rope type's partial rotary slice."""
        rope_parameters = config.rope_parameters[layer_type]
        base = rope_parameters["rope_theta"]
        partial_rotary_factor = rope_parameters.get("partial_rotary_factor", 1.0)
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        dim = int(head_dim * partial_rotary_factor)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float, device=device) / dim))
        return inv_freq, 1.0

    @torch.no_grad()
    @dynamic_rope_update
    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor, layer_type: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = getattr(self, f"{layer_type}_inv_freq")
        attention_scaling = getattr(self, f"{layer_type}_attention_scaling")
        inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            # No `cat([freqs, freqs])`: interleaved RoPE needs one theta per pair, and
            # `apply_rotary_pos_emb_interleaved` widens cos/sin next to the rotation.
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            cos = freqs.cos() * attention_scaling
            sin = freqs.sin() * attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


__all__ = [
    "DeepseekV4RotaryEmbedding",
    "apply_rotary_pos_emb_interleaved",
    "rotate_half_interleaved",
]
