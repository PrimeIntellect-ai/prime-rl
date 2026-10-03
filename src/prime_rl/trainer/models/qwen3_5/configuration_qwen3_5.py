from typing import Any, ClassVar, Literal

from pydantic import Field, model_validator

from prime_rl.trainer.models.config import PrimeModelConfig
from prime_rl.trainer.models.layers.rotary_emb import RopeParameters, standardize_rope_parameters

LayerType = Literal["linear_attention", "full_attention"]


class Qwen3_5RopeParameters(RopeParameters):
    partial_rotary_factor: float = 0.25
    mrope_section: list[int] | None = None
    """Rotary pairs per (temporal, height, width) axis; derived from the rotary dim when unset."""


class Qwen3_5TextConfig(PrimeModelConfig):
    model_type: ClassVar[str] = "qwen3_5_text"

    vocab_size: int = 248320
    hidden_size: int = 4096
    intermediate_size: int = 12288
    num_hidden_layers: int = 32
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int = 256
    hidden_act: str = "silu"
    max_position_embeddings: int = 32768
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    rope_parameters: Qwen3_5RopeParameters
    attention_bias: bool = False
    attn_output_gate: bool = True
    output_gate_type: str = "silu"
    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 32
    layer_types: list[LayerType] | None = None
    """Defaults to a full-attention layer every ``full_attention_interval`` layers, linear attention otherwise."""
    full_attention_interval: int = 4

    @model_validator(mode="before")
    @classmethod
    def _standardize_rope(cls, data: dict[str, Any]) -> dict[str, Any]:
        return standardize_rope_parameters(data, default_rope_theta=10_000.0)

    @model_validator(mode="after")
    def _resolve_defaults(self) -> "Qwen3_5TextConfig":
        if self.layer_types is None:
            self.layer_types = [
                "linear_attention" if (layer_idx + 1) % self.full_attention_interval else "full_attention"
                for layer_idx in range(self.num_hidden_layers)
            ]
        return self


class Qwen3_5MoeTextConfig(Qwen3_5TextConfig):
    model_type: ClassVar[str] = "qwen3_5_moe_text"

    moe_intermediate_size: int = 512
    shared_expert_intermediate_size: int = 512
    num_experts_per_tok: int = 8
    num_experts: int = 256
    load_balance_coeff: float | None = None


class Qwen3_5VisionConfig(PrimeModelConfig):
    depth: int = 27
    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_heads: int = 16
    in_channels: int = 3
    patch_size: int = 16
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    out_hidden_size: int = 3584
    num_position_embeddings: int = 2304


class Qwen3_5Config(PrimeModelConfig):
    """Vision-language composite: a Qwen3.5 language model fed by a vision encoder."""

    model_type: ClassVar[str] = "qwen3_5"

    text_config: Qwen3_5TextConfig = Field(default_factory=Qwen3_5TextConfig)
    vision_config: Qwen3_5VisionConfig = Field(default_factory=Qwen3_5VisionConfig)
    image_token_id: int = 248056


class Qwen3_5MoeConfig(Qwen3_5Config):
    model_type: ClassVar[str] = "qwen3_5_moe"

    text_config: Qwen3_5MoeTextConfig = Field(default_factory=Qwen3_5MoeTextConfig)


__all__ = [
    "Qwen3_5Config",
    "Qwen3_5MoeConfig",
    "Qwen3_5MoeTextConfig",
    "Qwen3_5RopeParameters",
    "Qwen3_5TextConfig",
    "Qwen3_5VisionConfig",
]
