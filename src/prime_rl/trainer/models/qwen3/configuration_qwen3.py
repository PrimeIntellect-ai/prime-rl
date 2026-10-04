from typing import Any, ClassVar, Literal

from pydantic import model_validator

from prime_rl.trainer.models.config import PrimeModelConfig
from prime_rl.trainer.models.layers.rotary_emb import RopeParameters, standardize_rope_parameters


class Qwen3Config(PrimeModelConfig):
    model_type: ClassVar[str] = "qwen3"

    vocab_size: int = 151936
    hidden_size: int = 4096
    intermediate_size: int = 22016
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int | None = None
    """Defaults to ``num_attention_heads``."""
    head_dim: int = 128
    hidden_act: str = "silu"
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    rope_parameters: RopeParameters
    attention_bias: bool = False
    use_sliding_window: bool = False
    sliding_window: int | None = 4096
    """Only applies when ``use_sliding_window``; reset to ``None`` otherwise."""
    max_window_layers: int = 28
    layer_types: list[Literal["full_attention", "sliding_attention"]] | None = None
    """Defaults to sliding attention from ``max_window_layers`` on when sliding windows are enabled."""

    @model_validator(mode="before")
    @classmethod
    def _standardize_rope(cls, data: dict[str, Any]) -> dict[str, Any]:
        return standardize_rope_parameters(data, default_rope_theta=10_000.0)

    @model_validator(mode="after")
    def _resolve_defaults(self) -> "Qwen3Config":
        if not self.use_sliding_window:
            self.sliding_window = None
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention"
                if self.sliding_window is not None and layer_idx >= self.max_window_layers
                else "full_attention"
                for layer_idx in range(self.num_hidden_layers)
            ]
        return self
