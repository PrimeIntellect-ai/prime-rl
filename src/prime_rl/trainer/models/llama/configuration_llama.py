from typing import Any, ClassVar

from pydantic import field_validator, model_validator

from prime_rl.trainer.models.config import PrimeModelConfig
from prime_rl.trainer.models.layers.rotary_emb import RopeParameters, standardize_rope_parameters


class LlamaConfig(PrimeModelConfig):
    model_type: ClassVar[str] = "llama"

    vocab_size: int = 32000
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int | None = None
    """Defaults to ``num_attention_heads``."""
    head_dim: int | None = None
    """Defaults to ``hidden_size // num_attention_heads``."""
    hidden_act: str = "silu"
    max_position_embeddings: int = 2048
    rms_norm_eps: float = 1e-6
    rope_parameters: RopeParameters
    attention_bias: bool = False
    mlp_bias: bool = False
    eos_token_id: int | list[int] | None = 2

    @field_validator("pad_token_id", mode="before")
    @classmethod
    def _unwrap_pad_token_id(cls, value: int | list[int] | None) -> int | None:
        # Llama 3.x checkpoints store a list of pad token ids.
        return value[0] if isinstance(value, list) else value

    @model_validator(mode="before")
    @classmethod
    def _standardize_rope(cls, data: dict[str, Any]) -> dict[str, Any]:
        return standardize_rope_parameters(data, default_rope_theta=10_000.0)

    @model_validator(mode="after")
    def _resolve_defaults(self) -> "LlamaConfig":
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        return self
