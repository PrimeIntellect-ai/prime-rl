from typing import Any, ClassVar, Literal

from pydantic import model_validator

from prime_rl.trainer.models.config import PrimeModelConfig
from prime_rl.trainer.models.layers.rotary_emb import RopeParameters, standardize_rope_parameters

GptOssLayerType = Literal["sliding_attention", "full_attention"]

_DEFAULT_ROPE_PARAMETERS = {
    "rope_type": "yarn",
    "factor": 32.0,
    "beta_fast": 32.0,
    "beta_slow": 1.0,
    "truncate": False,
    "original_max_position_embeddings": 4096,
}


class GptOssConfig(PrimeModelConfig):
    model_type: ClassVar[str] = "gpt_oss"

    vocab_size: int = 201088
    hidden_size: int = 2880
    intermediate_size: int = 2880
    num_hidden_layers: int = 36
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    head_dim: int = 64
    num_local_experts: int = 128
    num_experts_per_tok: int = 4
    max_position_embeddings: int = 131072
    sliding_window: int | None = 128
    layer_types: list[GptOssLayerType] | None = None
    """Defaults to alternating sliding and full attention, starting with sliding."""
    rope_parameters: RopeParameters
    rms_norm_eps: float = 1e-5
    attention_bias: bool = True
    initializer_range: float = 0.02

    @property
    def num_experts(self) -> int:
        return self.num_local_experts

    @model_validator(mode="before")
    @classmethod
    def _standardize_rope(cls, data: dict[str, Any]) -> dict[str, Any]:
        if data.get("rope_parameters") is None and data.get("rope_scaling") is None:
            data = {**data, "rope_parameters": _DEFAULT_ROPE_PARAMETERS}
        return standardize_rope_parameters(data, default_rope_theta=150000.0)

    @model_validator(mode="after")
    def _resolve_defaults(self) -> "GptOssConfig":
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention" if layer_idx % 2 == 0 else "full_attention"
                for layer_idx in range(self.num_hidden_layers)
            ]
        return self


__all__ = ["GptOssConfig"]
