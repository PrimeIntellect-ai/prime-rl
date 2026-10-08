from typing import Any, ClassVar

from pydantic import AliasChoices, Field, model_validator

from prime_rl.trainer.models.config import PrimeModelConfig
from prime_rl.trainer.models.layers.rotary_emb import RopeParameters, standardize_rope_parameters


class Qwen3MoeConfig(PrimeModelConfig):
    model_type: ClassVar[str] = "qwen3_moe"

    vocab_size: int = 151936
    hidden_size: int = 2048
    intermediate_size: int = 6144
    num_hidden_layers: int = 24
    num_attention_heads: int = 32
    num_key_value_heads: int = 4
    head_dim: int | None = None
    """Defaults to ``hidden_size // num_attention_heads``."""
    hidden_act: str = "silu"
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    rope_parameters: RopeParameters
    attention_bias: bool = False
    decoder_sparse_step: int = 1
    moe_intermediate_size: int = 768
    num_experts_per_tok: int = 8
    # Older checkpoints write `num_experts`; transformers 5 writes `num_local_experts`.
    num_experts: int = Field(128, validation_alias=AliasChoices("num_experts", "num_local_experts"))
    norm_topk_prob: bool = False
    mlp_only_layers: list[int] = []
    load_balance_coeff: float | None = None

    @model_validator(mode="before")
    @classmethod
    def _standardize_rope(cls, data: dict[str, Any]) -> dict[str, Any]:
        return standardize_rope_parameters(data, default_rope_theta=10_000.0)

    @model_validator(mode="after")
    def _resolve_defaults(self) -> "Qwen3MoeConfig":
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        return self
