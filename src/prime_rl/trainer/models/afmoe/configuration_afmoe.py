from typing import Any, ClassVar, Literal

from pydantic import model_validator

from prime_rl.trainer.models.config import PrimeModelConfig
from prime_rl.trainer.models.layers.moe import ScoreFuncType
from prime_rl.trainer.models.layers.rotary_emb import RopeParameters, standardize_rope_parameters

AfmoeLayerType = Literal["sliding_attention", "full_attention"]


class AfmoeConfig(PrimeModelConfig):
    model_type: ClassVar[str] = "afmoe"

    vocab_size: int = 200192
    hidden_size: int = 2048
    intermediate_size: int = 6144
    moe_intermediate_size: int = 1408
    num_hidden_layers: int = 32
    num_dense_layers: int = 1
    num_attention_heads: int = 16
    num_key_value_heads: int | None = None
    """Defaults to ``num_attention_heads``."""
    head_dim: int = 128
    hidden_act: str = "silu"
    max_position_embeddings: int = 16384
    rms_norm_eps: float = 1e-5
    rope_parameters: RopeParameters
    num_experts: int = 64
    num_experts_per_tok: int = 6
    num_shared_experts: int = 2
    score_func: ScoreFuncType = "sigmoid"
    route_norm: bool = True
    route_scale: float = 1.0
    score_before_experts: bool = False
    load_balance_coeff: float | None = 5e-4
    global_attn_every_n_layers: int = 4
    sliding_window: int = 1024
    mup_enabled: bool = False
    layer_types: list[AfmoeLayerType] | None = None
    """Defaults to sliding attention with every ``global_attn_every_n_layers``-th layer full attention."""

    @model_validator(mode="before")
    @classmethod
    def _standardize_rope(cls, data: dict[str, Any]) -> dict[str, Any]:
        return standardize_rope_parameters(data, default_rope_theta=10_000.0)

    @model_validator(mode="after")
    def _resolve_defaults(self) -> "AfmoeConfig":
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.layer_types is None:
            self.layer_types = [
                "full_attention" if (i + 1) % self.global_attn_every_n_layers == 0 else "sliding_attention"
                for i in range(self.num_hidden_layers)
            ]
        return self
