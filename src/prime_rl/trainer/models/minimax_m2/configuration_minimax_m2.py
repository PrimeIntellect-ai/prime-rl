from typing import Any, ClassVar, Literal

from pydantic import model_validator

from prime_rl.trainer.models.config import PrimeModelConfig
from prime_rl.trainer.models.layers.moe import ScoreFuncType
from prime_rl.trainer.models.layers.rotary_emb import RopeParameters, standardize_rope_parameters


class MiniMaxM2Config(PrimeModelConfig):
    model_type: ClassVar[str] = "minimax_m2"

    vocab_size: int = 200064
    hidden_size: int = 6144
    intermediate_size: int = 24576
    """Hidden size of each routed expert."""
    num_hidden_layers: int = 92
    num_attention_heads: int = 48
    num_key_value_heads: int = 8
    head_dim: int = 128
    hidden_act: str = "silu"
    max_position_embeddings: int = 131072
    rms_norm_eps: float = 1e-6
    rope_parameters: RopeParameters
    """Rotates the first ``rotary_dim`` (config.json key) channels of each head."""
    num_local_experts: int = 256
    num_experts_per_tok: int = 8
    scoring_func: ScoreFuncType = "sigmoid"
    use_routing_bias: bool = True
    use_qk_norm: bool = True
    qk_norm_type: Literal["per_head", "per_layer"] = "per_layer"
    attention_bias: bool = False
    eos_token_id: int | list[int] | None = 2

    @model_validator(mode="before")
    @classmethod
    def _standardize_rope(cls, data: dict[str, Any]) -> dict[str, Any]:
        partial_rotary_factor = data.get("rotary_dim", 64) / data.get("head_dim", 128)
        return standardize_rope_parameters(
            {**data, "partial_rotary_factor": partial_rotary_factor}, default_rope_theta=5_000_000.0
        )
