from typing import Any, ClassVar

from pydantic import model_validator

from prime_rl.trainer.models.config import PrimeModelConfig
from prime_rl.trainer.models.layers.rotary_emb import RopeParameters, standardize_rope_parameters


class Glm4MoeConfig(PrimeModelConfig):
    model_type: ClassVar[str] = "glm4_moe"

    vocab_size: int = 151552
    hidden_size: int = 4096
    intermediate_size: int = 10944
    num_hidden_layers: int = 46
    num_attention_heads: int = 96
    num_key_value_heads: int = 8
    head_dim: int | None = None
    """Defaults to ``hidden_size // num_attention_heads``."""
    hidden_act: str = "silu"
    max_position_embeddings: int = 131072
    rms_norm_eps: float = 1e-5
    rope_parameters: RopeParameters
    attention_bias: bool = False
    use_qk_norm: bool = False
    moe_intermediate_size: int = 1408
    num_experts_per_tok: int = 8
    n_shared_experts: int = 1
    n_routed_experts: int = 128
    routed_scaling_factor: float = 1.0
    first_k_dense_replace: int = 1
    norm_topk_prob: bool = True

    @model_validator(mode="before")
    @classmethod
    def _standardize_rope(cls, data: dict[str, Any]) -> dict[str, Any]:
        # GLM-4 MoE applies RoPE to half of each head unless the checkpoint says otherwise.
        data = {"partial_rotary_factor": 0.5, **data}
        return standardize_rope_parameters(data, default_rope_theta=10_000.0)

    @model_validator(mode="after")
    def _resolve_defaults(self) -> "Glm4MoeConfig":
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        return self
