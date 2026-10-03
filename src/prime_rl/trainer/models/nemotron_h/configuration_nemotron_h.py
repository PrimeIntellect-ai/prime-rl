from typing import Any, ClassVar, Literal

from pydantic import model_validator

from prime_rl.trainer.models.config import PrimeModelConfig

NemotronHLayerType = Literal["mamba", "moe", "attention"]

PATTERN_TO_LAYER_TYPE: dict[str, NemotronHLayerType] = {
    "M": "mamba",
    "E": "moe",
    "*": "attention",
}


class NemotronHConfig(PrimeModelConfig):
    model_type: ClassVar[str] = "nemotron_h"

    vocab_size: int = 131072
    hidden_size: int = 4096
    layers_block_type: list[NemotronHLayerType]
    """Defaults to the ``hybrid_override_pattern`` string (``M`` mamba, ``E`` moe, ``*`` attention), else ``"ME*E"``."""
    num_hidden_layers: int | None = None
    """Defaults to ``len(layers_block_type)``; a smaller value truncates ``layers_block_type``."""

    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    attention_bias: bool = False

    intermediate_size: int = 21504
    mlp_hidden_act: str = "relu2"
    mlp_bias: bool = False

    ssm_state_size: int = 128
    mamba_num_heads: int = 128
    mamba_head_dim: int = 64
    mamba_hidden_act: str = "silu"
    n_groups: int = 8
    conv_kernel: int = 4
    time_step_min: float = 0.001
    time_step_max: float = 0.1
    time_step_limit: tuple[float, float] | None = (0.0, float("inf"))
    time_step_floor: float = 1e-4
    use_conv_bias: bool = True
    chunk_size: int = 128

    n_routed_experts: int = 8
    moe_intermediate_size: int = 7688
    moe_shared_expert_intermediate_size: int = 7688
    moe_latent_size: int | None = None
    num_experts_per_tok: int = 2
    routed_scaling_factor: float = 1.0
    norm_topk_prob: bool = True
    load_balance_coeff: float | None = None

    use_bias: bool = False
    initializer_range: float = 0.02
    layer_norm_epsilon: float = 1e-5

    pad_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 2

    @model_validator(mode="before")
    @classmethod
    def _expand_pattern(cls, data: dict[str, Any]) -> dict[str, Any]:
        if data.get("layers_block_type") is None:
            pattern = data.get("hybrid_override_pattern") or "ME*E"
            data = {**data, "layers_block_type": [PATTERN_TO_LAYER_TYPE[token] for token in pattern]}
        return data

    @model_validator(mode="after")
    def _resolve_defaults(self) -> "NemotronHConfig":
        if self.num_hidden_layers is None:
            self.num_hidden_layers = len(self.layers_block_type)
        if self.num_hidden_layers > len(self.layers_block_type):
            raise ValueError(
                f"num_hidden_layers={self.num_hidden_layers} exceeds the {len(self.layers_block_type)} layer types"
            )
        self.layers_block_type = self.layers_block_type[: self.num_hidden_layers]
        return self


__all__ = ["NemotronHConfig"]
