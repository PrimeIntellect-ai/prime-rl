from typing import Any, ClassVar, Literal

from pydantic import model_validator

from prime_rl.trainer.models.config import PrimeModelConfig
from prime_rl.trainer.models.layers.rotary_emb import RopeParameters, standardize_rope_dict

LagunaLayerType = Literal["full_attention", "sliding_attention"]

_DEFAULT_ROPE_PARAMETERS: dict[LagunaLayerType, dict[str, Any]] = {
    "full_attention": {"rope_type": "default", "rope_theta": 500000.0},
    "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
}


class LagunaConfig(PrimeModelConfig):
    """Configuration for Poolside Laguna MoE models."""

    model_type: ClassVar[str] = "laguna"

    vocab_size: int = 100352
    hidden_size: int = 2048
    intermediate_size: int = 8192
    num_hidden_layers: int = 40
    num_attention_heads: int = 48
    num_attention_heads_per_layer: list[int] | None = None
    """Defaults to ``num_attention_heads`` for every layer."""
    num_key_value_heads: int = 8
    head_dim: int = 128
    hidden_act: str = "silu"
    max_position_embeddings: int = 131072
    rms_norm_eps: float = 1e-6
    layer_types: list[LagunaLayerType]
    """Defaults to full attention for every layer."""
    rope_parameters: dict[LagunaLayerType, RopeParameters]
    """One RoPE per attention layer type in ``layer_types``."""
    sliding_window: int | None = 512
    attention_bias: bool = False
    gating: bool | Literal["per-head", "per-element"] = True
    """Attention output gating: one gate per head broadcast across head_dim (``True``/``"per-head"``), one gate
    per (head, head_dim) channel (``"per-element"``), or none (``False``)."""
    mlp_layer_types: list[Literal["dense", "sparse"]] | None = None
    """Defaults to a dense first layer followed by sparse layers."""
    moe_intermediate_size: int = 512
    shared_expert_intermediate_size: int = 512
    num_experts: int = 256
    num_experts_per_tok: int = 8
    moe_routed_scaling_factor: float = 1.0
    moe_apply_router_weight_on_input: bool = False
    moe_router_logit_softcapping: float = 0.0
    load_balance_coeff: float | None = 1e-3

    @model_validator(mode="before")
    @classmethod
    def _standardize_rope(cls, data: dict[str, Any]) -> dict[str, Any]:
        num_hidden_layers = data.get("num_hidden_layers", cls.model_fields["num_hidden_layers"].default)
        layer_types = data.get("layer_types") or ["full_attention"] * num_hidden_layers
        rope = data.get("rope_parameters") or data.get("rope_scaling") or {}
        # `rope` is either one dict per layer type or a single dict shared by all layer types.
        is_per_layer_type = any(layer_type in rope for layer_type in layer_types)
        partial_rotary_factor = data.get("partial_rotary_factor")

        rope_parameters = {}
        for layer_type in dict.fromkeys(layer_types):
            params = {**_DEFAULT_ROPE_PARAMETERS.get(layer_type, {})}
            params.update(rope.get(layer_type, {}) if is_per_layer_type else rope)
            if partial_rotary_factor is not None:
                params.setdefault("partial_rotary_factor", partial_rotary_factor)
            rope_parameters[layer_type] = standardize_rope_dict(params, rope_theta=None)
        # vLLM ignores this override and derives YaRN scaling from factor; match it for rollout/trainer parity.
        rope_parameters.get("full_attention", {}).pop("attention_factor", None)
        return {**data, "layer_types": layer_types, "rope_parameters": rope_parameters}

    @model_validator(mode="after")
    def _resolve_defaults(self) -> "LagunaConfig":
        if self.num_attention_heads_per_layer is None:
            self.num_attention_heads_per_layer = [self.num_attention_heads] * self.num_hidden_layers
        if self.mlp_layer_types is None:
            self.mlp_layer_types = ["dense"] + ["sparse"] * (self.num_hidden_layers - 1)
        self._validate_architecture()
        return self

    def _validate_architecture(self) -> None:
        if self.moe_apply_router_weight_on_input:
            raise NotImplementedError("moe_apply_router_weight_on_input=True is not supported by PrimeRL Laguna.")
        if len(self.num_attention_heads_per_layer) != self.num_hidden_layers:
            raise ValueError(
                f"num_attention_heads_per_layer length ({len(self.num_attention_heads_per_layer)}) "
                f"must equal num_hidden_layers ({self.num_hidden_layers})."
            )
        for num_heads in self.num_attention_heads_per_layer:
            if num_heads % self.num_key_value_heads != 0:
                raise ValueError(
                    f"Per-layer attention head count ({num_heads}) must be divisible by "
                    f"num_key_value_heads ({self.num_key_value_heads})."
                )
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"layer_types length ({len(self.layer_types)}) must equal num_hidden_layers ({self.num_hidden_layers})."
            )
        if len(self.mlp_layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"mlp_layer_types length ({len(self.mlp_layer_types)}) "
                f"must equal num_hidden_layers ({self.num_hidden_layers})."
            )


__all__ = ["LagunaConfig"]
