from typing import Any, Literal

from transformers.configuration_utils import PretrainedConfig


class LagunaConfig(PretrainedConfig):
    """Configuration for Poolside Laguna MoE models."""

    model_type = "laguna"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"num_local_experts": "num_experts"}

    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.g_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.self_attn.q_norm": "replicated_with_grad_allreduce",
        "layers.*.self_attn.k_norm": "replicated_with_grad_allreduce",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
        "layers.*.mlp.experts.gate_up_proj": "packed_colwise",
        "layers.*.mlp.experts.down_proj": "rowwise",
        "layers.*.mlp.experts": "moe_tp_experts",
        "layers.*.shared_expert.w1": "colwise",
        "layers.*.shared_expert.w2": "rowwise",
        "layers.*.shared_expert.w3": "colwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size: int = 100352,
        hidden_size: int = 2048,
        intermediate_size: int = 8192,
        num_hidden_layers: int = 40,
        num_attention_heads: int = 48,
        num_key_value_heads: int = 8,
        hidden_act: str = "silu",
        max_position_embeddings: int = 131072,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        rope_parameters: dict[str, Any] | None = None,
        rope_scaling: dict[str, Any] | None = None,
        sliding_window: int | None = 512,
        attention_dropout: float = 0.0,
        moe_intermediate_size: int = 512,
        shared_expert_intermediate_size: int = 512,
        num_experts_per_tok: int = 8,
        num_experts: int = 256,
        output_router_logits: bool = False,
        router_aux_loss_coef: float = 0.001,
        layer_types: list[str] | None = None,
        pad_token_id: int | None = None,
        bos_token_id: int | None = None,
        eos_token_id: int | list[int] | None = None,
        head_dim: int = 128,
        attention_bias: bool = False,
        partial_rotary_factor: float | None = None,
        num_attention_heads_per_layer: list[int] | None = None,
        mlp_layer_types: list[str] | None = None,
        moe_routed_scaling_factor: float = 1.0,
        moe_apply_router_weight_on_input: bool = False,
        moe_router_logit_softcapping: float = 0.0,
        load_balance_coeff: float | None = 1e-3,
        use_grouped_mm: bool = True,
        **kwargs,
    ):
        raw_rope_parameters = rope_parameters if rope_parameters is not None else rope_scaling

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.sliding_window = sliding_window
        self.attention_dropout = attention_dropout
        self.moe_intermediate_size = moe_intermediate_size
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.layer_types = layer_types or ["full_attention"] * num_hidden_layers
        self.head_dim = head_dim
        self.attention_bias = attention_bias
        self.partial_rotary_factor = partial_rotary_factor
        self.num_attention_heads_per_layer = num_attention_heads_per_layer or [num_attention_heads] * num_hidden_layers
        self.mlp_layer_types = mlp_layer_types or ["dense"] + ["sparse"] * (num_hidden_layers - 1)
        self.moe_routed_scaling_factor = moe_routed_scaling_factor
        self.moe_apply_router_weight_on_input = moe_apply_router_weight_on_input
        self.moe_router_logit_softcapping = moe_router_logit_softcapping
        self.load_balance_coeff = load_balance_coeff
        self.use_grouped_mm = use_grouped_mm

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        self.rope_parameters = raw_rope_parameters
        self._normalize_rope_parameters()
        self.rope_scaling = self.rope_parameters
        self.validate_architecture()

    def _normalize_rope_parameters(self) -> None:
        default_rope_params: dict[Literal["full_attention", "sliding_attention"], dict[str, Any]] = {
            "full_attention": {"rope_type": "default", "rope_theta": 500000.0},
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
        }
        layer_types = set(self.layer_types)
        rope_params = self.rope_parameters or {}
        is_nested = isinstance(rope_params, dict) and any(key in layer_types for key in rope_params)
        if is_nested:
            nested = {}
            for layer_type in layer_types:
                params = dict(default_rope_params.get(layer_type, {}))
                params.update(rope_params.get(layer_type, {}))
                nested[layer_type] = params
        else:
            nested = {}
            for layer_type in layer_types:
                params = dict(default_rope_params.get(layer_type, {}))
                params.update(rope_params)
                nested[layer_type] = params

        if self.partial_rotary_factor is not None:
            for params in nested.values():
                params.setdefault("partial_rotary_factor", self.partial_rotary_factor)

        for params in nested.values():
            params.setdefault("rope_type", "default")

        self.rope_parameters = nested
        self.partial_rotary_factor = None

    def convert_rope_params_to_dict(self, **kwargs):
        return kwargs

    def _validate_yarn_rope_parameters(self, rope_parameters: dict, ignore_keys=None) -> None:
        flat_rope_parameters = self.rope_parameters
        self.rope_parameters = rope_parameters
        try:
            super()._validate_yarn_rope_parameters(rope_parameters, ignore_keys=ignore_keys)
        finally:
            self.rope_parameters = flat_rope_parameters

    def validate_architecture(self) -> None:
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
