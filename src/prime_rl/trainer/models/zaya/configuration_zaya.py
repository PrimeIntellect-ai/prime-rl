from __future__ import annotations

from typing import Any, Literal

from transformers.configuration_utils import PretrainedConfig


class ZayaConfig(PretrainedConfig):
    model_type = "zaya"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        cca=True,
        num_query_groups=2,
        use_cache=True,
        attention_bias=False,
        lm_head_bias=False,
        vocab_size=262272,
        hidden_size=2048,
        ffn_hidden_size=4096,
        num_hidden_layers=40,
        num_experts=16,
        num_attention_heads=8,
        head_dim=128,
        activation_func="swiglu",
        max_position_embeddings=131072,
        norm_epsilon=1e-05,
        pad_token_id=0,
        bos_token_id=2,
        eos_token_id=106,
        tie_word_embeddings=True,
        rope_theta=5000000,
        attention_dropout=0.0,
        moe_router_topk=1,
        num_experts_per_tok=1,
        moe_intermediate_size=None,
        normalization="RMSNorm",
        router_hidden_size=None,
        zaya_mlp_expansion=None,
        zaya_use_mod=True,
        zaya_high_prec=True,
        zaya_use_eda=True,
        add_bias_linear=False,
        gated_linear_unit=True,
        scale_residual_merge=True,
        fused_add_norm=False,
        residual_in_fp32=True,
        apply_rope_fusion=True,
        bias_activation_fusion=True,
        activation_func_fp8_input_store=False,
        sliding_window=None,
        rope_scaling=None,
        rope_parameters=None,
        partial_rotary_factor=0.5,
        layer_types: list[str] | None = None,
        num_key_value_heads=2,
        clamp_temp=False,
        cca_time0=2,
        cca_time1=2,
        swa_layers=None,
        swa_rotary_base=None,
        rms_norm_eps: float | None = None,
        hidden_act: str = "silu",
        initializer_range: float = 0.02,
        output_router_logits: bool = False,
        _attn_implementation="eager",
        use_grouped_mm=True,
        load_balance_coeff=None,
        **kwargs,
    ):
        raw_rope_parameters = rope_parameters if rope_parameters is not None else rope_scaling

        if moe_router_topk != num_experts_per_tok:
            raise ValueError("moe_router_topk must match num_experts_per_tok")
        if moe_router_topk != 1:
            raise ValueError("PrimeRL Zaya currently supports moe_router_topk == 1")
        if add_bias_linear:
            raise ValueError("PrimeRL Zaya currently supports add_bias_linear=False")
        if attention_bias:
            raise ValueError("PrimeRL Zaya currently supports attention_bias=False")
        if router_hidden_size is None:
            router_hidden_size = 256 if zaya_mlp_expansion is None else zaya_mlp_expansion
        if zaya_mlp_expansion is not None and zaya_mlp_expansion != router_hidden_size:
            raise ValueError("zaya_mlp_expansion must match router_hidden_size")

        self.cca = cca
        self.num_query_groups = num_query_groups
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.lm_head_bias = lm_head_bias
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_experts = num_experts
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        if self.head_dim is None:
            raise ValueError("ZayaConfig requires head_dim")
        if self.num_query_groups != self.num_key_value_heads:
            raise ValueError("num_query_groups must match num_key_value_heads")

        self.activation_func = activation_func
        self.max_position_embeddings = max_position_embeddings
        self.norm_epsilon = norm_epsilon
        self.normalization = normalization
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings
        self.attention_dropout = attention_dropout
        self.moe_router_topk = moe_router_topk
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = (
            self.ffn_hidden_size // 2 if moe_intermediate_size is None else moe_intermediate_size
        )
        self.router_hidden_size = router_hidden_size
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.output_router_logits = output_router_logits
        self.rms_norm_eps = float(norm_epsilon) if rms_norm_eps is None else float(rms_norm_eps)
        self.zaya_mlp_expansion = router_hidden_size
        self.zaya_use_mod = zaya_use_mod
        self.zaya_high_prec = zaya_high_prec
        self.zaya_use_eda = zaya_use_eda
        self.add_bias_linear = add_bias_linear
        self.gated_linear_unit = gated_linear_unit
        self.scale_residual_merge = scale_residual_merge
        self.fused_add_norm = fused_add_norm
        self.residual_in_fp32 = residual_in_fp32
        self.apply_rope_fusion = apply_rope_fusion
        self.bias_activation_fusion = bias_activation_fusion
        self.activation_func_fp8_input_store = activation_func_fp8_input_store
        self.sliding_window = sliding_window
        self.partial_rotary_factor = partial_rotary_factor
        self.rope_theta = rope_theta
        self.clamp_temp = clamp_temp
        self.cca_time0 = cca_time0
        self.cca_time1 = cca_time1
        self.swa_layers = swa_layers
        self.swa_rotary_base = swa_rotary_base
        self._attn_implementation = _attn_implementation
        self.use_grouped_mm = use_grouped_mm
        self.load_balance_coeff = load_balance_coeff
        self.layer_types = layer_types or ["hybrid"] * self.num_hidden_layers

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=self.tie_word_embeddings,
            **kwargs,
        )

        self.rope_parameters = raw_rope_parameters
        self._normalize_rope_parameters()
        self.rope_scaling = self.rope_parameters
        self.validate_architecture()

    def _normalize_rope_parameters(self) -> None:
        default_rope_params: dict[Literal["hybrid", "hybrid_sliding"], dict[str, Any]] = {
            "hybrid": {
                "rope_type": "default",
                "rope_theta": float(self.rope_theta),
                "partial_rotary_factor": self.partial_rotary_factor,
            },
            "hybrid_sliding": {
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": self.partial_rotary_factor,
            },
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

        for params in nested.values():
            if "type" in params:
                params.setdefault("rope_type", params.pop("type"))
            params.setdefault("rope_type", "default")
            params.setdefault("rope_theta", float(self.rope_theta))
            params.setdefault("partial_rotary_factor", self.partial_rotary_factor)

        self.rope_parameters = nested

    def convert_rope_params_to_dict(self, **kwargs):
        return kwargs

    def validate_architecture(self) -> None:
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be a multiple of num_key_value_heads")
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("layer_types must have one entry per hidden layer")
        if invalid_layer_types := set(self.layer_types) - {"hybrid", "hybrid_sliding"}:
            raise ValueError(f"layer_types contains unsupported values: {sorted(invalid_layer_types)}")
        if "hybrid_sliding" in self.layer_types and self.sliding_window is None:
            raise ValueError("sliding_window must be set when layer_types contains hybrid_sliding")
        if self.sliding_window is not None and self.sliding_window <= 0:
            raise ValueError("sliding_window must be a strictly positive integer")


__all__ = ["ZayaConfig"]
