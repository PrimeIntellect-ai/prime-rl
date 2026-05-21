from __future__ import annotations

from transformers.configuration_utils import PretrainedConfig


class ZayaConfig(PretrainedConfig):
    model_type = "zaya"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        use_cache=True,
        attention_bias=False,
        lm_head_bias=False,
        vocab_size=262272,
        hidden_size=2048,
        num_hidden_layers=40,
        num_experts=16,
        num_attention_heads=8,
        head_dim=128,
        max_position_embeddings=131072,
        pad_token_id=0,
        bos_token_id=2,
        eos_token_id=106,
        tie_word_embeddings=True,
        attention_dropout=0.0,
        num_experts_per_tok=1,
        moe_intermediate_size=2048,
        router_hidden_size=256,
        zaya_use_eda=True,
        sliding_window=None,
        rope_parameters=None,
        partial_rotary_factor=0.5,
        layer_types: list[str] | None = None,
        num_key_value_heads=2,
        cca_time0=2,
        cca_time1=2,
        rms_norm_eps: float = 1e-05,
        hidden_act: str = "silu",
        initializer_range: float = 0.02,
        output_router_logits: bool = False,
        _attn_implementation="eager",
        use_grouped_mm=True,
        load_balance_coeff=None,
        **kwargs,
    ):
        if attention_bias:
            raise ValueError("PrimeRL Zaya currently supports attention_bias=False")
        if num_experts_per_tok != 1:
            raise ValueError("PrimeRL Zaya currently supports num_experts_per_tok == 1")

        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.lm_head_bias = lm_head_bias
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_experts = num_experts
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        if self.head_dim is None:
            raise ValueError("ZayaConfig requires head_dim")

        self.max_position_embeddings = max_position_embeddings
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings
        self.attention_dropout = attention_dropout
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size
        self.router_hidden_size = router_hidden_size
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.output_router_logits = output_router_logits
        self.rms_norm_eps = float(rms_norm_eps)
        self.norm_epsilon = self.rms_norm_eps
        self.zaya_use_eda = zaya_use_eda
        self.sliding_window = sliding_window
        self.partial_rotary_factor = partial_rotary_factor
        self.cca_time0 = cca_time0
        self.cca_time1 = cca_time1
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

        self._attn_implementation = _attn_implementation
        self.rope_parameters = rope_parameters
        self.rope_scaling = self.rope_parameters
        self.validate_architecture()

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
