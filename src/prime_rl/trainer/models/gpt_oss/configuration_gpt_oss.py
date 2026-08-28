from transformers.configuration_utils import PretrainedConfig


class GptOssConfig(PretrainedConfig):
    model_type = "gpt_oss"
    attribute_map = {"num_experts": "num_local_experts"}

    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.experts.gate_proj": "colwise",
        "layers.*.mlp.experts.up_proj": "colwise",
        "layers.*.mlp.experts.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }
    base_model_ep_plan = {
        "layers.*.mlp.router": "ep_router",
        "layers.*.mlp.experts.gate_proj": "grouped_gemm",
        "layers.*.mlp.experts.gate_proj_bias": "grouped_gemm",
        "layers.*.mlp.experts.up_proj": "grouped_gemm",
        "layers.*.mlp.experts.up_proj_bias": "grouped_gemm",
        "layers.*.mlp.experts.down_proj": "grouped_gemm",
        "layers.*.mlp.experts.down_proj_bias": "grouped_gemm",
        "layers.*.mlp.experts": "moe_tp_experts",
    }

    def __init__(
        self,
        vocab_size: int = 201088,
        hidden_size: int = 2880,
        intermediate_size: int = 2880,
        num_hidden_layers: int = 36,
        num_attention_heads: int = 64,
        num_key_value_heads: int = 8,
        head_dim: int = 64,
        num_local_experts: int = 128,
        num_experts_per_tok: int = 4,
        hidden_act: str = "silu",
        swiglu_limit: float = 7.0,
        max_position_embeddings: int = 131072,
        initial_context_length: int = 4096,
        sliding_window: int | None = 128,
        layer_types: list[str] | None = None,
        rope_theta: float = 150000.0,
        rope_scaling: dict | None = None,
        rope_parameters: dict | None = None,
        rms_norm_eps: float = 1e-5,
        attention_bias: bool = True,
        attention_dropout: float = 0.0,
        initializer_range: float = 0.02,
        router_aux_loss_coef: float = 0.001,
        output_router_logits: bool = False,
        use_cache: bool = True,
        pad_token_id: int | None = None,
        bos_token_id: int | None = None,
        eos_token_id: int | list[int] | None = None,
        tie_word_embeddings: bool = False,
        **kwargs,
    ) -> None:
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.num_local_experts = num_local_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.hidden_act = hidden_act
        self.swiglu_limit = swiglu_limit
        self.max_position_embeddings = max_position_embeddings
        self.initial_context_length = initial_context_length
        self.sliding_window = sliding_window
        self.layer_types = layer_types or [
            "sliding_attention" if layer_idx % 2 == 0 else "full_attention" for layer_idx in range(num_hidden_layers)
        ]

        if rope_parameters is None:
            rope_parameters = rope_scaling
        if rope_parameters is None:
            rope_parameters = {
                "rope_type": "yarn",
                "factor": 32.0,
                "beta_fast": 32.0,
                "beta_slow": 1.0,
                "truncate": False,
                "original_max_position_embeddings": 4096,
            }
        self.rope_parameters = dict(rope_parameters)
        self.rope_parameters.setdefault("rope_theta", rope_theta)

        self.rms_norm_eps = rms_norm_eps
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.router_aux_loss_coef = router_aux_loss_coef
        self.output_router_logits = output_router_logits
        self.use_cache = use_cache
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


__all__ = ["GptOssConfig"]
