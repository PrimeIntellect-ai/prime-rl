from transformers.configuration_utils import PretrainedConfig


class Qwen3_5TextConfig(PretrainedConfig):
    model_type = "qwen3_5_text"
    base_config_key = "text_config"
    keys_to_ignore_at_inference = ["past_key_values"]
    ignore_keys_at_rope_validation = {"mrope_section", "mrope_interleaved"}

    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.self_attn.q_norm": "replicated_with_grad_allreduce",
        "layers.*.self_attn.k_norm": "replicated_with_grad_allreduce",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size: int = 248320,
        hidden_size: int = 4096,
        intermediate_size: int = 12288,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 4,
        head_dim: int = 256,
        hidden_act: str = "silu",
        max_position_embeddings: int = 32768,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 10_000.0,
        rope_scaling: dict | None = None,
        rope_parameters: dict | None = None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        attn_output_gate: bool = True,
        output_gate_type: str = "silu",
        linear_conv_kernel_dim: int = 4,
        linear_key_head_dim: int = 128,
        linear_value_head_dim: int = 128,
        linear_num_key_heads: int = 16,
        linear_num_value_heads: int = 32,
        layer_types: list[str] | None = None,
        full_attention_interval: int = 4,
        mamba_ssm_dtype: str = "float32",
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
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.attn_output_gate = attn_output_gate
        self.output_gate_type = output_gate_type

        if rope_parameters is None:
            rope_parameters = rope_scaling
        if rope_parameters is None:
            rope_parameters = {
                "rope_type": "default",
                "rope_theta": rope_theta,
                "partial_rotary_factor": kwargs.pop("partial_rotary_factor", 0.25),
                "mrope_interleaved": True,
            }
        self.rope_parameters = dict(rope_parameters)
        self.rope_parameters.setdefault("rope_type", "default")
        self.rope_parameters.setdefault("rope_theta", rope_theta)
        self.rope_parameters.setdefault("partial_rotary_factor", kwargs.pop("partial_rotary_factor", 0.25))
        self.rope_parameters.setdefault("mrope_interleaved", True)

        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads
        self.layer_types = layer_types or [
            "linear_attention" if (layer_idx + 1) % full_attention_interval else "full_attention"
            for layer_idx in range(num_hidden_layers)
        ]
        self.mamba_ssm_dtype = mamba_ssm_dtype
        self.use_cache = use_cache

        kwargs.pop("mtp_num_hidden_layers", None)
        kwargs.pop("mtp_use_dedicated_embeddings", None)
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class Qwen3_5MoeTextConfig(Qwen3_5TextConfig):
    model_type = "qwen3_5_moe_text"

    base_model_tp_plan = {
        **Qwen3_5TextConfig.base_model_tp_plan,
        "layers.*.mlp.experts.gate_proj": "grouped_gemm",
        "layers.*.mlp.experts.up_proj": "grouped_gemm",
        "layers.*.mlp.experts.down_proj": "grouped_gemm",
        "layers.*.mlp.experts": "moe_tp_experts",
        "layers.*.mlp.shared_expert.gate_proj": "colwise",
        "layers.*.mlp.shared_expert.up_proj": "colwise",
        "layers.*.mlp.shared_expert.down_proj": "rowwise",
    }
    base_model_ep_plan = {
        "layers.*.mlp.router": "ep_router",
        "layers.*.mlp.experts.gate_proj": "grouped_gemm",
        "layers.*.mlp.experts.up_proj": "grouped_gemm",
        "layers.*.mlp.experts.down_proj": "grouped_gemm",
        "layers.*.mlp.experts": "moe_tp_experts",
    }

    def __init__(
        self,
        moe_intermediate_size: int = 512,
        shared_expert_intermediate_size: int = 512,
        num_experts_per_tok: int = 8,
        num_experts: int = 256,
        output_router_logits: bool = False,
        router_aux_loss_coef: float = 0.001,
        load_balance_coeff: float | None = None,
        **kwargs,
    ) -> None:
        self.moe_intermediate_size = moe_intermediate_size
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.load_balance_coeff = load_balance_coeff
        kwargs.pop("mlp_only_layers", None)
        super().__init__(**kwargs)


class Qwen3_5VisionConfig(PretrainedConfig):
    model_type = "qwen3_5_vision"
    base_config_key = "vision_config"

    def __init__(
        self,
        depth: int = 27,
        hidden_size: int = 1152,
        hidden_act: str = "gelu_pytorch_tanh",
        intermediate_size: int = 4304,
        num_heads: int = 16,
        in_channels: int = 3,
        patch_size: int = 16,
        spatial_merge_size: int = 2,
        temporal_patch_size: int = 2,
        out_hidden_size: int = 3584,
        num_position_embeddings: int = 2304,
        initializer_range: float = 0.02,
        **kwargs,
    ) -> None:
        self.depth = depth
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.out_hidden_size = out_hidden_size
        self.num_position_embeddings = num_position_embeddings
        self.initializer_range = initializer_range
        kwargs.pop("deepstack_visual_indexes", None)
        super().__init__(**kwargs)


class Qwen3_5Config(PretrainedConfig):
    model_type = "qwen3_5"
    sub_configs = {"vision_config": Qwen3_5VisionConfig, "text_config": Qwen3_5TextConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config: dict | PretrainedConfig | None = None,
        vision_config: dict | PretrainedConfig | None = None,
        image_token_id: int = 248056,
        video_token_id: int = 248057,
        vision_start_token_id: int = 248053,
        vision_end_token_id: int = 248054,
        tie_word_embeddings: bool = False,
        **kwargs,
    ) -> None:
        self.text_config = (
            self.sub_configs["text_config"](**text_config) if isinstance(text_config, dict) else text_config
        ) or self.sub_configs["text_config"]()
        self.vision_config = (
            self.sub_configs["vision_config"](**vision_config) if isinstance(vision_config, dict) else vision_config
        ) or self.sub_configs["vision_config"]()
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        kwargs.pop("language_model_only", None)
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


class Qwen3_5MoeConfig(Qwen3_5Config):
    model_type = "qwen3_5_moe"
    sub_configs = {"vision_config": Qwen3_5VisionConfig, "text_config": Qwen3_5MoeTextConfig}

    def __init__(
        self,
        text_config: dict | PretrainedConfig | None = None,
        vision_config: dict | PretrainedConfig | None = None,
        **kwargs,
    ) -> None:
        super().__init__(text_config=text_config, vision_config=vision_config, **kwargs)


__all__ = [
    "Qwen3_5Config",
    "Qwen3_5MoeConfig",
    "Qwen3_5MoeTextConfig",
    "Qwen3_5TextConfig",
    "Qwen3_5VisionConfig",
]
