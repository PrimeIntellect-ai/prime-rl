from transformers.configuration_utils import PretrainedConfig


def _default_layer_types(num_hidden_layers: int) -> list[str]:
    return ["deepseek_sparse_attention"] * num_hidden_layers


def _default_mlp_layer_types(num_hidden_layers: int, first_k_dense_replace: int) -> list[str]:
    dense_layers = min(first_k_dense_replace, num_hidden_layers)
    return ["dense"] * dense_layers + ["sparse"] * (num_hidden_layers - dense_layers)


def _as_dict(config) -> dict:
    if isinstance(config, PretrainedConfig):
        return config.to_dict()
    return dict(config)


class Glm5NextTextConfig(PretrainedConfig):
    model_type = "glm5_next_text"
    base_config_key = "text_config"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"num_local_experts": "n_routed_experts"}

    base_model_tp_plan = {
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.experts.gate_proj": "colwise",
        "layers.*.mlp.experts.up_proj": "colwise",
        "layers.*.mlp.experts.down_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(self, **kwargs):
        model_type = kwargs.pop("model_type", self.model_type)
        vocab_size = kwargs.pop("vocab_size", 154880)
        hidden_size = kwargs.pop("hidden_size", 4096)
        head_dim = kwargs.pop("head_dim", None)
        intermediate_size = kwargs.pop("intermediate_size", 12288)
        num_hidden_layers = kwargs.pop("num_hidden_layers", 45)
        num_attention_heads = kwargs.pop("num_attention_heads", 64)
        num_key_value_heads = kwargs.pop("num_key_value_heads", None)
        hidden_act = kwargs.pop("hidden_act", "silu")
        rms_norm_eps = kwargs.pop("rms_norm_eps", 1e-5)
        pad_token_id = kwargs.pop("pad_token_id", 154820)
        bos_token_id = kwargs.pop("bos_token_id", None)
        eos_token_id = kwargs.pop("eos_token_id", None)
        rope_parameters = kwargs.pop("rope_parameters", None)
        rope_theta = kwargs.pop("rope_theta", 10000.0)
        max_position_embeddings = kwargs.pop("max_position_embeddings", 1048576)
        tie_word_embeddings = kwargs.pop("tie_word_embeddings", False)

        moe_intermediate_size = kwargs.pop("moe_intermediate_size", 2048)
        moe_renormalize = kwargs.pop("moe_renormalize", True)
        norm_topk_prob = kwargs.pop("norm_topk_prob", moe_renormalize)
        scoring_func = kwargs.pop("scoring_func", "sigmoid")
        n_routed_experts = kwargs.pop("n_routed_experts", 288)
        num_experts_per_token = kwargs.pop("num_experts_per_token", 8)
        num_experts_per_tok = kwargs.pop("num_experts_per_tok", num_experts_per_token)
        n_shared_experts = kwargs.pop("n_shared_experts", 1)
        routed_scaling_factor = kwargs.pop("routed_scaling_factor", 2.5)
        topk_method = kwargs.pop("topk_method", "noaux_tc")
        first_k_dense_replace = kwargs.pop("first_k_dense_replace", 3)
        moe_layer_freq = kwargs.pop("moe_layer_freq", 1)
        use_grouped_topk = kwargs.pop("use_grouped_topk", True)
        n_group = kwargs.pop("n_group", 1)
        topk_group = kwargs.pop("topk_group", 1)
        router_aux_loss_coef = kwargs.pop("router_aux_loss_coef", 1e-3)
        moe_router_dtype = kwargs.pop("moe_router_dtype", None)
        output_router_logits = kwargs.pop("output_router_logits", False)

        mla = kwargs.pop("mla", True)
        q_lora_rank = kwargs.pop("q_lora_rank", 1536)
        kv_lora_rank = kwargs.pop("kv_lora_rank", 512)
        qk_nope_head_dim = kwargs.pop("qk_nope_head_dim", 256)
        qk_rope_head_dim = kwargs.pop("qk_rope_head_dim", 0)
        qk_head_dim = kwargs.pop("qk_head_dim", qk_nope_head_dim + qk_rope_head_dim)
        v_head_dim = kwargs.pop("v_head_dim", 256)
        mla_nope = kwargs.pop("mla_nope", kwargs.pop("mla_use_nope", True))
        num_nextn_predict_layers = kwargs.pop("num_nextn_predict_layers", 1)

        layer_types = kwargs.pop("layer_types", None)
        mlp_layer_types = kwargs.pop("mlp_layer_types", None)
        linear_cfg = kwargs.pop("linear_attn_config", {}) or {}
        linear_head_dim = kwargs.pop("linear_head_dim", linear_cfg.get("head_dim", 128))
        linear_num_heads = kwargs.pop("linear_num_heads", linear_cfg.get("num_heads", 64))
        linear_conv_kernel_dim = kwargs.pop("linear_conv_kernel_dim", linear_cfg.get("short_conv_kernel_size", 4))
        linear_lower_bound = kwargs.pop("linear_lower_bound", linear_cfg.get("gate_lower_bound", -5.0))

        index_head_dim = kwargs.pop("index_head_dim", 128)
        index_topk = kwargs.pop("index_topk", 2048)
        index_n_heads = kwargs.pop("index_n_heads", 32)
        index_dsa_use_layernorm = kwargs.pop("index_dsa_use_layernorm", True)
        index_kpool_compress = kwargs.pop("index_kpool_compress", True)
        index_kpool = kwargs.pop("index_kpool", 4)
        index_kpool_always_select_tail = kwargs.pop("index_kpool_always_select_tail", True)
        indexer_rope_interleave = kwargs.pop("indexer_rope_interleave", True)
        indexer_types = kwargs.pop("indexer_types", None)
        use_index_cache = kwargs.pop("use_index_cache", False)
        index_topk_freq = kwargs.pop("index_topk_freq", 1)
        index_topk_pattern = kwargs.pop("index_topk_pattern", None)

        mhc = kwargs.pop("mhc", True)
        mhc_num_residual_streams = kwargs.pop("mhc_num_residual_streams", kwargs.pop("hc_mult", 4))
        hc_eps = kwargs.pop("hc_eps", 1e-6)
        mhc_tau = kwargs.pop("mhc_tau", 0.05)
        hres_vwnstyle = kwargs.pop("hres_vwnstyle", True)
        mhc_no_norm_weight = kwargs.pop("mhc_no_norm_weight", False)
        mhc_sinkhorn_iterations = kwargs.pop("mhc_sinkhorn_iterations", kwargs.pop("hc_sinkhorn_iters", 20))
        mhc_post_mult_value = kwargs.pop("mhc_post_mult_value", 2.0)
        swiglu_limit = kwargs.pop("swiglu_limit", None)
        logit_scale = kwargs.pop("logit_scale", 1.0)
        use_cache = kwargs.pop("use_cache", True)
        attention_bias = kwargs.pop("attention_bias", False)
        attention_dropout = kwargs.pop("attention_dropout", 0.0)
        initializer_range = kwargs.pop("initializer_range", 0.02)

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        if layer_types is None:
            layer_types = _default_layer_types(num_hidden_layers)
        if mlp_layer_types is None:
            mlp_layer_types = _default_mlp_layer_types(num_hidden_layers, first_k_dense_replace)
        if len(layer_types) < num_hidden_layers:
            raise ValueError(
                f"GLM-5.3 layer_types has {len(layer_types)} entries for {num_hidden_layers} hidden layers"
            )
        if len(mlp_layer_types) < num_hidden_layers:
            raise ValueError(
                f"GLM-5.3 mlp_layer_types has {len(mlp_layer_types)} entries for {num_hidden_layers} hidden layers"
            )
        if hidden_act != "silu":
            raise ValueError(f"GLM-5.3 only supports silu hidden_act, got {hidden_act!r}")
        if scoring_func not in ("softmax", "sigmoid"):
            raise ValueError(f"Unknown GLM-5.3 MoE scoring_func {scoring_func!r}")

        self.model_type = model_type
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.head_dim = qk_rope_head_dim if head_dim in (None, 0) else head_dim
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.rms_norm_eps = rms_norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.rope_parameters = rope_parameters
        self.rope_theta = rope_theta

        self.mla = mla
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_head_dim
        self.v_head_dim = v_head_dim
        self.mla_nope = mla_nope

        self.n_routed_experts = n_routed_experts
        self.num_experts_per_token = num_experts_per_tok
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_renormalize = norm_topk_prob
        self.norm_topk_prob = norm_topk_prob
        self.n_shared_experts = n_shared_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.topk_method = topk_method
        self.scoring_func = scoring_func
        self.moe_intermediate_size = moe_intermediate_size
        self.first_k_dense_replace = first_k_dense_replace
        self.moe_layer_freq = moe_layer_freq
        self.use_grouped_topk = use_grouped_topk
        self.n_group = n_group
        self.topk_group = topk_group
        self.router_aux_loss_coef = router_aux_loss_coef
        self.load_balance_coeff = router_aux_loss_coef
        self.moe_router_dtype = moe_router_dtype
        self.output_router_logits = output_router_logits
        self.num_nextn_predict_layers = num_nextn_predict_layers

        self.glm5_layer_types = layer_types
        self.layer_types = [
            "linear_attention" if layer_type == "linear_attention" else "sparse" for layer_type in layer_types
        ]
        self.mlp_layer_types = mlp_layer_types
        self.linear_head_dim = linear_head_dim
        self.linear_num_heads = linear_num_heads
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_lower_bound = linear_lower_bound

        self.index_head_dim = index_head_dim
        self.index_topk = index_topk
        self.index_n_heads = index_n_heads
        self.index_dsa_use_layernorm = index_dsa_use_layernorm
        self.index_kpool_compress = index_kpool_compress
        self.index_kpool = index_kpool
        self.index_kpool_always_select_tail = index_kpool_always_select_tail
        self.indexer_rope_interleave = indexer_rope_interleave
        self.indexer_types = indexer_types
        self.use_index_cache = use_index_cache
        self.index_topk_freq = index_topk_freq
        self.index_topk_pattern = index_topk_pattern

        self.mhc = mhc
        self.mhc_num_residual_streams = mhc_num_residual_streams
        self.hc_mult = mhc_num_residual_streams
        self.hc_eps = hc_eps
        self.mhc_tau = mhc_tau
        self.hres_vwnstyle = hres_vwnstyle
        self.mhc_no_norm_weight = mhc_no_norm_weight
        self.mhc_sinkhorn_iterations = mhc_sinkhorn_iterations
        self.hc_sinkhorn_iters = mhc_sinkhorn_iterations
        self.mhc_post_mult_value = mhc_post_mult_value
        self.swiglu_limit = swiglu_limit
        self.logit_scale = logit_scale
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def is_mla(self):
        return self.mla

    @property
    def is_moe(self):
        return self.n_routed_experts is not None

    @property
    def is_linear_attn(self) -> bool:
        return any(layer_type == "linear_attention" for layer_type in self.layer_types)

    def is_kda_layer(self, layer_idx: int) -> bool:
        return self.glm5_layer_types[layer_idx] == "linear_attention"

    @property
    def layers_block_type(self):
        return [
            "linear_attention" if layer_type == "linear_attention" else "attention"
            for layer_type in self.glm5_layer_types
        ]


class Glm5NextConfig(Glm5NextTextConfig):
    model_type = "glm5_next"
    sub_configs = {"text_config": Glm5NextTextConfig}

    def __init__(self, text_config=None, vision_config=None, **kwargs):
        if text_config is not None:
            text_kwargs = _as_dict(text_config)
            for key in ("_attn_implementation", "attn_implementation"):
                if key in kwargs:
                    text_kwargs[key] = kwargs[key]
            super().__init__(**text_kwargs)
            self.text_config = Glm5NextTextConfig(**text_kwargs)
        else:
            super().__init__(**kwargs)
            self.text_config = Glm5NextTextConfig(**self.to_dict())
        self.model_type = Glm5NextConfig.model_type
        self.vision_config = vision_config
        self.image_token_id = kwargs.get("image_token_id")
        self.video_token_id = kwargs.get("video_token_id")
        self.image_start_token_id = kwargs.get("image_start_token_id")
        self.image_end_token_id = kwargs.get("image_end_token_id")
        self.video_start_token_id = kwargs.get("video_start_token_id")
        self.video_end_token_id = kwargs.get("video_end_token_id")

    def get_text_config(self, decoder: bool = False):
        return getattr(self, "text_config", self)
