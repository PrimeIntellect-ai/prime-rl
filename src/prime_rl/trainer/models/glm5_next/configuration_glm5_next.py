from transformers.configuration_utils import PretrainedConfig


class Glm5NextTextConfig(PretrainedConfig):
    model_type = "glm5_next_text"
    base_config_key = "text_config"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"num_local_experts": "n_routed_experts"}

    def __init__(
        self,
        vocab_size: int = 154880,
        hidden_size: int = 4096,
        intermediate_size: int = 12288,
        moe_intermediate_size: int = 2048,
        num_hidden_layers: int = 45,
        num_attention_heads: int = 64,
        num_key_value_heads: int = 64,
        hidden_act: str = "silu",
        max_position_embeddings: int = 1048576,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-5,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        layer_types: list[str] | None = None,
        mlp_layer_types: list[str] | None = None,
        first_k_dense_replace: int = 3,
        linear_attn_config: dict | None = None,
        q_lora_rank: int = 1536,
        kv_lora_rank: int = 512,
        qk_nope_head_dim: int = 256,
        qk_rope_head_dim: int = 0,
        v_head_dim: int = 256,
        index_n_heads: int = 32,
        index_head_dim: int = 128,
        index_topk: int = 2048,
        index_kpool: int = 4,
        index_kpool_compress: bool = True,
        index_kpool_always_select_tail: bool = True,
        indexer_types: list[str] | None = None,
        n_routed_experts: int = 288,
        num_experts_per_tok: int = 8,
        n_shared_experts: int = 1,
        routed_scaling_factor: float = 2.5,
        norm_topk_prob: bool = True,
        scoring_func: str = "sigmoid",
        n_group: int = 1,
        topk_group: int = 1,
        swiglu_limit: float = 10.0,
        hc_mult: int = 4,
        hc_sinkhorn_iters: int = 20,
        hc_eps: float = 1e-6,
        num_nextn_predict_layers: int = 1,
        pad_token_id: int | None = 154820,
        bos_token_id: int | None = None,
        eos_token_id: int | list[int] | None = None,
        tie_word_embeddings: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> None:
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        self.layer_types = layer_types or ["linear_attention"] * num_hidden_layers
        self.first_k_dense_replace = first_k_dense_replace
        self.mlp_layer_types = mlp_layer_types or [
            "dense" if layer_index < first_k_dense_replace else "sparse" for layer_index in range(num_hidden_layers)
        ]

        linear_attn_config = linear_attn_config or {}
        self.linear_attn_config = linear_attn_config
        self.linear_num_heads = linear_attn_config.get("num_heads", 64)
        self.linear_head_dim = linear_attn_config.get("head_dim", 128)
        self.linear_conv_kernel_dim = linear_attn_config.get("short_conv_kernel_size", 4)
        self.linear_lower_bound = linear_attn_config.get("gate_lower_bound", -5.0)

        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim

        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.index_topk = index_topk
        self.index_kpool = index_kpool
        self.index_kpool_compress = index_kpool_compress
        self.index_kpool_always_select_tail = index_kpool_always_select_tail
        self.indexer_types = indexer_types or ["full"] * num_hidden_layers

        self.n_routed_experts = n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.n_shared_experts = n_shared_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.norm_topk_prob = norm_topk_prob
        self.scoring_func = scoring_func
        self.n_group = n_group
        self.topk_group = topk_group
        self.swiglu_limit = swiglu_limit

        self.hc_mult = hc_mult
        self.hc_sinkhorn_iters = hc_sinkhorn_iters
        self.hc_eps = hc_eps
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.use_cache = use_cache

        # The remaining checkpoint keys pass through untouched so an export's config.json still
        # carries them: vLLM reads topk_method (selection bias), moe_router_dtype (fp32 routing)
        # and indexer_rope_interleave off it. head_dim is 0 in the checkpoint, which generic
        # head_dim readers (perf accounting) would take literally.
        for unused in ("dtype", "head_dim"):
            kwargs.pop(unused, None)
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class Glm5NextConfig(PretrainedConfig):
    model_type = "glm5_next"
    sub_configs = {"text_config": Glm5NextTextConfig}

    def __init__(
        self,
        text_config: Glm5NextTextConfig | dict | None = None,
        tie_word_embeddings: bool = False,
        **kwargs,
    ) -> None:
        if isinstance(text_config, dict):
            text_config = Glm5NextTextConfig(**text_config)
        self.text_config = text_config or Glm5NextTextConfig()

        # Vision keys are not modelled but pass through, so an export's config.json keeps them.
        kwargs.setdefault("pad_token_id", self.text_config.pad_token_id)
        kwargs.setdefault("bos_token_id", self.text_config.bos_token_id)
        kwargs.setdefault("eos_token_id", self.text_config.eos_token_id)
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


__all__ = ["Glm5NextConfig", "Glm5NextTextConfig"]
