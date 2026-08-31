from transformers.configuration_utils import PretrainedConfig


class Qwen3_8FlashNextTextConfig(PretrainedConfig):
    model_type = "qwen4_exp_text"
    base_config_key = "text_config"
    keys_to_ignore_at_inference = ["past_key_values"]
    ignore_keys_at_rope_validation = {"mrope_section", "mrope_interleaved"}

    def __init__(
        self,
        vocab_size: int = 248320,
        hidden_size: int = 2560,
        num_hidden_layers: int = 48,
        num_attention_heads: int = 24,
        num_key_value_heads: int = 2,
        head_dim: int = 256,
        hidden_act: str = "silu",
        max_position_embeddings: int = 262144,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        rope_parameters: dict | None = None,
        partial_rotary_factor: float = 0.25,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        layer_types: list[str] | None = None,
        full_attention_interval: int = 4,
        linear_conv_kernel_dim: int = 4,
        linear_key_head_dim: int = 128,
        linear_value_head_dim: int = 128,
        linear_num_key_heads: int = 16,
        linear_num_value_heads: int = 48,
        indexer_n_heads: int = 4,
        indexer_kv_heads: int = 1,
        indexer_head_dim: int = 128,
        indexer_budget: int = 2048,
        indexer_compress_ratio: int = 4,
        hc_count: int = 4,
        hc_lowrank: int = 320,
        ple_layer_ids: list[int] | None = None,
        ple_embed_dim: int = 2560,
        ple_conv_kernel_size: int = 4,
        ngram_size: int = 3,
        heads_per_ngram: int = 8,
        ngram_vocab_size_base: int = 20_000_000,
        make_ngram_vocab_size_divisible_by: int = 128,
        split_ngram_parts: int = 128,
        moe_intermediate_size: int = 640,
        shared_expert_intermediate_size: int = 640,
        num_experts_per_tok: int = 10,
        num_experts: int = 512,
        load_balance_coeff: float | None = None,
        pad_token_id: int | None = None,
        bos_token_id: int | None = 248044,
        eos_token_id: int = 248044,
        tie_word_embeddings: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> None:
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
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

        rope_parameters = dict(rope_parameters or {})
        self.rope_theta = rope_parameters.get("rope_theta", 10_000_000.0)
        self.partial_rotary_factor = rope_parameters.get("partial_rotary_factor", partial_rotary_factor)
        self.mrope_section = tuple(rope_parameters.get("mrope_section", (11, 11, 10)))
        self.rope_parameters = rope_parameters

        self.layer_types = layer_types or [
            "linear_attention" if (layer_index + 1) % full_attention_interval else "full_attention"
            for layer_index in range(num_hidden_layers)
        ]
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads
        self.indexer_n_heads = indexer_n_heads
        self.indexer_kv_heads = indexer_kv_heads
        self.indexer_head_dim = indexer_head_dim
        self.indexer_budget = indexer_budget
        self.indexer_compress_ratio = indexer_compress_ratio

        self.hc_count = hc_count
        self.hc_lowrank = hc_lowrank
        self.ple_layer_ids = ple_layer_ids or [2]
        self.ple_embed_dim = ple_embed_dim
        self.ple_conv_kernel_size = ple_conv_kernel_size
        self.ngram_size = ngram_size
        self.heads_per_ngram = heads_per_ngram
        self.ngram_vocab_size_base = ngram_vocab_size_base
        self.make_ngram_vocab_size_divisible_by = make_ngram_vocab_size_divisible_by
        self.split_ngram_parts = split_ngram_parts

        self.moe_intermediate_size = moe_intermediate_size
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts
        self.load_balance_coeff = load_balance_coeff
        self.use_cache = use_cache

        kwargs.pop("mamba_ssm_dtype", None)
        kwargs.pop("mtp", None)
        kwargs.pop("mtp_num_hidden_layers", None)
        kwargs.pop("mtp_use_dedicated_embeddings", None)
        kwargs.pop("output_gate_type", None)
        kwargs.pop("output_router_logits", None)
        kwargs.pop("router_aux_loss_coef", None)
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class Qwen3_8FlashNextConfig(PretrainedConfig):
    model_type = "qwen4_exp"
    sub_configs = {"text_config": Qwen3_8FlashNextTextConfig}

    def __init__(
        self,
        text_config: Qwen3_8FlashNextTextConfig | dict | None = None,
        tie_word_embeddings: bool = False,
        **kwargs,
    ) -> None:
        if isinstance(text_config, dict):
            text_config = Qwen3_8FlashNextTextConfig(**text_config)
        self.text_config = text_config or Qwen3_8FlashNextTextConfig()

        kwargs.pop("vision_config", None)
        kwargs.pop("image_token_id", None)
        kwargs.pop("video_token_id", None)
        kwargs.pop("vision_start_token_id", None)
        kwargs.pop("vision_end_token_id", None)
        kwargs.pop("language_model_only", None)
        super().__init__(
            pad_token_id=self.text_config.pad_token_id,
            bos_token_id=self.text_config.bos_token_id,
            eos_token_id=self.text_config.eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = ["Qwen3_8FlashNextConfig", "Qwen3_8FlashNextTextConfig"]
