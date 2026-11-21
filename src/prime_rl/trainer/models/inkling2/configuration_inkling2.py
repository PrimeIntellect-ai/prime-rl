from transformers.configuration_utils import PretrainedConfig


class Inkling2TextConfig(PretrainedConfig):
    model_type = "inkling2_text"
    base_config_key = "text_config"

    def __init__(
        self,
        vocab_size: int = 201024,
        unpadded_vocab_size: int = 200058,
        hidden_size: int = 4608,
        num_hidden_layers: int = 48,
        num_attention_heads: int = 48,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        swa_num_attention_heads: int = 48,
        swa_num_key_value_heads: int = 8,
        swa_head_dim: int = 128,
        sliding_window_size: int = 1024,
        local_layer_ids: list[int] | None = None,
        d_rel: int = 16,
        rel_extent: int = 1024,
        q_bias: bool = False,
        o_bias: bool = False,
        log_scaling_n_floor: float | None = 128000,
        log_scaling_alpha: float = 0.1,
        rms_norm_eps: float = 1e-6,
        pre_norm_pre_affine: bool = False,
        final_norm_pre_affine: bool = False,
        use_value_norm_no_gamma: bool = True,
        qk_l2_clip_threshold: float = 22.627416997969522,
        rel_l2_clip_threshold: float = 8.0,
        sconv_kernel_size: int = 4,
        final_logit_softcapping: float | None = 30.0,
        n_layers_per_attn_res_block: int = 6,
        attn_res_normalize_values: bool = True,
        n_routed_experts: int = 384,
        num_experts_per_tok: int = 8,
        n_shared_experts: int = 0,
        intermediate_size: int = 2304,
        mlp_gate_softcap: float = 4.0,
        mlp_up_softcap: float = 25.0,
        route_scale: float = 8.0,
        use_gate_bias: bool = True,
        model_max_length: int = 1048576,
        initializer_range: float = 0.02,
        tie_word_embeddings: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> None:
        unsupported = {
            "q_bias": q_bias,
            "o_bias": o_bias,
            "pre_norm_pre_affine": pre_norm_pre_affine,
            "final_norm_pre_affine": final_norm_pre_affine,
            "n_shared_experts": n_shared_experts,
        }
        enabled = [name for name, value in unsupported.items() if value]
        if enabled:
            raise ValueError(f"Inkling 2 options without a PrimeRL implementation: {enabled}")
        if not 0 < unpadded_vocab_size <= vocab_size:
            raise ValueError(f"unpadded_vocab_size={unpadded_vocab_size} must be in (0, vocab_size={vocab_size}]")

        self.vocab_size = vocab_size
        self.unpadded_vocab_size = unpadded_vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.swa_num_attention_heads = swa_num_attention_heads
        self.swa_num_key_value_heads = swa_num_key_value_heads
        self.swa_head_dim = swa_head_dim
        self.sliding_window_size = sliding_window_size
        # Every sixth layer attends over the whole document; the others over a sliding window.
        self.local_layer_ids = (
            list(local_layer_ids)
            if local_layer_ids is not None
            else [layer_idx for layer_idx in range(num_hidden_layers) if layer_idx % 6 != 5]
        )
        self.d_rel = d_rel
        self.rel_extent = rel_extent
        self.q_bias = q_bias
        self.o_bias = o_bias
        self.log_scaling_n_floor = log_scaling_n_floor
        self.log_scaling_alpha = log_scaling_alpha
        self.rms_norm_eps = rms_norm_eps
        self.pre_norm_pre_affine = pre_norm_pre_affine
        self.final_norm_pre_affine = final_norm_pre_affine
        self.use_value_norm_no_gamma = use_value_norm_no_gamma
        self.qk_l2_clip_threshold = qk_l2_clip_threshold
        self.rel_l2_clip_threshold = rel_l2_clip_threshold
        self.sconv_kernel_size = sconv_kernel_size
        self.final_logit_softcapping = final_logit_softcapping
        self.n_layers_per_attn_res_block = n_layers_per_attn_res_block
        self.attn_res_normalize_values = attn_res_normalize_values
        self.n_routed_experts = n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.n_shared_experts = n_shared_experts
        self.intermediate_size = intermediate_size
        self.mlp_gate_softcap = mlp_gate_softcap
        self.mlp_up_softcap = mlp_up_softcap
        self.route_scale = route_scale
        self.use_gate_bias = use_gate_bias
        self.model_max_length = model_max_length
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

    def is_local_layer(self, layer_idx: int) -> bool:
        return layer_idx in self.local_layer_ids


class Inkling2Config(PretrainedConfig):
    """Composite checkpoint config. Only the language model is trained; the vision tower (`vision_config`)
    and the speculator (`mtp_config`) are kept as-is."""

    model_type = "inkling2"
    sub_configs = {"text_config": Inkling2TextConfig}

    def __init__(
        self,
        text_config: Inkling2TextConfig | dict | None = None,
        vision_config: dict | None = None,
        mtp_config: dict | None = None,
        tie_word_embeddings: bool = False,
        **kwargs,
    ) -> None:
        if isinstance(text_config, dict):
            text_config = Inkling2TextConfig(**text_config)
        self.text_config = text_config or Inkling2TextConfig()
        self.vision_config = vision_config
        self.mtp_config = mtp_config
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


__all__ = ["Inkling2Config", "Inkling2TextConfig"]
