from transformers.configuration_utils import PretrainedConfig


class MuseGlimmerTextConfig(PretrainedConfig):
    model_type = "muse_glimmer_text"
    base_config_key = "text_config"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 202048,
        hidden_size: int = 6656,
        intermediate_size: int = 19968,
        num_hidden_layers: int = 52,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 2,
        head_dim: int = 128,
        hidden_activation: str = "silu",
        max_position_embeddings: int = 131072,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-5,
        post_norm_eps: float = 1e-8,
        rope_parameters: dict | None = None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        sliding_window: int = 2048,
        layer_types: list[str] | None = None,
        qk_scale_factor: float = 3.87,
        output_multiplier: float = 0.19611613513818404,
        final_logit_softcapping: float | None = 20.0,
        pad_token_id: int | None = None,
        bos_token_id: int | None = 200000,
        eos_token_id: int | list[int] | None = 200001,
        tie_word_embeddings: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> None:
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_activation = hidden_activation
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.post_norm_eps = post_norm_eps
        self.rope_parameters = rope_parameters or {"rope_type": "default", "rope_theta": 500000.0}
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.sliding_window = sliding_window
        # iRoPE: every fourth layer, counted back from the last, is a full-attention NoPE layer.
        self.layer_types = layer_types or [
            "full_attention" if (num_hidden_layers - 1 - layer_idx) % 4 == 0 else "sliding_attention"
            for layer_idx in range(num_hidden_layers)
        ]
        self.qk_scale_factor = qk_scale_factor
        self.output_multiplier = output_multiplier
        self.final_logit_softcapping = final_logit_softcapping
        self.use_cache = use_cache
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def query_scale(self) -> float:
        """Query multiplier applied after QK-norm, on top of the usual head_dim**-0.5 softmax scale.

        Checkpoints carry either the raw factor or the factor already divided by sqrt(head_dim);
        the raw one is the larger of the two, so anything at or above sqrt(head_dim) is divided.
        """
        sqrt_head_dim = self.head_dim**0.5
        if self.qk_scale_factor >= sqrt_head_dim:
            return self.qk_scale_factor / sqrt_head_dim
        return self.qk_scale_factor


class MuseGlimmerConfig(PretrainedConfig):
    """Composite checkpoint config. Only the text model is trained; `vision_config` is kept as-is."""

    model_type = "muse_glimmer"
    sub_configs = {"text_config": MuseGlimmerTextConfig}

    def __init__(
        self,
        text_config: MuseGlimmerTextConfig | dict | None = None,
        tie_word_embeddings: bool = False,
        **kwargs,
    ) -> None:
        if isinstance(text_config, dict):
            text_config = MuseGlimmerTextConfig(**text_config)
        self.text_config = text_config or MuseGlimmerTextConfig()
        kwargs.setdefault("pad_token_id", self.text_config.pad_token_id)
        kwargs.setdefault("bos_token_id", self.text_config.bos_token_id)
        kwargs.setdefault("eos_token_id", self.text_config.eos_token_id)
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


__all__ = ["MuseGlimmerConfig", "MuseGlimmerTextConfig"]
