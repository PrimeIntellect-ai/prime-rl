from transformers.configuration_utils import PretrainedConfig


class Qwen3_5MoeConfig(PretrainedConfig):
    r"""
    Configuration class for the custom PrimeRL Qwen3.5-MoE model.

    This matches the HuggingFace `Qwen3_5MoeTextConfig` architecture:
    hybrid attention (GatedDeltaNet linear + gated softmax), MoE with gated shared expert,
    and (1+weight) RMSNorm parameterization.

    Defaults match Qwen3.5-35B-A3B.
    """

    model_type = "qwen3_5_moe_text"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=248320,
        hidden_size=2048,
        num_hidden_layers=40,
        num_attention_heads=16,
        num_key_value_heads=2,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        head_dim=256,
        # Linear attention (GatedDeltaNet) params
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        # MoE params
        moe_intermediate_size=512,
        shared_expert_intermediate_size=512,
        num_experts_per_tok=8,
        num_experts=256,
        output_router_logits=False,
        router_aux_loss_coef=0.001,
        # Layer type configuration
        layer_types=None,
        # PrimeRL additions
        load_balance_coeff=None,
        use_grouped_mm=True,
        pad_token_id=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.head_dim = head_dim
        self.pad_token_id = pad_token_id

        # Partial RoPE: only rotate 25% of head_dim
        kwargs.setdefault("partial_rotary_factor", 0.25)

        # Layer types: default every 4th layer is full_attention, rest linear_attention
        self.layer_types = layer_types
        if self.layer_types is None:
            interval_pattern = kwargs.pop("full_attention_interval", 4)
            self.layer_types = [
                "linear_attention" if bool((i + 1) % interval_pattern) else "full_attention"
                for i in range(self.num_hidden_layers)
            ]

        # Linear attention (GatedDeltaNet) params
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads

        # MoE params
        self.moe_intermediate_size = moe_intermediate_size
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef

        # PrimeRL additions
        self.load_balance_coeff = load_balance_coeff
        self.use_grouped_mm = use_grouped_mm

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
