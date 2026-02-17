from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_rope_utils import rope_config_validation


class MiniMaxM2Config(PretrainedConfig):
    r"""
    Configuration class for MiniMax M2.1 MoE model.

    Args:
        vocab_size (`int`, *optional*, defaults to 200064):
            Vocabulary size of the model.
        hidden_size (`int`, *optional*, defaults to 6144):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 24576):
            Dimension of the MoE expert FFN.
        num_hidden_layers (`int`, *optional*, defaults to 92):
            Number of hidden layers in the Transformer.
        num_attention_heads (`int`, *optional*, defaults to 48):
            Number of attention heads.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            Number of key-value heads for GQA.
        head_dim (`int`, *optional*, defaults to 128):
            Dimension of each attention head.
        hidden_act (`str`, *optional*, defaults to `"silu"`):
            Activation function for the MLP.
        max_position_embeddings (`int`, *optional*, defaults to 131072):
            Maximum sequence length.
        rms_norm_eps (`float`, *optional*, defaults to 1e-6):
            Epsilon for RMS normalization.
        rope_theta (`float`, *optional*, defaults to 5000000):
            Base period for RoPE.
        rotary_dim (`int`, *optional*, defaults to 64):
            Dimension of the rotary embeddings. Converted to `partial_rotary_factor` for HF rope init.
        num_local_experts (`int`, *optional*, defaults to 256):
            Number of routed experts per layer.
        num_experts_per_tok (`int`, *optional*, defaults to 8):
            Number of experts selected per token.
        scoring_func (`str`, *optional*, defaults to `"sigmoid"`):
            Scoring function for the router.
        use_routing_bias (`bool`, *optional*, defaults to `True`):
            Whether to use e_score_correction_bias for load balancing.
        use_qk_norm (`bool`, *optional*, defaults to `True`):
            Whether to use QK normalization.
        qk_norm_type (`str`, *optional*, defaults to `"per_layer"`):
            Type of QK normalization. "per_layer" means each head has its own scale.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in attention projections.
        use_grouped_mm (`bool`, *optional*, defaults to `True`):
            Whether to use grouped matmul for experts.
    """

    model_type = "minimax_m2"
    keys_to_ignore_at_inference = ["past_key_values"]

    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size=200064,
        hidden_size=6144,
        intermediate_size=24576,
        num_hidden_layers=92,
        num_attention_heads=48,
        num_key_value_heads=8,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=131072,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=5000000,
        rope_scaling=None,
        rotary_dim=64,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        num_local_experts=256,
        num_experts_per_tok=8,
        scoring_func="sigmoid",
        use_routing_bias=True,
        use_qk_norm=True,
        qk_norm_type="per_layer",
        attention_bias=False,
        sliding_window=None,
        attention_dropout=0.0,
        output_router_logits=False,
        router_aux_loss_coef=0.001,
        router_jitter_noise=0.0,
        use_grouped_mm=True,
        **kwargs,
    ):
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
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.sliding_window = sliding_window
        self.attention_dropout = attention_dropout

        # Compute partial_rotary_factor from rotary_dim for HF rope init compatibility
        self.rotary_dim = rotary_dim
        self.partial_rotary_factor = rotary_dim / head_dim

        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)

        # MoE arguments
        self.num_local_experts = num_local_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.scoring_func = scoring_func
        self.use_routing_bias = use_routing_bias
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.router_jitter_noise = router_jitter_noise

        # Attention
        self.use_qk_norm = use_qk_norm
        self.qk_norm_type = qk_norm_type

        # Training
        self.use_grouped_mm = use_grouped_mm

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
