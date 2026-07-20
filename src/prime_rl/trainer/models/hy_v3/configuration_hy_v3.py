from transformers.configuration_utils import PretrainedConfig


class HYV3Config(PretrainedConfig):
    r"""
    Configuration class for Tencent Hy3 (HYV3) models. Instantiating a configuration with the
    defaults will yield a configuration close to that of
    [tencent/Hy3](https://huggingface.co/tencent/Hy3).

    Field names follow the hub `config.json` (glm4_moe-style: `first_k_dense_replace`, `qk_norm`,
    `route_norm`) rather than the transformers-native `HYV3Config`, so the hub checkpoint config
    loads directly. `mlp_layer_types` is derived from `first_k_dense_replace` for compatibility
    with the transformers `HYV3ForCausalLM` implementation used in verification.

    Args:
        vocab_size (`int`, *optional*, defaults to 120832):
            Vocabulary size of the HYV3 model.
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 13312):
            Dimension of the dense MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 80):
            Number of hidden layers in the Transformer decoder (excluding the MTP layer).
        num_attention_heads (`int`, *optional*, defaults to 64):
            Number of attention heads for each attention layer.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            Number of key/value heads for Grouped Query Attention.
        head_dim (`int`, *optional*, defaults to 128):
            Dimension of each attention head.
        hidden_act (`str`, *optional*, defaults to `"silu"`):
            The non-linear activation function in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 262144):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.006):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the rms normalization layers.
        rope_parameters (`dict`, *optional*):
            RoPE configuration, e.g. `{"rope_type": "default", "rope_theta": 11158840.0}`.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        mlp_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the dense MLP layers.
        moe_intermediate_size (`int`, *optional*, defaults to 1536):
            Intermediate size of each routed expert.
        num_experts (`int`, *optional*, defaults to 192):
            Number of routed experts.
        num_experts_per_tok (`int`, *optional*, defaults to 8):
            Number of experts each token is routed to.
        num_shared_experts (`int`, *optional*, defaults to 1):
            Number of shared experts.
        router_scaling_factor (`float`, *optional*, defaults to 2.826):
            Scaling factor applied to the (normalized) top-k routing weights.
        route_norm (`bool`, *optional*, defaults to `True`):
            Whether to normalize the top-k routing weights to sum to 1.
        first_k_dense_replace (`int`, *optional*, defaults to 1):
            Number of dense layers before the MoE layers start.
        mlp_layer_types (`list[str]`, *optional*):
            Per-layer MLP type (`"dense"` or `"sparse"`). Derived from `first_k_dense_replace`
            if not given.
        qk_norm (`bool`, *optional*, defaults to `True`):
            Whether to use per-head query-key RMS normalization in the attention.
        num_nextn_predict_layers (`int`, *optional*, defaults to 1):
            Number of MTP (multi-token prediction) layers in the checkpoint. MTP layers are only
            used for speculative decoding at inference time and are dropped for training.
        use_grouped_mm (`bool`, *optional*, defaults to `True`):
            Whether to use grouped GEMM for the experts computation. The hub config sets this to
            `False` (it refers to the HF eager experts implementation there); the trainer
            overrides it from `ModelConfig.moe_use_grouped_mm` before model construction.
    """

    model_type = "hy_v3"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"num_local_experts": "num_experts"}

    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size=120832,
        hidden_size=4096,
        intermediate_size=13312,
        num_hidden_layers=80,
        num_attention_heads=64,
        num_key_value_heads=8,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=262144,
        initializer_range=0.006,
        rms_norm_eps=1e-5,
        use_cache=True,
        tie_word_embeddings=False,
        rope_parameters=None,
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        moe_intermediate_size=1536,
        num_experts=192,
        num_experts_per_tok=8,
        num_shared_experts=1,
        router_scaling_factor=2.826,
        route_norm=True,
        first_k_dense_replace=1,
        mlp_layer_types=None,
        qk_norm=True,
        enable_moe_fp32_combine=False,
        num_nextn_predict_layers=1,
        use_grouped_mm=True,
        pad_token_id=None,
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
        self.rope_parameters = rope_parameters or {"rope_type": "default", "rope_theta": 11158840.0}
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias
        self.pad_token_id = pad_token_id

        # MoE arguments
        self.moe_intermediate_size = moe_intermediate_size
        # vLLM reads the expert intermediate size under the hub config's Tencent-style name
        self.expert_hidden_dim = moe_intermediate_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.num_shared_experts = num_shared_experts
        self.router_scaling_factor = router_scaling_factor
        self.route_norm = route_norm
        self.first_k_dense_replace = first_k_dense_replace
        if mlp_layer_types is None:
            mlp_layer_types = [
                "dense" if layer_idx < first_k_dense_replace else "sparse" for layer_idx in range(num_hidden_layers)
            ]
        self.mlp_layer_types = mlp_layer_types

        self.qk_norm = qk_norm
        self.enable_moe_fp32_combine = enable_moe_fp32_combine
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.use_grouped_mm = use_grouped_mm

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
