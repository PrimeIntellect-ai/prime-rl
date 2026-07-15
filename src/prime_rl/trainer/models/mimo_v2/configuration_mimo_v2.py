from transformers.configuration_utils import PretrainedConfig


class MiMoV2Config(PretrainedConfig):
    r"""
    Configuration class for Xiaomi MiMo-V2 models. Instantiating a configuration with the defaults
    will yield a configuration close to that of
    [XiaomiMiMo/MiMo-V2.5](https://huggingface.co/XiaomiMiMo/MiMo-V2.5).

    Field names follow the hub `config.json`. Only the text backbone is modeled; the hub
    checkpoint's vision/audio towers and MTP layers are dropped during weight conversion.

    Args:
        vocab_size (`int`, *optional*, defaults to 152576):
            Vocabulary size of the MiMo-V2 model.
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 16384):
            Dimension of the dense MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 48):
            Number of hidden layers in the Transformer decoder (excluding MTP layers).
        hybrid_layer_pattern (`list[int]`, *optional*):
            Per-layer attention type: 0 = full attention, 1 = sliding-window attention.
            Defaults to all-full if not given.
        num_attention_heads (`int`, *optional*, defaults to 64):
            Number of attention heads on full-attention layers.
        num_key_value_heads (`int`, *optional*, defaults to 4):
            Number of key/value heads on full-attention layers.
        head_dim (`int`, *optional*, defaults to 192):
            Query/key head dimension on full-attention layers.
        v_head_dim (`int`, *optional*, defaults to 128):
            Value head dimension on full-attention layers.
        swa_num_attention_heads (`int`, *optional*, defaults to 64):
            Number of attention heads on sliding-window layers.
        swa_num_key_value_heads (`int`, *optional*, defaults to 8):
            Number of key/value heads on sliding-window layers.
        swa_head_dim (`int`, *optional*, defaults to 192):
            Query/key head dimension on sliding-window layers.
        swa_v_head_dim (`int`, *optional*, defaults to 128):
            Value head dimension on sliding-window layers.
        sliding_window (`int`, *optional*, defaults to 128):
            Sliding window size for SWA layers.
        partial_rotary_factor (`float`, *optional*, defaults to 0.334):
            Fraction of the query/key head dimension that receives rotary embeddings.
        rope_theta (`float`, *optional*, defaults to 10000000.0):
            RoPE base for full-attention layers.
        swa_rope_theta (`float`, *optional*, defaults to 10000.0):
            RoPE base for sliding-window layers.
        attention_value_scale (`float`, *optional*):
            Scalar multiplier applied to the value states before attention.
        add_full_attention_sink_bias (`bool`, *optional*, defaults to `False`):
            Whether full-attention layers carry a per-head attention sink bias.
        add_swa_attention_sink_bias (`bool`, *optional*, defaults to `True`):
            Whether sliding-window layers carry a per-head attention sink bias.
        attention_projection_layout (`str`, *optional*, defaults to `"fused_qkv"`):
            Checkpoint layout of the attention projections. The PrimeRL implementation always uses
            a fused qkv projection module; this field is kept for hub config compatibility.
        moe_layer_freq (`list[int]`, *optional*):
            Per-layer MLP type: 1 = MoE layer, 0 = dense layer. Defaults to all-dense if not given.
        n_routed_experts (`int`, *optional*, defaults to 256):
            Number of routed experts.
        num_experts_per_tok (`int`, *optional*, defaults to 8):
            Number of experts each token is routed to.
        moe_intermediate_size (`int`, *optional*, defaults to 2048):
            Intermediate size of each routed expert.
        norm_topk_prob (`bool`, *optional*, defaults to `True`):
            Whether to normalize the top-k routing weights to sum to 1.
        routed_scaling_factor (`float`, *optional*):
            Scaling factor applied to the routing weights. `None` means 1.0.
        scoring_func (`str`, *optional*, defaults to `"sigmoid"`):
            Router scoring function. Only `"sigmoid"` is supported.
        topk_method (`str`, *optional*, defaults to `"noaux_tc"`):
            Expert selection method. Only `"noaux_tc"` with `n_group=1` is supported.
        layernorm_epsilon (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the rms normalization layers.
    """

    model_type = "mimo_v2"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"num_local_experts": "n_routed_experts", "rms_norm_eps": "layernorm_epsilon"}

    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size=152576,
        hidden_size=4096,
        intermediate_size=16384,
        num_hidden_layers=48,
        hybrid_layer_pattern=None,
        num_attention_heads=64,
        num_key_value_heads=4,
        head_dim=192,
        v_head_dim=128,
        swa_num_attention_heads=64,
        swa_num_key_value_heads=8,
        swa_head_dim=192,
        swa_v_head_dim=128,
        sliding_window=128,
        partial_rotary_factor=0.334,
        rope_theta=10000000.0,
        swa_rope_theta=10000.0,
        attention_bias=False,
        attention_dropout=0.0,
        attention_value_scale=None,
        add_full_attention_sink_bias=False,
        add_swa_attention_sink_bias=True,
        attention_projection_layout="fused_qkv",
        hidden_act="silu",
        max_position_embeddings=1048576,
        initializer_range=0.02,
        layernorm_epsilon=1e-5,
        use_cache=True,
        tie_word_embeddings=False,
        moe_layer_freq=None,
        n_routed_experts=256,
        num_experts_per_tok=8,
        moe_intermediate_size=2048,
        norm_topk_prob=True,
        routed_scaling_factor=None,
        scoring_func="sigmoid",
        topk_method="noaux_tc",
        n_group=1,
        topk_group=1,
        use_grouped_mm=True,
        pad_token_id=None,
        vision_config=None,
        audio_config=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.hybrid_layer_pattern = hybrid_layer_pattern or [0] * num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.v_head_dim = v_head_dim
        self.swa_num_attention_heads = swa_num_attention_heads
        self.swa_num_key_value_heads = swa_num_key_value_heads
        self.swa_head_dim = swa_head_dim
        self.swa_v_head_dim = swa_v_head_dim
        self.sliding_window = sliding_window
        # vLLM reads the window size under the hub config's alternate name
        self.sliding_window_size = sliding_window
        self.partial_rotary_factor = partial_rotary_factor
        self.rope_theta = rope_theta
        self.swa_rope_theta = swa_rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.attention_value_scale = attention_value_scale
        self.add_full_attention_sink_bias = add_full_attention_sink_bias
        self.add_swa_attention_sink_bias = add_swa_attention_sink_bias
        self.attention_projection_layout = attention_projection_layout
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layernorm_epsilon = layernorm_epsilon
        self.use_cache = use_cache
        self.pad_token_id = pad_token_id
        # Multimodal towers are not modeled; kept for hub config and remote-code compatibility
        self.vision_config = vision_config
        self.audio_config = audio_config

        # Used by HF masking utils and renderers to identify the attention type per layer
        self.layer_types = [
            "sliding_attention" if pattern == 1 else "full_attention" for pattern in self.hybrid_layer_pattern
        ]

        # MoE arguments
        self.moe_layer_freq = moe_layer_freq or [0] * num_hidden_layers
        self.n_routed_experts = n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.scoring_func = scoring_func
        self.topk_method = topk_method
        self.n_group = n_group
        self.topk_group = topk_group
        self.use_grouped_mm = use_grouped_mm

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
