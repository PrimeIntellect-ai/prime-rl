
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_rope_utils import rope_config_validation


class AfMoeConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`AfmoeModel`]. It is used to instantiate an
    AFMoE model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of [arcee-ai/Trinity-Mini](https://huggingface.co/arcee-ai/Trinity-Mini).

    AFMoE is an Adaptive Feedforward MoE (Mixture of Experts) model with token-choice routing, shared experts, and a
    hybrid attention mechanism combining sliding window and full attention patterns.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 200192):
            Vocabulary size of the AFMoE model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`AfmoeModel`].
        hidden_size (`int`, *optional*, defaults to 2048):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 6144):
            Dimension of the dense MLP representations.
        moe_intermediate_size (`int`, *optional*, defaults to 1408):
            Intermediate size of the routed expert MLPs.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer decoder.
        num_dense_layers (`int`, *optional*, defaults to 1):
            Number of initial dense layers before MoE layers begin. Layers with index < num_dense_layers will use
            standard dense MLPs instead of MoE.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details, check out [this
            paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to
            `num_attention_heads`.
        head_dim (`int`, *optional*, defaults to 128):
            The dimension of each attention head.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the MLP blocks.
        max_position_embeddings (`int`, *optional*, defaults to 16384):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the RMS normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_parameters (`RopeParameters`, *optional*):
            Dictionary containing the configuration parameters for the RoPE embeddings. The dictionary should contain
            a value for `rope_theta` and optionally parameters used for scaling in case you want to use RoPE
            with longer `max_position_embeddings`.
        num_experts (`int`, *optional*, defaults to 64):
            Number of routed experts in MoE layers.
        num_experts_per_tok (`int`, *optional*, defaults to 6):
            Number of experts to route each token to. This is the top-k value for the token-choice routing.
        num_shared_experts (`int`, *optional*, defaults to 2):
            Number of shared experts that are always activated for all tokens.
        route_scale (`float`, *optional*, defaults to 1.0):
            Scaling factor applied to routing weights.
        global_attn_every_n_layers (`int`, *optional*, defaults to 4):
            The frequency of full attention layers. Every Nth layer will use full attention, while others use sliding
            window attention.
        sliding_window (`int`, *optional*, defaults to 1024):
            Sliding window size for local attention layers.
        layer_types (`list[str]`, *optional*):
            A list that explicitly maps each layer index with its attention type. Each element should be either
            "sliding_attention" or "full_attention". If not provided, it will be automatically generated based on
            `global_attn_every_n_layers`.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        mup_enabled (`bool`, *optional*, defaults to `False`):
            Whether to enable muP (Maximal Update Parametrization) input scaling. When enabled, input embeddings
            are scaled by `sqrt(hidden_size)`.

    Example:
    ```python
    >>> from transformers import AfmoeModel, AfmoeConfig

    >>> # Initializing an AFMoE configuration
    >>> configuration = AfmoeConfig()

    >>> # Initializing a model from the afmoe-small-sft-v1 style configuration
    >>> model = AfmoeModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    """
    model_type = "afmoe"
    keys_to_ignore_at_inference = ["past_key_values"]
    # Default tensor parallel plan for base model `AFMoE`
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.experts.*.gate_proj": "colwise",
        "layers.*.mlp.experts.*.up_proj": "colwise",
        "layers.*.mlp.experts.*.down_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size=200192,
        hidden_size=2048,
        intermediate_size=6144,
        moe_intermediate_size=1408,
        num_hidden_layers=32,
        num_dense_layers=1,
        num_attention_heads=16,
        num_key_value_heads=None,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=16384,
        initializer_range=0.02,
        rms_norm_eps=1e-05,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        rope_parameters=None,
        num_experts=64,
        num_experts_per_tok=6,
        num_shared_experts=2,
        route_scale=1.0,
        global_attn_every_n_layers=4,
        sliding_window=1024,
        layer_types=None,
        attention_dropout=0.0,
        mup_enabled=False,
        norm_topk_prob=True,
        use_qk_norm=False,
        attention_bias=False,
        use_grouped_mm=True,
        load_balance_coeff=1e-3,
        pad_token_id=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_dense_layers = num_dense_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.rope_parameters = rope_parameters
        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, move it to 'rope_type'.
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)

        # MoE specific 
        self.moe_intermediate_size = moe_intermediate_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.num_shared_experts = num_shared_experts
        self.route_scale = route_scale
        self.norm_topk_prob = norm_topk_prob
        self.use_grouped_mm = use_grouped_mm
        self.load_balance_coeff = load_balance_coeff
        self.attention_types = False

        # Attention and normalization specific
        self.use_qk_norm = use_qk_norm
        self.attention_bias = attention_bias

        # Attention specific
        self.attention_dropout = attention_dropout
        self.global_attn_every_n_layers = global_attn_every_n_layers
        self.sliding_window = sliding_window
        self.mup_enabled = mup_enabled
        self.layer_types = layer_types
        if num_key_value_heads is None:
            self.num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            pad_token_id=pad_token_id,
            **kwargs,
        )

