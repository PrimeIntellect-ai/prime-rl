from transformers.configuration_utils import PretrainedConfig, layer_type_validation
from transformers.modeling_rope_utils import rope_config_validation


class GptOssConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`GptOssModel`]. It is used to instantiate a
    GPT OSS model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the GPT-OSS-120B model.
    
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    
    Args:
        vocab_size (`int`, *optional*, defaults to 200000):
            Vocabulary size of the GPT OSS model (o200k_harmony tokenizer).
        hidden_size (`int`, *optional*, defaults to 8192):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 28672):
            Dimension of the MLP representations in dense layers.
        num_hidden_layers (`int`, *optional*, defaults to 36):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 64):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            Number of key_value heads for Grouped Query Attention (GQA). GPT OSS uses GQA with group size 8.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 131072):
            The maximum sequence length that this model might ever be used with (128k context).
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-5):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings.
        attention_bias (`bool`, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        use_local_attention (`bool`, *optional*, defaults to `True`):
            Whether to use locally banded sparse attention pattern (alternating with dense attention).
        local_attention_window (`int`, *optional*, defaults to 4096):
            The window size for locally banded sparse attention.
        moe_intermediate_size (`int`, *optional*, defaults to 3584):
            Intermediate size of the routed expert.
        num_experts_per_tok (`int`, *optional*, defaults to 4):
            Number of selected experts per token.
        num_experts (`int`, *optional*, defaults to 128):
            Number of routed experts (128 for 120B model, 32 for 20B model).
        norm_topk_prob (`bool`, *optional*, defaults to `True`):
            Whether to normalize the topk probabilities.
        output_router_logits (`bool`, *optional*, defaults to `False`):
            Whether or not the router logits should be returned by the model.
        router_aux_loss_coef (`float`, *optional*, defaults to 0.001):
            The aux loss factor for the total loss.
        decoder_sparse_step (`int`, *optional*, defaults to 1):
            The frequency of the MoE layer (1 means all layers are MoE).
        use_grouped_mm (`bool`, *optional*, defaults to `True`):
            Whether to use grouped matrix multiplication for MoE.
    
    ```python
    >>> from transformers import GptOssModel, GptOssConfig
    >>> # Initializing a GPT OSS style configuration
    >>> configuration = GptOssConfig()
    >>> # Initializing a model from the GPT-OSS-120B style configuration
    >>> model = GptOssModel(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "gpt_oss"
    keys_to_ignore_at_inference = ["past_key_values"]

    # Default tensor parallel plan for base model `GptOss`
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
        vocab_size=201088,
        hidden_size=2880,
        intermediate_size=2880,
        num_hidden_layers=36,
        num_attention_heads=64,
        num_key_value_heads=8,
        hidden_act="silu",
        max_position_embeddings=16384,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        use_local_attention=True,
        local_attention_window=4096,
        sliding_window=128,
        moe_intermediate_size=3584,
        num_experts_per_tok=4,
        num_experts=64,
        norm_topk_prob=True,
        output_router_logits=False,
        router_aux_loss_coef=0.001,
        decoder_sparse_step=1,
        load_balance_coeff=None,
        use_grouped_mm=True,
        layer_types=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
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
        
        # Attention pattern settings
        self.use_local_attention = use_local_attention
        self.local_attention_window = local_attention_window if use_local_attention else None
        self.sliding_window = sliding_window
        self.layer_types = layer_types
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention" if bool((i + 1) % 2) else "full_attention" for i in range(self.num_hidden_layers)
            ]
        layer_type_validation(self.layer_types, self.num_hidden_layers)
        
        
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)

        # MoE arguments
        self.moe_intermediate_size = moe_intermediate_size
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts
        self.num_local_experts = num_experts
        self.norm_topk_prob = norm_topk_prob
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.decoder_sparse_step = decoder_sparse_step
        self.load_balance_coeff = load_balance_coeff
        self.use_grouped_mm = use_grouped_mm

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )