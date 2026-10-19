from transformers.configuration_utils import PretrainedConfig


class PrimeMoeConfig(PretrainedConfig):
    r"""
    Configuration of the pretraining MoE (`PrimeMoeForCausalLM`): GLM-4.5's architecture (attention with optional QK
    norm, sigmoid-scored token-choice experts with a selection bias, shared experts, leading dense layers) plus
    Gated Residual connections and null experts. The fields shared with `Glm4MoeConfig` have the same names.

    Args:
        vocab_size (`int`): Vocabulary size.
        hidden_size (`int`): Width of the residual stream.
        intermediate_size (`int`): Hidden size of the dense MLP layers.
        num_hidden_layers (`int`): Number of decoder layers.
        num_attention_heads (`int`): Number of query heads.
        num_key_value_heads (`int`): Number of key/value heads (grouped-query attention).
        head_dim (`int`, *optional*): Dimension of each attention head. Defaults to `hidden_size // num_attention_heads`.
        partial_rotary_factor (`float`): Share of each head's dimensions that RoPE rotates.
        hidden_act (`str`): Activation of the gated MLPs.
        max_position_embeddings (`int`): Maximum sequence length.
        initializer_range (`float`): Unused by prime-rl's initialization; kept for Hugging Face compatibility.
        rms_norm_eps (`float`): Epsilon of the RMS norms.
        tie_word_embeddings (`bool`): Whether the LM head shares the input embedding.
        rope_theta (`float`): RoPE base period.
        rope_scaling (`dict`, *optional*): RoPE scaling configuration.
        attention_bias (`bool`): Whether the attention projections have biases.
        attention_dropout (`float`): Attention dropout (unused in training).
        moe_intermediate_size (`int`): Hidden size of each routed and shared expert.
        num_experts_per_tok (`int`): Experts each token selects (`top_k`), null experts included.
        n_shared_experts (`int`): Number of shared experts; their hidden sizes add up.
        n_routed_experts (`int`): Number of routed experts.
        routed_scaling_factor (`float`): Scale of the routed experts' output.
        first_k_dense_replace (`int`): Number of leading dense layers.
        norm_topk_prob (`bool`): Whether the selected experts' scores are normalized to sum to 1.
        use_qk_norm (`bool`): Whether queries and keys are RMS-normalized.
        hc_count (`int`): Number of residual streams. Above 1, Gated Residual connections (Qwen3.8-Next) replace the
            pre-norms and the final norm.
        hc_lowrank (`int`, *optional*): Rank of the Gated Residual read gates; required when `hc_count` is above 1.
        n_null_experts (`int`): Zero-output experts the router can select next to the routed experts (AdaMoE); a
            token that selects them uses fewer routed experts.
    """

    model_type = "prime_moe"

    def __init__(
        self,
        vocab_size=151552,
        hidden_size=4096,
        intermediate_size=10944,
        num_hidden_layers=46,
        num_attention_heads=96,
        partial_rotary_factor=0.5,
        num_key_value_heads=8,
        head_dim=None,
        hidden_act="silu",
        max_position_embeddings=131072,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        moe_intermediate_size=1408,
        num_experts_per_tok=8,
        n_shared_experts=1,
        n_routed_experts=128,
        routed_scaling_factor=1.0,
        first_k_dense_replace=1,
        norm_topk_prob=True,
        use_qk_norm=False,
        hc_count=1,
        hc_lowrank=None,
        n_null_experts=0,
        pad_token_id=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.partial_rotary_factor = partial_rotary_factor
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_attention_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.pad_token_id = pad_token_id

        self.moe_intermediate_size = moe_intermediate_size
        self.num_experts_per_tok = num_experts_per_tok
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.first_k_dense_replace = first_k_dense_replace
        self.norm_topk_prob = norm_topk_prob
        self.use_qk_norm = use_qk_norm

        self.hc_count = hc_count
        self.hc_lowrank = hc_lowrank
        self.n_null_experts = n_null_experts

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
