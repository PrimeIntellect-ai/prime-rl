import warnings

from transformers.configuration_utils import PretrainedConfig


class GlmMoeDsaConfig(PretrainedConfig):
    r"""
    Configuration class for the GLM-5 (GlmMoeDsa) model.

    GLM-5 uses DeepSeek V3.2-style Multi-head Latent Attention (MLA) with LoRA-compressed
    KV projections, combined with Mixture-of-Experts feed-forward layers.

    Args:
        vocab_size (`int`, defaults to 154880):
            Vocabulary size.
        hidden_size (`int`, defaults to 6144):
            Dimension of the hidden representations.
        intermediate_size (`int`, defaults to 12288):
            Dimension of the dense MLP representations.
        moe_intermediate_size (`int`, defaults to 2048):
            Dimension of the MoE expert representations.
        num_hidden_layers (`int`, defaults to 78):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, defaults to 64):
            Number of attention heads.
        num_key_value_heads (`int`, defaults to 64):
            Number of key-value heads (after kv_b_proj up-projection).
        n_shared_experts (`int`, defaults to 1):
            Number of shared experts.
        n_routed_experts (`int`, defaults to 256):
            Number of routed experts.
        routed_scaling_factor (`float`, defaults to 2.5):
            Scaling factor for routed experts.
        kv_lora_rank (`int`, defaults to 512):
            Rank of the LoRA compression for key-value projections.
        q_lora_rank (`int`, defaults to 2048):
            Rank of the LoRA compression for query projections.
        qk_rope_head_dim (`int`, defaults to 64):
            Dimension of query/key heads that use rotary position embeddings.
        v_head_dim (`int`, defaults to 256):
            Dimension of value heads.
        qk_nope_head_dim (`int`, defaults to 192):
            Dimension of query/key heads that don't use rotary position embeddings.
        n_group (`int`, defaults to 1):
            Number of groups for routed experts.
        topk_group (`int`, defaults to 1):
            Number of selected groups per token.
        num_experts_per_tok (`int`, defaults to 8):
            Number of active experts per token.
        first_k_dense_replace (`int`, defaults to 3):
            Number of initial layers that use dense MLP instead of MoE.
        norm_topk_prob (`bool`, defaults to `True`):
            Whether to normalize the top-k routing probabilities.
        hidden_act (`str`, defaults to `"silu"`):
            Activation function in the feed-forward layers.
        max_position_embeddings (`int`, defaults to 202752):
            Maximum sequence length.
        initializer_range (`float`, defaults to 0.02):
            Standard deviation for weight initialization.
        rms_norm_eps (`float`, defaults to 1e-5):
            Epsilon for RMS normalization.
        rope_interleave (`bool`, defaults to `True`):
            Whether to use interleaved (non-neox) RoPE style.
        attention_bias (`bool`, defaults to `False`):
            Whether to use bias in attention projections.
        attention_dropout (`float`, defaults to 0.0):
            Dropout for attention probabilities.
        index_n_heads (`int`, defaults to 32):
            Number of heads used by the sparse indexer.
        index_head_dim (`int`, defaults to 128):
            Head dimension used by the sparse indexer.
        indexer_rope_interleave (`bool`, defaults to `True`):
            Whether to use interleaved RoPE style in the sparse indexer.
        index_topk (`int`, defaults to 2048):
            Number of top tokens selected by the sparse indexer.
        topk_method (`str`, defaults to `"noaux_tc"`):
            MoE routing top-k method used by GLM-5 checkpoints.
        use_grouped_mm (`bool`, defaults to `True`):
            Whether to use grouped matrix multiplication for MoE.
    """

    model_type = "glm_moe_dsa"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"num_local_experts": "n_routed_experts"}

    base_model_tp_plan = {
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
        "layers": (["hidden_states"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size=154880,
        hidden_size=6144,
        intermediate_size=12288,
        moe_intermediate_size=2048,
        num_hidden_layers=78,
        num_attention_heads=64,
        num_key_value_heads=64,
        n_shared_experts=1,
        n_routed_experts=256,
        routed_scaling_factor=2.5,
        kv_lora_rank=512,
        q_lora_rank=2048,
        qk_rope_head_dim=64,
        v_head_dim=256,
        qk_nope_head_dim=192,
        n_group=1,
        topk_group=1,
        num_experts_per_tok=8,
        first_k_dense_replace=3,
        norm_topk_prob=True,
        hidden_act="silu",
        max_position_embeddings=202752,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        tie_word_embeddings=False,
        rope_interleave=True,
        rope_parameters=None,
        rope_theta=10000.0,
        mlp_layer_types=None,
        attention_bias=False,
        attention_dropout=0.0,
        index_n_heads=32,
        index_head_dim=128,
        indexer_rope_interleave=True,
        index_topk=2048,
        topk_method="noaux_tc",
        use_grouped_mm=True,
        pad_token_id=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads

        # MLA dimensions
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.head_dim = qk_rope_head_dim

        # MoE
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.n_group = n_group
        self.topk_group = topk_group
        self.num_experts_per_tok = num_experts_per_tok
        self.first_k_dense_replace = first_k_dense_replace
        self.norm_topk_prob = norm_topk_prob

        # MLP layer types: per-layer dense/sparse dispatch
        if mlp_layer_types is None:
            mlp_layer_types = ["dense"] * min(first_k_dense_replace, num_hidden_layers) + ["sparse"] * max(
                0, num_hidden_layers - first_k_dense_replace
            )
        self.mlp_layer_types = mlp_layer_types

        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_interleave = rope_interleave
        self.rope_parameters = rope_parameters
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.indexer_rope_interleave = indexer_rope_interleave
        self.index_topk = index_topk
        self.topk_method = topk_method
        self.use_grouped_mm = use_grouped_mm
        self.pad_token_id = pad_token_id

        if not self.use_grouped_mm:
            warnings.warn("not using grouped mm for moe is very slow, should only be used for debugging")

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
