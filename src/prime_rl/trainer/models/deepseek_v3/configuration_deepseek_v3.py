from transformers.configuration_utils import PretrainedConfig


class DeepseekV3Config(PretrainedConfig):
    model_type = "deepseek_v3"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=129280,
        hidden_size=7168,
        intermediate_size=18432,
        num_hidden_layers=61,
        num_attention_heads=16,
        num_key_value_heads=16,
        max_position_embeddings=163840,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        hidden_act="silu",
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=0,
        eos_token_id=1,
        tie_word_embeddings=False,
        first_k_dense_replace=3,
        q_lora_rank=None,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        moe_intermediate_size=2048,
        n_routed_experts=64,
        n_shared_experts=1,
        num_experts_per_tok=8,
        moe_layer_freq=1,
        n_group=1,
        topk_group=1,
        num_cycles=1,
        # num_experts_per_tok_k=0,
        scoring_func="sigmoid",
        norm_topk_prob=True,
        routed_scaling_factor=2.5,
        seq_scope=None,
        long_context_remap=None,
        rope_ver="v1",
        ep_size=1,
        num_nextn_predict_layers=1,
        load_balance_coeff=None,
        use_grouped_mm=True,
        rope_interleave=True,
        rope_parameters: dict | None = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache

        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.head_dim = qk_rope_head_dim  ### !!! Not actual head dim.
        self.v_head_dim = v_head_dim

        self.moe_intermediate_size = moe_intermediate_size
        self.n_routed_experts = n_routed_experts
        self.num_local_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_layer_freq = moe_layer_freq
        self.first_k_dense_replace = first_k_dense_replace
        self.n_group = n_group
        self.topk_group = topk_group
        self.num_cycles = num_cycles
        # self.num_experts_per_tok_k = num_experts_per_tok_k
        self.scoring_func = scoring_func
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.seq_scope = seq_scope
        self.long_context_remap = long_context_remap
        self.rope_ver = rope_ver
        self.ep_size = ep_size
        self.num_nextn_predict_layers = num_nextn_predict_layers

        self.load_balance_coeff = load_balance_coeff
        self.use_grouped_mm = use_grouped_mm
        self.rope_interleave = rope_interleave

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        assert self.qk_nope_head_dim + self.qk_rope_head_dim == self.qk_head_dim
        assert self.n_routed_experts % self.n_group == 0  # required for TopK router

        if hasattr(self, "rope_scaling") and isinstance(self.rope_scaling, dict):
            rope_conf = self.rope_scaling
            self.rope_type = rope_conf.get("rope_type", rope_conf.get("type"))
            for key in ["beta_fast", "beta_slow"]:
                # convert to float
                if key in rope_conf:
                    rope_conf[key] = 1.0 * rope_conf[key]
        else:
            self.rope_type = "default"


        self.rope_parameters = self.rope_scaling or self.rope_parameters
        self.rope_parameters = self.rope_parameters if self.rope_parameters is not None else {}
        
    @property
    def rope_total_dim(self):
        return self.num_attention_heads * self.qk_rope_head_dim


__all__ = ["DeepseekV3Config"]
