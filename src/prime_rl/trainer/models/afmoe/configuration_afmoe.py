from transformers.configuration_utils import PretrainedConfig, layer_type_validation
from transformers.modeling_rope_utils import rope_config_validation
from transformers.utils import logging

logger = logging.get_logger(__name__)

class AfmoeConfig(PretrainedConfig):
    """Configuration for AFMoE."""
    model_type = "afmoe"
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        num_hidden_layers: int = 32,
        vocab_size: int = 200192,
        hidden_size: int = 2048,
        intermediate_size: int = 6144,
        moe_intermediate_size=1408,
        num_dense_layers=1,
        num_attention_heads=16,
        num_key_value_heads=None,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=16384,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        num_experts=64,
        num_experts_per_tok=6,
        num_shared_experts=2,
        num_expert_groups=1,
        num_limited_groups=1,
        score_func="sigmoid",
        route_norm=True,
        route_scale=1.0,
        score_before_experts: bool = False,
        load_balance_coeff: float | None = None,
        use_grouped_mm: bool = True,
        global_attn_every_n_layers=4,
        sliding_window=1024,
        mup_enabled=False,
        layer_types=None,
        attention_dropout: float = 0.0,
        n_group: int = 1,
        topk_group: int = 1,
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
        # MoE specific
        self.moe_intermediate_size = moe_intermediate_size
        self.num_experts_per_tok = num_experts_per_tok
        self.n_group = n_group
        self.topk_group = topk_group
        self.num_experts = num_experts
        self.num_shared_experts = num_shared_experts
        self.num_expert_groups = num_expert_groups
        self.num_limited_groups = num_limited_groups
        self.score_func = score_func
        self.route_norm = route_norm
        self.route_scale = route_scale
        self.score_before_experts = score_before_experts
        self.load_balance_coeff = load_balance_coeff
        self.use_grouped_mm = use_grouped_mm
        # Attention specific
        self.attention_dropout = attention_dropout
        self.global_attn_every_n_layers = global_attn_every_n_layers
        self.sliding_window = sliding_window
        self.layer_types = layer_types
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention" if bool((i + 1) % global_attn_every_n_layers) else "full_attention" for i in range(self.num_hidden_layers)
            ]
        layer_type_validation(self.layer_types)

        # muP specific
        self.mup_enabled = mup_enabled

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        # Validate rope configs
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = ["AfmoeConfig"]
