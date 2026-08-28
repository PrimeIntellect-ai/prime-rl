from transformers.configuration_utils import PretrainedConfig


class NemotronHConfig(PretrainedConfig):
    model_type = "nemotron_h"

    PATTERN_TO_LAYER_TYPE = {
        "M": "mamba",
        "E": "moe",
        "*": "attention",
    }

    def __init__(
        self,
        vocab_size: int = 131072,
        hidden_size: int = 4096,
        num_hidden_layers: int | None = None,
        hybrid_override_pattern: str | None = None,
        layers_block_type: list[str] | None = None,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        max_position_embeddings: int = 4096,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        intermediate_size: int = 21504,
        mlp_hidden_act: str = "relu2",
        mlp_bias: bool = False,
        ssm_state_size: int = 128,
        mamba_num_heads: int = 128,
        mamba_head_dim: int = 64,
        mamba_hidden_act: str = "silu",
        n_groups: int = 8,
        conv_kernel: int = 4,
        expand: int = 2,
        time_step_min: float = 0.001,
        time_step_max: float = 0.1,
        time_step_limit: tuple[float, float] | list[float] | None = (0.0, float("inf")),
        time_step_floor: float = 1e-4,
        use_conv_bias: bool = True,
        chunk_size: int = 128,
        mamba_proj_bias: bool = False,
        n_routed_experts: int = 8,
        n_shared_experts: int = 1,
        moe_intermediate_size: int = 7688,
        moe_shared_expert_intermediate_size: int = 7688,
        moe_latent_size: int | None = None,
        moe_shared_expert_overlap: bool = True,
        num_experts_per_tok: int = 2,
        routed_scaling_factor: float = 1.0,
        n_group: int = 1,
        topk_group: int = 1,
        norm_topk_prob: bool = True,
        use_bias: bool = False,
        initializer_range: float = 0.02,
        layer_norm_epsilon: float = 1e-5,
        residual_in_fp32: bool = False,
        hidden_dropout: float = 0.0,
        rescale_prenorm_residual: bool = True,
        load_balance_coeff: float | None = None,
        tie_word_embeddings: bool = False,
        pad_token_id: int | None = 0,
        bos_token_id: int | None = 1,
        eos_token_id: int | list[int] | None = 2,
        **kwargs,
    ) -> None:
        if n_group != 1 or topk_group != 1:
            raise NotImplementedError("Nemotron-H grouped expert routing is not supported")
        if n_shared_experts != 1:
            raise NotImplementedError("Nemotron-H requires exactly one shared expert")
        if attention_dropout != 0.0 or hidden_dropout != 0.0:
            raise NotImplementedError("Nemotron-H custom training does not support dropout")

        if layers_block_type is None:
            pattern = hybrid_override_pattern or "ME*E"
            layers_block_type = [self.PATTERN_TO_LAYER_TYPE[token] for token in pattern]
        elif hybrid_override_pattern is not None:
            pattern_layer_types = [self.PATTERN_TO_LAYER_TYPE[token] for token in hybrid_override_pattern]
            if layers_block_type != pattern_layer_types:
                raise ValueError("layers_block_type and hybrid_override_pattern describe different layers")

        if num_hidden_layers is not None and num_hidden_layers != len(layers_block_type):
            raise ValueError(
                f"num_hidden_layers ({num_hidden_layers}) does not match "
                f"the configured layer count ({len(layers_block_type)} layers)"
            )

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.layers_block_type = layers_block_type

        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        self.intermediate_size = intermediate_size
        self.mlp_hidden_act = mlp_hidden_act
        self.mlp_bias = mlp_bias

        self.ssm_state_size = ssm_state_size
        self.mamba_num_heads = mamba_num_heads
        self.mamba_head_dim = mamba_head_dim
        self.mamba_hidden_act = mamba_hidden_act
        self.n_groups = n_groups
        self.conv_kernel = conv_kernel
        self.expand = expand
        self.time_step_min = time_step_min
        self.time_step_max = time_step_max
        self.time_step_limit = time_step_limit
        self.time_step_floor = time_step_floor
        self.use_conv_bias = use_conv_bias
        self.chunk_size = chunk_size
        self.mamba_proj_bias = mamba_proj_bias

        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.moe_intermediate_size = moe_intermediate_size
        self.moe_shared_expert_intermediate_size = moe_shared_expert_intermediate_size
        self.moe_latent_size = moe_latent_size
        self.moe_shared_expert_overlap = moe_shared_expert_overlap
        self.num_experts_per_tok = num_experts_per_tok
        self.routed_scaling_factor = routed_scaling_factor
        self.n_group = n_group
        self.topk_group = topk_group
        self.norm_topk_prob = norm_topk_prob

        self.use_bias = use_bias
        self.initializer_range = initializer_range
        self.layer_norm_epsilon = layer_norm_epsilon
        self.residual_in_fp32 = residual_in_fp32
        self.hidden_dropout = hidden_dropout
        self.rescale_prenorm_residual = rescale_prenorm_residual
        self.load_balance_coeff = load_balance_coeff

        kwargs.pop("use_mamba_kernels", None)
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )

    @property
    def layer_types(self) -> list[str]:
        return self.layers_block_type

    @property
    def num_hidden_layers(self) -> int:
        return len(self.layers_block_type)

    @num_hidden_layers.setter
    def num_hidden_layers(self, value: int | None) -> None:
        if value is None or not hasattr(self, "layers_block_type"):
            return
        if value > len(self.layers_block_type):
            raise ValueError(f"Cannot increase Nemotron-H from {len(self.layers_block_type)} to {value} layers")
        self.layers_block_type = self.layers_block_type[:value]


__all__ = ["NemotronHConfig"]
