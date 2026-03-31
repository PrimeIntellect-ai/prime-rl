from transformers.configuration_utils import PretrainedConfig


class NemotronHConfig(PretrainedConfig):
    """Configuration for NemotronH (Nemotron-3-Super-120B-A12B) hybrid Mamba-Transformer-MoE model.

    The model architecture is defined by `layers_block_type`, a list where each element is
    one of "mamba", "attention", or "moe". The 120B model uses a pattern like MEMEMEM*EMEMEMEM*...
    (M=Mamba-2, E=LatentMoE, *=Attention) repeated across 88 layers.

    Args:
        vocab_size: Vocabulary size.
        hidden_size: Dimension of the hidden representations.
        layers_block_type: Explicit list of layer types. Determines num_hidden_layers.
        tie_word_embeddings: Whether to tie input/output embeddings.
        pad_token_id: Padding token id.
        bos_token_id: Beginning of sequence token id.
        eos_token_id: End of sequence token id.
        num_attention_heads: Number of attention heads.
        num_key_value_heads: Number of key/value heads for GQA.
        head_dim: Dimension of each attention head.
        max_position_embeddings: Maximum sequence length.
        attention_bias: Whether to use bias in attention layers.
        intermediate_size: MLP intermediate dimension (for shared expert).
        mlp_hidden_act: Activation function for MLP layers.
        mlp_bias: Whether to use bias in MLP layers.
        use_mamba_kernels: Whether to use fast mamba CUDA kernels.
        ssm_state_size: Mamba state space dimension.
        mamba_num_heads: Number of mamba heads.
        mamba_n_groups: Number of groups in mamba.
        mamba_head_dim: Dimension per mamba head.
        mamba_d_conv: Convolution kernel size.
        mamba_expand: Expansion factor.
        mamba_hidden_act: Activation for mamba layers.
        mamba_dt_min: Time step minimum.
        mamba_dt_max: Time step maximum.
        mamba_dt_limit: Time step limits.
        mamba_dt_init_floor: Floor for time step initialization.
        mamba_conv_bias: Whether to use bias in mamba convolution.
        mamba_proj_bias: Whether to use bias in mamba projections.
        mamba_chunk_size: Chunk size for mamba processing.
        n_routed_experts: Number of routed experts.
        n_shared_experts: Number of shared experts.
        moe_intermediate_size: Expert intermediate dimension.
        moe_shared_expert_intermediate_size: Shared expert intermediate dimension.
        moe_latent_size: Latent projection size (None means use hidden_size).
        num_experts_per_tok: Top-k routing parameter.
        routed_scaling_factor: Scaling factor for routed expert outputs.
        n_group: Number of groups for expert routing.
        topk_group: Top-k group parameter.
        norm_topk_prob: Whether to normalize top-k probabilities.
        use_bias: Global bias setting.
        initializer_range: Standard deviation for weight initialization.
        layer_norm_epsilon: Epsilon for layer normalization.
        residual_in_fp32: Whether to keep residuals in fp32.
        rescale_prenorm_residual: Whether to rescale pre-norm residuals.
        load_balance_coeff: Auxiliary-loss-free load balancing coefficient.
        use_grouped_mm: Whether to use grouped MM for experts.
    """

    model_type = "nemotron_h"
    keys_to_ignore_at_inference = ["past_key_values"]

    PATTERN_MAP = {"M": "mamba", "E": "moe", "*": "attention"}
    REVERSE_PATTERN_MAP = {"mamba": "M", "moe": "E", "attention": "*"}

    def __init__(
        self,
        # General
        vocab_size=131072,
        hidden_size=4096,
        layers_block_type=None,
        num_hidden_layers=None,
        tie_word_embeddings=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        # Attention
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        max_position_embeddings=4096,
        attention_bias=False,
        # MLP
        intermediate_size=21504,
        mlp_hidden_act="relu2",
        mlp_bias=False,
        # Mamba
        use_mamba_kernels=True,
        ssm_state_size=128,
        mamba_num_heads=128,
        mamba_n_groups=8,
        mamba_head_dim=64,
        mamba_d_conv=4,
        mamba_expand=2,
        mamba_hidden_act="silu",
        mamba_dt_min=0.001,
        mamba_dt_max=0.1,
        mamba_dt_limit=(0.0, float("inf")),
        mamba_dt_init_floor=1e-4,
        mamba_conv_bias=True,
        mamba_proj_bias=False,
        mamba_chunk_size=128,
        # MoE
        n_routed_experts=8,
        n_shared_experts=1,
        moe_intermediate_size=7688,
        moe_shared_expert_intermediate_size=7688,
        moe_latent_size=None,
        num_experts_per_tok=2,
        routed_scaling_factor=1.0,
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
        # General training
        use_bias=False,
        initializer_range=0.02,
        layer_norm_epsilon=1e-5,
        residual_in_fp32=False,
        rescale_prenorm_residual=True,
        # PrimeRL training features
        load_balance_coeff=None,
        use_grouped_mm=True,
        # RoPE
        rope_theta=10000.0,
        rope_scaling=None,
        **kwargs,
    ):
        # Handle hybrid_override_pattern for backward compatibility
        if "hybrid_override_pattern" in kwargs:
            pattern = kwargs.pop("hybrid_override_pattern")
            if layers_block_type is None:
                layers_block_type = [self.PATTERN_MAP[c] for c in pattern]

        if layers_block_type is None:
            layers_block_type = ["mamba", "moe", "attention", "moe"]

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.layers_block_type = layers_block_type

        # Attention config
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads is not None else num_attention_heads
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.attention_bias = attention_bias
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling

        # MLP config
        self.intermediate_size = intermediate_size
        self.mlp_hidden_act = mlp_hidden_act
        self.mlp_bias = mlp_bias

        # Mamba config - stored with HF-compatible attribute names used by NemotronHMamba2Mixer
        self.use_mamba_kernels = use_mamba_kernels
        self.ssm_state_size = ssm_state_size
        self.mamba_num_heads = mamba_num_heads
        self.n_groups = mamba_n_groups
        self.mamba_head_dim = mamba_head_dim
        self.conv_kernel = mamba_d_conv
        self.expand = mamba_expand
        self.mamba_hidden_act = mamba_hidden_act
        self.time_step_min = mamba_dt_min
        self.time_step_max = mamba_dt_max
        self.time_step_limit = mamba_dt_limit
        self.time_step_floor = mamba_dt_init_floor
        self.use_conv_bias = mamba_conv_bias
        self.mamba_proj_bias = mamba_proj_bias
        self.chunk_size = mamba_chunk_size

        # Zamba2MambaMixer compat aliases (read by parent __init__ before NemotronHMamba2Mixer overrides).
        # mamba_expand must give the correct intermediate_size = mamba_num_heads * mamba_head_dim
        # when Zamba2 computes int(mamba_expand * hidden_size); the config's raw "expand" field
        # does not satisfy this for all model sizes (e.g. Nemotron-3-Nano-30B).
        self.mamba_d_state = ssm_state_size
        self.mamba_d_conv = mamba_d_conv
        self.mamba_expand = (mamba_num_heads * mamba_head_dim) / hidden_size
        self.mamba_ngroups = mamba_n_groups
        self.mamba_headdim = mamba_head_dim
        self.n_mamba_heads = mamba_num_heads
        self.use_mem_eff_path = True
        self.add_bias_linear = use_bias

        # MoE config
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.moe_intermediate_size = moe_intermediate_size
        self.moe_shared_expert_intermediate_size = moe_shared_expert_intermediate_size
        self.moe_latent_size = moe_latent_size
        self.num_experts_per_tok = num_experts_per_tok
        self.routed_scaling_factor = routed_scaling_factor
        self.n_group = n_group
        self.topk_group = topk_group
        self.norm_topk_prob = norm_topk_prob

        # General training config
        self.use_bias = use_bias
        self.initializer_range = initializer_range
        self.layer_norm_epsilon = layer_norm_epsilon
        self.residual_in_fp32 = residual_in_fp32
        self.rescale_prenorm_residual = rescale_prenorm_residual

        # PrimeRL training features
        self.load_balance_coeff = load_balance_coeff
        self.use_grouped_mm = use_grouped_mm

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def num_hidden_layers(self) -> int:
        return len(self.layers_block_type)

    @num_hidden_layers.setter
    def num_hidden_layers(self, value):
        if value is not None and value < len(self.layers_block_type):
            self.layers_block_type = self.layers_block_type[:value]


__all__ = ["NemotronHConfig"]
