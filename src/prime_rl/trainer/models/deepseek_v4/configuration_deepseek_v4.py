from typing import Any

from transformers.configuration_utils import PretrainedConfig

DEEPSEEK_V4_LAYER_TYPES = (
    "sliding_attention",
    "compressed_sparse_attention",
    "heavily_compressed_attention",
)

DEEPSEEK_V4_MLP_LAYER_TYPES = ("hash_moe", "moe")


class DeepseekV4Config(PretrainedConfig):
    """Configuration for DeepSeek-V4 (Flash / Pro) models.

    DeepSeek-V4 differs from V3 in three structural ways, all controlled from here:

    1. The residual stream is `hc_mult` parallel streams tied together by
       manifold-constrained hyper-connections (mHC), governed by `hc_mult`,
       `hc_sinkhorn_iters` and `hc_eps`.
    2. Attention is a per-layer mix of compressed variants selected by `layer_types`,
       with the per-type compression rate given by `compress_rates` and a Lightning
       Indexer sized by `index_n_heads` / `index_head_dim` / `index_topk`.
    3. The MLP schedule in `mlp_layer_types` bootstraps the first few layers with a
       frozen hash router before switching to standard top-k routed MoE.

    Args:
        vocab_size: Vocabulary size.
        hidden_size: Dimension of the hidden representations.
        moe_intermediate_size: Intermediate dimension of each expert. Also aliased as
            `intermediate_size` (the shared expert reads that name).
        num_hidden_layers: Number of decoder layers.
        num_attention_heads: Number of attention query heads.
        num_key_value_heads: Number of key/value heads. V4 is MQA, so this is 1.
        head_dim: Dimension of each attention head.
        q_lora_rank: Rank of the query down-projection.
        partial_rotary_factor: Fraction of `head_dim` that gets RoPE. Defaults to
            `qk_rope_head_dim / head_dim` if `qk_rope_head_dim` is given, else 64/512.
        qk_rope_head_dim: Legacy way to express `partial_rotary_factor`. Only read when
            `partial_rotary_factor` is not given; the derived value is always stored as
            the `qk_rope_head_dim` attribute.
        rope_theta: RoPE base for the main self-attention rotary.
        compress_rope_theta: RoPE base for the compressed attention branches.
        rope_parameters: RoPE parameters, nested under the `main` and `compress` rope
            types. A flat dict is broadcast into both.
        rope_scaling: Legacy alias for `rope_parameters`.
        max_position_embeddings: Maximum sequence length.
        sliding_window: Local window size used by every attention block's
            sliding-window branch.
        o_groups: Number of head groups in the grouped output projection.
        o_lora_rank: Per-group intermediate dimension of the grouped output projection.
        layer_types: Per-layer attention schedule, entries drawn from
            `DEEPSEEK_V4_LAYER_TYPES`. Defaults to two heavily-compressed bootstrap
            layers followed by an alternating heavy/sparse interleave.
        compress_rates: Per-layer-type compression rate.
        index_n_heads: Number of Lightning Indexer query heads.
        index_head_dim: Lightning Indexer head dimension.
        index_topk: Number of compressed entries the Lightning Indexer keeps per query.
        num_experts_per_tok: Number of routed experts activated per token.
        n_routed_experts: Total number of routed experts.
        n_shared_experts: Number of always-on shared experts.
        scoring_func: Router activation, one of `sqrtsoftplus`, `softmax`, `sigmoid`.
        norm_topk_prob: Whether to renormalize the top-k routing probabilities.
        routed_scaling_factor: Scaling factor applied to the routed expert output.
        mlp_layer_types: Per-layer MLP schedule, entries drawn from
            `DEEPSEEK_V4_MLP_LAYER_TYPES`. Defaults to `num_hash_layers` hash-routed
            layers followed by standard MoE layers.
        num_hash_layers: Number of leading `hash_moe` layers used to build the
            `mlp_layer_types` default.
        swiglu_limit: Clamp applied to the expert gate/up pre-activations.
        hc_mult: Number of parallel residual streams carried by mHC.
        hc_sinkhorn_iters: Sinkhorn-Knopp iterations used to project the mHC combine
            matrix onto the doubly-stochastic manifold.
        hc_eps: Numerical floor for the Sinkhorn-Knopp normalization.
        hidden_act: Activation function of the MLPs.
        initializer_range: Standard deviation of the weight initializer.
        rms_norm_eps: Epsilon of the RMS normalization layers.
        use_cache: Whether to return past key/values.
        tie_word_embeddings: Whether to tie the input and output embeddings.
        attention_bias: Whether the attention projections carry a bias.
        mlp_bias: Whether the MLP projections carry a bias.
        attention_dropout: Dropout ratio on the attention probabilities.
        output_router_logits: Whether to return the router logits.
        router_aux_loss_coef: Coefficient of the router auxiliary loss.
        router_jitter_noise: Jitter noise added to the router inputs during training.
        use_grouped_mm: Whether the routed experts use a grouped matmul kernel.
    """

    model_type = "deepseek_v4"
    keys_to_ignore_at_inference = ["past_key_values"]
    # `num_local_experts` is the name FP8 / EP integrations read; `intermediate_size` is
    # what the shared-expert MLP reads, and V4 only ships `moe_intermediate_size`.
    attribute_map = {
        "num_local_experts": "n_routed_experts",
        "intermediate_size": "moe_intermediate_size",
    }
    # `rope_parameters` is keyed by rope type, not by `layer_types`.
    _rope_type_labels = ("main", "compress")
    # `qk_rope_head_dim` (64) / `head_dim` (512).
    default_partial_rotary_factor = 64 / 512

    def __init__(
        self,
        vocab_size: int = 129280,
        hidden_size: int = 4096,
        moe_intermediate_size: int = 2048,
        num_hidden_layers: int = 43,
        num_attention_heads: int = 64,
        num_key_value_heads: int = 1,
        head_dim: int = 512,
        q_lora_rank: int = 1024,
        partial_rotary_factor: float | None = None,
        qk_rope_head_dim: int | None = None,
        rope_theta: float = 10000.0,
        compress_rope_theta: float = 160000.0,
        rope_parameters: dict[str, Any] | None = None,
        rope_scaling: dict[str, Any] | None = None,
        max_position_embeddings: int = 1048576,
        sliding_window: int = 128,
        o_groups: int = 8,
        o_lora_rank: int = 1024,
        layer_types: list[str] | None = None,
        compress_rates: dict[str, int] | None = None,
        index_n_heads: int = 64,
        index_head_dim: int = 128,
        index_topk: int = 512,
        num_experts_per_tok: int = 6,
        n_routed_experts: int = 256,
        n_shared_experts: int = 1,
        scoring_func: str = "sqrtsoftplus",
        norm_topk_prob: bool = True,
        routed_scaling_factor: float = 1.5,
        mlp_layer_types: list[str] | None = None,
        num_hash_layers: int = 3,
        swiglu_limit: float = 10.0,
        hc_mult: int = 4,
        hc_sinkhorn_iters: int = 20,
        hc_eps: float = 1e-6,
        hidden_act: str = "silu",
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        attention_bias: bool = False,
        mlp_bias: bool = False,
        attention_dropout: float = 0.0,
        output_router_logits: bool = False,
        router_aux_loss_coef: float = 0.001,
        router_jitter_noise: float = 0.0,
        use_grouped_mm: bool = True,
        pad_token_id: int | None = None,
        bos_token_id: int | None = 0,
        eos_token_id: int | list[int] | None = 1,
        **kwargs,
    ):
        raw_rope_parameters = rope_parameters if rope_parameters is not None else rope_scaling

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.max_position_embeddings = max_position_embeddings

        # Attention
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.q_lora_rank = q_lora_rank
        self.rope_theta = rope_theta
        self.compress_rope_theta = compress_rope_theta
        self.sliding_window = sliding_window
        self.o_groups = o_groups
        self.o_lora_rank = o_lora_rank
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.index_topk = index_topk

        self.partial_rotary_factor = (
            partial_rotary_factor
            if partial_rotary_factor is not None
            else (qk_rope_head_dim / head_dim if qk_rope_head_dim is not None else self.default_partial_rotary_factor)
        )
        self.qk_rope_head_dim = int(head_dim * self.partial_rotary_factor)

        self.compress_rates = (
            dict(compress_rates)
            if compress_rates is not None
            else {"compressed_sparse_attention": 4, "heavily_compressed_attention": 128}
        )
        if layer_types is None:
            # Two heavily-compressed bootstrap layers, then alternate sparse / heavy.
            interleave = [
                "compressed_sparse_attention" if i % 2 else "heavily_compressed_attention"
                for i in range(max(num_hidden_layers - 2, 0))
            ]
            layer_types = ["heavily_compressed_attention"] * min(num_hidden_layers, 2) + interleave
        self.layer_types = list(layer_types)

        # MoE
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.scoring_func = scoring_func
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.num_hash_layers = num_hash_layers
        self.swiglu_limit = swiglu_limit
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.router_jitter_noise = router_jitter_noise
        self.use_grouped_mm = use_grouped_mm

        # Manifold-constrained hyper-connections
        self.hc_mult = hc_mult
        self.hc_sinkhorn_iters = hc_sinkhorn_iters
        self.hc_eps = hc_eps

        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.mlp_bias = mlp_bias

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        # Set after the base __init__: its `validate_layer_type` hook is registered at
        # class-decoration time on `PretrainedConfig` (so a subclass override never runs)
        # and only accepts the generic "dense" / "sparse" labels, not V4's hash_moe / moe.
        if mlp_layer_types is None:
            mlp_layer_types = ["hash_moe"] * min(num_hidden_layers, num_hash_layers) + ["moe"] * max(
                0, num_hidden_layers - num_hash_layers
            )
        self.mlp_layer_types = list(mlp_layer_types)
        self.rope_parameters = self._nest_rope_parameters(raw_rope_parameters)
        self.validate_architecture()

    def _nest_rope_parameters(self, raw_rope_parameters: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
        """Split RoPE parameters into the `main` and `compress` sets the rotary reads."""
        rope_parameters = raw_rope_parameters or {}
        if isinstance(rope_parameters.get("main"), dict) and isinstance(rope_parameters.get("compress"), dict):
            return {"main": dict(rope_parameters["main"]), "compress": dict(rope_parameters["compress"])}

        scaling = {key: value for key, value in rope_parameters.items() if key not in ("main", "compress")}
        main = {
            "rope_type": "default",
            "rope_theta": self.rope_theta,
            "partial_rotary_factor": self.partial_rotary_factor,
        }
        compress = {
            **scaling,
            "rope_theta": self.compress_rope_theta,
            "partial_rotary_factor": self.partial_rotary_factor,
        }
        # TODO(deepseek_v4): wire up YaRN for the compress rope type - deferred from step 1
        compress["rope_type"] = "default"
        return {"main": main, "compress": compress}

    def convert_rope_params_to_dict(self, **kwargs):
        # Dead code: `rope_theta`/`rope_scaling` never reach here via `**kwargs`, and
        # `self.rope_parameters` is overwritten by `_nest_rope_parameters` right after
        # `super().__init__()` returns regardless. Kept for parity with the same
        # (also-dead) override in `laguna/configuration_laguna.py`; see TODO.md.
        return kwargs

    def validate_architecture(self) -> None:
        for name, allowed in (
            ("layer_types", DEEPSEEK_V4_LAYER_TYPES),
            ("mlp_layer_types", DEEPSEEK_V4_MLP_LAYER_TYPES),
        ):
            types = getattr(self, name)
            if len(types) != self.num_hidden_layers:
                raise ValueError(
                    f"{name} length ({len(types)}) must equal num_hidden_layers ({self.num_hidden_layers})."
                )
            unknown = sorted({layer_type for layer_type in types if layer_type not in allowed})
            if unknown:
                raise ValueError(f"{name} entries must be one of {allowed}; got {unknown}.")
        for layer_type in set(self.layer_types) - {"sliding_attention"}:
            if layer_type not in self.compress_rates:
                raise ValueError(f"compress_rates is missing a rate for layer type {layer_type!r}.")


__all__ = ["DeepseekV4Config"]
