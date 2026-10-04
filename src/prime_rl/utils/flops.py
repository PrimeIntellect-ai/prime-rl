from typing import Any


def forward_flops(config: Any) -> tuple[int, int]:
    """Matmul FLOPs of one forward pass, as ``(per token, per query-key pair)``.

    A sequence of ``n`` tokens costs ``linear * n + quadratic * pairs``, where ``pairs`` is
    ``n * n`` with full attention and about ``n * n / 2`` with causal attention.
    ``linear`` is twice the active matmul parameters: attention projections, the dense or
    routed + shared expert MLPs, the routers and the LM head.
    """
    config = getattr(config, "text_config", config)
    hidden = config.hidden_size
    heads = config.num_attention_heads
    layers = config.num_hidden_layers

    if getattr(config, "kv_lora_rank", None) is not None:  # MLA
        qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        v_head_dim = config.v_head_dim
        if config.q_lora_rank is None:
            q = hidden * heads * qk_head_dim
        else:
            q = config.q_lora_rank * (hidden + heads * qk_head_dim)
        kv = hidden * (config.kv_lora_rank + config.qk_rope_head_dim) + config.kv_lora_rank * heads * (
            config.qk_nope_head_dim + v_head_dim
        )
    else:
        qk_head_dim = v_head_dim = getattr(config, "head_dim", None) or hidden // heads
        q = hidden * heads * qk_head_dim
        kv = 2 * hidden * config.num_key_value_heads * qk_head_dim
    o = heads * v_head_dim * hidden

    intermediate = getattr(config, "intermediate_size", None) or config.moe_intermediate_size
    top_k = getattr(config, "num_experts_per_tok", None)
    if top_k:
        num_experts = getattr(config, "n_routed_experts", None) or getattr(config, "num_experts", None) or 0
        layer_types = getattr(config, "mlp_layer_types", None)
        num_dense = layer_types.count("dense") if layer_types else getattr(config, "first_k_dense_replace", 0)
        moe_intermediate = getattr(config, "moe_intermediate_size", None) or intermediate
        num_shared = getattr(config, "n_shared_experts", None) or getattr(config, "num_shared_experts", None) or 0
        shared_intermediate = (
            getattr(config, "moe_shared_expert_intermediate_size", None)
            or getattr(config, "shared_expert_intermediate_size", None)
            or num_shared * moe_intermediate
        )
        moe_mlp = 3 * hidden * (top_k * moe_intermediate + shared_intermediate) + num_experts * hidden
    else:
        num_dense, moe_mlp = layers, 0
    mlp = num_dense * 3 * hidden * intermediate + (layers - num_dense) * moe_mlp

    active_params = layers * (q + kv + o) + mlp + config.vocab_size * hidden
    return 2 * active_params, 2 * layers * heads * (qk_head_dim + v_head_dim)
