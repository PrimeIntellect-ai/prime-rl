"""Model-aware cost estimation for sequence packing and load balancing.

The forward-pass FLOPs of a transformer for a single sequence of length ``n`` are
exactly of the form ``c1 * n + c2 * n**2``: every projection, FFN and LM-head term
is linear in ``n`` while attention is quadratic. Both coefficients depend only on
the (fixed) model config, so we compute them once and return a closure that scores
a packed bin as ``c1 * sum(n_i) + c2 * sum(n_i**2)`` over its sequences.

The quadratic term is summed per sequence (``sum(n_i**2)``), never over the packed
length (``sum(n_i)**2``), because packed sequences do not attend across boundaries.

Estimates feed only relative load balancing across data-parallel workers (they are
never reported), so only ordering and additivity matter. The FLOP accounting follows
slime's model-aware packing estimate in spirit.
"""

from collections.abc import Callable, Sequence
from typing import Any


def _text_config(model_config: Any) -> Any:
    """Unwrap a multimodal model's text config, else return the config unchanged."""
    return getattr(model_config, "text_config", model_config)


def _is_mla(config: Any) -> bool:
    return bool(getattr(config, "multi_latent_attention", False) or hasattr(config, "q_lora_rank"))


def _kv_channels(config: Any) -> int:
    return getattr(config, "kv_channels", getattr(config, "head_dim", config.hidden_size // config.num_attention_heads))


def _qkv_projection_flops_per_token(config: Any) -> int:
    """Linear-in-seqlen FLOPs of the Q/K/V projections (per token)."""
    hidden_size = config.hidden_size
    num_attention_heads = config.num_attention_heads
    kv_channels = _kv_channels(config)
    is_mla = _is_mla(config)
    qk_head_dim = getattr(config, "qk_head_dim", 0)
    qk_pos_emb_head_dim = getattr(config, "qk_pos_emb_head_dim", 0)

    if is_mla and getattr(config, "q_lora_rank", None) is not None:
        q_flops = 2 * config.q_lora_rank * (hidden_size + num_attention_heads * (qk_head_dim + qk_pos_emb_head_dim))
    else:
        q_head_dim = (qk_head_dim + qk_pos_emb_head_dim) if is_mla else kv_channels
        q_flops = 2 * hidden_size * num_attention_heads * q_head_dim

    if is_mla and getattr(config, "kv_lora_rank", None) is not None:
        v_head_dim = getattr(config, "v_head_dim", 0)
        kv_flops = 2 * (
            config.kv_lora_rank * (hidden_size + num_attention_heads * (qk_head_dim + v_head_dim))
            + hidden_size * qk_pos_emb_head_dim
        )
    else:
        num_query_groups = getattr(
            config, "num_query_groups", getattr(config, "num_key_value_heads", num_attention_heads)
        )
        kv_flops = 4 * hidden_size * num_query_groups * kv_channels
    return q_flops + kv_flops


def _attention_flops_per_token_squared(config: Any) -> int:
    """Quadratic-in-seqlen FLOPs of the attention scores and values (per token^2)."""
    num_attention_heads = config.num_attention_heads
    kv_channels = _kv_channels(config)
    if _is_mla(config):
        qk_head_dim = getattr(config, "qk_head_dim", 0)
        qk_pos_emb_head_dim = getattr(config, "qk_pos_emb_head_dim", 0)
        v_head_dim = getattr(config, "v_head_dim", kv_channels)
        return num_attention_heads * (qk_head_dim + qk_pos_emb_head_dim) + num_attention_heads * v_head_dim
    return 2 * num_attention_heads * kv_channels


def _ffn_flops_per_token(hidden_size: int, ffn_hidden_size: int) -> int:
    return 6 * hidden_size * ffn_hidden_size


def _dense_ffn_hidden_size(config: Any) -> int:
    return getattr(
        config, "ffn_hidden_size", getattr(config, "intermediate_size", getattr(config, "moe_intermediate_size", 0))
    )


def _moe_ffn_hidden_size(config: Any) -> int:
    """Effective FFN width of an MoE layer: routed experts (topk) plus shared experts."""
    dense_ffn = _dense_ffn_hidden_size(config)
    routed_topk = getattr(config, "moe_router_topk", getattr(config, "num_experts_per_tok", 1))
    moe_ffn = getattr(config, "moe_ffn_hidden_size", getattr(config, "moe_intermediate_size", dense_ffn))
    return moe_ffn * routed_topk + (getattr(config, "moe_shared_expert_intermediate_size", None) or 0)


def _count_dense_and_moe_layers(config: Any) -> tuple[int, int]:
    """Split the model's layers into (dense, moe) counts."""
    num_experts = getattr(config, "num_experts", getattr(config, "n_routed_experts", None))
    if num_experts is None:
        return config.num_hidden_layers, 0

    moe_layer_freq = getattr(config, "moe_layer_freq", None)
    if isinstance(moe_layer_freq, list):
        num_dense = sum(1 for freq in moe_layer_freq if freq == 0)
        num_moe = sum(1 for freq in moe_layer_freq if freq > 0)
    elif isinstance(moe_layer_freq, int):
        num_dense = sum(1 for i in range(config.num_hidden_layers) if i % moe_layer_freq != 0)
        num_moe = config.num_hidden_layers - num_dense
    elif getattr(config, "first_k_dense_replace", None) is not None:
        num_dense = config.first_k_dense_replace
        num_moe = config.num_hidden_layers - num_dense
    else:
        num_dense = 0
        num_moe = config.num_hidden_layers
    return num_dense, num_moe


def _packing_cost_coeffs(config: Any) -> tuple[int, int]:
    """Return the (linear, quadratic) coefficients of the per-sequence forward FLOPs."""
    hidden_size = config.hidden_size
    num_dense_layers, num_moe_layers = _count_dense_and_moe_layers(config)
    qkv_per_token = _qkv_projection_flops_per_token(config)
    attn_per_token_squared = _attention_flops_per_token_squared(config)

    def layer_linear(ffn_hidden_size: int) -> int:
        return qkv_per_token + 2 * hidden_size * hidden_size + _ffn_flops_per_token(hidden_size, ffn_hidden_size)

    linear = (
        num_dense_layers * layer_linear(_dense_ffn_hidden_size(config))
        + num_moe_layers * layer_linear(_moe_ffn_hidden_size(config))
        + 2 * hidden_size * config.vocab_size
    )
    quadratic = (num_dense_layers + num_moe_layers) * attn_per_token_squared
    return linear, quadratic


def build_bin_cost(model_config: Any | None) -> Callable[[Sequence[int]], int]:
    """Build a closure scoring a packed bin by estimated forward compute.

    With ``model_config=None`` the linear/quadratic coefficients are ``(1, 0)``, so
    the cost reduces to the token count and balancing falls back to sequence length.
    """
    if model_config is None:
        linear, quadratic = 1, 0
    else:
        linear, quadratic = _packing_cost_coeffs(_text_config(model_config))

    def bin_cost(seqlens: Sequence[int]) -> int:
        return linear * sum(seqlens) + quadratic * sum(n * n for n in seqlens)

    return bin_cost
