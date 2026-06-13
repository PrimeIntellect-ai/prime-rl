"""HF<->prime weight conversion for GLM-MoE-DSA.

GLM-MoE-DSA's hf<->tt MoE conversion is identical to GLM-4 MoE (router rename,
expert-bias rename, fused-or-per-expert routed experts, shared experts with the
same backward leading-dim squeeze, and dropping the prime-only
``tokens_per_expert`` buffer), so :func:`conversion_chain` reuses
:func:`prime_rl.trainer.models.glm4_moe.converting_glm4_moe.glm_moe_layer_ops`.

The attention / MLA (DSA) weights (e.g. ``self_attn.q_a_proj.weight``,
``self_attn.kv_a_proj_with_mqa.weight``) are *not* renamed in the hf<->tt
direction — they pass through untouched. The ``convert_tt_layer_to_vllm_kernel``
fusion below is a separate, lossy terminal path (FP8/MLA) and is intentionally
not encoded in the declarative chain.
"""

from __future__ import annotations

import torch
from torch import Tensor

from prime_rl.trainer.models.conversion_ops import ConvOp
from prime_rl.trainer.models.fp8 import quantize_to_fp8_blockwise
from prime_rl.trainer.models.glm4_moe.converting_glm4_moe import glm_moe_layer_ops


def conversion_chain(config) -> list[ConvOp]:
    ops: list[ConvOp] = []
    for i in range(config.num_hidden_layers):
        ops.extend(glm_moe_layer_ops(i))
    return ops


def convert_tt_layer_to_vllm_kernel(
    state_dict: dict[str, Tensor],
    layer_idx: int,
    quantize_fp8: bool = False,
) -> dict[str, Tensor]:
    """Convert a single GLM layer from PrimeRL format to vLLM kernel format."""
    out: dict[str, Tensor] = {}
    prefix = f"model.layers.{layer_idx}"

    def add(name: str, tensor: Tensor) -> None:
        out[name] = tensor

    def add_maybe_fp8(name: str, tensor: Tensor) -> None:
        if quantize_fp8 and tensor.ndim == 2:
            fp8_weight, scale = quantize_to_fp8_blockwise(tensor)
            out[name] = fp8_weight
            scale_name = name.removesuffix(".weight") + ".weight_scale_inv"
            out[scale_name] = scale
            return
        out[name] = tensor

    for name in [f"{prefix}.input_layernorm.weight", f"{prefix}.post_attention_layernorm.weight"]:
        if name in state_dict:
            add(name, state_dict[name])

    q_a_key = f"{prefix}.self_attn.q_a_proj.weight"
    kv_a_key = f"{prefix}.self_attn.kv_a_proj_with_mqa.weight"
    if q_a_key in state_dict and kv_a_key in state_dict:
        add_maybe_fp8(
            f"{prefix}.self_attn.fused_qkv_a_proj.weight", torch.cat([state_dict[q_a_key], state_dict[kv_a_key]], dim=0)
        )

    for suffix in ["q_a_layernorm.weight", "kv_a_layernorm.weight"]:
        key = f"{prefix}.self_attn.{suffix}"
        if key in state_dict:
            add(key, state_dict[key])

    for suffix in ["q_b_proj.weight", "kv_b_proj.weight", "o_proj.weight"]:
        key = f"{prefix}.self_attn.{suffix}"
        if key in state_dict:
            add_maybe_fp8(key, state_dict[key])

    indexer_wq_b_key = f"{prefix}.self_attn.indexer.wq_b.weight"
    if indexer_wq_b_key in state_dict:
        add_maybe_fp8(indexer_wq_b_key, state_dict[indexer_wq_b_key])

    indexer_wk_key = f"{prefix}.self_attn.indexer.wk.weight"
    indexer_weights_proj_key = f"{prefix}.self_attn.indexer.weights_proj.weight"
    if indexer_wk_key in state_dict and indexer_weights_proj_key in state_dict:
        add(
            f"{prefix}.self_attn.indexer.wk_weights_proj.weight",
            torch.cat([state_dict[indexer_wk_key], state_dict[indexer_weights_proj_key]], dim=0),
        )
    for suffix in ["indexer.k_norm.weight", "indexer.k_norm.bias"]:
        key = f"{prefix}.self_attn.{suffix}"
        if key in state_dict:
            add(key, state_dict[key])

    gate_key = f"{prefix}.mlp.gate_proj.weight"
    up_key = f"{prefix}.mlp.up_proj.weight"
    down_key = f"{prefix}.mlp.down_proj.weight"
    if gate_key in state_dict and up_key in state_dict:
        add_maybe_fp8(f"{prefix}.mlp.gate_up_proj.weight", torch.cat([state_dict[gate_key], state_dict[up_key]], dim=0))
        if down_key in state_dict:
            add_maybe_fp8(f"{prefix}.mlp.down_proj.weight", state_dict[down_key])

    router_key = f"{prefix}.mlp.router.gate.weight"
    if router_key in state_dict:
        add(f"{prefix}.mlp.gate.weight", state_dict[router_key])
    expert_bias_key = f"{prefix}.mlp.expert_bias"
    if expert_bias_key in state_dict:
        add(f"{prefix}.mlp.gate.e_score_correction_bias", state_dict[expert_bias_key])

    w1_key = f"{prefix}.mlp.experts.w1"
    w2_key = f"{prefix}.mlp.experts.w2"
    w3_key = f"{prefix}.mlp.experts.w3"
    if w1_key in state_dict and w2_key in state_dict and w3_key in state_dict:
        w1 = state_dict[w1_key]
        w2 = state_dict[w2_key]
        w3 = state_dict[w3_key]
        w13 = torch.cat([w1, w3], dim=1)

        if quantize_fp8:
            w13_fp8: list[Tensor] = []
            w13_scales: list[Tensor] = []
            w2_fp8: list[Tensor] = []
            w2_scales: list[Tensor] = []
            for expert_idx in range(w1.shape[0]):
                expert_w13_fp8, expert_w13_scales = quantize_to_fp8_blockwise(w13[expert_idx])
                expert_w2_fp8, expert_w2_scales = quantize_to_fp8_blockwise(w2[expert_idx])
                w13_fp8.append(expert_w13_fp8)
                w13_scales.append(expert_w13_scales)
                w2_fp8.append(expert_w2_fp8)
                w2_scales.append(expert_w2_scales)

            out[f"{prefix}.mlp.experts.w13_weight"] = torch.stack(w13_fp8)
            out[f"{prefix}.mlp.experts.w13_weight_scale_inv"] = torch.stack(w13_scales)
            out[f"{prefix}.mlp.experts.w2_weight"] = torch.stack(w2_fp8)
            out[f"{prefix}.mlp.experts.w2_weight_scale_inv"] = torch.stack(w2_scales)
        else:
            out[f"{prefix}.mlp.experts.w13_weight"] = w13
            out[f"{prefix}.mlp.experts.w2_weight"] = w2

    sw1_key = f"{prefix}.mlp.shared_expert.w1"
    sw2_key = f"{prefix}.mlp.shared_expert.w2"
    sw3_key = f"{prefix}.mlp.shared_expert.w3"
    if sw1_key in state_dict and sw2_key in state_dict and sw3_key in state_dict:
        sw1 = state_dict[sw1_key]
        sw2 = state_dict[sw2_key]
        sw3 = state_dict[sw3_key]
        if sw1.ndim == 3:
            sw1 = sw1.squeeze(0)
            sw2 = sw2.squeeze(0)
            sw3 = sw3.squeeze(0)
        add_maybe_fp8(f"{prefix}.mlp.shared_experts.gate_up_proj.weight", torch.cat([sw1, sw3], dim=0))
        add_maybe_fp8(f"{prefix}.mlp.shared_experts.down_proj.weight", sw2)

    return out
