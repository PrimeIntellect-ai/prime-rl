"""PrimeRL -> vLLM kernel-format conversion for GLM-4 MoE.

The MoE/MLP/norm handling mirrors GLM-MoE-DSA's converter (their MoE layout is
identical); only attention differs — GLM-4 MoE uses standard GQA with separate
q/k/v projections (optionally biased), fused into vLLM's ``qkv_proj``.
"""

from __future__ import annotations

import torch
from torch import Tensor

from prime_rl.trainer.models.fp8 import quantize_to_vllm_kernel_format


def convert_tt_layer_to_vllm_kernel(
    state_dict: dict[str, Tensor],
    layer_idx: int,
    quantize_fp8: bool = False,
) -> dict[str, Tensor]:
    """Convert a single GLM-4 MoE layer from PrimeRL format to vLLM kernel format."""
    out: dict[str, Tensor] = {}
    prefix = f"model.layers.{layer_idx}"

    def add(name: str, tensor: Tensor) -> None:
        out[name] = tensor

    def add_maybe_fp8(name: str, tensor: Tensor) -> None:
        if quantize_fp8 and tensor.ndim == 2:
            fp8_weight, scale = quantize_to_vllm_kernel_format(tensor)
            out[name] = fp8_weight
            scale_name = name.removesuffix(".weight") + ".weight_scale_inv"
            out[scale_name] = scale
            return
        out[name] = tensor

    for name in [f"{prefix}.input_layernorm.weight", f"{prefix}.post_attention_layernorm.weight"]:
        if name in state_dict:
            add(name, state_dict[name])

    q_key = f"{prefix}.self_attn.q_proj.weight"
    k_key = f"{prefix}.self_attn.k_proj.weight"
    v_key = f"{prefix}.self_attn.v_proj.weight"
    if q_key in state_dict and k_key in state_dict and v_key in state_dict:
        add_maybe_fp8(
            f"{prefix}.self_attn.qkv_proj.weight",
            torch.cat([state_dict[q_key], state_dict[k_key], state_dict[v_key]], dim=0),
        )
    q_bias = f"{prefix}.self_attn.q_proj.bias"
    k_bias = f"{prefix}.self_attn.k_proj.bias"
    v_bias = f"{prefix}.self_attn.v_proj.bias"
    if q_bias in state_dict and k_bias in state_dict and v_bias in state_dict:
        add(
            f"{prefix}.self_attn.qkv_proj.bias",
            torch.cat([state_dict[q_bias], state_dict[k_bias], state_dict[v_bias]], dim=0),
        )

    for suffix in ["q_norm.weight", "k_norm.weight"]:
        key = f"{prefix}.self_attn.{suffix}"
        if key in state_dict:
            add(key, state_dict[key])

    o_key = f"{prefix}.self_attn.o_proj.weight"
    if o_key in state_dict:
        add_maybe_fp8(o_key, state_dict[o_key])

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
        w13 = torch.cat([w1, state_dict[w3_key]], dim=1)

        if quantize_fp8:
            w13_fp8: list[Tensor] = []
            w13_scales: list[Tensor] = []
            w2_fp8: list[Tensor] = []
            w2_scales: list[Tensor] = []
            for expert_idx in range(w1.shape[0]):
                expert_w13_fp8, expert_w13_scales = quantize_to_vllm_kernel_format(w13[expert_idx])
                expert_w2_fp8, expert_w2_scales = quantize_to_vllm_kernel_format(w2[expert_idx])
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
