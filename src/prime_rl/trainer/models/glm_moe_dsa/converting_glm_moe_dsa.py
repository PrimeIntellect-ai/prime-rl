import torch
from torch import Tensor

from prime_rl.trainer.models.fp8 import quantize_to_fp8_blockwise


def get_max_layer_num(state_dict: dict[str, Tensor]) -> int:
    return max(int(i.split(".")[2]) for i in state_dict.keys() if "model.layers." in i) + 1


def _is_moe_layer(state_dict: dict[str, Tensor], layer_idx: int) -> bool:
    """Check if a layer is an MoE layer by looking for the router gate weight."""
    return f"model.layers.{layer_idx}.mlp.gate.weight" in state_dict


def convert_hf_layer_to_tt(state_dict: dict[str, Tensor], layer_idx: int):
    i = layer_idx

    if not _is_moe_layer(state_dict, i):
        return

    # Router: gate.weight -> router.gate.weight
    state_dict[f"model.layers.{i}.mlp.router.gate.weight"] = state_dict[f"model.layers.{i}.mlp.gate.weight"]
    del state_dict[f"model.layers.{i}.mlp.gate.weight"]

    # Routed experts: fused or per-expert format -> stacked w1/w2/w3
    if f"model.layers.{i}.mlp.experts.gate_up_proj" in state_dict:
        gate_up_proj = state_dict[f"model.layers.{i}.mlp.experts.gate_up_proj"]
        down_proj = state_dict[f"model.layers.{i}.mlp.experts.down_proj"]

        num_experts, fused_dim, dim = gate_up_proj.shape
        moe_dim = fused_dim // 2

        w1 = gate_up_proj[:, :moe_dim, :]
        w3 = gate_up_proj[:, moe_dim:, :]
        w2 = down_proj

        del state_dict[f"model.layers.{i}.mlp.experts.gate_up_proj"]
        del state_dict[f"model.layers.{i}.mlp.experts.down_proj"]
    else:
        num_experts = len([j for j in state_dict.keys() if f"model.layers.{i}.mlp.experts" in j]) // 3
        if num_experts == 0:
            return

        dim, moe_dim = state_dict[f"model.layers.{i}.mlp.experts.0.down_proj.weight"].shape
        dtype = state_dict[f"model.layers.{i}.mlp.experts.0.down_proj.weight"].dtype
        w1 = torch.empty((num_experts, moe_dim, dim), dtype=dtype)
        w2 = torch.empty((num_experts, dim, moe_dim), dtype=dtype)
        w3 = torch.empty((num_experts, moe_dim, dim), dtype=dtype)
        for j in range(num_experts):
            w1[j].copy_(state_dict[f"model.layers.{i}.mlp.experts.{j}.gate_proj.weight"])
            w2[j].copy_(state_dict[f"model.layers.{i}.mlp.experts.{j}.down_proj.weight"])
            w3[j].copy_(state_dict[f"model.layers.{i}.mlp.experts.{j}.up_proj.weight"])

            del state_dict[f"model.layers.{i}.mlp.experts.{j}.gate_proj.weight"]
            del state_dict[f"model.layers.{i}.mlp.experts.{j}.down_proj.weight"]
            del state_dict[f"model.layers.{i}.mlp.experts.{j}.up_proj.weight"]

    state_dict[f"model.layers.{i}.mlp.experts.w1"] = w1
    state_dict[f"model.layers.{i}.mlp.experts.w2"] = w2
    state_dict[f"model.layers.{i}.mlp.experts.w3"] = w3

    # Shared experts
    state_dict[f"model.layers.{i}.mlp.shared_expert.w1"] = state_dict[
        f"model.layers.{i}.mlp.shared_experts.gate_proj.weight"
    ]
    state_dict[f"model.layers.{i}.mlp.shared_expert.w2"] = state_dict[
        f"model.layers.{i}.mlp.shared_experts.down_proj.weight"
    ]
    state_dict[f"model.layers.{i}.mlp.shared_expert.w3"] = state_dict[
        f"model.layers.{i}.mlp.shared_experts.up_proj.weight"
    ]
    del state_dict[f"model.layers.{i}.mlp.shared_experts.gate_proj.weight"]
    del state_dict[f"model.layers.{i}.mlp.shared_experts.down_proj.weight"]
    del state_dict[f"model.layers.{i}.mlp.shared_experts.up_proj.weight"]

    # Expert bias for load balancing
    state_dict[f"model.layers.{i}.mlp.expert_bias"] = state_dict[f"model.layers.{i}.mlp.gate.e_score_correction_bias"]
    del state_dict[f"model.layers.{i}.mlp.gate.e_score_correction_bias"]


def convert_hf_to_tt_moe(state_dict: dict[str, Tensor]):
    num_layers = get_max_layer_num(state_dict)
    for i in range(num_layers):
        convert_hf_layer_to_tt(state_dict, i)


def convert_tt_layer_to_hf(state_dict: dict[str, Tensor], layer_index: int):
    i = layer_index

    # Expert bias
    if f"model.layers.{i}.mlp.expert_bias" in state_dict:
        state_dict[f"model.layers.{i}.mlp.gate.e_score_correction_bias"] = state_dict[
            f"model.layers.{i}.mlp.expert_bias"
        ]
        del state_dict[f"model.layers.{i}.mlp.expert_bias"]
    if f"model.layers.{i}.mlp.tokens_per_expert" in state_dict:
        del state_dict[f"model.layers.{i}.mlp.tokens_per_expert"]

    # Shared experts
    if f"model.layers.{i}.mlp.shared_expert.w1" in state_dict:
        state_dict[f"model.layers.{i}.mlp.shared_experts.gate_proj.weight"] = state_dict[
            f"model.layers.{i}.mlp.shared_expert.w1"
        ]
        state_dict[f"model.layers.{i}.mlp.shared_experts.down_proj.weight"] = state_dict[
            f"model.layers.{i}.mlp.shared_expert.w2"
        ]
        state_dict[f"model.layers.{i}.mlp.shared_experts.up_proj.weight"] = state_dict[
            f"model.layers.{i}.mlp.shared_expert.w3"
        ]

        if state_dict[f"model.layers.{i}.mlp.shared_experts.up_proj.weight"].shape[0] == 1:
            state_dict[f"model.layers.{i}.mlp.shared_experts.up_proj.weight"] = state_dict[
                f"model.layers.{i}.mlp.shared_experts.up_proj.weight"
            ][0]
            state_dict[f"model.layers.{i}.mlp.shared_experts.down_proj.weight"] = state_dict[
                f"model.layers.{i}.mlp.shared_experts.down_proj.weight"
            ][0]
            state_dict[f"model.layers.{i}.mlp.shared_experts.gate_proj.weight"] = state_dict[
                f"model.layers.{i}.mlp.shared_experts.gate_proj.weight"
            ][0]
        del state_dict[f"model.layers.{i}.mlp.shared_expert.w1"]
        del state_dict[f"model.layers.{i}.mlp.shared_expert.w2"]
        del state_dict[f"model.layers.{i}.mlp.shared_expert.w3"]

    # Router
    if f"model.layers.{i}.mlp.router.gate.weight" in state_dict:
        state_dict[f"model.layers.{i}.mlp.gate.weight"] = state_dict[f"model.layers.{i}.mlp.router.gate.weight"]
        del state_dict[f"model.layers.{i}.mlp.router.gate.weight"]

        # Routed experts - convert to per-expert format (compatible with vLLM and transformers)
        w1 = state_dict.pop(f"model.layers.{i}.mlp.experts.w1")  # (num_experts, moe_dim, dim)
        w2 = state_dict.pop(f"model.layers.{i}.mlp.experts.w2")  # (num_experts, dim, moe_dim)
        w3 = state_dict.pop(f"model.layers.{i}.mlp.experts.w3")  # (num_experts, moe_dim, dim)

        num_experts = w1.shape[0]
        for j in range(num_experts):
            state_dict[f"model.layers.{i}.mlp.experts.{j}.gate_proj.weight"] = w1[j]
            state_dict[f"model.layers.{i}.mlp.experts.{j}.down_proj.weight"] = w2[j]
            state_dict[f"model.layers.{i}.mlp.experts.{j}.up_proj.weight"] = w3[j]


def convert_tt_to_hf_moe(state_dict: dict[str, Tensor]):
    num_layers = get_max_layer_num(state_dict)
    for i in range(num_layers):
        convert_tt_layer_to_hf(state_dict, i)


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

    for suffix in ["indexer.wq_b.weight", "indexer.wk.weight"]:
        key = f"{prefix}.self_attn.{suffix}"
        if key in state_dict:
            add_maybe_fp8(key, state_dict[key])
    for suffix in ["indexer.k_norm.weight", "indexer.k_norm.bias", "indexer.weights_proj.weight"]:
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
