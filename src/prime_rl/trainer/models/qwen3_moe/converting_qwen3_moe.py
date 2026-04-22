import torch
from torch import Tensor

from prime_rl.trainer.models.fp8 import quantize_to_fp8_blockwise


def get_max_layer_num(state_dict: dict[str, Tensor]) -> int:
    """Get the maximum number of layers in the model."""
    return max(int(i.split(".")[2]) for i in state_dict.keys() if "model.layers." in i) + 1


def convert_hf_layer_to_tt(state_dict: dict[str, Tensor], layer_idx: int):
    """Convert a layer from HF to TT format in-place."""
    i = layer_idx

    # Check if this is a MoE layer by looking for gate weight
    if f"model.layers.{i}.mlp.gate.weight" not in state_dict:
        return

    state_dict[f"model.layers.{i}.mlp.router.gate.weight"] = state_dict[f"model.layers.{i}.mlp.gate.weight"]
    del state_dict[f"model.layers.{i}.mlp.gate.weight"]

    # Check if this is the new fused format (transformers 5.0+) or old per-expert format
    if f"model.layers.{i}.mlp.experts.gate_up_proj" in state_dict:
        # New fused format: gate_up_proj has shape (num_experts, 2*moe_dim, dim)
        gate_up_proj = state_dict[f"model.layers.{i}.mlp.experts.gate_up_proj"]
        down_proj = state_dict[f"model.layers.{i}.mlp.experts.down_proj"]

        num_experts, fused_dim, dim = gate_up_proj.shape
        moe_dim = fused_dim // 2

        # Split gate_up_proj into w1 (gate) and w3 (up)
        w1 = gate_up_proj[:, :moe_dim, :]  # Gate: (num_experts, moe_dim, dim)
        w3 = gate_up_proj[:, moe_dim:, :]  # Up: (num_experts, moe_dim, dim)
        w2 = down_proj  # Down: (num_experts, dim, moe_dim)

        del state_dict[f"model.layers.{i}.mlp.experts.gate_up_proj"]
        del state_dict[f"model.layers.{i}.mlp.experts.down_proj"]
    else:
        # Old per-expert format
        num_experts = len([j for j in state_dict.keys() if f"model.layers.{i}.mlp.experts" in j]) // 3
        if num_experts == 0:
            return

        dim, moe_dim = state_dict[f"model.layers.{i}.mlp.experts.0.down_proj.weight"].shape
        w1 = torch.empty(
            (num_experts, moe_dim, dim), dtype=state_dict[f"model.layers.{i}.mlp.experts.0.down_proj.weight"].dtype
        )  # Gate
        w2 = torch.empty(
            (num_experts, dim, moe_dim), dtype=state_dict[f"model.layers.{i}.mlp.experts.0.down_proj.weight"].dtype
        )  # Down
        w3 = torch.empty(
            (num_experts, moe_dim, dim), dtype=state_dict[f"model.layers.{i}.mlp.experts.0.down_proj.weight"].dtype
        )  # Up
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


def convert_tt_layer_to_hf(state_dict: dict[str, Tensor], layer_index: int):
    """Convert a layer from TT to HF format in-place."""
    i = layer_index
    if f"model.layers.{i}.mlp.router.gate.weight" not in state_dict:
        return

    # Gate / Router
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


def convert_hf_to_tt_moe(state_dict: dict[str, Tensor]):
    """Convert MoE weights from HF to TT format in-place."""
    num_layers = get_max_layer_num(state_dict)
    for i in range(num_layers):
        convert_hf_layer_to_tt(state_dict, i)


def convert_tt_to_hf_moe(state_dict: dict[str, Tensor]):
    """Convert MoE weights from TT to HF format in-place."""
    num_layers = get_max_layer_num(state_dict)
    for i in range(num_layers):
        convert_tt_layer_to_hf(state_dict, i)


def _emit_weight(out: dict[str, Tensor], name: str, tensor: Tensor, quantize_fp8: bool) -> None:
    """Emit a 2D projection weight, optionally FP8-quantized with block scales."""
    if quantize_fp8:
        fp8_weight, scale = quantize_to_fp8_blockwise(tensor)
        out[name] = fp8_weight
        out[name.removesuffix(".weight") + ".weight_scale_inv"] = scale
    else:
        out[name] = tensor


def _emit_moe_experts(
    out: dict[str, Tensor],
    prefix: str,
    w1: Tensor,
    w2: Tensor,
    w3: Tensor,
    quantize_fp8: bool,
) -> None:
    """Emit fused MoE expert weights in vLLM FusedMoE W13 layout.

    vLLM's DEEPGEMM FP8 MoE path keeps weights in `[gate; up]` (W13) order;
    FLASHINFER paths flip to W31 during `process_weights_after_loading`, so
    inference must select the DEEPGEMM backend (`VLLM_USE_DEEP_GEMM=1`) for
    RL weight reloads to stay valid across broadcasts.
    """
    w13 = torch.cat([w1, w3], dim=1)
    if not quantize_fp8:
        out[f"{prefix}.mlp.experts.w13_weight"] = w13
        out[f"{prefix}.mlp.experts.w2_weight"] = w2
        return

    num_experts = w1.shape[0]
    w13_fp8: list[Tensor] = []
    w13_scales: list[Tensor] = []
    w2_fp8: list[Tensor] = []
    w2_scales: list[Tensor] = []
    for expert_idx in range(num_experts):
        q13, s13 = quantize_to_fp8_blockwise(w13[expert_idx])
        q2, s2 = quantize_to_fp8_blockwise(w2[expert_idx])
        w13_fp8.append(q13)
        w13_scales.append(s13)
        w2_fp8.append(q2)
        w2_scales.append(s2)

    out[f"{prefix}.mlp.experts.w13_weight"] = torch.stack(w13_fp8)
    out[f"{prefix}.mlp.experts.w13_weight_scale_inv"] = torch.stack(w13_scales)
    out[f"{prefix}.mlp.experts.w2_weight"] = torch.stack(w2_fp8)
    out[f"{prefix}.mlp.experts.w2_weight_scale_inv"] = torch.stack(w2_scales)


def _is_dense_layer(state_dict: dict[str, Tensor], layer_idx: int) -> bool:
    """Dense MLP layers have a fused gate_proj; MoE layers route through mlp.router."""
    return f"model.layers.{layer_idx}.mlp.gate_proj.weight" in state_dict


def convert_tt_layer_to_vllm_kernel(
    state_dict: dict[str, Tensor],
    layer_idx: int,
    quantize_fp8: bool = False,
) -> dict[str, Tensor]:
    """Convert a single Qwen3MoE layer from PrimeRL format to vLLM kernel format."""
    prefix = f"model.layers.{layer_idx}"
    out: dict[str, Tensor] = {
        f"{prefix}.input_layernorm.weight": state_dict[f"{prefix}.input_layernorm.weight"],
        f"{prefix}.post_attention_layernorm.weight": state_dict[f"{prefix}.post_attention_layernorm.weight"],
        f"{prefix}.self_attn.q_norm.weight": state_dict[f"{prefix}.self_attn.q_norm.weight"],
        f"{prefix}.self_attn.k_norm.weight": state_dict[f"{prefix}.self_attn.k_norm.weight"],
    }

    qkv = torch.cat(
        [
            state_dict[f"{prefix}.self_attn.q_proj.weight"],
            state_dict[f"{prefix}.self_attn.k_proj.weight"],
            state_dict[f"{prefix}.self_attn.v_proj.weight"],
        ],
        dim=0,
    )
    _emit_weight(out, f"{prefix}.self_attn.qkv_proj.weight", qkv, quantize_fp8)
    _emit_weight(out, f"{prefix}.self_attn.o_proj.weight", state_dict[f"{prefix}.self_attn.o_proj.weight"], quantize_fp8)

    if _is_dense_layer(state_dict, layer_idx):
        gate_up = torch.cat(
            [state_dict[f"{prefix}.mlp.gate_proj.weight"], state_dict[f"{prefix}.mlp.up_proj.weight"]],
            dim=0,
        )
        _emit_weight(out, f"{prefix}.mlp.gate_up_proj.weight", gate_up, quantize_fp8)
        _emit_weight(out, f"{prefix}.mlp.down_proj.weight", state_dict[f"{prefix}.mlp.down_proj.weight"], quantize_fp8)
    else:
        out[f"{prefix}.mlp.gate.weight"] = state_dict[f"{prefix}.mlp.router.gate.weight"]
        _emit_moe_experts(
            out,
            prefix,
            state_dict[f"{prefix}.mlp.experts.w1"],
            state_dict[f"{prefix}.mlp.experts.w2"],
            state_dict[f"{prefix}.mlp.experts.w3"],
            quantize_fp8,
        )

    return out
