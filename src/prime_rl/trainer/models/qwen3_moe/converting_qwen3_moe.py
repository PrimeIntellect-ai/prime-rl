import torch
from torch import Tensor

from prime_rl.trainer.models.conversion_spec import ConversionSpec, QuantizationSpec


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


# NIXL ConversionSpec tables for Qwen3MoE.
#
# Qwen3-235B-A22B-*-FP8 keeps every linear in block-FP8; only the layernorms
# and the router gate remain in bf16. vLLM fuses qkv into ``self_attn.qkv_proj``
# and (for dense layers) gate+up into ``mlp.gate_up_proj``. MoE layers use
# vLLM's FusedMoE ``w13_weight`` / ``w2_weight`` 3D stacked buffers; the
# ``_weight_scale_inv`` scale suffix (vs ``.weight_scale_inv`` on 2D linears)
# matches vLLM's FusedMoE naming convention.
_BASE: tuple[ConversionSpec, ...] = (
    ConversionSpec("input_layernorm.weight", ("input_layernorm.weight",)),
    ConversionSpec("post_attention_layernorm.weight", ("post_attention_layernorm.weight",)),
    ConversionSpec("self_attn.q_norm.weight", ("self_attn.q_norm.weight",)),
    ConversionSpec("self_attn.k_norm.weight", ("self_attn.k_norm.weight",)),
    ConversionSpec(
        "self_attn.qkv_proj.weight",
        ("self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight"),
        quantization=QuantizationSpec(torch.float8_e4m3fn, ".weight_scale_inv"),
    ),
    ConversionSpec(
        "self_attn.o_proj.weight",
        ("self_attn.o_proj.weight",),
        quantization=QuantizationSpec(torch.float8_e4m3fn, ".weight_scale_inv"),
    ),
)


_DENSE: tuple[ConversionSpec, ...] = (
    ConversionSpec(
        "mlp.gate_up_proj.weight",
        ("mlp.gate_proj.weight", "mlp.up_proj.weight"),
        quantization=QuantizationSpec(torch.float8_e4m3fn, ".weight_scale_inv"),
    ),
    ConversionSpec(
        "mlp.down_proj.weight",
        ("mlp.down_proj.weight",),
        quantization=QuantizationSpec(torch.float8_e4m3fn, ".weight_scale_inv"),
    ),
)


_SPARSE: tuple[ConversionSpec, ...] = (
    ConversionSpec("mlp.gate.weight", ("mlp.router.gate.weight",)),
    ConversionSpec(
        "mlp.experts.w13_weight",
        ("mlp.experts.w1", "mlp.experts.w3"),
        cat_dim=1,
        quantization=QuantizationSpec(torch.float8_e4m3fn, "_weight_scale_inv"),
    ),
    ConversionSpec(
        "mlp.experts.w2_weight",
        ("mlp.experts.w2",),
        quantization=QuantizationSpec(torch.float8_e4m3fn, "_weight_scale_inv"),
    ),
)
