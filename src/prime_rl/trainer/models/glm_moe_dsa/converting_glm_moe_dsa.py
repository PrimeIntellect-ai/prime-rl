import torch
from torch import Tensor


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


def _quantize_to_fp8_blockwise(weight: Tensor, block_size: int = 128) -> tuple[Tensor, Tensor]:
    """Quantize a 2D weight tensor to FP8 e4m3 with block-wise scaling."""
    rows, cols = weight.shape
    br = bc = block_size
    pad_r = (br - rows % br) % br
    pad_c = (bc - cols % bc) % bc
    if pad_r > 0 or pad_c > 0:
        padded = torch.zeros(rows + pad_r, cols + pad_c, dtype=weight.dtype, device=weight.device)
        padded[:rows, :cols] = weight
    else:
        padded = weight.clone()
    pr, pc = padded.shape
    blocks = padded.reshape(pr // br, br, pc // bc, bc).permute(0, 2, 1, 3)
    max_abs = blocks.float().abs().amax(dim=(2, 3))
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    scale = (max_abs / fp8_max).clamp(min=1e-12)
    blocks_fp8 = (blocks.float() / scale[:, :, None, None]).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
    return blocks_fp8.permute(0, 2, 1, 3).reshape(pr, pc)[:rows, :cols].contiguous(), scale.float().contiguous()


def convert_tt_layer_to_vllm_kernel(
    state_dict: dict[str, Tensor],
    layer_idx: int,
    quantize_fp8: bool = False,
) -> dict[str, Tensor]:
    """Convert a single layer from PrimeRL format to vLLM kernel format.

    Handles: fusing q_a_proj+kv_a_proj_with_mqa, gate+up projections,
    stacking MoE experts, and optional FP8 quantization.
    """
    out: dict[str, Tensor] = {}
    p = f"model.layers.{layer_idx}"

    def add(name: str, tensor: Tensor) -> None:
        out[name] = tensor

    def add_maybe_fp8(name: str, tensor: Tensor) -> None:
        if quantize_fp8 and tensor.ndim == 2:
            fp8_w, scale = _quantize_to_fp8_blockwise(tensor.cuda())
            out[name] = fp8_w
            scale_name = (
                name[: -len(".weight")] + ".weight_scale_inv" if name.endswith(".weight") else name + "_scale_inv"
            )
            out[scale_name] = scale
        else:
            out[name] = tensor

    # Norms
    for key in [f"{p}.input_layernorm.weight", f"{p}.post_attention_layernorm.weight"]:
        if key in state_dict:
            add(key, state_dict[key])

    # Attention: fuse q_a_proj + kv_a_proj_with_mqa -> fused_qkv_a_proj
    q_a_key = f"{p}.self_attn.q_a_proj.weight"
    kv_a_key = f"{p}.self_attn.kv_a_proj_with_mqa.weight"
    if q_a_key in state_dict and kv_a_key in state_dict:
        fused = torch.cat([state_dict[q_a_key], state_dict[kv_a_key]], dim=0)
        add_maybe_fp8(f"{p}.self_attn.fused_qkv_a_proj.weight", fused)

    for suffix in ["q_a_layernorm.weight", "kv_a_layernorm.weight"]:
        key = f"{p}.self_attn.{suffix}"
        if key in state_dict:
            add(key, state_dict[key])

    for suffix in ["q_b_proj.weight", "kv_b_proj.weight", "o_proj.weight"]:
        key = f"{p}.self_attn.{suffix}"
        if key in state_dict:
            add_maybe_fp8(key, state_dict[key])

    # Indexer (sparse MLA)
    for suffix in ["indexer.wq_b.weight", "indexer.wk.weight"]:
        key = f"{p}.self_attn.{suffix}"
        if key in state_dict:
            add_maybe_fp8(key, state_dict[key])
    for suffix in ["indexer.k_norm.weight", "indexer.k_norm.bias", "indexer.weights_proj.weight"]:
        key = f"{p}.self_attn.{suffix}"
        if key in state_dict:
            add(key, state_dict[key])

    # Dense MLP: fuse gate_proj + up_proj -> gate_up_proj
    gate_key = f"{p}.mlp.gate_proj.weight"
    up_key = f"{p}.mlp.up_proj.weight"
    if gate_key in state_dict and up_key in state_dict:
        gate_up = torch.cat([state_dict[gate_key], state_dict[up_key]], dim=0)
        add_maybe_fp8(f"{p}.mlp.gate_up_proj.weight", gate_up)
        add_maybe_fp8(f"{p}.mlp.down_proj.weight", state_dict[f"{p}.mlp.down_proj.weight"])

    # MoE: router
    router_key = f"{p}.mlp.router.gate.weight"
    if router_key in state_dict:
        add(f"{p}.mlp.gate.weight", state_dict[router_key])
    expert_bias_key = f"{p}.mlp.expert_bias"
    if expert_bias_key in state_dict:
        add(f"{p}.mlp.gate.e_score_correction_bias", state_dict[expert_bias_key])

    # MoE: routed experts w1+w3 -> w13, w2
    w1_key = f"{p}.mlp.experts.w1"
    if w1_key in state_dict:
        w1 = state_dict[w1_key].cuda()
        w3 = state_dict[f"{p}.mlp.experts.w3"].cuda()
        w2 = state_dict[f"{p}.mlp.experts.w2"].cuda()
        w13 = torch.cat([w1, w3], dim=1)
        n_experts = w1.shape[0]

        if quantize_fp8:
            w13_fp8, w13_s, w2_fp8, w2_s = [], [], [], []
            for j in range(n_experts):
                f8, s = _quantize_to_fp8_blockwise(w13[j])
                w13_fp8.append(f8)
                w13_s.append(s)
                f8, s = _quantize_to_fp8_blockwise(w2[j])
                w2_fp8.append(f8)
                w2_s.append(s)
            out[f"{p}.mlp.experts.w13_weight"] = torch.stack(w13_fp8)
            out[f"{p}.mlp.experts.w13_weight_scale_inv"] = torch.stack(w13_s)
            out[f"{p}.mlp.experts.w2_weight"] = torch.stack(w2_fp8)
            out[f"{p}.mlp.experts.w2_weight_scale_inv"] = torch.stack(w2_s)
        else:
            out[f"{p}.mlp.experts.w13_weight"] = w13
            out[f"{p}.mlp.experts.w2_weight"] = w2

    # MoE: shared experts w1+w3 -> gate_up_proj, w2 -> down_proj
    sw1_key = f"{p}.mlp.shared_expert.w1"
    if sw1_key in state_dict:
        sw1 = state_dict[sw1_key].cuda()
        sw3 = state_dict[f"{p}.mlp.shared_expert.w3"].cuda()
        sw2 = state_dict[f"{p}.mlp.shared_expert.w2"].cuda()
        if sw1.dim() == 3:
            sw1, sw3, sw2 = sw1.squeeze(0), sw3.squeeze(0), sw2.squeeze(0)
        add_maybe_fp8(f"{p}.mlp.shared_experts.gate_up_proj.weight", torch.cat([sw1, sw3], dim=0))
        add_maybe_fp8(f"{p}.mlp.shared_experts.down_proj.weight", sw2)

    return out
