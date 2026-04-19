from dataclasses import dataclass

import torch
from torch import Tensor

from prime_rl.trainer.models.fp8 import BLOCK_SIZE, ceil_div, fp8_block_quantize, grouped_fp8_block_quantize


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


# --------------------------------------------------------------------------- #
# tt (prime-rl) → vLLM FP8 kernel format.
#
# The mapping is deterministic (see mapper.py at the repo root). The only
# branch is dense-layer vs sparse-layer; every destination tensor is either
# a direct copy, a concat of two sources, or an FP8 block-quantize of the
# (possibly concatted) source.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _Spec:
    dst: str  # destination suffix (after "model.layers.{i}.")
    sources: tuple[str, ...]  # source suffixes (after "model.layers.{i}.")
    cat_dim: int = 0
    quantize: bool = False


_BASE: tuple[_Spec, ...] = (
    _Spec("input_layernorm.weight", ("input_layernorm.weight",)),
    _Spec("post_attention_layernorm.weight", ("post_attention_layernorm.weight",)),
    _Spec("self_attn.q_a_layernorm.weight", ("self_attn.q_a_layernorm.weight",)),
    _Spec("self_attn.kv_a_layernorm.weight", ("self_attn.kv_a_layernorm.weight",)),
    _Spec(
        "self_attn.fused_qkv_a_proj.weight",
        ("self_attn.q_a_proj.weight", "self_attn.kv_a_proj_with_mqa.weight"),
        quantize=True,
    ),
    _Spec("self_attn.q_b_proj.weight", ("self_attn.q_b_proj.weight",), quantize=True),
    _Spec("self_attn.kv_b_proj.weight", ("self_attn.kv_b_proj.weight",), quantize=True),
    _Spec("self_attn.o_proj.weight", ("self_attn.o_proj.weight",), quantize=True),
    _Spec("self_attn.indexer.wq_b.weight", ("self_attn.indexer.wq_b.weight",), quantize=True),
    _Spec("self_attn.indexer.wk.weight", ("self_attn.indexer.wk.weight",), quantize=True),
    _Spec("self_attn.indexer.k_norm.weight", ("self_attn.indexer.k_norm.weight",)),
    _Spec("self_attn.indexer.k_norm.bias", ("self_attn.indexer.k_norm.bias",)),
    _Spec("self_attn.indexer.weights_proj.weight", ("self_attn.indexer.weights_proj.weight",)),
)


_SPARSE: tuple[_Spec, ...] = (
    _Spec("mlp.gate.weight", ("mlp.router.gate.weight",)),
    _Spec("mlp.gate.e_score_correction_bias", ("mlp.expert_bias",)),
    _Spec(
        "mlp.shared_experts.gate_up_proj.weight",
        ("mlp.shared_expert.w1", "mlp.shared_expert.w3"),
        quantize=True,
    ),
    _Spec("mlp.shared_experts.down_proj.weight", ("mlp.shared_expert.w2",), quantize=True),
    _Spec("mlp.experts.w13_weight", ("mlp.experts.w1", "mlp.experts.w3"), cat_dim=1, quantize=True),
    _Spec("mlp.experts.w2_weight", ("mlp.experts.w2",), quantize=True),
)


_DENSE: tuple[_Spec, ...] = (
    _Spec("mlp.gate_up_proj.weight", ("mlp.gate_proj.weight", "mlp.up_proj.weight"), quantize=True),
    _Spec("mlp.down_proj.weight", ("mlp.down_proj.weight",), quantize=True),
)


def _scale_shape(shape: torch.Size) -> tuple[int, ...]:
    if len(shape) == 3:
        g, r, c = shape
        return (g, ceil_div(r, BLOCK_SIZE), ceil_div(c, BLOCK_SIZE))
    r, c = shape
    return (ceil_div(r, BLOCK_SIZE), ceil_div(c, BLOCK_SIZE))


def convert_tt_layer_to_vllm_kernel(
    state_dict: dict[str, Tensor],
    layer_idx: int,
    out_buffers: dict[str, Tensor] | None = None,
) -> dict[str, Tensor]:
    """Convert a single GLM MoE DSA layer from prime-rl format to vLLM FP8 kernel format.

    The mapping is deterministic; the only branch is dense-layer vs sparse-layer.
    ``out_buffers`` is always populated correctly — any destination the caller
    didn't pre-register is allocated here with the right shape/dtype, then
    filled in place.
    """
    prefix = f"model.layers.{layer_idx}"
    is_sparse = f"{prefix}.mlp.router.gate.weight" in state_dict
    specs = _BASE + (_SPARSE if is_sparse else _DENSE)

    if out_buffers is None:
        out_buffers = {}

    for spec in specs:
        dst = f"{prefix}.{spec.dst}"
        srcs = [state_dict[f"{prefix}.{s}"] for s in spec.sources]
        # Shared-expert tensors live as (1, M, N); drop the leading singleton so
        # they slot into the 2D destinations above.
        srcs = [s.squeeze(0) if s.ndim == 3 and s.shape[0] == 1 else s for s in srcs]
        tensor = srcs[0] if len(srcs) == 1 else torch.cat(srcs, dim=spec.cat_dim)

        dtype = torch.float8_e4m3fn if spec.quantize else tensor.dtype
        if dst not in out_buffers:
            out_buffers[dst] = torch.empty(tensor.shape, dtype=dtype, device=tensor.device)
        buf = out_buffers[dst]

        if spec.quantize:
            scale_name = dst + "_scale_inv"
            if scale_name not in out_buffers:
                out_buffers[scale_name] = torch.empty(
                    _scale_shape(tensor.shape), dtype=torch.float32, device=tensor.device
                )
            sf = out_buffers[scale_name]
            quantize = grouped_fp8_block_quantize if tensor.ndim == 3 else fp8_block_quantize
            quantize(tensor, out=buf, sf=sf)
        else:
            buf.copy_(tensor)

    return out_buffers
