from typing import Generator

import torch
from torch import Tensor
from torch.nn import Module
from vllm.logger import init_logger
from vllm.model_executor.model_loader.utils import process_weights_after_loading

logger = init_logger("vllm.inference.vllm.worker_weight_transfer")


def load_weights_checkpoint(model: Module, state_iter: Generator[tuple[str, torch.Tensor], None, None]) -> None:
    model.load_weights(state_iter)  # type: ignore


def postprocess_weights_checkpoint(model: Module, model_config, device: torch.device) -> None:
    process_weights_after_loading(model, model_config, device)


def build_expert_map(model: Module) -> dict[str, torch.Tensor]:
    """Map FusedMoE module names to global expert indices local to this worker."""
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE

    expert_slices: dict[str, torch.Tensor] = {}
    for module_name, module in model.named_modules():
        if not isinstance(module, FusedMoE):
            continue
        if module._expert_map is None:
            continue

        global_indices = torch.where(module._expert_map >= 0)[0]
        local_indices = module._expert_map[global_indices]
        global_indices = global_indices[local_indices.argsort()]
        expert_slices[module_name] = global_indices
    return expert_slices


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _weight_block_size(module: Module) -> tuple[int, int]:
    block_size = getattr(module, "weight_block_size", None)
    if block_size is None:
        return (128, 128)
    return (int(block_size[0]), int(block_size[1]))


def _slice_scale_dim(tensor: Tensor, dim: int, start: int, size: int, block: int) -> Tensor:
    return tensor.narrow(dim, start // block, _ceil_div(size, block))


def _module_name_for_weight(name: str) -> str | None:
    for suffix in (
        ".w13_weight_scale_inv",
        ".w2_weight_scale_inv",
        ".w13_weight",
        ".w2_weight",
        ".weight_scale_inv",
        ".weight_scale",
        ".weight",
    ):
        if name.endswith(suffix):
            return name.removesuffix(suffix)
    return None


def _slice_qkv_tensor(name: str, module: Module, tensor: Tensor) -> Tensor:
    head_size = module.head_size
    v_head_size = getattr(module, "v_head_size", head_size)

    q_total = module.total_num_heads * head_size
    k_total = module.total_num_kv_heads * head_size

    q_size = module.num_heads * head_size
    k_size = module.num_kv_heads * head_size
    v_size = module.num_kv_heads * v_head_size

    q_start = module.tp_rank * q_size
    kv_rank = module.tp_rank // module.num_kv_head_replicas
    k_start = q_total + kv_rank * k_size
    v_start = q_total + k_total + kv_rank * v_size

    if name.endswith(".weight_scale_inv"):
        block_n, _ = _weight_block_size(module)
        return torch.cat(
            [
                _slice_scale_dim(tensor, 0, q_start, q_size, block_n),
                _slice_scale_dim(tensor, 0, k_start, k_size, block_n),
                _slice_scale_dim(tensor, 0, v_start, v_size, block_n),
            ],
            dim=0,
        ).contiguous()

    return torch.cat(
        [
            tensor.narrow(0, q_start, q_size),
            tensor.narrow(0, k_start, k_size),
            tensor.narrow(0, v_start, v_size),
        ],
        dim=0,
    ).contiguous()


def _slice_row_parallel_tensor(name: str, module: Module, tensor: Tensor) -> Tensor:
    start = module.tp_rank * module.input_size_per_partition
    size = module.input_size_per_partition

    if name.endswith(".weight_scale_inv"):
        _, block_k = _weight_block_size(module)
        return _slice_scale_dim(tensor, 1, start, size, block_k).contiguous()

    return tensor.narrow(1, start, size).contiguous()


def _slice_merged_column_tensor(name: str, module: Module, tensor: Tensor) -> Tensor:
    pieces: list[Tensor] = []
    offset = 0
    block_n, _ = _weight_block_size(module)

    for output_size in module.output_sizes:
        shard_size = output_size // module.tp_size
        start = offset + module.tp_rank * shard_size
        if name.endswith(".weight_scale_inv"):
            pieces.append(_slice_scale_dim(tensor, 0, start, shard_size, block_n))
        else:
            pieces.append(tensor.narrow(0, start, shard_size))
        offset += output_size

    return torch.cat(pieces, dim=0).contiguous()


def _slice_ep_expert_tensor(name: str, expert_slices: dict[str, Tensor], tensor: Tensor) -> Tensor | None:
    for module_name, global_indices in expert_slices.items():
        if name.startswith(f"{module_name}."):
            return tensor[global_indices.to(tensor.device)].contiguous()
    return None


def _try_load_vocab_parallel_weight(name: str, param: Tensor, tensor: Tensor, modules: dict[str, Module]) -> bool:
    from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding

    module_name = _module_name_for_weight(name)
    module = modules.get(module_name or "")
    if not isinstance(module, VocabParallelEmbedding):
        return False

    weight_loader = getattr(param, "weight_loader")
    weight_loader(param, tensor)
    return True


def _slice_tp_tensor(name: str, modules: dict[str, Module], tensor: Tensor) -> Tensor | None:
    from vllm.model_executor.layers.linear import MergedColumnParallelLinear, QKVParallelLinear, RowParallelLinear

    module_name = _module_name_for_weight(name)
    module = modules.get(module_name or "")
    if isinstance(module, QKVParallelLinear):
        return _slice_qkv_tensor(name, module, tensor)
    if isinstance(module, RowParallelLinear):
        return _slice_row_parallel_tensor(name, module, tensor)
    if isinstance(module, MergedColumnParallelLinear):
        return _slice_merged_column_tensor(name, module, tensor)
    return None


@torch.no_grad()
def load_weights_kernel(model: Module, state_iter: Generator[tuple[str, torch.Tensor], None, None]) -> None:
    """Load vLLM kernel-format tensors for TP inference with EP routed experts."""
    params = dict(model.named_parameters())
    modules = dict(model.named_modules())
    expert_slices = build_expert_map(model)

    loaded = 0
    skipped: list[str] = []
    shape_mismatches: list[str] = []

    for name, tensor in state_iter:
        if name not in params:
            skipped.append(name)
            continue

        param = params[name]
        if param.shape != tensor.shape:
            if _try_load_vocab_parallel_weight(name, param, tensor, modules):
                loaded += 1
                continue

            sliced = _slice_ep_expert_tensor(name, expert_slices, tensor)
            if sliced is None:
                sliced = _slice_tp_tensor(name, modules, tensor)

            if sliced is None:
                shape_mismatches.append(f"{name}: param={list(param.shape)} != received={list(tensor.shape)}")
                continue

            tensor = sliced
            if param.shape != tensor.shape:
                shape_mismatches.append(f"{name}: param={list(param.shape)} != received={list(tensor.shape)}")
                continue

        param.copy_(tensor)
        loaded += 1

    if shape_mismatches:
        raise ValueError(f"Kernel weight transfer had {len(shape_mismatches)} shape mismatches: {shape_mismatches}")
    if skipped:
        raise ValueError(f"Kernel weight transfer skipped {len(skipped)} weights not found in model: {skipped}")
    logger.debug(f"Kernel weight transfer copied {loaded} weights in-place")


@torch.no_grad()
def update_mla_absorbed_weights(model: Module) -> None:
    """Recompute MLA absorbed KV weights after in-place kv_b_proj updates."""
    from vllm.model_executor.layers.quantization.utils.quant_utils import get_and_maybe_dequant_weights

    for name, module in model.named_modules():
        has_absorbed_weights = hasattr(module, "W_UV") or hasattr(module, "W_UK_T")
        if not has_absorbed_weights or not hasattr(module, "kv_b_proj"):
            continue

        if hasattr(module, "W_UV"):
            out_dtype = module.W_UV.dtype
        else:
            out_dtype = torch.bfloat16

        kv_b_proj_weight = get_and_maybe_dequant_weights(module.kv_b_proj, out_dtype=out_dtype).T
        kv_b_proj_weight = kv_b_proj_weight.view(
            module.kv_lora_rank,
            module.num_heads,
            module.qk_nope_head_dim + module.v_head_dim,
        )
        w_uk, w_uv = kv_b_proj_weight.split([module.qk_nope_head_dim, module.v_head_dim], dim=-1)

        if hasattr(module, "W_UV"):
            module.W_UV.copy_(w_uv.transpose(0, 1))
        if hasattr(module, "W_UK_T"):
            module.W_UK_T.copy_(w_uk.permute(1, 2, 0))

        logger.debug(f"Updated MLA absorbed weights for module {name}")


def postprocess_weights_kernel(model: Module, _model_config, _device: torch.device) -> None:
    update_mla_absorbed_weights(model)
