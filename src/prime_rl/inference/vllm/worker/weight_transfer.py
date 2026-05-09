from contextlib import contextmanager
from typing import Generator, Iterable

import torch
from torch.nn import Module
from vllm.logger import init_logger
from vllm.model_executor.model_loader.reload import finalize_layerwise_reload, initialize_layerwise_reload
from vllm.model_executor.model_loader.utils import process_weights_after_loading

logger = init_logger("vllm.inference.vllm.worker_weight_transfer")


@contextmanager
def eager_online_fp8_conversion(model_config):
    if getattr(model_config, "quantization", None) != "fp8_per_block":
        yield
        return

    from vllm.model_executor.layers.quantization.online import fp8

    original = fp8.per_block_cast_to_fp8
    eager = getattr(original, "__wrapped__", None)
    if eager is None:
        yield
        return

    logger.info("Running online FP8 layerwise conversion without torch.compile")
    fp8.per_block_cast_to_fp8 = eager
    try:
        yield
    finally:
        fp8.per_block_cast_to_fp8 = original


@contextmanager
def current_vllm_config(vllm_config):
    if vllm_config is None:
        yield
        return

    from vllm.config import set_current_vllm_config

    with set_current_vllm_config(vllm_config):
        yield


def load_weights_checkpoint(model: Module, state_iter: Generator[tuple[str, torch.Tensor], None, None]) -> None:
    model.load_weights(state_iter)  # type: ignore


def postprocess_weights_checkpoint(model: Module, model_config, device: torch.device) -> None:
    process_weights_after_loading(model, model_config, device)


def load_weights_checkpoint_or_layerwise(
    model: Module,
    state_iter: Iterable[tuple[str, torch.Tensor]],
    model_config,
    device: torch.device,
    layerwise: bool,
    vllm_config=None,
) -> None:
    """Reload checkpoint-format weights, optionally through vLLM layerwise processing."""
    if getattr(model_config, "quantization", None) == "fp8_per_block" and not layerwise:
        raise ValueError("vLLM fp8_per_block online quantization requires weight_broadcast.layerwise = true.")

    if layerwise:
        logger.info("Reloading checkpoint-format weights with vLLM layerwise processing")
        with torch.device(device), current_vllm_config(vllm_config), eager_online_fp8_conversion(model_config):
            initialize_layerwise_reload(model)
            model.load_weights(state_iter)  # type: ignore
            finalize_layerwise_reload(model, model_config)
        return

    load_weights_checkpoint(model, state_iter)
    postprocess_weights_checkpoint(model, model_config, device)


def _invert_logical_to_physical_map(logical_to_physical_map: torch.Tensor, num_physical_experts: int) -> torch.Tensor:
    """Build a physical expert -> logical expert map from vLLM EPLB state."""
    physical_to_logical = torch.full(
        (num_physical_experts,),
        -1,
        dtype=torch.long,
        device=logical_to_physical_map.device,
    )
    logical_indices = torch.arange(
        logical_to_physical_map.shape[0],
        dtype=torch.long,
        device=logical_to_physical_map.device,
    )[:, None].expand_as(logical_to_physical_map)
    physical_indices = logical_to_physical_map.to(torch.long)
    invalid = (physical_indices < -1) | (physical_indices >= num_physical_experts)
    if invalid.any():
        invalid_indices = physical_indices[invalid].unique().tolist()
        raise ValueError(f"EPLB maps to invalid physical experts: {invalid_indices}")

    valid = physical_indices >= 0
    physical_to_logical[physical_indices[valid]] = logical_indices[valid]
    return physical_to_logical


def _build_expert_source_indices(module) -> torch.Tensor | None:
    if module._expert_map is None:
        return None

    physical_indices = torch.where(module._expert_map >= 0)[0]
    local_indices = module._expert_map[physical_indices]
    physical_indices = physical_indices[local_indices.argsort()]

    eplb_layer_state = getattr(module, "eplb_state", None)
    logical_to_physical_map = getattr(eplb_layer_state, "logical_to_physical_map", None)
    if logical_to_physical_map is None:
        return physical_indices

    physical_to_logical = _invert_logical_to_physical_map(logical_to_physical_map, module.global_num_experts)
    logical_indices = physical_to_logical[physical_indices.to(physical_to_logical.device)]
    if (logical_indices < 0).any():
        missing = physical_indices[(logical_indices < 0).to(physical_indices.device)].tolist()
        raise ValueError(f"EPLB has no logical mapping for local physical experts: {missing}")

    return logical_indices.to(physical_indices.device)


def build_expert_map(model: Module) -> dict[str, torch.Tensor]:
    """Map FusedMoE module names to source expert indices local to this worker."""
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE

    source_indices_by_module: dict[str, torch.Tensor] = {}
    for module_name, module in model.named_modules():
        if not isinstance(module, FusedMoE):
            continue
        source_indices = _build_expert_source_indices(module)
        if source_indices is None:
            continue
        source_indices_by_module[module_name] = source_indices
    return source_indices_by_module


@torch.no_grad()
def load_weights_kernel(model: Module, state_iter: Generator[tuple[str, torch.Tensor], None, None]) -> None:
    """Load vLLM kernel-format tensors using in-place copy_ updates."""
    params = dict(model.named_parameters())
    expert_source_indices = build_expert_map(model)

    loaded = 0
    skipped: list[str] = []
    shape_mismatches: list[str] = []

    for name, tensor in state_iter:
        if name not in params:
            skipped.append(name)
            continue

        param = params[name]
        if param.shape != tensor.shape:
            for module_name, source_indices in expert_source_indices.items():
                if not name.startswith(f"{module_name}."):
                    continue
                tensor = tensor[source_indices.to(tensor.device)]
                break

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
