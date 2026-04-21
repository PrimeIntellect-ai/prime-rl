from typing import Generator

import torch
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


def _iter_mla_absorbed_modules(model: Module):
    for name, module in model.named_modules():
        has_absorbed_weights = hasattr(module, "W_UV") or hasattr(module, "W_UK_T")
        if not has_absorbed_weights or not hasattr(module, "kv_b_proj"):
            continue
        yield name, module


def _compute_mla_absorbed_weights(
    module: Module,
    out_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    from vllm.model_executor.layers.quantization.utils.quant_utils import get_and_maybe_dequant_weights

    kv_b_proj_weight = get_and_maybe_dequant_weights(module.kv_b_proj, out_dtype=out_dtype).T
    kv_b_proj_weight = kv_b_proj_weight.view(
        module.kv_lora_rank,
        module.num_heads,
        module.qk_nope_head_dim + module.v_head_dim,
    )
    w_uk, w_uv = kv_b_proj_weight.split([module.qk_nope_head_dim, module.v_head_dim], dim=-1)
    return w_uv.transpose(0, 1), w_uk.permute(1, 2, 0)


@torch.no_grad()
def update_mla_absorbed_weights(model: Module) -> None:
    """Recompute MLA absorbed KV weights after in-place kv_b_proj updates."""
    for name, module in _iter_mla_absorbed_modules(model):
        if hasattr(module, "W_UV"):
            out_dtype = module.W_UV.dtype
        else:
            out_dtype = torch.bfloat16

        w_uv, w_uk_t = _compute_mla_absorbed_weights(module, out_dtype)

        if hasattr(module, "W_UV"):
            module.W_UV.copy_(w_uv)
        if hasattr(module, "W_UK_T"):
            module.W_UK_T.copy_(w_uk_t)

        logger.debug(f"Updated MLA absorbed weights for module {name}")


@torch.no_grad()
def assert_mla_absorbed_weights_match(model: Module) -> int:
    """Verify that live MLA absorbed tensors match vLLM's canonical recompute."""
    checked = 0
    for name, module in _iter_mla_absorbed_modules(model):
        if hasattr(module, "W_UV"):
            out_dtype = module.W_UV.dtype
        else:
            out_dtype = torch.bfloat16

        expected_w_uv, expected_w_uk_t = _compute_mla_absorbed_weights(module, out_dtype)

        if hasattr(module, "W_UV") and not torch.equal(module.W_UV, expected_w_uv):
            diff = (module.W_UV - expected_w_uv).abs().max().item()
            raise RuntimeError(f"MLA absorbed W_UV mismatch for module {name}: max_abs_diff={diff}")
        if hasattr(module, "W_UK_T") and not torch.equal(module.W_UK_T, expected_w_uk_t):
            diff = (module.W_UK_T - expected_w_uk_t).abs().max().item()
            raise RuntimeError(f"MLA absorbed W_UK_T mismatch for module {name}: max_abs_diff={diff}")
        checked += 1
    return checked

