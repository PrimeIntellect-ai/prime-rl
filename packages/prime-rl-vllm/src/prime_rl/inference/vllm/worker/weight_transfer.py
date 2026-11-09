from typing import Iterable

import torch
from torch.nn import Module
from vllm.config import set_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.model_loader.reload import finalize_layerwise_reload, initialize_layerwise_reload

logger = init_logger("vllm.inference.vllm.worker_weight_transfer")


def load_weights_checkpoint_layerwise(
    model: Module,
    state_iter: Iterable[tuple[str, torch.Tensor]],
    model_config,
    vllm_config,
) -> None:
    logger.info("Reloading checkpoint-format weights with vLLM layerwise processing")
    device = next(model.parameters()).device
    with torch.device(device), set_current_vllm_config(vllm_config):
        initialize_layerwise_reload(model)
        model.load_weights(state_iter)  # type: ignore
        finalize_layerwise_reload(model, model_config)


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
