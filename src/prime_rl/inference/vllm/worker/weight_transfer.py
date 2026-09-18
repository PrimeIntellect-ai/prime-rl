from typing import TYPE_CHECKING, Iterable

import torch
from torch.nn import Module
from vllm.config import set_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.model_loader.reload import finalize_layerwise_reload, initialize_layerwise_reload
from vllm.model_executor.model_loader.reload.layerwise import get_layerwise_info

from prime_rl.inference.vllm.eplb import local_expert_destinations, remap_expert_weight_loader

if TYPE_CHECKING:
    from vllm.distributed.eplb.eplb_state import EplbState

logger = init_logger("vllm.inference.vllm.worker_weight_transfer")


def load_weights_checkpoint_layerwise(
    model: Module,
    state_iter: Iterable[tuple[str, torch.Tensor]],
    model_config,
    vllm_config,
    eplb_state: "EplbState | None" = None,
) -> None:
    logger.info("Reloading checkpoint-format weights with vLLM layerwise processing")
    device = next(model.parameters()).device
    if eplb_state is not None:
        eplb_state.drain_async()
        torch.cuda.synchronize(device)
    weight_loaders = []
    moe_kernels = []
    with torch.device(device), set_current_vllm_config(vllm_config):
        initialize_layerwise_reload(model)
        if eplb_state is not None:
            state = eplb_state.model_states[model_config.compute_hash()]
            placements = state.physical_to_logical_map.cpu().tolist()
            moved = sum(
                logical_id != physical_id % state.model.num_routed_experts
                for placement in placements
                for physical_id, logical_id in enumerate(placement)
            )
            logger.info("Reloading EPLB experts with %d physical slots moved from their initial placement", moved)
            for layer, placement in zip(state.model.moe_layers, placements, strict=True):
                experts = layer.routed_experts
                quant_method = experts.quant_method
                moe_kernels.append((quant_method, quant_method.moe_quant_config, quant_method.moe_kernel))
                expert_map = experts._expert_map
                global_to_local = expert_map.cpu().tolist() if expert_map is not None else list(range(len(placement)))
                destinations = local_expert_destinations(placement, global_to_local, state.model.num_routed_experts)
                kernel_parameters, _ = get_layerwise_info(experts).kernel_tensors
                for name, param in experts.named_parameters(recurse=False):
                    # vLLM reuses these metadata parameters on subsequent reloads.
                    weight_loaders.append((param, param.weight_loader))
                    param.weight_loader = remap_expert_weight_loader(param.weight_loader, destinations)
                    # A layer can finish before the checkpoint stream reaches its
                    # remaining nonlocal experts. Keep those calls in the completed
                    # layerwise loader after vLLM restores the kernel parameters.
                    kernel_param = kernel_parameters[name]
                    weight_loaders.append((kernel_param, kernel_param.weight_loader))
                    kernel_param.weight_loader = param.weight_loader
        model.load_weights(state_iter)  # type: ignore
        finalize_layerwise_reload(model, model_config)
        for param, loader in weight_loaders:
            param.weight_loader = loader
        # Quantization rebuilds kernels against temporary scales. Keep the kernels
        # that reference the restored storage registered with EPLB and NIXL.
        for quant_method, quant_config, kernel in moe_kernels:
            quant_method.moe_quant_config = quant_config
            quant_method.moe_kernel = kernel


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
