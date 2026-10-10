from typing import Iterable

import torch
from torch.nn import Module
from vllm.config import ParallelConfig, set_current_vllm_config
from vllm.distributed import get_pp_group, get_tensor_model_parallel_rank
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
def compute_weight_checksums(model: Module, chunk_numel: int = 2**22) -> dict[str, int]:
    """Position-weighted integer sum of each parameter's raw bits, computed on GPU.

    Exact and deterministic, so engines with the same parallel layout and weights
    produce identical values. Any changed or reordered element changes the sum."""
    names, sums = [], []
    for name, param in model.named_parameters():
        words = param.detach().reshape(-1).view(torch.uint8)
        if words.numel() % 4 == 0:
            words = words.view(torch.int32)
        total = torch.zeros((), dtype=torch.int64, device=words.device)
        for start in range(0, words.numel(), chunk_numel):
            chunk = words[start : start + chunk_numel].long()
            positions = torch.arange(start + 1, start + 1 + chunk.numel(), dtype=torch.int64, device=chunk.device)
            total += (chunk * positions).sum()
        names.append(name)
        sums.append(total)
    return dict(zip(names, torch.stack(sums).tolist()))


def collect_weight_checksums(model: Module, parallel_config: ParallelConfig) -> dict:
    """Checksums of this worker's parameters, keyed by the shard it holds. Workers of
    different engines with the same shard must hold identical weights. With expert
    parallelism each DP rank holds different experts, so the DP rank is part of the shard."""
    shard = f"tp{get_tensor_model_parallel_rank()}-pp{get_pp_group().rank_in_group}"
    if parallel_config.enable_expert_parallel:
        shard = f"dp{parallel_config.data_parallel_rank}-{shard}"
    return {"shard": shard, "checksums": compute_weight_checksums(model)}


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
