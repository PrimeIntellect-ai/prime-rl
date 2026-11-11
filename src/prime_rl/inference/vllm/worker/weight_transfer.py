from typing import Iterable

import torch
from torch.nn import Module
from vllm.config import set_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.model_loader.reload import finalize_layerwise_reload, initialize_layerwise_reload
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

logger = init_logger("vllm.inference.vllm.worker_weight_transfer")


FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e4m3fnuz)


def check_fp8_source(name: str, param_dtype: torch.dtype, source_dtype: torch.dtype) -> None:
    """A pre-quantized (fp8) param only takes fp8 checkpoint tensors. Loading a bf16/fp32 tensor would cast
    it into the fp8 param without its scale, which silently corrupts the layer."""
    if param_dtype in FP8_DTYPES and source_dtype not in FP8_DTYPES:
        raise ValueError(
            f"Cannot load a {source_dtype} tensor into the fp8 param {name} of a pre-quantized checkpoint. "
            'The trainer sends bf16 weights: serve the BF16 checkpoint with `quantization = "fp8_per_block"`.'
        )


def _guard_fp8_params(model: Module) -> None:
    """Wrap the loaders of the fp8 params of pre-quantized layers with ``check_fp8_source``. The wrapper keeps the
    layerwise loader's name and ``__wrapped__`` so vLLM's reload unwraps it like its own."""
    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            if param.dtype not in FP8_DTYPES:
                continue
            loader = getattr(param, "weight_loader", default_weight_loader)
            name = f"{module_name}.{param_name}"

            def guarded(*args, loader=loader, name=name, param_dtype=param.dtype, **kwargs):
                source = kwargs["loaded_weight"] if "loaded_weight" in kwargs else args[1]
                check_fp8_source(name, param_dtype, source.dtype)
                return loader(*args, **kwargs)

            guarded.__name__ = loader.__name__
            guarded.__wrapped__ = loader
            param.weight_loader = guarded


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
        _guard_fp8_params(model)
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
