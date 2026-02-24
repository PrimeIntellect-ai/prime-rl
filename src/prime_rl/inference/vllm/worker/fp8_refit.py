from collections.abc import Iterable, Iterator

import torch
from torch.nn import Module

from prime_rl.inference.vllm.worker.kernels.fp8 import quantize_weight_to_block_fp8

_FP8_DTYPE = torch.float8_e4m3fn


def reset_fp8_process_flags(model: Module) -> None:
    """Ensure vLLM FP8 post-processing runs again after each refit."""
    for layer in model.modules():
        if hasattr(layer, "_already_called_process_weights_after_loading"):
            delattr(layer, "_already_called_process_weights_after_loading")


def _build_packed_reverse_mapping(model: Module) -> dict[str, str]:
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    reverse_mapping: dict[str, str] = {}
    for fused_name, original_names in packed_modules_mapping.items():
        for original_name in original_names:
            reverse_mapping[original_name] = fused_name
    return reverse_mapping


def _get_remapped_scale_name(weight_name: str, packed_reverse_mapping: dict[str, str]) -> str | None:
    name_parts = weight_name.split(".")
    if len(name_parts) < 2:
        return None

    module_name = name_parts[-2]
    fused_module_name = packed_reverse_mapping.get(module_name)
    if fused_module_name is None:
        return None
    remapped_weight_name = ".".join([*name_parts[:-2], fused_module_name, name_parts[-1]])
    return f"{remapped_weight_name}_scale_inv"


def convert_weights_for_fp8_refit(
    model: Module,
    weights: Iterable[tuple[str, torch.Tensor]],
) -> Iterable[tuple[str, torch.Tensor]]:
    """Convert incoming BF16 weights to blockwise FP8 for FP8-quantized vLLM models."""
    reset_fp8_process_flags(model)
    parameter_names = {param_name for param_name, _ in model.named_parameters()}
    packed_reverse_mapping = _build_packed_reverse_mapping(model)
    return _iter_converted_fp8_refit_weights(weights, parameter_names, packed_reverse_mapping)


def _iter_converted_fp8_refit_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
    parameter_names: set[str],
    packed_reverse_mapping: dict[str, str],
) -> Iterator[tuple[str, torch.Tensor]]:
    emitted_names: set[str] = set()
    for name, tensor in weights:
        if name in emitted_names:
            continue

        scale_name = f"{name}_scale_inv"
        remapped_scale_name = _get_remapped_scale_name(name, packed_reverse_mapping)
        has_exact_scale = scale_name in parameter_names
        has_remapped_scale = remapped_scale_name is not None and remapped_scale_name in parameter_names
        has_target_scale = has_exact_scale or has_remapped_scale
        should_quantize = (
            name.endswith("weight") and tensor.ndim == 2 and tensor.dtype != _FP8_DTYPE and has_target_scale
        )

        if should_quantize:
            qweight, weight_scale_inv = quantize_weight_to_block_fp8(tensor)
            emitted_names.add(name)
            emitted_names.add(scale_name)
            yield name, qweight
            yield scale_name, weight_scale_inv
        else:
            emitted_names.add(name)
            yield name, tensor


def _resolve_layerwise_load_device(load_config: object, device_config: object) -> torch.device:
    load_device = getattr(load_config, "device", None) or getattr(device_config, "device", None)
    if load_device is None:
        raise AttributeError("model_runner load_config/device_config must define a load device")
    if load_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(load_device)


def load_checkpoint_weights_layerwise(
    model_runner: object,
    model: Module,
    weights: Iterable[tuple[str, torch.Tensor]],
) -> object:
    """Load checkpoint-format weights through vLLM's layerwise reload pipeline."""
    from vllm.model_executor.model_loader.reload import finalize_layerwise_reload, initialize_layerwise_reload

    weights_iter = weights
    quant_config = model_runner.vllm_config.quant_config
    if quant_config is not None and quant_config.get_name() == "fp8":
        weights_iter = convert_weights_for_fp8_refit(model, weights)

    vllm_config = model_runner.vllm_config
    load_config = vllm_config.load_config
    device_config = vllm_config.device_config
    load_device = _resolve_layerwise_load_device(load_config, device_config)
    with torch.device(load_device):
        initialize_layerwise_reload(model)
        load_result = model.load_weights(weights_iter)
        finalize_layerwise_reload(model, model_runner.model_config)
    return load_result
