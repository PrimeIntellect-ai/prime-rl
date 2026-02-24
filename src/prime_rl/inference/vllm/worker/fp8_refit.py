from collections.abc import Iterable, Iterator
from typing import Protocol

import torch
import triton
import triton.language as tl
from torch.nn import Module

_FP8_DTYPE = torch.float8_e4m3fn
_FP8_MAX = torch.finfo(_FP8_DTYPE).max
_FP8_MIN = -_FP8_MAX
_FP8_BLOCK_SIZE = (128, 128)


def unwrap_worker_model(model: Module) -> Module:
    """Return the underlying model used by vLLM weight updates."""
    if hasattr(model, "runnable"):
        return model.runnable  # type: ignore[return-value]
    return model


def reset_fp8_process_flags(model: Module) -> None:
    """Ensure vLLM FP8 post-processing runs again after each refit."""
    for layer in model.modules():
        if hasattr(layer, "_already_called_process_weights_after_loading"):
            delattr(layer, "_already_called_process_weights_after_loading")


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def _require_hopper_cuda_device(device: torch.device) -> None:
    if device.type != "cuda":
        raise RuntimeError("FP8 Triton quantization requires CUDA.")
    major, minor = torch.cuda.get_device_capability(device=device)
    if major < 9:
        raise RuntimeError(
            f"FP8 Triton quantization requires Hopper GPUs (SM90+), got compute capability {major}.{minor}."
        )


# Adapted from https://github.com/THUDM/slime/blob/main/slime/backends/megatron_utils/kernels/fp8_kernel.py
@triton.jit
def _blockwise_cast_to_fp8_triton(  # noqa: N802
    x_ptr,
    y_ptr,
    s_ptr,
    stride_xm,
    stride_xn,
    stride_ym,
    stride_yn,
    stride_sm,
    stride_sn,
    m_size,
    n_size,
    eps,
    fp8_min_value,
    fp8_max_value,
    BLOCK_M: tl.constexpr = 32,  # noqa: N803
    BLOCK_N: tl.constexpr = 128,  # noqa: N803
):
    pid_m = tl.cast(tl.program_id(axis=0), tl.int64)
    pid_n = tl.cast(tl.program_id(axis=1), tl.int64)
    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = off_m < m_size
    mask_n = off_n < n_size
    mask = mask_m[:, None] & mask_n[None, :]

    x = tl.load(
        x_ptr + off_m[:, None] * stride_xm + off_n[None, :] * stride_xn,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    absmax = tl.maximum(tl.max(tl.abs(x)), eps)
    scale = absmax / fp8_max_value
    scale_inv = 1.0 / scale
    y_q = tl.clamp(x * scale_inv, fp8_min_value, fp8_max_value).to(y_ptr.dtype.element_ty)

    tl.store(y_ptr + off_m[:, None] * stride_ym + off_n[None, :] * stride_yn, y_q, mask=mask)
    tl.store(s_ptr + pid_m * stride_sm + pid_n * stride_sn, scale)


def _get_triton_quantization_device(weight: torch.Tensor) -> torch.device:
    if weight.device.type == "cuda":
        return weight.device
    if not torch.cuda.is_available():
        raise RuntimeError("FP8 Triton quantization requires CUDA.")
    return torch.device("cuda")


def blockwise_cast_to_fp8_triton(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    _require_hopper_cuda_device(weight.device)
    block_m, block_n = _FP8_BLOCK_SIZE
    m_size, n_size = weight.shape
    qweight = torch.empty(m_size, n_size, device=weight.device, dtype=_FP8_DTYPE)
    scale = torch.empty(
        ceil_div(m_size, block_m),
        ceil_div(n_size, block_n),
        dtype=torch.float32,
        device=weight.device,
    )

    if weight.is_contiguous():
        launch_kwargs = {"BLOCK_M": block_m, "BLOCK_N": block_n, "num_warps": 8, "num_stages": 2}
    else:
        launch_kwargs = {"BLOCK_M": block_m, "BLOCK_N": block_n, "num_warps": 1, "num_stages": 4}

    grid = (triton.cdiv(m_size, block_m), triton.cdiv(n_size, block_n))
    _blockwise_cast_to_fp8_triton[grid](
        weight,
        qweight,
        scale,
        *weight.stride(),
        *qweight.stride(),
        *scale.stride(),
        m_size,
        n_size,
        1e-10,
        _FP8_MIN,
        _FP8_MAX,
        **launch_kwargs,
    )
    return qweight, scale


def quantize_weight_to_block_fp8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D tensor to FP8 and return (qweight, weight_scale_inv)."""
    if weight.ndim != 2:
        raise ValueError(f"Expected a 2D weight tensor, got shape={tuple(weight.shape)}")
    target_device = _get_triton_quantization_device(weight)
    return blockwise_cast_to_fp8_triton(weight.to(device=target_device))


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


def maybe_convert_weights_for_fp8_refit(
    model_runner: object,
    model: Module,
    weights: Iterable[tuple[str, torch.Tensor]],
) -> Iterable[tuple[str, torch.Tensor]]:
    """Convert incoming BF16 weights to blockwise FP8 for FP8-quantized vLLM models."""
    quant_config = model_runner.vllm_config.quant_config
    if quant_config is None or quant_config.get_name() != "fp8":
        return weights

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


def _resolve_layerwise_load_device(load_config: object | None, device_config: object | None) -> torch.device:
    load_device = load_config.device or device_config.device
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

    weights_iter = maybe_convert_weights_for_fp8_refit(model_runner, model, weights)

    vllm_config = model_runner.vllm_config
    load_config = vllm_config.load_config
    device_config = vllm_config.device_config
    load_device = _resolve_layerwise_load_device(load_config, device_config)
    with torch.device(load_device):
        initialize_layerwise_reload(model)
        try:
            return model.load_weights(weights_iter)
        finally:
            finalize_layerwise_reload(model, model_runner.model_config)
