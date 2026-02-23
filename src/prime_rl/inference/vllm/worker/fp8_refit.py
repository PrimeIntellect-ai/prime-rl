from collections.abc import Callable, Iterable, Iterator

import torch
from torch.nn import Module

_TRITON_BLOCKWISE_CAST_TO_FP8: Callable[[torch.Tensor, tuple[int, int]], tuple[torch.Tensor, torch.Tensor]] | None = (
    None
)


def unwrap_worker_model(model: Module) -> Module:
    """Return the underlying model used by vLLM weight updates."""
    if hasattr(model, "runnable"):
        return model.runnable  # type: ignore[return-value]
    return model


def _get_fp8_block_size(model_runner: object) -> tuple[int, int] | None:
    """Return FP8 block size when the runner uses blockwise FP8 quantization."""
    vllm_config = getattr(model_runner, "vllm_config", None)
    quant_config = getattr(vllm_config, "quant_config", None)
    if quant_config is None:
        return None
    quant_name_getter = getattr(quant_config, "get_name", None)
    quant_name = quant_name_getter() if callable(quant_name_getter) else None
    if quant_name != "fp8" and quant_config.__class__.__name__ != "Fp8Config":
        return None
    weight_block_size = getattr(quant_config, "weight_block_size", None)
    if (
        isinstance(weight_block_size, (list, tuple))
        and len(weight_block_size) == 2
        and isinstance(weight_block_size[0], int)
        and isinstance(weight_block_size[1], int)
    ):
        return (weight_block_size[0], weight_block_size[1])
    return None


def reset_fp8_process_flags(model: Module) -> None:
    """Ensure vLLM FP8 post-processing runs again after each refit."""
    for layer in model.modules():
        if hasattr(layer, "_already_called_process_weights_after_loading"):
            delattr(layer, "_already_called_process_weights_after_loading")


def _ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def _require_hopper_cuda_device(device: torch.device) -> None:
    if device.type != "cuda":
        raise RuntimeError("FP8 Triton quantization requires CUDA.")
    major, minor = torch.cuda.get_device_capability(device=device)
    if major < 9:
        raise RuntimeError(
            f"FP8 Triton quantization requires Hopper GPUs (SM90+), got compute capability {major}.{minor}."
        )


def _get_triton_quantization_device(weight: torch.Tensor) -> torch.device:
    if weight.device.type == "cuda":
        return weight.device
    if not torch.cuda.is_available():
        raise RuntimeError("FP8 Triton quantization requires CUDA.")
    return torch.device("cuda")


def _get_triton_blockwise_cast_to_fp8() -> Callable[[torch.Tensor, tuple[int, int]], tuple[torch.Tensor, torch.Tensor]]:
    global _TRITON_BLOCKWISE_CAST_TO_FP8
    if _TRITON_BLOCKWISE_CAST_TO_FP8 is not None:
        return _TRITON_BLOCKWISE_CAST_TO_FP8

    import triton
    import triton.language as tl

    fp8_dtype = torch.float8_e4m3fn
    fp8_max = torch.finfo(fp8_dtype).max
    fp8_min = -fp8_max

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

    def _cast(x: torch.Tensor, block_size: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
        _require_hopper_cuda_device(x.device)
        block_m, block_n = block_size
        m_size, n_size = x.shape
        y = torch.empty(m_size, n_size, device=x.device, dtype=fp8_dtype)
        s = torch.empty(_ceil_div(m_size, block_m), _ceil_div(n_size, block_n), dtype=torch.float32, device=x.device)

        def grid(meta):
            return (triton.cdiv(m_size, meta["BLOCK_M"]), triton.cdiv(n_size, meta["BLOCK_N"]))

        if x.is_contiguous():
            launch_kwargs = {"BLOCK_M": block_m, "BLOCK_N": block_n, "num_warps": 8, "num_stages": 2}
        else:
            launch_kwargs = {"BLOCK_M": block_m, "BLOCK_N": block_n, "num_warps": 1, "num_stages": 4}
        _blockwise_cast_to_fp8_triton[grid](
            x,
            y,
            s,
            *x.stride(),
            *y.stride(),
            *s.stride(),
            m_size,
            n_size,
            1e-10,
            fp8_min,
            fp8_max,
            **launch_kwargs,
        )
        return y, s

    _TRITON_BLOCKWISE_CAST_TO_FP8 = _cast
    return _TRITON_BLOCKWISE_CAST_TO_FP8


def quantize_weight_to_block_fp8(
    weight: torch.Tensor, block_size: tuple[int, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D tensor to FP8 and return (qweight, weight_scale_inv)."""
    if weight.ndim != 2:
        raise ValueError(f"Expected a 2D weight tensor, got shape={tuple(weight.shape)}")

    block_m, block_n = block_size
    if block_m <= 0 or block_n <= 0:
        raise ValueError(f"Invalid block size: {block_size}")

    target_device = _get_triton_quantization_device(weight)
    triton_cast = _get_triton_blockwise_cast_to_fp8()
    return triton_cast(weight.to(device=target_device), block_size)


def _build_packed_reverse_mapping(model: Module) -> dict[str, str]:
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    if not isinstance(packed_modules_mapping, dict):
        return {}

    reverse_mapping: dict[str, str] = {}
    for fused_name, original_names in packed_modules_mapping.items():
        if not isinstance(fused_name, str) or not isinstance(original_names, (list, tuple)):
            continue
        for original_name in original_names:
            if isinstance(original_name, str):
                reverse_mapping[original_name] = fused_name
    return reverse_mapping


def _get_scale_name_candidates(weight_name: str, packed_reverse_mapping: dict[str, str]) -> tuple[str, ...]:
    scale_name = f"{weight_name}_scale_inv"
    candidates = [scale_name]

    name_parts = weight_name.split(".")
    if len(name_parts) >= 2:
        module_name = name_parts[-2]
        fused_module_name = packed_reverse_mapping.get(module_name)
        if fused_module_name is not None:
            remapped_weight_name = ".".join([*name_parts[:-2], fused_module_name, name_parts[-1]])
            remapped_scale_name = f"{remapped_weight_name}_scale_inv"
            if remapped_scale_name not in candidates:
                candidates.append(remapped_scale_name)

    return tuple(candidates)


def maybe_convert_weights_for_fp8_refit(
    model_runner: object,
    model: Module,
    weights: Iterable[tuple[str, torch.Tensor]],
) -> Iterable[tuple[str, torch.Tensor]]:
    """Convert incoming BF16 weights to blockwise FP8 for FP8-quantized vLLM models."""
    block_size = _get_fp8_block_size(model_runner)
    if block_size is None:
        return weights

    reset_fp8_process_flags(model)
    parameter_names = {param_name for param_name, _ in model.named_parameters()}
    packed_reverse_mapping = _build_packed_reverse_mapping(model)

    def _iter() -> Iterator[tuple[str, torch.Tensor]]:
        emitted_names: set[str] = set()
        for name, tensor in weights:
            if name in emitted_names:
                continue

            scale_name = f"{name}_scale_inv"
            scale_name_candidates = _get_scale_name_candidates(name, packed_reverse_mapping)
            has_exact_scale = scale_name in parameter_names
            has_remapped_scale = any(
                candidate != scale_name and candidate in parameter_names for candidate in scale_name_candidates
            )
            should_quantize = (
                name.endswith("weight") and tensor.ndim == 2 and tensor.dtype != torch.float8_e4m3fn and has_exact_scale
            )

            if should_quantize:
                qweight, weight_scale_inv = quantize_weight_to_block_fp8(tensor, block_size)
                emitted_names.add(name)
                emitted_names.add(scale_name)
                yield name, qweight
                yield scale_name, weight_scale_inv
            elif has_remapped_scale and name.endswith("weight"):
                # For packed modules (e.g., q_proj -> qkv_proj), vLLM expects a
                # fused block-scale tensor layout; emitting per-shard *_scale_inv
                # here causes shape mismatches during model.load_weights.
                emitted_names.add(name)
                yield name, tensor
            else:
                emitted_names.add(name)
                yield name, tensor

    return _iter()


def _initialize_layerwise_reload(model: Module) -> None:
    from vllm.model_executor.model_loader.reload import initialize_layerwise_reload

    initialize_layerwise_reload(model)


def _finalize_layerwise_reload(model: Module, model_config: object) -> None:
    from vllm.model_executor.model_loader.reload import finalize_layerwise_reload

    finalize_layerwise_reload(model, model_config)


def load_checkpoint_weights_layerwise(
    model_runner: object,
    model: Module,
    weights: Iterable[tuple[str, torch.Tensor]],
) -> object:
    """Load checkpoint-format weights through vLLM's layerwise reload pipeline."""
    model_config = getattr(model_runner, "model_config", None)
    if model_config is None:
        raise AttributeError("model_runner.model_config is required for layerwise reload")
    vllm_config = getattr(model_runner, "vllm_config", None)
    if vllm_config is None:
        raise AttributeError("model_runner.vllm_config is required for layerwise reload")

    weights_iter = maybe_convert_weights_for_fp8_refit(model_runner, model, weights)

    load_config = getattr(vllm_config, "load_config", None)
    device_config = getattr(vllm_config, "device_config", None)
    load_device = getattr(load_config, "device", None) or getattr(device_config, "device", None) or torch.device("cuda")
    with torch.device(load_device):
        _initialize_layerwise_reload(model)
        try:
            return model.load_weights(weights_iter)
        finally:
            _finalize_layerwise_reload(model, model_config)
