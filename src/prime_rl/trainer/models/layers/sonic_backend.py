from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable

import torch

_REQUIRED_MODULES = ("sonicmoe", "quack", "cutlass", "cuda.bindings", "triton")
_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16)


@dataclass(frozen=True)
class SonicRuntimeInfo:
    is_supported: bool
    code: str
    message: str


@dataclass(frozen=True)
class SonicBindings:
    moe_general_routing_inputs: Callable[..., tuple[torch.Tensor, torch.Tensor]]
    swiglu_activation: Any


def _module_missing(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is None


def _cuda_version_at_least_12_9(cuda_version: str | None) -> bool:
    if cuda_version is None:
        return False

    try:
        major_str, minor_str, *_ = cuda_version.split(".")
        major = int(major_str)
        minor = int(minor_str)
    except (TypeError, ValueError):
        return False

    return (major, minor) >= (12, 9)


def check_sonic_runtime(
    *,
    hidden_size: int,
    intermediate_size: int,
    top_k: int,
    input_dtype: torch.dtype,
    score_before_experts: bool,
    has_grouped_experts: bool,
    device: torch.device,
) -> SonicRuntimeInfo:
    if not has_grouped_experts:
        return SonicRuntimeInfo(
            is_supported=False,
            code="unsupported_experts_module",
            message="SonicMoE v1 requires unwrapped GroupedExperts modules.",
        )

    if score_before_experts:
        return SonicRuntimeInfo(
            is_supported=False,
            code="score_before_experts_not_supported",
            message="SonicMoE v1 requires score_before_experts=false.",
        )

    if input_dtype not in _SUPPORTED_DTYPES:
        return SonicRuntimeInfo(
            is_supported=False,
            code="unsupported_dtype",
            message=f"SonicMoE only supports fp16/bf16 inputs (got {input_dtype}).",
        )

    if top_k <= 0:
        return SonicRuntimeInfo(
            is_supported=False,
            code="invalid_topk",
            message=f"Invalid top-k value ({top_k}).",
        )

    if hidden_size < 512 or hidden_size % 64 != 0:
        return SonicRuntimeInfo(
            is_supported=False,
            code="unsupported_hidden_size",
            message=(f"SonicMoE currently requires hidden_size >= 512 and hidden_size % 64 == 0 (got {hidden_size})."),
        )

    if intermediate_size % 64 != 0:
        return SonicRuntimeInfo(
            is_supported=False,
            code="unsupported_intermediate_size",
            message=(f"SonicMoE currently requires intermediate_size % 64 == 0 (got {intermediate_size})."),
        )

    if not torch.cuda.is_available():
        return SonicRuntimeInfo(
            is_supported=False,
            code="cuda_unavailable",
            message="CUDA is required for SonicMoE.",
        )

    runtime_device = device if device.type == "cuda" else torch.device("cuda")
    device_capability = torch.cuda.get_device_capability(runtime_device)
    if device_capability[0] not in (9, 10):
        return SonicRuntimeInfo(
            is_supported=False,
            code="unsupported_gpu",
            message=(
                f"SonicMoE currently supports Hopper/Blackwell only (got compute capability {device_capability})."
            ),
        )

    if not _cuda_version_at_least_12_9(torch.version.cuda):
        return SonicRuntimeInfo(
            is_supported=False,
            code="unsupported_cuda_version",
            message=(f"SonicMoE requires CUDA 12.9+ (got torch.version.cuda={torch.version.cuda!r})."),
        )

    missing_modules = [name for name in _REQUIRED_MODULES if _module_missing(name)]
    if missing_modules:
        return SonicRuntimeInfo(
            is_supported=False,
            code="missing_dependencies",
            message=f"Missing SonicMoE runtime dependencies: {', '.join(missing_modules)}",
        )

    return SonicRuntimeInfo(
        is_supported=True,
        code="ok",
        message="SonicMoE runtime checks passed.",
    )


@lru_cache(maxsize=1)
def load_sonic_bindings() -> tuple[SonicBindings | None, SonicRuntimeInfo]:
    try:
        from sonicmoe.enums import ActivationType
        from sonicmoe.functional import moe_general_routing_inputs
    except Exception as exc:  # noqa: BLE001 - keep broad for optional runtime deps
        return (
            None,
            SonicRuntimeInfo(
                is_supported=False,
                code="import_error",
                message=f"Unable to import SonicMoE runtime: {exc}",
            ),
        )

    return (
        SonicBindings(
            moe_general_routing_inputs=moe_general_routing_inputs,
            swiglu_activation=ActivationType.SWIGLU,
        ),
        SonicRuntimeInfo(
            is_supported=True,
            code="ok",
            message="SonicMoE runtime imported successfully.",
        ),
    )
