from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

_SUPPORTED_GPU_ARCHS = (9, 10)
_QUACK_KERNELS_ENABLED = False
_QUACK_IMPORT_ERROR: Exception | None = None


def _raise_missing_quack(*args: Any, **kwargs: Any) -> Any:
    raise RuntimeError(f"Unable to import quack runtime: {_QUACK_IMPORT_ERROR}")


try:
    from quack import cross_entropy as quack_cross_entropy
    from quack import rmsnorm as quack_rmsnorm
    from quack.gemm_interface import gemm as quack_gemm
    from quack.linear import gated_linear_func as quack_gated_linear_func
    from quack.linear import linear_gated_func as quack_linear_gated_func
    from quack.linear_cross_entropy import chunked_linear_cross_entropy as quack_chunked_linear_cross_entropy
except Exception as exc:  # noqa: BLE001 - keep broad for optional runtime deps
    _QUACK_IMPORT_ERROR = exc
    quack_cross_entropy = _raise_missing_quack
    quack_rmsnorm = _raise_missing_quack
    quack_gemm = _raise_missing_quack
    quack_gated_linear_func = _raise_missing_quack
    quack_linear_gated_func = _raise_missing_quack
    quack_chunked_linear_cross_entropy = _raise_missing_quack


@dataclass(frozen=True)
class QuackRuntimeInfo:
    is_supported: bool
    code: str
    message: str


def set_quack_kernels_enabled(enabled: bool) -> None:
    global _QUACK_KERNELS_ENABLED
    _QUACK_KERNELS_ENABLED = enabled


def quack_kernels_enabled() -> bool:
    return _QUACK_KERNELS_ENABLED


def check_quack_imports() -> QuackRuntimeInfo:
    if _QUACK_IMPORT_ERROR is not None:
        return QuackRuntimeInfo(
            is_supported=False,
            code="import_error",
            message=f"Unable to import quack runtime: {_QUACK_IMPORT_ERROR}",
        )

    return QuackRuntimeInfo(
        is_supported=True,
        code="ok",
        message="quack runtime imported successfully.",
    )


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


def check_quack_runtime(device: torch.device | None = None) -> QuackRuntimeInfo:
    if not torch.cuda.is_available():
        return QuackRuntimeInfo(
            is_supported=False,
            code="cuda_unavailable",
            message="CUDA is required for quack kernels.",
        )

    runtime_device = device if device is not None and device.type == "cuda" else torch.device("cuda")
    capability = torch.cuda.get_device_capability(runtime_device)
    if capability[0] not in _SUPPORTED_GPU_ARCHS:
        return QuackRuntimeInfo(
            is_supported=False,
            code="unsupported_gpu",
            message=(f"quack kernels currently support Hopper/Blackwell only (got compute capability {capability})."),
        )

    if not _cuda_version_at_least_12_9(torch.version.cuda):
        return QuackRuntimeInfo(
            is_supported=False,
            code="unsupported_cuda_version",
            message=f"quack kernels require CUDA 12.9+ (got torch.version.cuda={torch.version.cuda!r}).",
        )

    return QuackRuntimeInfo(
        is_supported=True,
        code="ok",
        message="quack runtime checks passed.",
    )
