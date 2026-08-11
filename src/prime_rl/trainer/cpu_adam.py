from functools import cache
from pathlib import Path
from types import ModuleType

import torch
from torch.utils.cpp_extension import load


def _cpu_build_options() -> tuple[str, list[str]]:
    capability = torch.backends.cpu.get_cpu_capability().upper()
    if capability == "AVX512":
        return "avx512", [
            "-DCPU_CAPABILITY=AVX512",
            "-DCPU_CAPABILITY_AVX512",
            "-mavx512f",
            "-mavx512dq",
            "-mavx512vl",
            "-mavx512bw",
            "-mfma",
        ]
    if capability == "AVX2":
        return "avx2", [
            "-DCPU_CAPABILITY=AVX2",
            "-DCPU_CAPABILITY_AVX2",
            "-mavx2",
            "-mfma",
            "-mf16c",
        ]
    return "default", []


@cache
def _extension() -> ModuleType:
    capability, capability_flags = _cpu_build_options()
    return load(
        name=f"prime_rl_cpu_adam_{capability}",
        sources=[str(Path(__file__).with_suffix(".cpp"))],
        extra_cflags=["-O3", "-fopenmp", *capability_flags],
        extra_ldflags=["-fopenmp"],
        verbose=False,
    )


def load_cpu_adamw_kernel() -> None:
    _extension()


@torch.no_grad()
def add_bfloat16_(destination: torch.Tensor, source: torch.Tensor) -> None:
    _extension().add_bfloat16_(destination, source)


@torch.no_grad()
def copy_bfloat16_(destination: torch.Tensor, source: torch.Tensor) -> None:
    _extension().copy_bfloat16_(destination, source)


@torch.no_grad()
def copy_or_add_bfloat16_multi_(
    destinations: list[torch.Tensor],
    sources: list[torch.Tensor],
    add: list[bool],
) -> None:
    _extension().copy_or_add_bfloat16_multi_(destinations, sources, add)


@torch.no_grad()
def adamw_step(
    params: list[torch.Tensor],
    grads: list[torch.Tensor],
    exp_avgs: list[torch.Tensor],
    exp_avg_sqs: list[torch.Tensor],
    state_steps: list[torch.Tensor],
    compute_params: list[torch.Tensor] | None = None,
    *,
    lr: float,
    beta1: float,
    beta2: float,
    weight_decay: float,
    eps: float,
    gradient_scale: float,
) -> None:
    _extension().adamw_step(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        state_steps,
        compute_params or [],
        lr,
        beta1,
        beta2,
        weight_decay,
        eps,
        gradient_scale,
    )
