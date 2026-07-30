from __future__ import annotations

import importlib
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path
from types import ModuleType

from prime_kernels._spec import Arch, KernelSpec
from prime_kernels._spec import load as _load_manifest

__all__ = [
    "Arch",
    "KernelSpec",
    "KERNELS",
    "is_available",
    "load",
    "spec",
    "status",
    "unavailable_reason",
]

_SPECS: dict[str, KernelSpec] = _load_manifest(Path(__file__).parent)

KERNELS: tuple[str, ...] = tuple(_SPECS)


def spec(name: str) -> KernelSpec:
    if name not in _SPECS:
        raise KeyError(f"unknown kernel {name!r}, have {', '.join(KERNELS) or '<none>'}")
    return _SPECS[name]


def is_built(name: str) -> bool:
    directory = spec(name).path
    return any((directory / f"_C{suffix}").exists() for suffix in EXTENSION_SUFFIXES)


def unavailable_reason(name: str, device: int | None = None) -> str | None:
    import torch

    kernel = spec(name)
    if not is_built(name):
        return f"{name} was not compiled into this install of prime-kernels"
    if not torch.cuda.is_available():
        return f"{name} requires a CUDA device ({kernel.sm_list}), none is available"
    capability = torch.cuda.get_device_capability(device)
    if not kernel.supports(capability):
        return f"{name} requires {kernel.sm_list}, this device is sm_{capability[0]}{capability[1]}"
    return None


def is_available(name: str, device: int | None = None) -> bool:
    return unavailable_reason(name, device) is None


def load(name: str, device: int | None = None) -> ModuleType:
    reason = unavailable_reason(name, device)
    if reason is not None:
        raise RuntimeError(reason)
    return importlib.import_module(spec(name).module)


def status(device: int | None = None) -> dict[str, str]:
    return {name: unavailable_reason(name, device) or "available" for name in KERNELS}
