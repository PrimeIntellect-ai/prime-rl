from __future__ import annotations

import contextlib
import contextvars
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from safetensors.torch import load_file
from torch import Tensor

_ACTIVE_TTT_ADAPTER: contextvars.ContextVar["FrozenLoRAAdapter | None"] = contextvars.ContextVar(
    "active_ttt_trainer_adapter", default=None
)


@dataclass
class LoRAModuleState:
    a: Tensor
    b: Tensor
    _device_cache: dict[tuple[str, torch.dtype], tuple[Tensor, Tensor]] = field(default_factory=dict, repr=False)

    def to(self, device: torch.device, dtype: torch.dtype, *, cache: bool) -> tuple[Tensor, Tensor]:
        key = (str(device), dtype)
        if cache:
            cached = self._device_cache.get(key)
            if cached is None:
                cached = (
                    self.a.to(device=device, dtype=dtype, non_blocking=True),
                    self.b.to(device=device, dtype=dtype, non_blocking=True),
                )
                self._device_cache[key] = cached
            return cached
        return (
            self.a.to(device=device, dtype=dtype, non_blocking=True),
            self.b.to(device=device, dtype=dtype, non_blocking=True),
        )

    def clear_device_cache(self) -> None:
        self._device_cache.clear()


@dataclass
class FrozenLoRAAdapter:
    path: str
    modules: dict[str, LoRAModuleState]


def _adapter_module_name(key: str, suffix: str) -> str | None:
    if not key.endswith(suffix):
        return None
    name = key[: -len(suffix)]
    for prefix in ("base_model.model.", "model."):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    return name


def _normalize_runtime_name(name: str) -> str:
    parts = [p for p in name.split(".") if p not in {"_orig_mod", "module"}]
    return ".".join(parts)


def _find_adapter_state(module_name: str, adapter: FrozenLoRAAdapter) -> LoRAModuleState | None:
    normalized = _normalize_runtime_name(module_name)
    state = adapter.modules.get(normalized)
    if state is not None:
        return state
    for adapter_name, adapter_state in adapter.modules.items():
        if normalized.endswith(adapter_name) or adapter_name.endswith(normalized):
            return adapter_state
    return None


class TTTTrainerAdapterManager:
    """Trainer-side frozen LoRA adapter hooks for exact TTT replay.

    The adapters are constants: gradients flow through the base model and hidden
    states, but not into the sampled TTT adapter tensors.
    """

    def __init__(self, model: nn.Module, *, cache_device_tensors: bool = False) -> None:
        self.model = model
        self.cache_device_tensors = cache_device_tensors
        self.cache: dict[str, FrozenLoRAAdapter] = {}
        self._handles = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                self._handles.append(module.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, module_name: str):
        def hook(module: nn.Linear, inputs: tuple[Any, ...], output: Tensor) -> Tensor:
            adapter = _ACTIVE_TTT_ADAPTER.get()
            if adapter is None:
                return output
            state = _find_adapter_state(module_name, adapter)
            if state is None:
                return output
            x = inputs[0]
            a, b = state.to(x.device, x.dtype, cache=self.cache_device_tensors)
            delta = torch.matmul(torch.matmul(x, a.transpose(0, 1)), b.transpose(0, 1))
            return output + delta.to(dtype=output.dtype, device=output.device)

        return hook

    def load_adapter(self, adapter_path: str | Path) -> FrozenLoRAAdapter:
        path = Path(adapter_path).as_posix()
        cached = self.cache.get(path)
        if cached is not None:
            return cached

        state_dict = load_file(Path(path) / "adapter_model.safetensors", device="cpu")
        modules: dict[str, dict[str, Tensor]] = {}
        for key, tensor in state_dict.items():
            a_name = _adapter_module_name(key, ".lora_A.weight")
            if a_name is not None:
                modules.setdefault(a_name, {})["a"] = tensor.detach()
                continue
            b_name = _adapter_module_name(key, ".lora_B.weight")
            if b_name is not None:
                modules.setdefault(b_name, {})["b"] = tensor.detach()

        adapter_modules = {
            name: LoRAModuleState(a=parts["a"], b=parts["b"])
            for name, parts in modules.items()
            if "a" in parts and "b" in parts
        }
        adapter = FrozenLoRAAdapter(path=path, modules=adapter_modules)
        self.cache[path] = adapter
        return adapter

    @contextlib.contextmanager
    def active(self, adapter_path: str | Path):
        adapter = self.load_adapter(adapter_path)
        token = _ACTIVE_TTT_ADAPTER.set(adapter)
        try:
            yield
        finally:
            _ACTIVE_TTT_ADAPTER.reset(token)
            if not self.cache_device_tensors:
                for state in adapter.modules.values():
                    state.clear_device_cache()

    def evict(self, paths: set[str] | list[str]) -> None:
        for path in paths:
            adapter = self.cache.pop(Path(path).as_posix(), None)
            if adapter is None:
                continue
            for state in adapter.modules.values():
                state.clear_device_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def collect_trace_adapter_paths(trace: list[dict]) -> list[str]:
    paths = []
    for entry in trace:
        for key in ("adapter_path", "prompt_adapter_path"):
            path = entry.get(key)
            if path:
                paths.append(str(path))
        for span in entry.get("prompt_adapter_spans") or []:
            path = span.get("adapter_path")
            if path:
                paths.append(str(path))
    return paths


def gather_consumed_adapter_paths(local_paths: set[str] | list[str]) -> list[str]:
    paths = sorted({Path(path).as_posix() for path in local_paths if path})
    if not (dist.is_available() and dist.is_initialized()):
        return paths

    gathered: list[list[str] | None] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, paths)
    merged: set[str] = set()
    for item in gathered:
        if item:
            merged.update(item)
    return sorted(merged)


def cleanup_consumed_adapters(
    manager: TTTTrainerAdapterManager,
    paths: set[str] | list[str],
    *,
    delete_from_disk: bool,
) -> int:
    all_paths = gather_consumed_adapter_paths(paths)
    manager.evict(all_paths)
    deleted = 0
    is_rank_zero = not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0
    if delete_from_disk and is_rank_zero:
        for path in all_paths:
            directory = Path(path)
            if directory.exists():
                shutil.rmtree(directory)
                deleted += 1
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    return deleted
