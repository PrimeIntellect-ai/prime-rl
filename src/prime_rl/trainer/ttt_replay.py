"""Frozen TTT-adapter replay: run a micro batch's forward under the exact LoRA adapter its
tokens were sampled with, while training only the base weights.

The RL half of test-time training (see `verifiers.v1.ttt` / `prime_rl.ttt`): every branch of
a TTT rollout was sampled under one adapter version, checkpointed by the TTT service in PEFT
format. The packer keeps one adapter per micro batch (`MicroBatch.ttt_adapter_path`), and
this manager applies it around the forward — so the trainer's logprobs (and thus the
importance ratio) are computed under the same weights the sampler used. The adapter acts as
context, exactly like the tokens do.

Mechanism: forward hooks on the target `nn.Linear` modules add the adapter contribution
`scale * (x @ A^T @ B^T)` from plain no-grad tensors. No parameters, no buffers, no wrapped
modules — so it composes with FSDP/AC/compile untouched, the base weights keep their
gradients, and the adapter path still backpropagates *through* the activations (a frozen
adapter is a fixed function of x, not a stop-gradient).

Not compatible with policy-LoRA training (`model.lora`): stacking a trainable adapter on the
frozen ones is deliberately out of scope — the config validator rejects the combination.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import torch
from torch import nn

from prime_rl.utils.logger import get_logger

# PEFT checkpoint key anatomy: "base_model.model.<module_path>.lora_A.weight" where
# <module_path> is relative to the wrapped CausalLM — the same path the trainer model uses.
_PEFT_PREFIX = "base_model.model."


def _module_path(key: str) -> tuple[str, str] | None:
    """A PEFT tensor key -> ``(module_path, "A"|"B")``, or None for a non-LoRA key."""
    if not key.startswith(_PEFT_PREFIX):
        return None
    rest = key[len(_PEFT_PREFIX) :]
    for infix, which in ((".lora_A.weight", "A"), (".lora_B.weight", "B")):
        if rest.endswith(infix):
            return rest[: -len(infix)], which
    return None


class TTTReplayManager:
    """Owns the model's replay hooks and the active adapter.

    ``activate(path)`` is called once per micro batch (idempotent per path): it loads the
    PEFT checkpoint (small rank-r tensors; LRU-cached on device), resolves each tensor to
    its module, and arms the hooks. ``activate(None)`` disarms them — the forward is then
    exactly the base model. Hooks are installed lazily, only on modules an adapter actually
    touches, and are permanent no-ops while unarmed."""

    def __init__(self, model: nn.Module, device: torch.device, cache_size: int = 4) -> None:
        self.model = model
        self.device = device
        self.cache_size = cache_size
        self._modules: dict[str, nn.Module] = dict(model.named_modules())
        self._hooked: set[int] = set()
        # module -> (A [r,in], B [out,r], scale); consulted by the hooks each forward.
        self._active: dict[nn.Module, tuple[torch.Tensor, torch.Tensor, float]] = {}
        # path -> (per-module tensors, scale); adapters are tiny, keep a few resident.
        self._cache: OrderedDict[str, tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], float]] = OrderedDict()
        self._current: str | None = None

    def activate(self, path: str | None) -> None:
        if path == self._current:
            return
        self._active.clear()
        self._current = path
        if path is None:
            return
        tensors, scale = self._load(path)
        for module_path, (a, b) in tensors.items():
            module = self._modules.get(module_path)
            if module is None:
                raise ValueError(
                    f"TTT replay: adapter {path} targets module {module_path!r}, which "
                    "does not exist on the trainer model — the TTT service and the trainer "
                    "must run the same architecture."
                )
            self._ensure_hook(module)
            self._active[module] = (a, b, scale)

    def _ensure_hook(self, module: nn.Module) -> None:
        if id(module) in self._hooked:
            return

        def hook(mod: nn.Module, args: tuple, output: torch.Tensor) -> torch.Tensor:
            slot = self._active.get(mod)
            if slot is None:
                return output
            a, b, scale = slot
            x = args[0]
            return output + (x.to(a.dtype) @ a.T @ b.T).to(output.dtype) * scale

        module.register_forward_hook(hook)
        self._hooked.add(id(module))

    def _load(self, path: str) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], float]:
        if path in self._cache:
            self._cache.move_to_end(path)
            return self._cache[path]
        import json

        import safetensors.torch

        ckpt = Path(path)
        raw = safetensors.torch.load_file(ckpt / "adapter_model.safetensors")
        config = json.loads((ckpt / "adapter_config.json").read_text())
        scale = float(config["lora_alpha"]) / float(config["r"])
        halves: dict[str, dict[str, torch.Tensor]] = {}
        for key, tensor in raw.items():
            resolved = _module_path(key)
            if resolved is None:
                continue
            module_path, which = resolved
            halves.setdefault(module_path, {})[which] = tensor.to(
                self.device, dtype=torch.bfloat16, non_blocking=False
            ).requires_grad_(False)
        tensors: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for module_path, pair in halves.items():
            if "A" not in pair or "B" not in pair:
                raise ValueError(f"TTT replay: adapter {path} has an incomplete A/B pair for {module_path!r}")
            tensors[module_path] = (pair["A"], pair["B"])
        if not tensors:
            raise ValueError(f"TTT replay: adapter {path} contains no LoRA tensors")
        self._cache[path] = (tensors, scale)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        get_logger().debug(f"TTT replay: loaded adapter {path} ({len(tensors)} modules, scale={scale})")
        return tensors, scale
