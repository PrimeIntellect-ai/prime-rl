"""Frozen TTT-adapter replay: run a micro batch's forward under the exact LoRA adapter its
tokens were sampled with (one adapter per micro batch), while training only the base
weights — so the trainer's logprobs/importance ratio use the sampler's weights. Mechanism:
forward hooks on the target `nn.Linear`s add `scale * (x @ A^T @ B^T)` from plain no-grad
tensors, composing with FSDP/AC/compile untouched (adapter still backpropagates *through*
the activations). Incompatible with policy-LoRA training — the config validator rejects it.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import torch
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import _CHECKPOINT_PREFIX

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


# Kept out of any compiled graph: the adapter tensors change per micro batch, and tracing
# them into a compiled block would either bake them in or trigger recompiles.
@torch._dynamo.disable
def _adapter_delta(
    x: torch.Tensor, a: torch.Tensor, b: torch.Tensor, scale: float, out_dtype: torch.dtype
) -> torch.Tensor:
    # The trainer can keep FP32 master parameters outside an FSDP forward while FSDP (or
    # autocast) actually executes the module in BF16. ``output.dtype`` is the observable
    # compute dtype at the hook boundary; casting the frozen checkpoint tensors here
    # matches the TTT MultiLoRA/vLLM forward instead of accidentally replaying in the
    # first model parameter's storage dtype.
    return (x.to(out_dtype) @ a.to(out_dtype).T @ b.to(out_dtype).T) * scale


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
        # Strip AC wrapper segments so PEFT module paths resolve: checkpoint_wrapper renames
        # e.g. "layers.0.self_attn.q_proj" to "layers.0._checkpoint_wrapped_module.self_attn.q_proj".
        # named_modules yields the wrapper before the inner module, so on the (rare) collision
        # of a wrapper path with its inner module the later — inner — entry wins, as it should.
        self._modules: dict[str, nn.Module] = {
            name.replace(_CHECKPOINT_PREFIX, ""): mod for name, mod in model.named_modules()
        }
        self._hooked: set[int] = set()
        # module -> (A [r,in], B [out,r], scale); consulted by the hooks each forward.
        self._active: dict[nn.Module, tuple[torch.Tensor, torch.Tensor, float]] = {}
        # path -> (per-module tensors, scale); adapters are tiny, keep a few resident.
        self._cache: OrderedDict[str, tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], float]] = OrderedDict()
        self._current: str | None = None

    def activate(self, path: str | None) -> None:
        if path == self._current:
            return
        # Clear first, mutate last: a failed load/resolve must leave the manager disarmed
        # (base model), never half-armed or pretending the previous adapter is active.
        self._active.clear()
        self._current = None
        if path is None:
            return
        tensors, scale = self._load(path)
        new_active: dict[nn.Module, tuple[torch.Tensor, torch.Tensor, float]] = {}
        for module_path, (a, b) in tensors.items():
            module = self._modules.get(module_path)
            if module is None:
                raise ValueError(
                    f"TTT replay: adapter {path} targets module {module_path!r}, which "
                    "does not exist on the trainer model — the TTT service and the trainer "
                    "must run the same architecture."
                )
            if not isinstance(module, nn.Linear):
                raise ValueError(
                    f"TTT replay: adapter {path} targets {module_path!r}, resolved as "
                    f"{type(module).__name__}, but frozen replay only supports nn.Linear modules"
                )
            self._ensure_hook(module)
            new_active[module] = (a, b, scale)
        self._active.update(new_active)
        self._current = path

    def install_hooks_on_all_linears(self) -> None:
        """Eagerly hook every ``nn.Linear`` — required before ``torch.compile``: dynamo skips
        guards on module hooks by default, so a hook installed after a block was compiled is
        silently ignored (base-model forward with the trainer believing the adapter is on).
        Unarmed hooks cost one dict lookup per linear per forward."""
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                self._ensure_hook(module)

    def _ensure_hook(self, module: nn.Module) -> None:
        if id(module) in self._hooked:
            return

        def hook(mod: nn.Module, args: tuple, output: torch.Tensor) -> torch.Tensor:
            slot = self._active.get(mod)
            if slot is None:
                return output
            a, b, scale = slot
            return output + _adapter_delta(args[0], a, b, scale, output.dtype)

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
        # The hook math only implements plain LoRA (scale * x A^T B^T): reject PEFT variants
        # that would silently replay the wrong function.
        problems = []
        if config.get("peft_type") not in (None, "LORA"):
            problems.append(f"peft_type={config['peft_type']!r} (only LORA is supported)")
        for flag in ("use_rslora", "use_dora"):
            if config.get(flag):
                problems.append(f"{flag}=true")
        for pattern in ("rank_pattern", "alpha_pattern"):
            if config.get(pattern):
                problems.append(f"non-empty {pattern}")
        if config.get("modules_to_save") is not None:
            problems.append("modules_to_save is set")
        if problems:
            raise ValueError(f"TTT replay: adapter {path} is not plain-LoRA replayable: {', '.join(problems)}")
        scale = float(config["lora_alpha"]) / float(config["r"])
        halves: dict[str, dict[str, torch.Tensor]] = {}
        unknown_keys = []
        for key, tensor in raw.items():
            resolved = _module_path(key)
            if resolved is None:
                unknown_keys.append(key)
                continue
            module_path, which = resolved
            halves.setdefault(module_path, {})[which] = tensor.to(self.device, non_blocking=False).requires_grad_(False)
        if unknown_keys:
            # A key we can't map would silently drop part of the adapter — fail instead.
            raise ValueError(f"TTT replay: adapter {path} has unrecognized tensor keys: {unknown_keys}")
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
