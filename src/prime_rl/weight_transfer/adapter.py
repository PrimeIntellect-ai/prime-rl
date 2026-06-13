"""Metadata-level adapter from prime-rl trainer naming to HF checkpoint naming.

The bake feeds the trainer's *native* state-dict entries into vLLM's
``model.load_weights``, which only understands HF checkpoint names. For most
tensors the names already coincide. The exceptions are prime-rl's uniform
MoE training layout — shared by every custom MoE model in
``prime_rl.trainer.models``:

* ``<layer>.mlp.experts.w1|w2|w3`` — experts stacked into one 3-D tensor
  ``(num_experts, ...)`` for grouped GEMM,
* ``<layer>.mlp.router.gate.*`` — the router under a ``router`` submodule.

This module bridges that convention *generically*: stacked expert tensors
are exploded into per-expert lazy views (op chain ``[j]``) under the
standard HF per-expert names, and the router prefix is renamed. No data
moves, no per-model code runs — all real layout logic (fused QKV, merged
gate/up, EP routing) stays inside vLLM's own loaders via the recorded op
chains.
"""

from __future__ import annotations

import re

import torch

from prime_rl.weight_transfer.lazy import BakeRecorder, LazyWeight

# prime-rl stacks experts as w1 (gate), w2 (down), w3 (up) — the Llama-style
# convention used by every custom MoE model in prime_rl.trainer.models.
_STACKED_EXPERT_PROJ = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}
_STACKED_EXPERT_RE = re.compile(r"^(?P<prefix>.+\.mlp\.experts)\.(?P<key>w[123])$")
_ROUTER_PREFIX = ".mlp.router.gate."
_HF_ROUTER_PREFIX = ".mlp.gate."


def make_hf_named_lazy_weights(
    metas: list[tuple[str, torch.dtype, tuple[int, ...]]],
    device: torch.device,
    recorder: BakeRecorder,
) -> list[tuple[str, LazyWeight]]:
    """Build HF-named lazy placeholders over trainer-native tensor metadata.

    Each returned placeholder records ops against its *trainer* source tensor
    (``LazyWeight._name`` stays the trainer name); the HF name is only the
    key handed to ``model.load_weights``.
    """
    weights: list[tuple[str, LazyWeight]] = []
    for name, dtype, shape in metas:
        match = _STACKED_EXPERT_RE.match(name)
        if match and len(shape) == 3:
            proj = _STACKED_EXPERT_PROJ[match.group("key")]
            root = LazyWeight(name=name, shape=torch.Size(shape), dtype=dtype, device=device, recorder=recorder)
            for expert_id in range(shape[0]):
                hf_name = f"{match.group('prefix')}.{expert_id}.{proj}.weight"
                weights.append((hf_name, root[expert_id]))
        elif _ROUTER_PREFIX in name:
            hf_name = name.replace(_ROUTER_PREFIX, _HF_ROUTER_PREFIX)
            weights.append(
                (hf_name, LazyWeight(name=name, shape=torch.Size(shape), dtype=dtype, device=device, recorder=recorder))
            )
        else:
            weights.append(
                (name, LazyWeight(name=name, shape=torch.Size(shape), dtype=dtype, device=device, recorder=recorder))
            )
    return weights
