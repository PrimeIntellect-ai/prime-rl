"""Declarative, invertible state-dict conversion operators.

Every prime-rl model converts between the HuggingFace checkpoint layout and
prime-rl's training ("tt") layout. Historically each model hand-wrote a pair of
imperative ``convert_hf_*`` / ``convert_*_to_hf`` functions; this module
replaces them with a declarative chain of small operators, each of which knows
how to go *both* directions:

    forward  (``hf_to_tt``):  HF checkpoint  ->  prime training layout
    backward (``tt_to_hf``):  prime training layout  ->  HF checkpoint

A model declares a flat list of ``ConvOp``\\ s (with concrete, fully-templated
keys). :func:`apply_hf_to_tt` plays them in order; :func:`apply_tt_to_hf` plays
each op's backward in reverse order. The conversion is thus defined once and the
inverse falls out for free.

Design constraints:

* **Sharding-aware, no gathers.** Pure name ops (:class:`Rename`,
  :class:`PrefixRename`, :class:`Drop`) are value-agnostic, so they operate on
  DTensors/local shards untouched. The only value-touching op,
  :class:`Stack`, stacks/unstacks along a new (expert) dim and takes an
  ``expert_offset`` so a rank can unstack just its *local* experts into
  globally-numbered per-expert keys — no all-gather. Any op that genuinely
  needs a full tensor is the caller's responsibility to gather first (via the
  trainer's ``_resolve_dtensors``); none of the ops here gather on their own.
* **Every op is present-guarded** — it no-ops when its inputs are absent — so
  the same per-layer op list can be emitted for every layer and simply skips
  dense / non-matching layers, exactly as the imperative loops did.
* **"Backward" means "what the imperative ``*_to_hf`` did"**, not a strict
  mathematical inverse: a few conversions are intentionally lossy (e.g.
  NemotronH shifts a router bias one way and does not undo it). Such ops carry
  an explicit backward (see :class:`MapValue`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable

import torch
from torch import Tensor

StateDict = dict[str, Tensor]


class ConvOp(ABC):
    @abstractmethod
    def hf_to_tt(self, sd: StateDict) -> None: ...

    @abstractmethod
    def tt_to_hf(self, sd: StateDict) -> None: ...


def apply_hf_to_tt(sd: StateDict, ops: list[ConvOp]) -> StateDict:
    """Play the chain forward (HF -> prime), in place."""
    for op in ops:
        op.hf_to_tt(sd)
    return sd


def apply_tt_to_hf(sd: StateDict, ops: list[ConvOp]) -> StateDict:
    """Play each op's backward in reverse order (prime -> HF), in place."""
    for op in reversed(ops):
        op.tt_to_hf(sd)
    return sd


# --------------------------------------------------------------------------- #
# Name operators (value-agnostic — safe on DTensors / shards, no gather)
# --------------------------------------------------------------------------- #


@dataclass
class Rename(ConvOp):
    """Rename a single key. ``hf`` is the HF name, ``tt`` the prime name."""

    hf: str
    tt: str

    def hf_to_tt(self, sd: StateDict) -> None:
        if self.hf in sd:
            sd[self.tt] = sd.pop(self.hf)

    def tt_to_hf(self, sd: StateDict) -> None:
        if self.tt in sd:
            sd[self.hf] = sd.pop(self.tt)


@dataclass
class PrefixRename(ConvOp):
    """Rename every key under a prefix (e.g. ``backbone.`` <-> ``model.``,
    ``mixer.`` <-> ``mamba.``, ``block_sparse_moe.`` <-> ``mlp.``)."""

    hf: str
    tt: str

    @staticmethod
    def _swap(sd: StateDict, old: str, new: str) -> None:
        for key in [k for k in sd if k.startswith(old)]:
            sd[new + key[len(old) :]] = sd.pop(key)

    def hf_to_tt(self, sd: StateDict) -> None:
        self._swap(sd, self.hf, self.tt)

    def tt_to_hf(self, sd: StateDict) -> None:
        self._swap(sd, self.tt, self.hf)


@dataclass
class Drop(ConvOp):
    """Drop keys (by exact name or prefix) that have no counterpart in the
    other format. Symmetric: removes matching keys whenever present, in either
    direction — covers prime-only runtime buffers dropped on the way to HF
    (``tokens_per_expert``, ``reorderer``) and HF-only keys dropped on the way
    to prime (e.g. NemotronH ``mtp.*`` multi-token-prediction heads). Only use
    for keys that are genuinely absent from one side, so neither direction
    needs to recreate them."""

    name: str
    is_prefix: bool = False

    def _drop(self, sd: StateDict) -> None:
        for key in [k for k in sd if (k.startswith(self.name) if self.is_prefix else k == self.name)]:
            del sd[key]

    def hf_to_tt(self, sd: StateDict) -> None:
        self._drop(sd)

    def tt_to_hf(self, sd: StateDict) -> None:
        self._drop(sd)


# --------------------------------------------------------------------------- #
# Value operators
# --------------------------------------------------------------------------- #


@dataclass
class Stack(ConvOp):
    """Stack a variable-cardinality ``{e}``-indexed group of keys into one
    tensor along a NEW leading dim, and unstack back.

    ``item`` is the per-element key pattern with an ``{e}`` placeholder, e.g.
    ``"…mlp.experts.{e}.gate_proj.weight"``; ``stacked`` is the combined key.
    Not MoE-specific — any indexed group (experts, layers, ...) fits.

    * forward: collect contiguous elements ``0..N-1`` and ``torch.stack`` them.
    * backward: unstack along ``dim`` into per-element keys, numbering from
      ``index_offset`` (0 for a full tensor; the global id of local row 0 when
      serving a shard — so a rank can unstack just its local slice into
      globally-numbered keys without gathering).
    """

    stacked: str
    item: str
    dim: int = 0
    index_offset: int = 0

    def hf_to_tt(self, sd: StateDict) -> None:
        items: list[Tensor] = []
        e = 0
        while self.item.format(e=e) in sd:
            items.append(sd.pop(self.item.format(e=e)))
            e += 1
        if items:
            sd[self.stacked] = torch.stack(items, dim=self.dim)

    def tt_to_hf(self, sd: StateDict) -> None:
        if self.stacked not in sd:
            return
        t = sd.pop(self.stacked)
        for e in range(t.shape[self.dim]):
            sd[self.item.format(e=self.index_offset + e)] = t.select(self.dim, e)


@dataclass
class SplitConcat(ConvOp):
    """Split one tensor into fixed parts along an EXISTING dim, and concat back.

    ``parts`` is ``[(key, size), ...]``; ``size=None`` means an equal split
    across all parts. Forward (HF -> prime) splits ``combined`` into the part
    keys (views); backward concatenates them back. Used e.g. for the
    transformers-v5 fused ``gate_up_proj`` (two equal halves on dim 1), but is
    a general structural primitive.
    """

    combined: str
    parts: list[tuple[str, "int | None"]]
    dim: int = 0

    def _sizes(self, total: int) -> list[int]:
        if all(size is None for _, size in self.parts):
            n = len(self.parts)
            assert total % n == 0, f"{self.combined}: dim {self.dim} size {total} not divisible by {n} parts"
            return [total // n] * n
        return [int(size) for _, size in self.parts]

    def hf_to_tt(self, sd: StateDict) -> None:
        if self.combined not in sd:
            return
        t = sd.pop(self.combined)
        offset = 0
        for (key, _), size in zip(self.parts, self._sizes(t.shape[self.dim])):
            sd[key] = t.narrow(self.dim, offset, size)
            offset += size

    def tt_to_hf(self, sd: StateDict) -> None:
        if not all(key in sd for key, _ in self.parts):
            return
        tensors = [sd.pop(key) for key, _ in self.parts]
        sd[self.combined] = torch.cat(tensors, dim=self.dim)


@dataclass
class Synthetic(ConvOp):
    """A prime-only tensor with no HF counterpart, created on forward and
    dropped on backward (e.g. NemotronH's dummy ``experts.w3`` of shape (0,)).

    ``factory`` builds the tensor from the current state dict (so it can match
    device/dtype of a sibling)."""

    tt: str
    factory: Callable[[StateDict], Tensor]

    def hf_to_tt(self, sd: StateDict) -> None:
        sd[self.tt] = self.factory(sd)

    def tt_to_hf(self, sd: StateDict) -> None:
        sd.pop(self.tt, None)


@dataclass
class MapValue(ConvOp):
    """Apply a value transform to one key. ``forward`` runs HF->prime,
    ``backward`` runs prime->HF. Use for genuinely non-structural conversions
    (e.g. a bias shift). ``backward`` may be the identity when the imperative
    converter does not undo the transform (a deliberately lossy roundtrip)."""

    tt: str  # the prime-side key the value lives under (after any rename)
    forward: Callable[[Tensor], Tensor]
    backward: Callable[[Tensor], Tensor]

    def hf_to_tt(self, sd: StateDict) -> None:
        if self.tt in sd:
            sd[self.tt] = self.forward(sd[self.tt])

    def tt_to_hf(self, sd: StateDict) -> None:
        if self.tt in sd:
            sd[self.tt] = self.backward(sd[self.tt])


@dataclass
class Cast(ConvOp):
    """Declare the dtype required on each side of a logical weight.

    Besides normal checkpoint conversion, this is consumed by the NIXL
    publisher to choose the dtype of the registered RDMA staging allocation.
    The cast therefore happens before raw bytes are exposed to inference.
    """

    name: str
    hf_dtype: torch.dtype
    tt_dtype: "torch.dtype | None" = None

    def hf_to_tt(self, sd: StateDict) -> None:
        if self.tt_dtype is not None and self.name in sd:
            sd[self.name] = sd[self.name].to(dtype=self.tt_dtype)

    def tt_to_hf(self, sd: StateDict) -> None:
        if self.name in sd:
            sd[self.name] = sd[self.name].to(dtype=self.hf_dtype)


@dataclass
class SqueezeLeading(ConvOp):
    """Backward-only: if a prime key carries a leading singleton dim that HF
    doesn't expect, drop it (GLM shared experts stored as ``(1, …)``). Forward
    is a no-op — prime tolerates either shape and the dim is only stripped when
    emitting HF."""

    key: str

    def hf_to_tt(self, sd: StateDict) -> None:
        return None

    def tt_to_hf(self, sd: StateDict) -> None:
        if self.key in sd and sd[self.key].shape[0] == 1:
            sd[self.key] = sd[self.key][0]


# --------------------------------------------------------------------------- #
# Control flow
# --------------------------------------------------------------------------- #


@dataclass
class Conditional(ConvOp):
    """Run ``then`` or ``else_`` depending on a predicate over the state dict.

    The predicate is evaluated independently in each direction against the
    *current* state dict (``hf_to_tt`` sees HF keys, ``tt_to_hf`` sees prime
    keys), so the branch chosen on the way back can legitimately differ from the
    way out — which is exactly what the imperative converters do (e.g. accept a
    fused *or* per-expert HF input, but always emit per-expert)."""

    predicate: Callable[[StateDict], bool]
    then: list[ConvOp]
    else_: list[ConvOp] = field(default_factory=list)

    def hf_to_tt(self, sd: StateDict) -> None:
        apply_hf_to_tt(sd, self.then if self.predicate(sd) else self.else_)

    def tt_to_hf(self, sd: StateDict) -> None:
        branch = self.then if self.predicate(sd) else self.else_
        apply_tt_to_hf(sd, branch)


@dataclass
class Sequence(ConvOp):
    """Bundle several ops into one (forward in order, backward reversed). Lets a
    helper return a single ``ConvOp`` built from base ops."""

    ops: list[ConvOp]

    def hf_to_tt(self, sd: StateDict) -> None:
        apply_hf_to_tt(sd, self.ops)

    def tt_to_hf(self, sd: StateDict) -> None:
        apply_tt_to_hf(sd, self.ops)


def key_present(name: str) -> Callable[[StateDict], bool]:
    return lambda sd: name in sd


# --------------------------------------------------------------------------- #
# Shared composition helper for Llama-style routed experts (prime w1=gate,
# w2=down, w3=up). Used by several models' converting_<model>.py.
# --------------------------------------------------------------------------- #

GATE_DOWN_UP = (("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj"))


def routed_experts_op(
    prefix: str,
    *,
    hf_experts: str,
    tt_experts: str,
    proj_order=GATE_DOWN_UP,
    hf_proj_suffix: str = ".weight",
    fused: bool = False,
) -> ConvOp:
    """Compose the routed-experts conversion for one layer from base ops.

    Per-expert HF weights <-> stacked prime tensors is a :class:`Stack` per
    proj. ``hf_experts``/``tt_experts`` are the (relative) expert container
    names (e.g. ``mlp.experts`` / ``block_sparse_moe.experts``); ``proj_order``
    maps prime ``wN`` to the HF per-expert proj name.

    With ``fused`` set, the transformers-v5 fused ``gate_up_proj`` input is also
    accepted: a :class:`Conditional` on its presence splits it (dim 1) into
    w1/w3 and renames ``down_proj`` -> w2; otherwise the per-expert ``Stack``s
    run. Prime always stores split w1/w2/w3, so the backward path always goes
    through the per-expert ``Stack`` unstack (the fused key is absent in prime,
    so the conditional takes the else branch)."""
    stacks = [
        Stack(stacked=f"{prefix}.{tt_experts}.{wn}", item=f"{prefix}.{hf_experts}.{{e}}.{hf_proj}{hf_proj_suffix}")
        for wn, hf_proj in proj_order
    ]
    if not fused:
        return Sequence(stacks)
    gate_up = f"{prefix}.{hf_experts}.gate_up_proj"
    fused_then = [
        SplitConcat(
            combined=gate_up,
            parts=[(f"{prefix}.{tt_experts}.w1", None), (f"{prefix}.{tt_experts}.w3", None)],
            dim=1,
        ),
        Rename(f"{prefix}.{hf_experts}.down_proj", f"{prefix}.{tt_experts}.w2"),
    ]
    return Conditional(key_present(gate_up), then=fused_then, else_=stacks)
