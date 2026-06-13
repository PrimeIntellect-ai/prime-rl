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
  :class:`MoEExperts`, stacks/unstacks along the expert dim and takes an
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
class MoEExperts(ConvOp):
    """Stack per-expert HF weights into prime's fused 3-D tensors, and back.

    ``projs`` maps each prime stacked key to its HF per-expert key pattern
    (``{e}`` is the expert index), e.g.
    ``{"…mlp.experts.w1": "…mlp.experts.{e}.gate_proj.weight"}``.

    * forward: for each proj, collect contiguous experts ``0..N-1`` and
      ``torch.stack`` them into the prime key. If ``fused`` is given and its key
      is present, the transformers-v5 fused layout is split instead (one
      ``(E, 2*I, H)`` gate_up tensor → two stacked halves) — this is the
      forward-only alternative input.
    * backward: unstack each prime tensor into per-expert HF keys, numbering
      experts from ``expert_offset`` (0 for a full tensor; the global id of
      local row 0 when serving a shard).

    Sharding: backward never indexes a sharded dim across ranks — it unstacks
    whatever local rows it holds and labels them with the global offset, so no
    gather is needed.
    """

    projs: dict[str, str]
    fused: "FusedGateUp | None" = None
    expert_offset: int = 0
    dim: int = 0

    def hf_to_tt(self, sd: StateDict) -> None:
        if self.fused is not None and self.fused.gate_up in sd:
            self.fused.split(sd, self.projs, self.dim)
            return
        for tt_key, hf_pattern in self.projs.items():
            experts: list[Tensor] = []
            e = 0
            while hf_pattern.format(e=e) in sd:
                experts.append(sd.pop(hf_pattern.format(e=e)))
                e += 1
            if experts:
                sd[tt_key] = torch.stack(experts, dim=self.dim)

    def tt_to_hf(self, sd: StateDict) -> None:
        for tt_key, hf_pattern in self.projs.items():
            if tt_key not in sd:
                continue
            stacked = sd.pop(tt_key)
            for e in range(stacked.shape[self.dim]):
                sd[hf_pattern.format(e=self.expert_offset + e)] = stacked.select(self.dim, e)


@dataclass
class FusedGateUp:
    """The transformers-v5 fused expert layout: one ``gate_up_proj`` of shape
    ``(E, 2*I, H)`` (gate then up along dim 1) plus a separate ``down_proj``.
    Only consumed in the forward (HF -> prime) direction; prime always stores
    the split w1/w2/w3, so backward goes through per-expert keys."""

    gate_up: str  # HF key, e.g. "…mlp.experts.gate_up_proj"
    down: str  # HF key, e.g. "…mlp.experts.down_proj"
    w_gate: str  # prime key for gate (w1)
    w_down: str  # prime key for down (w2)
    w_up: str  # prime key for up (w3)
    split_dim: int = 1

    def split(self, sd: StateDict, projs: dict[str, str], stack_dim: int) -> None:
        gate_up = sd.pop(self.gate_up)
        down = sd.pop(self.down)
        half = gate_up.shape[self.split_dim] // 2
        sd[self.w_gate] = gate_up.narrow(self.split_dim, 0, half)
        sd[self.w_up] = gate_up.narrow(self.split_dim, half, half)
        sd[self.w_down] = down


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


def key_present(name: str) -> Callable[[StateDict], bool]:
    return lambda sd: name in sd
