"""Optimizer steps that update a subset of the parameters at a time.

``CPUOffloadOptimizer`` streams offloaded optimizer state through the GPU in chunks, so it updates one
chunk of parameters at a time. An optimizer opts in by implementing ``SupportsSubsetStep`` (e.g.
prime-rl's ``Muon``); torch ``AdamW`` and ``SignSGD`` get an adapter here (``subset_step_for``).

Partial updates can also run between the calls of an ``UpdateHook``, e.g. for the sparse weight broadcast to
diff the weights the step updates as it goes (``step_params_with_hook``).

The contract:

- ``begin_subset_step()`` replaces ``step()`` for one step. It captures everything the step reads that
  the trainer may change before the last partial update runs (group hyperparameters, gradients) and
  advances step counters, so afterwards the param groups look as after ``step()``.
- ``units`` partitions the parameters the step updates into the smallest sets that must update
  together (e.g. a batch sharing one collective). A step that communicates must return the same
  units, in the same order, on every rank it communicates with: decide from global shapes, never from
  local shard sizes.
- ``step_params(params)`` updates exactly ``params``, a union of whole units, bitwise as the full step
  would. It reads each state tensor from ``optimizer.state`` at call time (the caller swaps them for
  GPU copies), enqueues its work on the current stream without syncing the host, and is called at
  most once per unit. Every rank calls it with the same units in the same order.
- The streamed state is every tensor in ``optimizer.state[param]`` with at least one dimension (a
  DTensor's local tensor); scalar step counters stay where they are. ``step_params`` never creates
  state: the caller runs a plain ``step()`` while a parameter with a gradient has none.
"""

import math
from collections.abc import Callable, Collection
from typing import Protocol, TypeAlias, runtime_checkable

import torch
from torch.distributed.tensor import DTensor
from torch.optim import AdamW, Optimizer

from prime_rl.trainer.optim.sign_sgd import SignSGD

# Called right before some parameters update, on the stream of the update; the callable it returns is called right
# after the update
UpdateHook: TypeAlias = Callable[[list[torch.Tensor]], Callable[[], None]]

# Elements per rank of the parameters of one hooked update
HOOK_NUMEL = 1 << 26


class SubsetStep(Protocol):
    """One optimizer step, run as several partial updates."""

    units: list[list[torch.Tensor]]

    def step_params(self, params: Collection[torch.Tensor]) -> None: ...


@runtime_checkable
class SupportsSubsetStep(Protocol):
    def begin_subset_step(self) -> SubsetStep: ...


class GroupSubsetStep:
    """Partial updates of a torch optimizer whose ``step()`` updates exactly the parameters of its param
    groups that have a gradient: each update narrows every group to the requested parameters and
    reinstates the hyperparameters and gradients captured at step time around one ``step()``."""

    def __init__(self, optimizer: Optimizer, units: list[list[torch.Tensor]]):
        self.optimizer = optimizer
        self.units = units
        self.hyperparams = [{k: v for k, v in group.items() if k != "params"} for group in optimizer.param_groups]
        self.grads = {param: param.grad for unit in units for param in unit}

    def step_params(self, params: Collection[torch.Tensor]) -> None:
        params = set(params)
        groups = self.optimizer.param_groups
        current = [dict(group) for group in groups]
        current_grads = {param: param.grad for param in params}
        for group, hyperparams in zip(groups, self.hyperparams):
            group.update(hyperparams)
            group["params"] = [param for param in group["params"] if param in params]
        for param in params:
            param.grad = self.grads.pop(param)
        try:
            self.optimizer.step()
        finally:
            for param in params:
                param.grad = current_grads[param]
            for group, values in zip(groups, current):
                group.update(values)


class ParameterwiseSubsetStep:
    """torch AdamW and SignSGD update every parameter on its own (AdamW keeps its step counters in the state)."""

    def __init__(self, optimizer: AdamW | SignSGD):
        self.optimizer = optimizer

    def begin_subset_step(self) -> SubsetStep:
        groups = self.optimizer.param_groups
        units = [[param] for group in groups for param in group["params"] if param.grad is not None]
        return GroupSubsetStep(self.optimizer, units)


def subset_step_for(optimizer: Optimizer) -> SupportsSubsetStep | None:
    """The optimizer itself if it implements partial steps, an adapter for known optimizers, else None."""
    if isinstance(optimizer, SupportsSubsetStep):
        return optimizer
    if isinstance(optimizer, (AdamW, SignSGD)):
        return ParameterwiseSubsetStep(optimizer)
    return None


def nominal_numel(tensor: torch.Tensor) -> int:
    """Elements of a tensor's local shard, the same on every rank: an even split of each sharded mesh dim."""
    numel = tensor.numel()
    if isinstance(tensor, DTensor):
        shards = math.prod(
            tensor.device_mesh.size(dim) for dim, placement in enumerate(tensor.placements) if placement.is_shard()
        )
        numel = -(-numel // shards)
    return numel


def step_params_with_hook(subset_step: SubsetStep, params: Collection[torch.Tensor], hook: UpdateHook) -> None:
    """Update ``params``, a union of whole units of ``subset_step``, in pieces of whole units, each between
    ``hook(piece)`` and the callable it returns. A piece is closed once it holds ``HOOK_NUMEL`` elements per rank by
    ``nominal_numel``, so every rank cuts the same pieces and a step that communicates runs the same collectives
    everywhere."""
    selected = set(params)
    pieces, numel = [[]], 0
    for unit in subset_step.units:
        if unit[0] not in selected:
            continue
        if numel >= HOOK_NUMEL:
            pieces.append([])
            numel = 0
        pieces[-1] += unit
        numel += sum(nominal_numel(param) for param in unit)
    for piece in filter(None, pieces):
        finish = hook(piece)
        subset_step.step_params(piece)
        finish()


def step_with_hook(optimizer: SupportsSubsetStep, hook: UpdateHook) -> None:
    """One step of ``optimizer`` as partial updates between the calls of ``hook`` (``step_params_with_hook``)."""
    subset_step = optimizer.begin_subset_step()
    step_params_with_hook(subset_step, [param for unit in subset_step.units for param in unit], hook)


class HookedSubsetStep:
    """A ``SubsetStep`` whose partial updates run between the calls of ``hook`` (``step_params_with_hook``)."""

    def __init__(self, subset_step: SubsetStep, hook: UpdateHook):
        self.subset_step = subset_step
        self.hook = hook
        self.units = subset_step.units

    def step_params(self, params: Collection[torch.Tensor]) -> None:
        step_params_with_hook(self.subset_step, params, self.hook)


class UpdateHooks:
    """Optimizer mixin: with ``update_hook`` set, ``step()`` runs as partial updates between the calls of the hook
    (``step_with_hook``)."""

    update_hook: UpdateHook | None = None

    def step(self, closure=None):
        hook = self.update_hook
        if hook is None:
            return super().step(closure)
        assert closure is None
        # The partial updates of torch optimizers call ``step()`` (``GroupSubsetStep``), which must not hook again
        self.update_hook = None
        try:
            step_with_hook(subset_step_for(self), hook)
        finally:
            self.update_hook = hook
