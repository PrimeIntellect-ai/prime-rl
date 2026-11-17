"""Optimizer steps that update a subset of the parameters at a time.

``CPUOffloadOptimizer`` streams offloaded optimizer state through the GPU in chunks, so it updates one
chunk of parameters at a time. An optimizer opts in by implementing ``SupportsSubsetStep`` (e.g.
prime-rl's ``Muon``); torch ``AdamW`` gets an adapter here (``subset_step_for``).

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

from collections.abc import Collection
from typing import Protocol, runtime_checkable

import torch
from torch.optim import AdamW, Optimizer


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


class AdamWSubsetStep:
    """torch AdamW updates every parameter on its own and keeps its step counters in the state."""

    def __init__(self, optimizer: AdamW):
        self.optimizer = optimizer

    def begin_subset_step(self) -> SubsetStep:
        groups = self.optimizer.param_groups
        units = [[param] for group in groups for param in group["params"] if param.grad is not None]
        return GroupSubsetStep(self.optimizer, units)


def subset_step_for(optimizer: Optimizer) -> SupportsSubsetStep | None:
    """The optimizer itself if it implements partial steps, an adapter for known optimizers, else None."""
    if isinstance(optimizer, SupportsSubsetStep):
        return optimizer
    if isinstance(optimizer, AdamW):
        return AdamWSubsetStep(optimizer)
    return None
