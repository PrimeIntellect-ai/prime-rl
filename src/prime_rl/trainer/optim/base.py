from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from typing import TypeAlias

import torch
from torch.distributed.tensor import DTensor
from torch.optim import Optimizer

# Called right before the optimizer updates some parameters, on the stream of the update; the callable it returns is
# called right after the update
UpdateHook: TypeAlias = Callable[[list[torch.Tensor]], Callable[[], None]]


class ChunkedUpdates:
    """Optimizer mixin: with ``update_hook`` set, ``step`` updates the parameters a chunk of about ``CHUNK_NUMEL``
    local elements at a time and calls the hook around each chunk's update."""

    CHUNK_NUMEL = 1 << 26
    update_hook: UpdateHook | None = None

    def update_units(self, group: dict) -> Iterator[tuple[list[torch.Tensor], int]]:
        """The parameters of a group to update, in units that a chunk does not split, with their local numel."""
        for param in group["params"]:
            if param.grad is not None:
                yield [param], (param.to_local() if isinstance(param, DTensor) else param).numel()

    def step(self, closure=None):
        if self.update_hook is None:
            return super().step(closure)
        assert closure is None
        groups = self.param_groups
        try:
            for group in groups:
                chunks, numel = [[]], 0
                for params, unit_numel in self.update_units(group):
                    if numel >= self.CHUNK_NUMEL:
                        chunks.append([])
                        numel = 0
                    chunks[-1] += params
                    numel += unit_numel
                stepped = group
                for chunk in filter(None, chunks):
                    # Every chunk steps a copy of the group, so per-group state (e.g. Muon's step count) advances once
                    stepped = {**group, "params": chunk}
                    self.param_groups = [stepped]
                    finish = self.update_hook(chunk)
                    super().step()
                    finish()
                group.update({key: value for key, value in stepped.items() if key != "params"})
        finally:
            self.param_groups = groups


class OffloadOptimizer(ABC):
    @property
    @abstractmethod
    def base_optimizer(self) -> Optimizer: ...

    @abstractmethod
    def checkpoint_optimizer(self) -> Optimizer: ...

    def prepare_checkpoint_save(self) -> None:
        pass

    def finish_checkpoint_save(self) -> None:
        pass

    @abstractmethod
    def finish_checkpoint_load(self) -> None: ...

    def finish_model_only_checkpoint_load(self) -> None:
        pass


OptimizerLike: TypeAlias = Optimizer | OffloadOptimizer
