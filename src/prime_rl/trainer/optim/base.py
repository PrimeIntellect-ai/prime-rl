from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TypeAlias

import torch
from torch.distributed.tensor import DTensor
from torch.optim import Optimizer

# Called right before the optimizer updates some parameters, on the stream of the update; the callable it returns is
# called right after the update
UpdateHook: TypeAlias = Callable[[list[torch.Tensor]], Callable[[], None]]


class ChunkedUpdates:
    """Optimizer mixin: with ``update_hook`` set, ``step`` updates the parameters a chunk of up to ``CHUNK_NUMEL``
    local elements at a time and calls the hook around each chunk's update."""

    CHUNK_NUMEL = 1 << 28
    update_hook: UpdateHook | None = None

    def step(self, closure=None):
        if self.update_hook is None:
            return super().step(closure)
        assert closure is None
        groups = self.param_groups
        try:
            for group in groups:
                chunk, numel = [], 0
                for param in group["params"]:
                    if param.grad is None:
                        continue
                    chunk.append(param)
                    numel += (param.to_local() if isinstance(param, DTensor) else param).numel()
                    if numel >= self.CHUNK_NUMEL:
                        self._step_params(group, chunk)
                        chunk, numel = [], 0
                if chunk:
                    self._step_params(group, chunk)
        finally:
            self.param_groups = groups

    def _step_params(self, group: dict, params: list[torch.Tensor]) -> None:
        self.param_groups = [{**group, "params": params}]
        finish = self.update_hook(params)
        super().step()
        finish()


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
