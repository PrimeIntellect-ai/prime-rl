from abc import ABC, abstractmethod
from concurrent.futures import Future
from typing import TypeAlias

from torch.optim import Optimizer


class OffloadOptimizer(ABC):
    _pending_save: Future | None = None

    @property
    @abstractmethod
    def base_optimizer(self) -> Optimizer: ...

    @abstractmethod
    def checkpoint_optimizer(self) -> Optimizer: ...

    def hold_state_updates(self, save: Future) -> None:
        """Delay the next in-place update of the CPU optimizer state until ``save`` has written it."""
        self._pending_save = save

    def _wait_for_pending_save(self) -> None:
        if self._pending_save is not None:
            self._pending_save.result()
            self._pending_save = None

    def prepare_checkpoint_save(self) -> None:
        pass

    def finish_checkpoint_save(self) -> None:
        pass

    @abstractmethod
    def finish_checkpoint_load(self) -> None: ...

    def finish_model_only_checkpoint_load(self) -> None:
        pass


OptimizerLike: TypeAlias = Optimizer | OffloadOptimizer
