from abc import ABC, abstractmethod
from typing import TypeAlias

from torch.optim import Optimizer


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
