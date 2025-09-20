from abc import ABC, abstractmethod

from prime_rl.inference.config import InferenceConfig


class BaseBackend(ABC):
    @abstractmethod
    def startup(self, config: InferenceConfig) -> None: ...

    @abstractmethod
    async def update_weights(self, path: str) -> None: ...

    @abstractmethod
    async def reload_weights(self) -> None: ...

    @abstractmethod
    async def flush_cache(self) -> None: ...
