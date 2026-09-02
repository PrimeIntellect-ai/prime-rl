from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Literal, overload

from prime_rl.utils.config import BaseConfig
from prime_rl.utils.logger import get_logger

if TYPE_CHECKING:
    import verifiers.v1 as vf

Kind = Literal["train", "eval"]
Subset = Literal["all", "effective"]


class Monitor(ABC):
    """Base class for monitors."""

    def __init__(self, config: BaseConfig):
        self.config = config
        self.logger = get_logger()

    async def init(self, **kwargs: Any) -> None:
        """Initialize run. Overrides name their own kwargs."""

    @overload
    async def log(self, data: dict[str, Any], step: int | None) -> None: ...

    @overload
    async def log(self, data: vf.Episode | list[vf.Episode], step: int, kind: Kind, subset: Subset) -> None: ...

    async def log(
        self,
        data: dict[str, Any] | vf.Episode | list[vf.Episode],
        step: int | None,
        kind: Kind = "train",
        subset: Subset = "effective",
    ) -> None:
        """Log scalar metrics, or episodes."""
        if isinstance(data, dict):
            await self.log_metrics(data, step=step)
        else:
            episodes = data if isinstance(data, list) else [data]
            assert step is not None
            await self.log_episodes(episodes, step=step, kind=kind, subset=subset)

    @abstractmethod
    async def log_metrics(self, metrics: dict[str, Any], step: int | None) -> None:
        """Log scalar metrics. ``step=None`` logs a time-keyed row (e.g. inference
        metrics, which are sampled on wall time rather than the training step)."""

    @abstractmethod
    async def log_episodes(self, episodes: list[vf.Episode], step: int, kind: Kind, subset: Subset) -> None:
        """Log episodes."""

    async def log_annotations(self, updates: list[dict[str, Any]]) -> None:
        """Log trace updates — post-hoc facts about traces this run already logged.
        Monitors that only carry scalars ignore them."""

    async def log_eval_epoch(self, env_name: str, step: int, episodes: list[vf.Episode]) -> None:
        """Log one finished eval epoch: every episode ``env_name`` produced for ``step``,
        errored ones included. Fires once per epoch, after the episodes streamed through
        ``log``. Monitors that carry episodes as they arrive ignore it."""

    async def finalize(self) -> None:
        """Finalize run."""
