from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Literal

from prime_rl.utils.config import BaseConfig
from prime_rl.utils.logger import get_logger

if TYPE_CHECKING:
    from prime_rl.env.heads import EpisodeHead, EpisodeRecord

Kind = Literal["train", "eval"]


class Monitor(ABC):
    """Base class for monitors."""

    def __init__(self, config: BaseConfig):
        self.config = config
        self.logger = get_logger()

    async def init(self, **kwargs: Any) -> None:
        """Initialize run. Overrides name their own kwargs."""

    async def log(self, metrics: dict[str, Any], step: int | None) -> None:
        """Log scalar metrics."""
        await self.log_metrics(metrics, step=step)

    @abstractmethod
    async def log_metrics(self, metrics: dict[str, Any], step: int | None) -> None:
        """Log scalar metrics. ``step=None`` logs a time-keyed row (e.g. inference
        metrics, which are sampled on wall time rather than the training step)."""

    def wants_record(self, kind: Kind, group_id: str | None) -> bool:
        """Whether this monitor persists the episodes of a group, from their records."""
        return False

    def wants_live(self, kind: Kind, group_id: str | None) -> bool:
        """Whether this monitor shows a group's in-flight traces whole, not just their progress."""
        return False

    async def log_records(self, records: list[tuple[EpisodeHead, EpisodeRecord]], step: int, kind: Kind) -> None:
        """Persist finished episodes as they arrive, each from the record its env worker
        serialized. Monitors that only carry scalars ignore them."""

    async def log_annotations(self, updates: list[dict[str, Any]]) -> None:
        """Log trace updates — post-hoc facts about traces this run already logged.
        Monitors that only carry scalars ignore them."""

    async def log_live(self, events: list[dict[str, Any]]) -> None:
        """Log the env worker pools' stream of in-flight traces: ``{"delta", "dispatch"}`` for
        each delta as it arrived, ``{"done": trace_id}`` when a trace's episode finished.
        Monitors that only carry finished work ignore it."""

    async def log_eval_plan(self, env_name: str, step: int, expected: int) -> None:
        """Log how many episodes the eval epoch of ``env_name`` at ``step`` will produce,
        known once its tasks are counted. Monitors that only carry results ignore it."""

    async def log_eval_epoch(self, env_name: str, step: int, episodes: list[EpisodeHead]) -> None:
        """Log one finished eval epoch: every episode ``env_name`` produced for ``step``,
        errored ones included. Fires once per epoch, after the episodes streamed through
        ``log_records``. Monitors that carry episodes as they arrive ignore it."""

    async def finalize(self, paused: bool = False) -> None:
        """Finalize run. A paused run stopped to resume later and did not complete."""
