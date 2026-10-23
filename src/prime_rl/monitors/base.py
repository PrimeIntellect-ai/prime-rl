from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, overload

from prime_rl.utils.config import BaseConfig
from prime_rl.utils.logger import get_logger

if TYPE_CHECKING:
    import verifiers.v1 as vf

Kind = Literal["train", "eval"]


@dataclass(frozen=True)
class MonitorContext:
    """What every monitor of a process knows about the run it logs."""

    output_dir: Path
    """The run directory."""
    producer: str | None = None
    """The process logging: ``orchestrator``, ``trainer``, ``online-eval`` or ``eval``."""
    run_config: BaseConfig | None = None
    """The process's resolved config."""
    train_env_names: list[str] = field(default_factory=list)
    eval_env_names: list[str] = field(default_factory=list)
    overview_flavor: Literal["rl", "sft", "eval"] = "rl"


class Monitor(ABC):
    """Base class for monitors. Every hook but ``log_metrics`` defaults to a no-op, so a
    monitor implements only what it carries."""

    def __init__(self, config: BaseConfig, context: MonitorContext):
        self.config = config
        self.context = context
        self.logger = get_logger()

    async def init(self) -> None:
        """Open the monitor's run. Network or file I/O, before the first log."""

    @overload
    async def log(self, data: dict[str, Any], step: int | None) -> None: ...

    @overload
    async def log(self, data: vf.Episode | list[vf.Episode], step: int, kind: Kind) -> None: ...

    async def log(
        self,
        data: dict[str, Any] | vf.Episode | list[vf.Episode],
        step: int | None,
        kind: Kind = "train",
    ) -> None:
        """Log scalar metrics, or episodes."""
        if isinstance(data, dict):
            await self.log_metrics(data, step=step)
            return
        episodes = data if isinstance(data, list) else [data]
        if episodes:
            assert step is not None
            await self.log_episodes(episodes, step=step, kind=kind)

    @abstractmethod
    async def log_metrics(self, metrics: dict[str, Any], step: int | None) -> None:
        """Log scalar metrics. ``step=None`` logs a time-keyed row (e.g. inference
        metrics, which are sampled on wall time rather than the training step)."""

    async def log_episodes(self, episodes: list[vf.Episode], step: int, kind: Kind) -> None:
        """Log episodes as they arrive: every episode exactly once, errored and filtered
        ones included. What the run learns about them later (batch membership,
        advantages) arrives through ``log_annotations``."""

    async def log_annotations(self, updates: list[dict[str, Any]]) -> None:
        """Log trace updates — post-hoc facts about traces this run already logged."""

    async def log_live(self, events: list[dict[str, Any]]) -> None:
        """Log the env servers' stream of in-flight traces: ``{"delta", "dispatch"}`` for
        each delta as it arrived, ``{"done": trace_id}`` when a trace's episode finished."""

    async def log_eval_plan(self, env_name: str, step: int, expected: int) -> None:
        """Log how many episodes the eval epoch of ``env_name`` at ``step`` will produce,
        known once its tasks are counted."""

    async def log_eval_epoch(self, env_name: str, step: int, episodes: list[vf.Episode]) -> None:
        """Log one finished eval epoch: every episode ``env_name`` produced for ``step``,
        errored ones included. Fires once per epoch, after the episodes streamed through
        ``log``."""

    async def finalize(self, paused: bool = False) -> None:
        """Finalize run. A paused run stopped to resume later and did not complete."""
