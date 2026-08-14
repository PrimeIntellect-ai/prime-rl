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

    def init(self, **kwargs: Any) -> None:
        """Initialize external resources. Overrides name their own kwargs."""

    @overload
    def log(self, data: dict[str, Any], step: int) -> None: ...

    @overload
    def log(self, data: list[vf.Episode], step: int, kind: Kind, subset: Subset) -> None: ...

    def log(
        self, data: dict[str, Any] | list[vf.Episode], step: int, kind: Kind = "train", subset: Subset = "effective"
    ) -> None:
        """Log a dict of scalar metrics, or episodes with their cohort coordinates
        (train/eval x all/effective)."""
        if isinstance(data, dict):
            self.log_metrics(data, step=step)
        else:
            self.log_episodes(data, step=step, kind=kind, subset=subset)

    @abstractmethod
    def log_metrics(self, metrics: dict[str, Any], step: int) -> None:
        """Log scalar metrics for one step."""

    def log_episodes(self, episodes: list[vf.Episode], step: int, kind: Kind, subset: Subset) -> None:
        """Log episodes from one cohort (train/eval x all/effective). No-op unless the monitor supports it."""

    def finalize(self) -> None:
        """Finalize the run on the monitor's backend. No-op unless the monitor supports it."""
