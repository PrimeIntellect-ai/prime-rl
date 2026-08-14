from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from prime_rl.utils.config import BaseConfig
from prime_rl.utils.logger import get_logger

if TYPE_CHECKING:
    from prime_rl.orchestrator.types import Rollout


def sanitize(obj: Any, dropped_paths: list[str]) -> Any:
    """Recursively drop non-finite floats (NaN/inf), which are not valid JSON.
    Appends the dotted path of each dropped value to ``dropped_paths``."""

    def keep(item: Any, path: str) -> bool:
        if isinstance(item, float) and not math.isfinite(item):
            dropped_paths.append(path)
            return False
        return True

    def walk(value: Any, path: str) -> Any:
        if isinstance(value, dict):
            return {
                key: walk(item, child)
                for key, item in value.items()
                if keep(item, child := f"{path}.{key}" if path else key)
            }
        if isinstance(value, list):
            return [walk(item, child) for index, item in enumerate(value) if keep(item, child := f"{path}[{index}]")]
        return value

    return walk(obj, "")


class Monitor(ABC):
    """Base class for metric monitors.

    Construction takes only the monitor's config and must be side-effect free —
    runtime context is passed to ``init``, which creates the external resources
    (run registration, connections, file handles) and is called once per monitor
    by ``monitors.setup``. An ``init`` that raises crashes the run — a
    configured monitor must work.
    """

    run_id: str | None = None
    """External identifier of the run this monitor reports to (platform / W&B), when it has one."""

    def __init__(self, config: BaseConfig):
        self.config = config
        self.logger = get_logger()

    def init(self, **kwargs: Any) -> None:
        """Initialize external resources. Overrides name their own kwargs."""

    @abstractmethod
    def log(self, metrics: dict[str, Any], step: int) -> None:
        """Log scalar metrics for one step."""

    def log_episodes(self, rollouts: list[Rollout], step: int) -> None:
        """Log full episodes. No-op unless the monitor supports it."""

    def finalize(self) -> None:
        """Finalize the run on the monitor's backend. No-op unless the monitor supports it."""
