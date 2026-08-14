from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from prime_rl.utils.config import BaseConfig
from prime_rl.utils.logger import get_logger

if TYPE_CHECKING:
    from prime_rl.orchestrator.types import Rollout


_DROPPED_JSON_VALUE = object()


def drop_non_finite_json_values(value: Any, dropped_paths: list[str], path: str = "") -> Any:
    """Recursively drop non-finite floats (NaN/inf) from a JSON-serializable value.

    Appends the dotted path of each dropped value to `dropped_paths`. Used before
    serializing metric payloads that must be strict JSON (the public Prime API and
    the local `metrics.jsonl` sink), since NaN/Infinity are not valid JSON.
    """
    if isinstance(value, float) and not math.isfinite(value):
        dropped_paths.append(path)
        return _DROPPED_JSON_VALUE

    if isinstance(value, dict):
        return {
            key: sanitized_item
            for key, item in value.items()
            if (
                sanitized_item := drop_non_finite_json_values(
                    item,
                    dropped_paths,
                    f"{path}.{key}" if path else str(key),
                )
            )
            is not _DROPPED_JSON_VALUE
        }

    if isinstance(value, list):
        return [
            sanitized_item
            for idx, item in enumerate(value)
            if (sanitized_item := drop_non_finite_json_values(item, dropped_paths, f"{path}[{idx}]"))
            is not _DROPPED_JSON_VALUE
        ]

    return value


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
