from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

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

    Construction must be side-effect free — external resources (run registration,
    connections, file handles) are created in ``init``, which ``monitors.setup``
    calls once per monitor. An ``init`` that raises disables the monitor.
    """

    run_id: str | None = None
    """External identifier of the run this monitor reports to (platform / W&B), when it has one."""

    def init(self) -> None:
        """Initialize external resources."""

    @abstractmethod
    def log(self, metrics: dict[str, Any], step: int) -> None:
        """Log scalar metrics for one step."""

    def log_episodes(self, rollouts: list[Rollout], step: int) -> None:
        """Log full episodes. No-op unless the monitor supports it."""

    def finalize(self) -> None:
        """Finalize the run on the monitor's backend. No-op unless the monitor supports it."""
