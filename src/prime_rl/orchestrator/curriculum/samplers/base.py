"""Task sampler interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import verifiers.v1 as vf

if TYPE_CHECKING:
    from prime_rl.orchestrator.types import Rollout


class TaskSampler(Iterator[vf.Task], ABC):
    """Base class for user-authored task selection policies."""

    @abstractmethod
    def __next__(self) -> vf.Task:
        """Choose the next task."""
        raise NotImplementedError

    def observe(self, group: list[Rollout]) -> None:
        """Update sampling state from a finalized group."""

    def state_dict(self) -> dict[str, Any]:
        """Return checkpoint state owned by this sampler."""
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore checkpoint state before sampling resumes."""

    def metrics(self) -> dict[str, float]:
        """Return metrics relative to this sampler's namespace."""
        return {}
