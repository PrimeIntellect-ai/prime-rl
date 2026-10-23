"""Task sampler interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any

import verifiers.v1 as vf


class TaskSampler(Iterator[vf.Task], ABC):
    """Base class for user-authored task selection policies."""

    tracks_inflight = False

    def has_available(self, excluded: set[str]) -> bool:
        return True

    def next_available(self, excluded: set[str]) -> vf.Task | None:
        return next(self)

    def requeue_abandoned(self, key: str, initial_sweep: bool) -> None:
        """Recover a draw whose network calls are not restored."""

    @abstractmethod
    def __next__(self) -> vf.Task:
        """Choose the next task."""
        raise NotImplementedError

    def observe(self, group: list[vf.Episode]) -> None:
        """Update sampling state from a finalized group."""

    def observe_episode(self, episode: vf.Episode) -> None:
        """Observe a completed training episode before waiting for its whole group."""

    def on_train_batch(self, episodes: list[vf.Episode]) -> None:
        """Observe the effective cohort after its payload has been sent to the trainer."""

    def state_dict(self) -> dict[str, Any]:
        """Return checkpoint state owned by this sampler."""
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore checkpoint state before sampling resumes."""

    def metrics(self) -> dict[str, float]:
        """Return metrics relative to this sampler's namespace."""
        return {}
