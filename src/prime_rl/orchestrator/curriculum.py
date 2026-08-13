"""User-authored task sampling and admission policy."""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import verifiers.v1 as vf

from prime_rl.utils.utils import import_object

if TYPE_CHECKING:
    from prime_rl.configs.orchestrator import CurriculumConfig
    from prime_rl.orchestrator.types import Rollout


@dataclass(frozen=True)
class CurriculumResult:
    """One finalized task group, after its training samples and credit are built."""

    task_key: str
    rollouts: tuple[Rollout, ...]

    @classmethod
    def from_rollouts(cls, rollouts: list[Rollout]) -> CurriculumResult:
        if not rollouts:
            raise ValueError("A curriculum result needs at least one rollout")
        keys = {rollout.task.key for rollout in rollouts}
        if None in keys:
            raise ValueError("A curriculum result is missing Task.key")
        if len(keys) != 1:
            raise ValueError(f"A curriculum result contains multiple task keys: {keys}")
        task_key = keys.pop()
        assert task_key is not None
        return cls(task_key=task_key, rollouts=tuple(rollouts))


class Curriculum:
    """Default task curriculum and base class for user-authored policies.

    Override :meth:`sample` to choose tasks and :meth:`on_result` to update
    policy state or reject a finalized group. Returning ``False`` from
    :meth:`on_result` keeps the group out of the training batch; the engine
    continues sampling until the batch is full.
    """

    def __init__(self, tasks: Sequence[vf.Task] | Iterator[vf.Task], *, seed: int = 42) -> None:
        self.rng = random.Random(seed)
        self.epoch = 1
        self.cursor = 0

        if isinstance(tasks, Sequence):
            if not tasks:
                raise ValueError("A finite curriculum needs at least one task")
            self.tasks: tuple[vf.Task, ...] | None = tuple(tasks)
            self.task_iterator: Iterator[vf.Task] | None = None
            keys = [task.key for task in self.tasks]
            duplicates = {key for key, count in Counter(keys).items() if count > 1}
            if duplicates:
                raise ValueError(f"Task keys must be unique within a taskset: {sorted(duplicates)}")
            self.tasks_by_key = dict(zip(keys, self.tasks))
            self._epoch_tasks = self._shuffle()
        else:
            self.tasks = None
            self.task_iterator = tasks
            self.tasks_by_key: dict[str, vf.Task] = {}
            self._epoch_tasks: list[vf.Task] | None = None

    def _shuffle(self) -> list[vf.Task] | None:
        if self.tasks is None:
            return None
        tasks = list(self.tasks)
        random.Random(self.epoch).shuffle(tasks)
        return tasks

    def sample(self) -> vf.Task:
        """Choose the next task.

        The default preserves prime-rl's epoch-shuffled finite iteration and
        sequential infinite-taskset iteration.
        """
        if self._epoch_tasks is None:
            assert self.task_iterator is not None
            task = next(self.task_iterator)
            self.cursor += 1
            return task
        if self.cursor >= len(self._epoch_tasks):
            self.epoch += 1
            self.cursor = 0
            self._epoch_tasks = self._shuffle()
            assert self._epoch_tasks is not None
        task = self._epoch_tasks[self.cursor]
        self.cursor += 1
        return task

    def on_result(self, result: CurriculumResult) -> bool:
        """Update policy state and return whether this group should train."""
        return True

    def state_dict(self) -> dict[str, Any]:
        """Return checkpoint state owned by this curriculum."""
        return {
            "rng": self.rng.getstate(),
            "epoch": self.epoch,
            "cursor": self.cursor,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore checkpoint state before sampling resumes."""
        if "rng" in state_dict:
            self.rng.setstate(state_dict["rng"])
        self.epoch = state_dict["epoch"]
        self.cursor = state_dict["cursor"]
        if self.tasks is None:
            assert self.task_iterator is not None
            for _ in range(self.cursor):
                next(self.task_iterator)
        else:
            self._epoch_tasks = self._shuffle()

    def metrics(self) -> dict[str, float]:
        """Return metrics relative to this curriculum's env namespace."""
        return {}


def setup_curriculum(
    config: CurriculumConfig | None,
    tasks: Sequence[vf.Task] | Iterator[vf.Task],
) -> Curriculum:
    curriculum_type = Curriculum if config is None else import_object(config.import_path)
    kwargs = {} if config is None else config.kwargs
    curriculum = curriculum_type(tasks, **kwargs)
    if not isinstance(curriculum, Curriculum):
        raise TypeError(f"{curriculum_type.__module__}.{curriculum_type.__name__} must subclass Curriculum")
    return curriculum
