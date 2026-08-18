"""Standard task iteration."""

from __future__ import annotations

import itertools
from collections import Counter
from collections.abc import Iterator, Sequence
from typing import Any

import verifiers.v1 as vf

from prime_rl.orchestrator.curriculum.base import TaskSampler


class StandardSampler(TaskSampler):
    """Advance the task iterator, cycling finite tasksets in source order."""

    def __init__(self, tasks: Sequence[vf.Task] | Iterator[vf.Task]) -> None:
        self.tasks = tuple(tasks) if isinstance(tasks, Sequence) else None
        if self.tasks is not None:
            if not self.tasks:
                raise ValueError("A finite curriculum needs at least one task")
            keys = [task.key for task in self.tasks]
            duplicates = {key for key, count in Counter(keys).items() if count > 1}
            if duplicates:
                raise ValueError(f"Task keys must be unique within a taskset: {sorted(duplicates)}")
            self.task_iterator = itertools.cycle(self.tasks)
        else:
            self.task_iterator = tasks
        self.cursor = 0

    def __next__(self) -> vf.Task:
        task = next(self.task_iterator)
        self.cursor += 1
        return task

    def state_dict(self) -> dict[str, Any]:
        return {"cursor": self.cursor}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        cursor = state_dict["cursor"]
        if self.tasks is not None:
            self.task_iterator = itertools.cycle(self.tasks)
            offset = cursor % len(self.tasks)
        else:
            offset = cursor
        for _ in range(offset):
            next(self.task_iterator)
        self.cursor = cursor
