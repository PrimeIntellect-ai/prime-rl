"""Standard task iteration."""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Iterator, Sequence
from typing import Any

import verifiers.v1 as vf

from prime_rl.orchestrator.curriculum.samplers.base import TaskSampler
from prime_rl.utils.logger import get_logger


class StandardSampler(TaskSampler):
    """Cycle finite tasksets in source order; advance a task iterator.

    A finite taskset keeps a sampling count per task key and yields a least-sampled
    task, breaking ties by source order. A pass yields every task at the lowest count
    once, so a fixed taskset yields the same order as ``itertools.cycle``.

    On resume, the counts are reconciled with the current taskset: removed tasks are
    dropped, and new tasks start at the lowest count among the surviving tasks. New
    tasks thus join the current pass, instead of being sampled first to catch up or
    waiting for the other tasks to finish a pass. With no surviving tasks, counts
    restart at zero.

    A task iterator (a streaming taskset) has no task keys up front, so it checkpoints
    the number of tasks it yielded and skips that many on resume. That offset assumes
    the stream is unchanged: a stream that gained or lost tasks resumes at the same
    position, not at the same task.
    """

    def __init__(self, tasks: Sequence[vf.Task] | Iterator[vf.Task]) -> None:
        self.tasks = tuple(tasks) if isinstance(tasks, Sequence) else None
        if self.tasks is not None:
            if not self.tasks:
                raise ValueError("A finite curriculum needs at least one task")
            # Task.key hashes the task data, so compute each key once.
            self.keys = [task.key for task in self.tasks]
            duplicates = {key for key, count in Counter(self.keys).items() if count > 1}
            if duplicates:
                raise ValueError(f"Task keys must be unique within a taskset: {sorted(duplicates)}")
            self.tasks_by_key = dict(zip(self.keys, self.tasks))
            self.counts = dict.fromkeys(self.keys, 0)
            self.start_pass()
        else:
            self.task_iterator = tasks
            self.cursor = 0

    def start_pass(self) -> None:
        """Queue the least-sampled tasks in source order: one scan of the taskset per pass."""
        lowest = min(self.counts.values())
        self.remaining = deque(index for index, key in enumerate(self.keys) if self.counts[key] == lowest)

    def __next__(self) -> vf.Task:
        if self.tasks is None:
            task = next(self.task_iterator)
            self.cursor += 1
            return task
        if not self.remaining:
            self.start_pass()
        index = self.remaining.popleft()
        self.counts[self.keys[index]] += 1
        return self.tasks[index]

    def find_task(self, key: str) -> vf.Task | None:
        return None if self.tasks is None else self.tasks_by_key.get(key)

    def state_dict(self) -> dict[str, Any]:
        if self.tasks is None:
            return {"cursor": self.cursor}
        return {"counts": dict(self.counts)}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if self.tasks is None:
            if "cursor" not in state_dict:
                get_logger().warning(
                    "The checkpoint holds no stream offset, so the task stream starts from the beginning"
                )
                return
            self.cursor = state_dict["cursor"]
            for _ in range(self.cursor):
                next(self.task_iterator)
            return
        if "counts" in state_dict:
            counts = state_dict["counts"]
        else:
            # A cursor into a cycle over the checkpointed taskset: each task was sampled
            # once per completed pass, plus once if the current pass already reached it.
            cursor = state_dict["cursor"]
            passes, position = divmod(cursor, len(self.keys))
            counts = {key: passes + int(index < position) for index, key in enumerate(self.keys)}
            get_logger().info(
                f"Converted the checkpointed task cursor {cursor} to sampling counts. The conversion assumes "
                "the taskset is unchanged since the checkpoint: if tasks were added, removed or reordered, "
                "the resumed order is approximate"
            )
        surviving = {key: counts[key] for key in self.keys if key in counts}
        start = min(surviving.values(), default=0)
        self.counts = {key: surviving.get(key, start) for key in self.keys}
        added, removed = len(self.keys) - len(surviving), len(counts) - len(surviving)
        if added or removed:
            get_logger().info(
                f"Reconciled sampling counts with the taskset: {added} new task(s) start at count {start}, "
                f"{removed} removed task(s) dropped"
            )
        self.start_pass()
