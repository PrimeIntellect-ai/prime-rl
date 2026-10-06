"""TaskFeed: pull an unbounded taskset off the event loop.

An unbounded taskset is a plain iterator, and its ``next()`` may block - e.g. a
generator that waits until more tasks are ready. A feed calls it on its own thread and
holds what it read in a small queue, so the dispatcher polls without ever blocking:
``ready()`` says whether a task can be taken now, ``__next__`` takes it. Reading one
task ahead keeps the generator deciding what to yield close to when it is dispatched."""

from __future__ import annotations

import queue
import threading
from collections.abc import Iterator

import verifiers.v1 as vf

_END = object()


class TaskFeed(Iterator[vf.Task]):
    def __init__(self, tasks: Iterator[vf.Task], *, name: str) -> None:
        self._tasks = tasks
        self._queue: queue.Queue = queue.Queue(maxsize=1)
        self._next: object | None = None
        self._error: BaseException | None = None
        self.exhausted = False
        """Whether the taskset ended and every task it yielded was taken."""
        threading.Thread(target=self._pull, name=f"task-feed-{name}", daemon=True).start()

    def _pull(self) -> None:
        try:
            for task in self._tasks:
                self._queue.put(task)
        except BaseException as error:
            self._error = error
        self._queue.put(_END)

    def ready(self) -> bool:
        """Whether a task can be taken now. Raises the taskset's own error."""
        if self._next is None:
            try:
                self._next = self._queue.get_nowait()
            except queue.Empty:
                return False
        if self._next is _END:
            if self._error is not None:
                raise self._error
            self.exhausted = True
            return False
        return True

    def __next__(self) -> vf.Task:
        if not self.ready():
            if self.exhausted:
                raise StopIteration
            raise RuntimeError("no task is ready; check ready() before taking one")
        task, self._next = self._next, None
        return task  # type: ignore[return-value]
