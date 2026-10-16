"""TaskFeed: read an infinite taskset off the event loop.

An infinite taskset's ``next()`` may block, e.g. a generator waiting for its next task
to be ready. A feed calls it on its own thread, one task ahead, so the dispatcher polls
without blocking."""

from __future__ import annotations

import queue
import threading
from collections.abc import Iterator

import verifiers.v1 as vf

_END = object()


class TaskFeed:
    def __init__(self, tasks: Iterator[vf.Task], *, name: str) -> None:
        self._queue: queue.Queue = queue.Queue(maxsize=1)
        self._error: BaseException | None = None
        self.done = False
        """Whether the taskset ended and every task it yielded was taken."""
        threading.Thread(target=self._pull, args=(tasks,), name=f"task-feed-{name}", daemon=True).start()

    def _pull(self, tasks: Iterator[vf.Task]) -> None:
        try:
            for task in tasks:
                self._queue.put(task)
        except BaseException as error:
            self._error = error
        self._queue.put(_END)

    def poll(self) -> vf.Task | None:
        """The next task if one is ready, else None. Raises the taskset's own error."""
        if not self.done:
            try:
                task = self._queue.get_nowait()
            except queue.Empty:
                return None
            if task is not _END:
                return task
            self.done = True
        if self._error is not None:
            raise self._error
        return None
