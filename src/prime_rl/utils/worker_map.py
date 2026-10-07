import multiprocessing
import os
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Iterator, MutableMapping, Sequence
from concurrent.futures import Future, ProcessPoolExecutor
from typing import Any, Generic, TypeVar

import torch

T = TypeVar("T")
R = TypeVar("R")
M = TypeVar("M", bound=MutableMapping[str, Any])

# Set by init_worker in each worker process; ProcessPoolExecutor has no other place for per-worker state.
worker_fn: Callable[[Any], Any] | None = None


def exit_when_parent_dies(parent_pid: int) -> None:
    while os.getppid() == parent_pid:
        time.sleep(5.0)
    os._exit(1)


def init_worker(parent_pid: int, fn: Callable[[Any], Any]) -> None:
    """Set up a forked worker: use one torch thread so workers don't oversubscribe the CPU cores the main process needs (as torch's DataLoader does), exit if the parent dies, and keep ``fn`` for tasks."""
    global worker_fn
    torch.set_num_threads(1)
    threading.Thread(target=exit_when_parent_dies, args=(parent_pid,), daemon=True).start()
    worker_fn = fn


def call_worker_fn(item: Any) -> Any:
    return worker_fn(item)


class WorkerMap(Generic[T, R]):
    """Ordered, bounded map of ``fn`` over an iterator, run in persistent worker processes, or inline when ``num_workers`` is 0.

    Workers are forked and must not use CUDA. ``fn`` is inherited through the fork without pickling, so it
    may be a closure or partial; each item and result are pickled.
    """

    def __init__(self, num_workers: int, fn: Callable[[T], R]):
        self.num_workers = num_workers
        self.fn = fn
        self.executor: ProcessPoolExecutor | None = None
        if num_workers > 0:
            self.executor = ProcessPoolExecutor(
                max_workers=num_workers,
                # fork lets workers inherit fn without pickling it.
                mp_context=multiprocessing.get_context("fork"),
                initializer=init_worker,
                initargs=(os.getpid(), fn),
            )

    def __call__(self, items: Iterable[T]) -> Iterator[R]:
        """Yield ``fn(item)`` in input order, with at most ``2 * num_workers`` items submitted but not yet yielded.

        The factor 2 plays the role of torch DataLoader's ``prefetch_factor`` (batches loaded ahead per worker),
        hard-coded to its torch default for simplicity. Any bound above ``num_workers`` keeps every worker busy while
        the caller holds a result; the extra slack absorbs items that take unusually long.
        """
        if self.executor is None:
            yield from map(self.fn, items)
            return
        max_in_flight = 2 * self.num_workers
        pending: deque[Future[R]] = deque()
        try:
            for item in items:
                pending.append(self.executor.submit(call_worker_fn, item))
                if len(pending) >= max_in_flight:
                    yield pending.popleft().result()
            while pending:
                yield pending.popleft().result()
        finally:
            for future in pending:
                future.cancel()

    def close(self) -> None:
        if self.executor is not None:
            self.executor.shutdown(cancel_futures=True)

    def __enter__(self) -> "WorkerMap[T, R]":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def prepare(
    worker_map: WorkerMap[dict[str, Any], dict[str, Any]], items: Sequence[M], input_keys: Sequence[str]
) -> Iterator[M]:
    """Yield each item updated in place with ``worker_map``'s output on its ``input_keys`` fields.

    Workers receive only ``input_keys`` rather than whole items: sending a micro batch to a worker would move all of
    its tensors into shared memory (/dev/shm) until the step ends, costing a copy and an open file per tensor.
    """
    inputs = ({key: item[key] for key in input_keys} for item in items)
    for item, updates in zip(items, worker_map(inputs)):
        item.update(updates)
        yield item
