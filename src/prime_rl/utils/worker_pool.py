import multiprocessing
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future, ProcessPoolExecutor
from typing import Any, TypeVar

T = TypeVar("T")
R = TypeVar("R")


class WorkerPool:
    """Ordered, bounded map over an iterator, run in persistent worker processes, or inline when ``num_workers`` is 0.

    Workers are forked and must not use CUDA. ``fn`` and each item and result are pickled, so ``fn``
    must be a module-level function; ``initargs`` are inherited through the fork without pickling.
    """

    def __init__(
        self,
        num_workers: int,
        initializer: Callable[..., None] | None = None,
        initargs: tuple[Any, ...] = (),
    ):
        self.num_workers = num_workers
        self.executor: ProcessPoolExecutor | None = None
        if num_workers == 0:
            if initializer is not None:
                initializer(*initargs)
        else:
            self.executor = ProcessPoolExecutor(
                max_workers=num_workers,
                mp_context=multiprocessing.get_context("fork"),
                initializer=initializer,
                initargs=initargs,
            )

    def imap(self, fn: Callable[[T], R], items: Iterable[T], max_in_flight: int | None = None) -> Iterator[R]:
        """Yield ``fn(item)`` in input order, pulling at most ``max_in_flight`` items ahead of the consumer."""
        if self.executor is None:
            yield from map(fn, items)
            return
        max_in_flight = max_in_flight or 2 * self.num_workers
        pending: deque[Future[R]] = deque()
        try:
            for item in items:
                pending.append(self.executor.submit(fn, item))
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

    def __enter__(self) -> "WorkerPool":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
