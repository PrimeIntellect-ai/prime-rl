from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")


class WorkerPool:
    """Persistent background threads that map a function over an iterator in order, or inline when ``num_workers`` is 0.

    ``map_fn`` runs concurrently on up to ``num_workers`` threads, so it must be thread-safe.
    """

    def __init__(self, num_workers: int):
        self.num_workers = num_workers
        self.executor: ThreadPoolExecutor | None = None
        if num_workers > 0:
            # Threads share the trainer's memory, so items and results pass by reference: no pickling,
            # no /dev/shm copies, and no forked copy-on-write memory, unlike worker processes.
            self.executor = ThreadPoolExecutor(max_workers=num_workers, thread_name_prefix="worker-pool")

    def __call__(self, map_fn: Callable[[T], R], items: Iterable[T]) -> Iterator[R]:
        """Yield ``map_fn(item)`` in input order, with at most ``2 * num_workers`` items submitted but not yet yielded.

        The factor 2 plays the role of torch DataLoader's ``prefetch_factor`` (batches loaded ahead per worker),
        hard-coded to its torch default for simplicity. Any bound above ``num_workers`` keeps every worker busy while
        the caller holds a result; the extra slack absorbs items that take unusually long.
        """
        if self.executor is None:
            yield from map(map_fn, items)
            return
        max_in_flight = 2 * self.num_workers
        pending: deque[Future[R]] = deque()
        try:
            for item in items:
                pending.append(self.executor.submit(map_fn, item))
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
