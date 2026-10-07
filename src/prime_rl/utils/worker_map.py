from collections import deque
from collections.abc import Callable, Iterable, Iterator, MutableMapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Generic, TypeVar

T = TypeVar("T")
R = TypeVar("R")
M = TypeVar("M", bound=MutableMapping[str, Any])


class WorkerMap(Generic[T, R]):
    """Ordered, bounded map of ``fn`` over an iterator, run in persistent background threads, or inline when ``num_workers`` is 0.

    ``fn`` runs concurrently on up to ``num_workers`` threads, so it must be thread-safe.
    """

    def __init__(self, num_workers: int, fn: Callable[[T], R]):
        self.num_workers = num_workers
        self.fn = fn
        self.executor: ThreadPoolExecutor | None = None
        if num_workers > 0:
            # Threads share the trainer's memory, so items and results pass by reference: no pickling,
            # no /dev/shm copies, and no forked copy-on-write memory, unlike worker processes.
            self.executor = ThreadPoolExecutor(max_workers=num_workers, thread_name_prefix="worker-map")

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
                pending.append(self.executor.submit(self.fn, item))
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
