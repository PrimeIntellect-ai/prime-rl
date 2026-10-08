import itertools
import threading
import time

import pytest

from prime_rl.utils.worker_pool import WorkerPool


def _slowest_first(item: int) -> int:
    time.sleep(0.05 / (item + 1))
    return item


def _fail_on_two(item: int) -> int:
    if item == 2:
        raise KeyError(item)
    return item


@pytest.mark.parametrize("num_workers", [0, 2])
def test_worker_pool_preserves_order(num_workers: int):
    with WorkerPool(num_workers) as workers:
        assert list(workers(_slowest_first, range(6))) == list(range(6))


def test_worker_pool_runs_items_concurrently():
    both_workers_busy = threading.Barrier(2, timeout=5)

    def wait_for_other_worker(item: int) -> int:
        both_workers_busy.wait()
        return item

    with WorkerPool(2) as workers:
        assert list(workers(wait_for_other_worker, range(4))) == list(range(4))


@pytest.mark.parametrize("num_workers", [0, 2])
def test_worker_pool_propagates_worker_exception(num_workers: int):
    with WorkerPool(num_workers) as workers, pytest.raises(KeyError):
        list(workers(_fail_on_two, range(4)))


@pytest.mark.parametrize(("num_workers", "expected_pulls"), [(0, 4), (2, 7)])
def test_worker_pool_pulls_lazily_and_bounded(num_workers: int, expected_pulls: int):
    pulled = []
    processed = []

    def source():
        for item in itertools.count():
            pulled.append(item)
            yield item

    def record_slowest_first(item: int) -> int:
        processed.append(item)
        return _slowest_first(item)

    with WorkerPool(num_workers) as workers:
        results = list(itertools.islice(workers(record_slowest_first, source()), 4))
    processed_at_close = list(processed)
    time.sleep(0.1)

    assert results == [0, 1, 2, 3]
    assert len(pulled) == expected_pulls
    assert set(processed) <= set(pulled)
    assert processed == processed_at_close
