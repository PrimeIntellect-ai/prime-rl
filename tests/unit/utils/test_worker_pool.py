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


@pytest.mark.parametrize("num_workers", [0, 2])
def test_worker_pool_reads_input_lazily(num_workers: int):
    pulled = []

    def source():
        for item in itertools.count():
            pulled.append(item)
            yield item

    consumed = 4
    with WorkerPool(num_workers) as workers:
        assert list(itertools.islice(workers(lambda item: item, source()), consumed)) == list(range(consumed))

    max_in_flight = 2 * num_workers
    assert len(pulled) == (consumed + max_in_flight - 1 if num_workers else consumed)


def test_worker_pool_stops_work_after_close():
    processed = []

    def record_slowly(item: int) -> int:
        processed.append(item)
        time.sleep(0.01)
        return item

    with WorkerPool(2) as workers:
        list(itertools.islice(workers(record_slowly, itertools.count()), 4))
    processed_at_close = list(processed)
    time.sleep(0.1)

    assert processed == processed_at_close
