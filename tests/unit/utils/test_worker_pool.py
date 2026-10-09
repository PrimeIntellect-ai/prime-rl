import itertools
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
