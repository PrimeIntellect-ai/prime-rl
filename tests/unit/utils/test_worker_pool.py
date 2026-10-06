import itertools
import time

import pytest

from prime_rl.utils.worker_pool import WorkerPool

_offset = 0


def _set_offset(offset: int) -> None:
    global _offset
    _offset = offset


def _add_offset_slowest_first(item: int) -> int:
    time.sleep(0.05 / (item + 1))
    return item + _offset


def _fail_on_two(item: int) -> int:
    if item == 2:
        raise KeyError(item)
    return item


@pytest.mark.parametrize("num_workers", [0, 2])
def test_imap_preserves_order_and_applies_initializer(num_workers: int):
    with WorkerPool(num_workers, _set_offset, (100,)) as pool:
        assert list(pool.imap(_add_offset_slowest_first, range(6))) == [100, 101, 102, 103, 104, 105]


@pytest.mark.parametrize("num_workers", [0, 2])
def test_imap_propagates_worker_exception(num_workers: int):
    with WorkerPool(num_workers) as pool, pytest.raises(KeyError):
        list(pool.imap(_fail_on_two, range(4)))


@pytest.mark.parametrize("num_workers", [0, 2])
def test_imap_pulls_lazily_and_bounded(num_workers: int):
    pulled = []

    def source():
        for item in itertools.count():
            pulled.append(item)
            yield item

    max_in_flight = 3
    with WorkerPool(num_workers, _set_offset, (0,)) as pool:
        results = list(itertools.islice(pool.imap(_add_offset_slowest_first, source(), max_in_flight), 4))

    assert results == [0, 1, 2, 3]
    assert len(pulled) <= len(results) + max_in_flight
