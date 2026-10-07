import itertools
import threading
import time

import pytest

from prime_rl.utils.worker_map import WorkerMap


def _slowest_first(item: int) -> int:
    time.sleep(0.05 / (item + 1))
    return item


def _fail_on_two(item: int) -> int:
    if item == 2:
        raise KeyError(item)
    return item


@pytest.mark.parametrize("num_workers", [0, 2])
def test_worker_map_preserves_order(num_workers: int):
    offset = 100
    both_workers_busy = threading.Barrier(2, timeout=5)

    def add_offset_slowest_first(item: int) -> int:
        if num_workers == 2:
            both_workers_busy.wait()
        return _slowest_first(item) + offset

    with WorkerMap(num_workers, add_offset_slowest_first) as worker_map:
        assert list(worker_map(range(6))) == [100, 101, 102, 103, 104, 105]


@pytest.mark.parametrize("num_workers", [0, 2])
def test_worker_map_propagates_worker_exception(num_workers: int):
    with WorkerMap(num_workers, _fail_on_two) as worker_map, pytest.raises(KeyError):
        list(worker_map(range(4)))


@pytest.mark.parametrize(("num_workers", "expected_pulls"), [(0, 4), (2, 7)])
def test_worker_map_pulls_lazily_and_bounded(num_workers: int, expected_pulls: int):
    pulled = []
    processed = []

    def source():
        for item in itertools.count():
            pulled.append(item)
            yield item

    def record_slowest_first(item: int) -> int:
        processed.append(item)
        return _slowest_first(item)

    with WorkerMap(num_workers, record_slowest_first) as worker_map:
        results = list(itertools.islice(worker_map(source()), 4))
    processed_at_close = list(processed)
    time.sleep(0.1)

    assert results == [0, 1, 2, 3]
    assert len(pulled) == expected_pulls
    assert set(processed) <= set(pulled)
    assert processed == processed_at_close
