import itertools
import time

import pytest
import torch

from prime_rl.utils.worker_map import WorkerMap


def _slowest_first(item: int) -> int:
    time.sleep(0.05 / (item + 1))
    return item


def _fail_on_two(item: int) -> int:
    if item == 2:
        raise KeyError(item)
    return item


def _torch_num_threads(item: int) -> int:
    return torch.get_num_threads()


@pytest.mark.parametrize("num_workers", [0, 2])
def test_worker_map_preserves_order_with_unpicklable_fn(num_workers: int):
    offset = 100

    def add_offset_slowest_first(item: int) -> int:
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

    def source():
        for item in itertools.count():
            pulled.append(item)
            yield item

    with WorkerMap(num_workers, _slowest_first) as worker_map:
        results = list(itertools.islice(worker_map(source()), 4))

    assert results == [0, 1, 2, 3]
    assert len(pulled) == expected_pulls


def test_workers_run_torch_single_threaded():
    with WorkerMap(2, _torch_num_threads) as worker_map:
        assert list(worker_map(range(4))) == [1, 1, 1, 1]
