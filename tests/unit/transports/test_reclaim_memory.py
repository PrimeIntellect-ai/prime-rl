"""CUDA memory reclamation tests."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from prime_rl.transports.weights.base import _GB, reclaim_memory_for_broadcast


def config(mode="if_needed", headroom_gb=64.0):
    return SimpleNamespace(reclaim_memory=mode, reclaim_headroom_gb=headroom_gb)


@pytest.fixture
def cuda():
    state = SimpleNamespace(free_device=0, reserved=0, allocated=0, emptied=0, synchronized=0)

    def empty_cache():
        state.emptied += 1

    def synchronize():
        state.synchronized += 1

    with patch("prime_rl.transports.weights.base.torch.cuda") as fake:
        fake.empty_cache.side_effect = empty_cache
        fake.synchronize.side_effect = synchronize
        fake.mem_get_info.side_effect = lambda: (state.free_device, 200 * _GB)
        fake.memory_reserved.side_effect = lambda: state.reserved
        fake.memory_allocated.side_effect = lambda: state.allocated
        yield state


def test_always_mode_empties_cache(cuda):
    cuda.free_device = 199 * _GB

    metrics = reclaim_memory_for_broadcast(config(mode="always"))

    assert cuda.emptied == 1
    assert metrics["time/reclaim_memory/emptied"] == 1.0


def test_sufficient_headroom_preserves_cache(cuda):
    cuda.free_device = 80 * _GB

    metrics = reclaim_memory_for_broadcast(config(headroom_gb=64))

    assert cuda.emptied == 0
    assert metrics["time/reclaim_memory/emptied"] == 0.0


def test_insufficient_headroom_empties_cache(cuda):
    cuda.free_device = 2 * _GB
    cuda.reserved = cuda.allocated = 100 * _GB

    reclaim_memory_for_broadcast(config(headroom_gb=64))

    assert cuda.emptied == 1


def test_cached_blocks_count_as_available(cuda):
    cuda.free_device = 10 * _GB
    cuda.reserved, cuda.allocated = 150 * _GB, 60 * _GB  # 90 GiB cached and idle

    reclaim_memory_for_broadcast(config(headroom_gb=64))

    assert cuda.emptied == 0


def test_synchronizes_before_emptying_cache(cuda):
    order = []
    with patch("prime_rl.transports.weights.base.torch.cuda") as fake:
        fake.synchronize.side_effect = lambda: order.append("synchronize")
        fake.empty_cache.side_effect = lambda: order.append("empty_cache")
        reclaim_memory_for_broadcast(config(mode="always"))

    assert order == ["synchronize", "empty_cache"]


def test_synchronize_and_empty_cache_are_reported_separately(cuda):
    metrics = reclaim_memory_for_broadcast(config(mode="always"))

    parts = metrics["time/reclaim_memory/synchronize"] + metrics["time/reclaim_memory/empty_cache"]
    assert parts == pytest.approx(metrics["time/reclaim_memory"], abs=1e-6)
    assert cuda.synchronized == 1
