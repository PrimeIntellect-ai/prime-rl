from collections import defaultdict
from types import SimpleNamespace

import pytest

from prime_rl.transport.ray import RayTrainingBatchReceiver, _RayTransportStore
from prime_rl.transport.types import TrainingBatch


def test_ray_transport_store_drains_training_batches():
    store = _RayTransportStore(max_queued_items=2)

    store.put_training_batch("sender-a", b"first")
    store.put_training_batch("sender-b", b"second")

    assert store.training_batch_count() == 2
    assert store.drain_training_batches() == [("sender-a", b"first"), ("sender-b", b"second")]
    assert store.training_batch_count() == 0


def test_ray_transport_store_limits_training_batches_per_sender():
    store = _RayTransportStore(max_queued_items=1)

    store.put_training_batch("sender-a", b"first")

    with pytest.raises(RuntimeError, match="queue is full"):
        store.put_training_batch("sender-a", b"second")

    store.put_training_batch("sender-b", b"allowed")
    assert store.training_batch_count() == 2


def test_ray_transport_store_pops_micro_batch_by_rank_and_step():
    store = _RayTransportStore(max_queued_items=2)

    store.put_micro_batch(data_rank=0, step=3, payload=b"rank-0-step-3")

    assert store.has_micro_batch(data_rank=0, step=3)
    assert store.pop_micro_batch(data_rank=0, step=3) == b"rank-0-step-3"
    assert store.pop_micro_batch(data_rank=0, step=3) is None


def test_ray_transport_store_limits_micro_batches_per_rank():
    store = _RayTransportStore(max_queued_items=1)

    store.put_micro_batch(data_rank=0, step=0, payload=b"first")

    with pytest.raises(RuntimeError, match="queue is full"):
        store.put_micro_batch(data_rank=0, step=1, payload=b"second")

    store.put_micro_batch(data_rank=1, step=0, payload=b"other-rank")
    assert store.has_micro_batch(data_rank=1, step=0)


def test_ray_transport_store_rejects_duplicate_micro_batch_step():
    store = _RayTransportStore(max_queued_items=2)

    store.put_micro_batch(data_rank=0, step=0, payload=b"first")

    with pytest.raises(RuntimeError, match="already queued"):
        store.put_micro_batch(data_rank=0, step=0, payload=b"duplicate")


def test_ray_training_batch_receiver_reset_run_clears_only_matching_pending_batches():
    receiver = object.__new__(RayTrainingBatchReceiver)
    receiver.multi_run_manager = SimpleNamespace(idx_2_id={0: "run-a", 1: "run-b"})
    receiver._pending = defaultdict(
        dict,
        {
            "run-a": {0: TrainingBatch(examples=[], step=0)},
            "run-b": {0: TrainingBatch(examples=[], step=0)},
        },
    )

    receiver.reset_run(0)

    assert "run-a" not in receiver._pending
    assert receiver._pending["run-b"] == {0: TrainingBatch(examples=[], step=0)}
