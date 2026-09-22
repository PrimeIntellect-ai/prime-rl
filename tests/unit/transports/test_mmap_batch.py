import asyncio
import json
import weakref
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event

import numpy as np
import pytest
import torch

from prime_rl.configs.shared import MMapTransportConfig
from prime_rl.transports.batch import setup_batch_receiver
from prime_rl.transports.batch.mmap import (
    MMapBatchReceiver,
    MMapBatchSender,
    MMapMicroBatches,
    compact_routing_ids,
    loss_token_counts,
)
from prime_rl.transports.batch.tensors import micro_batch_to_tensor
from prime_rl.transports.batch.types import MicroBatch, MMImageRef, MMRefs, RoutedExperts, SamplingMask
from prime_rl.utils.pathing import get_batch_dir, get_step_path


def make_batch():
    ids = np.array([0, 255, 2, 7, 1, 3, 4, 5], dtype=np.int64).reshape(4, 1, 2)
    return MicroBatch(
        input_ids=[10, 11, 12, 13],
        position_ids=[0, 1, 2, 3],
        loss_mask=[False, True, True, True],
        advantages=[0.0, 1.0, -1.0, 0.0],
        inference_logprobs=[0.0, -0.1, -0.2, -0.3],
        temperatures=[1.0, 1.0, 0.8, 0.8],
        env_names=["travel", "travel", "sec", "sec"],
        sequence_lengths=[2, 2],
        seq_lens=[2, 2],
        trace_ids=["a", "b"],
        branch_indices=[0, 1],
        ref_logprobs=[0.0, -0.2, -0.4, -0.6],
        rl_weights=[0.0, 1.0, 0.0, 1.0],
        ce_weights=[1.0, 0.0, 0.0, 0.0],
        ref_kl_weights=[0.0, 0.0, 1.0, 1.0],
        routed_experts=RoutedExperts(data=ids.tobytes(), shape=list(ids.shape), dtype="int64"),
        mm_refs=MMRefs(images=[MMImageRef(url="data:image/png;base64,dGVzdA==", offset=1, length=1)]),
        mm_token_type_ids=[0, 1, 0, 0],
        sampling_mask=SamplingMask(
            ids=np.array([4, 7, 9], dtype=np.int32).tobytes(),
            counts=np.array([0, 2, 0, 1], dtype=np.int32).tobytes(),
        ),
    )


@pytest.mark.parametrize(
    "values,dtype",
    [([0, 255], torch.uint8), ([0, 256], torch.uint16), ([-1, 255], torch.int16), ([0, 65536], torch.int32)],
)
def test_routing_compaction_is_lossless(values, dtype):
    original = torch.tensor(values, dtype=torch.int64)
    compact = compact_routing_ids(original)
    assert compact.dtype == dtype
    torch.testing.assert_close(compact.to(torch.int64), original)


@pytest.mark.parametrize("prefetch", [0, 2])
def test_mmap_roundtrip_preserves_training_inputs_and_loss_normalization(tmp_path, prefetch):
    config = MMapTransportConfig(readers_per_rank=2, prefetch_batches=prefetch)
    sender = MMapBatchSender(tmp_path, 1, 1, config)
    source = make_batch()
    expected = micro_batch_to_tensor(source)
    asyncio.run(sender.send([[source, source]]))
    receiver = setup_batch_receiver(tmp_path, 0, 1, config)
    batches = receiver.receive()
    assert len(batches) == 2
    assert batches.loss_counts == tuple(2 * n for n in loss_token_counts(expected))
    for actual in batches:
        for key in expected:
            if isinstance(expected[key], torch.Tensor):
                torch.testing.assert_close(actual[key].to(expected[key].dtype), expected[key])
            elif isinstance(expected[key], dict):
                for item in expected[key]:
                    torch.testing.assert_close(actual[key][item], expected[key][item])
            else:
                assert actual[key] == expected[key]
        assert actual["routed_experts"].dtype == torch.uint8
        assert actual["routed_experts"].untyped_storage().nbytes() == 8
    assert batches.seq_len == 4
    batches.close()


def test_mmap_does_not_retain_consumed_microbatches(tmp_path):
    sender = MMapBatchSender(tmp_path, 1, 1, MMapTransportConfig())
    asyncio.run(sender.send([[make_batch(), make_batch()]]))
    batches = MMapBatchReceiver(tmp_path, 0, 1).receive()
    batch = batches[0]
    reference = weakref.ref(batch["routed_experts"])
    del batch
    assert reference() is None
    # A later microbatch is read independently, not decoded with the whole step.
    path = get_step_path(get_batch_dir(tmp_path), 1) / "rank_0/1.pt"
    path.unlink()
    assert batches[0]["input_ids"].tolist() == [[10, 11, 12, 13]]
    with pytest.raises(FileNotFoundError):
        batches[1]


@pytest.mark.parametrize("prefetch", [0, 2])
def test_spool_waits_for_all_cp_readers_without_dropping_batches(tmp_path, prefetch):
    config = MMapTransportConfig(readers_per_rank=2, max_batch_bytes=100_000, max_pending_bytes=100_000)
    sender = MMapBatchSender(tmp_path, 1, 1, config)

    async def scenario():
        await sender.send([[make_batch()]])
        readers = [MMapBatchReceiver(tmp_path, 0, 1, prefetch) for _ in range(2)]
        readers[1].reader_id = 1
        batches = [reader.receive() for reader in readers]
        iterators = [iter(batch) for batch in batches]
        for iterator in iterators:
            next(iterator)
        next_send = asyncio.create_task(sender.send([[make_batch()]]))
        await asyncio.sleep(0.25)
        assert not next_send.done()
        batches[0].close()
        await asyncio.sleep(0.25)
        assert not next_send.done()
        batches[1].close()
        await asyncio.wait_for(next_send, timeout=5)
        assert not get_step_path(get_batch_dir(tmp_path), 1).exists()
        assert readers[0].can_receive()
        assert readers[0].receive()[0]["input_ids"].tolist() == [[10, 11, 12, 13]]
        with pytest.raises(RuntimeError, match="released"):
            batches[0][0]

    asyncio.run(scenario())


@pytest.mark.parametrize("workers", [1, 4])
def test_oversized_batch_is_not_published(tmp_path, workers):
    sender = MMapBatchSender(tmp_path, 2, 1, MMapTransportConfig(max_batch_bytes=1, write_workers=workers))
    with pytest.raises(ValueError, match="max_batch_bytes"):
        asyncio.run(sender.send([[make_batch()] for _ in range(2)]))
    assert not MMapBatchReceiver(tmp_path, 0, 1).can_receive()
    assert not list(get_batch_dir(tmp_path).iterdir())


@pytest.mark.parametrize("workers", [1, 4])
def test_manifest_accounts_for_all_rank_files(tmp_path, monkeypatch, workers):
    sender = MMapBatchSender(tmp_path, 2, 1, MMapTransportConfig(readers_per_rank=4, write_workers=workers))
    save = torch.save
    barrier = Barrier(2)

    def concurrent_save(batch, path):
        if workers > 1:
            barrier.wait(timeout=5)
        save(batch, path)

    monkeypatch.setattr(torch, "save", concurrent_save)
    other_rank = make_batch()
    other_rank.input_ids[0] = 99
    other_rank.loss_mask = [False] * 4
    asyncio.run(sender.send([[make_batch()], [other_rank]]))
    root = get_step_path(get_batch_dir(tmp_path), 1)
    manifest = json.loads((root / "ready.json").read_text())
    assert manifest["bytes"] == sum(p.stat().st_size for p in root.glob("rank_*/*.pt"))
    assert len(manifest["ranks"]) == 2
    assert manifest["readers_per_rank"] == 4
    assert manifest["ranks"][0]["loss_counts"][0] == 2
    assert manifest["ranks"][1]["loss_counts"][0] == 0
    assert MMapBatchReceiver(tmp_path, 1, 1).receive()[0]["input_ids"][0, 0] == 99


def test_prefetch_is_bounded_resident_and_releases_consumed_batches(tmp_path, monkeypatch):
    sender = MMapBatchSender(tmp_path, 1, 1, MMapTransportConfig())
    sources = [make_batch() for _ in range(6)]
    for index, source in enumerate(sources):
        source.input_ids[0] = index
    asyncio.run(sender.send([sources]))
    loads = []
    load = MMapMicroBatches._load

    def tracked_load(self, index, *, mmap):
        loads.append((index, mmap))
        return load(self, index, mmap=mmap)

    monkeypatch.setattr(MMapMicroBatches, "_load", tracked_load)
    batches = MMapBatchReceiver(tmp_path, 0, 1, prefetch_batches=2).receive()
    iterator = iter(batches)
    first = next(iterator)
    reference = weakref.ref(first["routed_experts"])
    del first
    assert reference() is None
    for future in batches._pending:
        future.result(timeout=5)
    assert loads == [(0, False), (1, False), (2, False)]
    assert len(batches._pending) == 2
    for index, batch in enumerate(iterator, start=1):
        assert batch["input_ids"][0, 0] == index
        assert len(batches._pending) <= 2
    assert batches._executor is None
    assert not batches._pending
    batches.close()


def test_prefetch_failure_propagates_without_acknowledging(tmp_path):
    sender = MMapBatchSender(tmp_path, 1, 1, MMapTransportConfig())
    asyncio.run(sender.send([[make_batch(), make_batch()]]))
    batches = MMapBatchReceiver(tmp_path, 0, 1, prefetch_batches=2).receive()
    (batches.step_dir / "rank_0/1.pt").unlink()
    iterator = iter(batches)
    next(iterator)
    with pytest.raises(FileNotFoundError):
        next(iterator)
    assert batches._executor is None
    assert not batches._pending
    assert not list(batches.step_dir.glob("*.done"))
    batches.close()


def test_close_joins_prefetch_before_acknowledging(tmp_path, monkeypatch):
    sender = MMapBatchSender(tmp_path, 1, 1, MMapTransportConfig())
    asyncio.run(sender.send([[make_batch(), make_batch(), make_batch()]]))
    batches = MMapBatchReceiver(tmp_path, 0, 1, prefetch_batches=2).receive()
    started, release, closing = Event(), Event(), Event()
    load = MMapMicroBatches._load

    def slow_load(self, index, *, mmap):
        if index == 1:
            started.set()
            assert release.wait(timeout=5)
        return load(self, index, mmap=mmap)

    monkeypatch.setattr(MMapMicroBatches, "_load", slow_load)
    iterator = iter(batches)
    next(iterator)
    assert started.wait(timeout=5)

    def close():
        closing.set()
        batches.close()

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(close)
        assert closing.wait(timeout=5)
        assert not future.done()
        assert not list(batches.step_dir.glob("*.done"))
        release.set()
        future.result(timeout=5)
    assert batches.closed
    assert not batches._pending
    assert batches._executor is None
    with pytest.raises(RuntimeError, match="released"):
        next(iterator)


def test_parallel_writer_failure_cleans_up_and_can_retry(tmp_path, monkeypatch):
    sender = MMapBatchSender(tmp_path, 2, 1, MMapTransportConfig(write_workers=2))
    save = torch.save
    barrier = Barrier(2)

    def failing_save(batch, path):
        barrier.wait(timeout=5)
        if path.parent.name == "rank_1":
            raise OSError("injected disk write failure")
        save(batch, path)

    with monkeypatch.context() as patch:
        patch.setattr(torch, "save", failing_save)
        with pytest.raises(OSError, match="injected disk write failure"):
            asyncio.run(sender.send([[make_batch()], [make_batch()]]))
    assert not list(get_batch_dir(tmp_path).iterdir())
    assert sender.current_step == 1
    asyncio.run(sender.send([[make_batch()], [make_batch()]]))
    assert MMapBatchReceiver(tmp_path, 0, 1).can_receive()
