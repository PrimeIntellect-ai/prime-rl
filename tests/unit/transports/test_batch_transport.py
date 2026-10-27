import asyncio
import socket
import threading
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from prime_rl.configs.shared import ZMQTransportConfig
from prime_rl.transports.batch.base import BatchReceiver, BatchSender
from prime_rl.transports.batch.filesystem import FileSystemBatchReceiver, FileSystemBatchSender
from prime_rl.transports.batch.journal import BatchJournal
from prime_rl.transports.batch.types import MicroBatch
from prime_rl.transports.batch.zmq import ZMQBatchReceiver, ZMQBatchSender


def _micro_batch(token: int) -> MicroBatch:
    return MicroBatch(
        input_ids=np.array([token], np.int32),
        loss_mask=np.array([True]),
        advantages=np.array([1.0], np.float32),
        inference_logprobs=np.array([0.0], np.float32),
        position_ids=np.array([0], np.int32),
        sequence_lengths=[1],
        temperatures=np.array([1.0], np.float32),
        env_names=["env"],
        seq_lens=[1],
    )


MICRO_BATCH_GRID = [[_micro_batch(0)], [_micro_batch(1)]]
# Two data ranks, each split across two trainer ranks (e.g. tensor parallel)
RANK_TO_DATA_RANK = [0, 0, 1, 1]


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _filesystem(output_dir: Path) -> tuple[BatchSender, list[BatchReceiver]]:
    sender = FileSystemBatchSender(output_dir, data_world_size=2)
    receivers = [
        FileSystemBatchReceiver(output_dir, data_rank, rank, world_size=len(RANK_TO_DATA_RANK))
        for rank, data_rank in enumerate(RANK_TO_DATA_RANK)
    ]
    return sender, receivers


def _zmq(output_dir: Path) -> tuple[BatchSender, list[BatchReceiver]]:
    transport = ZMQTransportConfig(host="127.0.0.1", port=_free_port())
    sender = ZMQBatchSender(output_dir, data_world_size=2, current_step=0, transport=transport)
    receivers = [
        ZMQBatchReceiver(
            output_dir,
            data_rank,
            rank,
            world_size=len(RANK_TO_DATA_RANK),
            current_step=0,
            transport=transport,
        )
        for rank, data_rank in enumerate(RANK_TO_DATA_RANK)
    ]
    return sender, receivers


async def _send_while_ranks_request(
    sender: BatchSender, receivers: list[BatchReceiver]
) -> tuple[bool, list[list[MicroBatch]]]:
    send = asyncio.create_task(sender.send(MICRO_BATCH_GRID))
    # every rank but the last asks, which covers both data ranks
    early_requests = [asyncio.create_task(asyncio.to_thread(receiver.wait)) for receiver in receivers[:-1]]
    await asyncio.sleep(0.5)
    sent_before_last_request = send.done()

    last_request = asyncio.create_task(asyncio.to_thread(receivers[-1].wait))
    await asyncio.wait_for(asyncio.gather(send, *early_requests, last_request), timeout=10)

    return sent_before_last_request, [receiver.receive() for receiver in receivers]


@pytest.mark.parametrize("setup_transport", [_filesystem, _zmq], ids=["filesystem", "zmq"])
@pytest.mark.parametrize("journal", [False, True], ids=["plain", "journal"])
def test_step_is_sent_only_after_every_trainer_rank_requests_it(
    tmp_path: Path, setup_transport: Callable[[Path], tuple[BatchSender, list[BatchReceiver]]], journal: bool
):
    sender, receivers = setup_transport(tmp_path)
    if journal:
        sender.journal = BatchJournal(tmp_path)

    result = asyncio.run(_send_while_ranks_request(sender, receivers))

    sent_before_last_request, received = result
    assert sent_before_last_request is False
    # Streams arrive as raw buffers: compare each rank's tokens with what its data rank was sent
    assert [[np.frombuffer(mb.input_ids, np.int32).tolist() for mb in batches] for batches in received] == [
        [mb.input_ids.tolist() for mb in MICRO_BATCH_GRID[data_rank]] for data_rank in RANK_TO_DATA_RANK
    ]
    if journal:
        batch_id = f"{sender.journal_session}:0"
        assert {mb.journal_batch_id for batches in received for mb in batches} == {batch_id}
        assert sender.journal.read(batch_id) == [sender.encode(received[rank]) for rank in (0, 2)]
        assert sender.journal_write_time > 0
        assert all(mb.journal_batch_id is None for batches in MICRO_BATCH_GRID for mb in batches)
    else:
        assert not (tmp_path / "batch_journal").exists()
        assert sender.journal_write_time == 0
    for transport in [sender, *receivers]:
        transport.close()


@pytest.mark.parametrize("setup_transport", [_filesystem, _zmq], ids=["filesystem", "zmq"])
def test_journal_disk_io_does_not_block_event_loop(tmp_path, setup_transport, monkeypatch):
    sender, receivers = setup_transport(tmp_path)
    sender.journal = BatchJournal(tmp_path)
    entered = threading.Event()
    release = threading.Event()
    write = sender.journal.write
    main_thread = threading.get_ident()

    def slow_write(batch_id, payloads):
        assert threading.get_ident() != main_thread
        entered.set()
        assert release.wait(timeout=5), "Event loop could not run while the disk writer was blocked"
        return write(batch_id, payloads)

    monkeypatch.setattr(sender.journal, "write", slow_write)

    async def run():
        sending = asyncio.create_task(sender.send(MICRO_BATCH_GRID))
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            assert not sending.done()
            requests = [asyncio.create_task(asyncio.to_thread(receiver.wait)) for receiver in receivers]
            await asyncio.sleep(0.05)
            assert not any(request.done() for request in requests)
            # This coroutine must run before the worker can finish its write.
            release.set()
            await asyncio.wait_for(asyncio.gather(sending, *requests), 10)
            for receiver in receivers:
                receiver.receive()
        finally:
            release.set()
            if not sending.done():
                sending.cancel()
                await asyncio.gather(sending, return_exceptions=True)

    try:
        asyncio.run(run())
    finally:
        for transport in [sender, *receivers]:
            transport.close()
