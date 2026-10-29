import asyncio
import socket
import threading
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

import msgspec
import numpy as np
import pytest
import zmq

from prime_rl.configs.shared import ZMQTransportConfig
from prime_rl.transports.batch.base import BatchReceiver, BatchSender
from prime_rl.transports.batch.filesystem import FileSystemBatchReceiver, FileSystemBatchSender
from prime_rl.transports.batch.journal import BatchCheckpoint, BatchJournal
from prime_rl.transports.batch.recovery import RecoveryRequired, TrainerGroup, TrainerHello
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
    group = TrainerGroup(uuid4().hex, len(RANK_TO_DATA_RANK), 0)
    receivers = [
        ZMQBatchReceiver(
            output_dir,
            data_rank,
            rank,
            world_size=len(RANK_TO_DATA_RANK),
            current_step=0,
            transport=transport,
            trainer_group=group,
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
            assert not sending.done()
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


@pytest.mark.parametrize("authorized", [False, True], ids=["plan-only", "replay"])
def test_replacement_replays_only_when_authorized(tmp_path, authorized):
    sender, receivers = _zmq(tmp_path)
    sender.journal = BatchJournal(tmp_path)
    replacements = []

    async def run():
        await _send_while_ranks_request(sender, receivers)
        checkpoint_path = tmp_path / "checkpoints" / "step_0" / "trainer"
        checkpoint_path.mkdir(parents=True)
        sender.journal.commit_checkpoint(checkpoint_path, 0, f"{sender.journal_session}:0")
        checkpoint = msgspec.json.decode((checkpoint_path / "batch.json").read_bytes(), type=BatchCheckpoint)
        await _send_while_ranks_request(sender, receivers)
        active = sender.handshake.active
        group = TrainerGroup(uuid4().hex, len(receivers), 1, checkpoint)
        if authorized:
            sender.authorize_recovery(active.attempt_id, group.attempt_id)
        port = int(sender.socket.getsockopt_string(zmq.LAST_ENDPOINT).rsplit(":", 1)[1])
        for rank, data_rank in enumerate(RANK_TO_DATA_RANK):
            receiver = ZMQBatchReceiver(
                tmp_path, data_rank, rank, len(receivers), 1, ZMQTransportConfig(host="127.0.0.1", port=port), group
            )
            receiver.socket.setsockopt(zmq.RCVTIMEO, 5000)
            replacements.append(receiver)
        sending = asyncio.create_task(sender.send(MICRO_BATCH_GRID))
        try:
            results = await asyncio.gather(
                *(asyncio.to_thread(receiver.wait) for receiver in replacements), return_exceptions=True
            )
            # Batch 1 was delivered to the old attempt and replays from the journal; batch 2 is live.
            replayed, live = (f"{sender.journal_session}:{step}" for step in (1, 2))
            if authorized:
                assert results == [None] * len(replacements)
                assert sender.handshake.active == group and not sending.done()
                for receiver in replacements:
                    assert [record.batch_id for record in receiver.replay_plan.batches] == [replayed]
                    batch = await asyncio.to_thread(receiver.receive)
                    assert sender.encode(batch) == sender.journal.read(replayed)[receiver.data_rank]
                await asyncio.wait_for(
                    asyncio.gather(sending, *(asyncio.to_thread(receiver.wait) for receiver in replacements)), 5
                )
                for receiver in replacements:
                    batch = await asyncio.to_thread(receiver.receive)
                    assert sender.encode(batch) == sender.journal.read(live)[receiver.data_rank]
                assert sender.replay_batches_sent == 1 and sender.current_step == 3
                await _send_while_ranks_request(sender, replacements)
                assert sender.current_step == 4 and sender.replay_batches_sent == 1
            else:
                assert all(isinstance(result, RecoveryRequired) for result in results)
                assert all([record.batch_id for record in result.plan.batches] == [replayed] for result in results)
                assert sender.handshake.active == active and not sending.done()
                # A probe does not displace the accepted trainer.
                await asyncio.wait_for(
                    asyncio.gather(sending, *(asyncio.to_thread(receiver.wait) for receiver in receivers)), 5
                )
                for receiver in receivers:
                    assert receiver.receive()[0].journal_batch_id == live
        finally:
            if not sending.done():
                sending.cancel()
                await asyncio.gather(sending, return_exceptions=True)

    try:
        asyncio.run(run())
    finally:
        for transport in [sender, *receivers, *replacements]:
            transport.close()


def test_recovery_authorization_checks_identity_and_fences_old_requests(tmp_path):
    sender, receivers = _zmq(tmp_path)
    group = receivers[0].trainer_group
    try:
        for rank, data_rank in enumerate(RANK_TO_DATA_RANK):
            sender.handshake.add(str(rank).encode(), TrainerHello(group, rank, data_rank))
        sender.handshake.accept()
        with pytest.raises(ValueError, match="journaling"):
            sender.authorize_recovery(group.attempt_id, uuid4().hex)
        sender.journal = BatchJournal(tmp_path)
        with pytest.raises(ValueError, match="active trainer"):
            sender.authorize_recovery(uuid4().hex, uuid4().hex)
        with pytest.raises(ValueError, match="new trainer"):
            sender.authorize_recovery(group.attempt_id, group.attempt_id)
        replacement = uuid4().hex
        sender.authorize_recovery(group.attempt_id, replacement)
        assert sender.failed_attempt == group.attempt_id
        assert sender.recovery_authorization == (group.attempt_id, replacement)
        with pytest.raises(ValueError, match="already authorized"):
            sender.authorize_recovery(group.attempt_id, uuid4().hex)
    finally:
        for transport in [sender, *receivers]:
            transport.close()
