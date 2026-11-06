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


def test_authorization_for_another_attempt_supersedes_an_admission_in_progress(tmp_path):
    """``authorize_recovery`` can reject the candidate while its automatic admission awaits the
    reconciliation; the sender tells its ranks, keeps serving and admits the authorized attempt."""
    sender, receivers = _zmq(tmp_path)
    sender.journal = BatchJournal(tmp_path)
    sender.auto_recover = True
    replacements = []

    async def run():
        await _send_while_ranks_request(sender, receivers)
        checkpoint_path = tmp_path / "checkpoints" / "step_0" / "trainer"
        checkpoint_path.mkdir(parents=True)
        sender.journal.commit_checkpoint(checkpoint_path, 0, f"{sender.journal_session}:0")
        checkpoint = msgspec.json.decode((checkpoint_path / "batch.json").read_bytes(), type=BatchCheckpoint)
        await _send_while_ranks_request(sender, receivers)
        active = sender.handshake.active
        for receiver in receivers:
            receiver.close()
        superseded = TrainerGroup(uuid4().hex, len(receivers), 1, checkpoint)
        authorized = TrainerGroup(uuid4().hex, len(receivers), 1, checkpoint)

        async def reconcile(group):
            if group == superseded:
                sender.authorize_recovery(active.attempt_id, authorized.attempt_id)

        sender.on_replacement = reconcile
        port = int(sender.socket.getsockopt_string(zmq.LAST_ENDPOINT).rsplit(":", 1)[1])

        def connect(group):
            attempt = []
            for rank, data_rank in enumerate(RANK_TO_DATA_RANK):
                receiver = ZMQBatchReceiver(
                    tmp_path, data_rank, rank, len(receivers), 1, ZMQTransportConfig(host="127.0.0.1", port=port), group
                )
                receiver.socket.setsockopt(zmq.RCVTIMEO, 10_000)
                attempt.append(receiver)
            replacements.extend(attempt)
            return asyncio.gather(*(asyncio.to_thread(receiver.wait) for receiver in attempt), return_exceptions=True)

        sending = asyncio.create_task(sender.send(MICRO_BATCH_GRID))
        try:
            results = await connect(superseded)
            assert all(
                isinstance(result, RuntimeError) and "another trainer attempt" in str(result) for result in results
            )
            assert not sender.serve_task.done()
            assert await connect(authorized) == [None] * len(receivers)
            assert sender.handshake.active == authorized
        finally:
            sending.cancel()
            await asyncio.gather(sending, return_exceptions=True)

    try:
        asyncio.run(run())
    finally:
        for transport in [sender, *receivers, *replacements]:
            transport.close()


def test_authorization_for_another_attempt_tells_a_candidate_still_collecting_its_ranks(tmp_path):
    """The ranks of a candidate that ``authorize_recovery`` rejects before all of them said hello get a
    reply too, instead of waiting for one forever while holding their GPUs."""
    sender, receivers = _zmq(tmp_path)
    sender.journal = BatchJournal(tmp_path)
    replacements = []

    async def run():
        await _send_while_ranks_request(sender, receivers)
        active = sender.handshake.active
        candidate = TrainerGroup(uuid4().hex, len(receivers), 1)
        port = int(sender.socket.getsockopt_string(zmq.LAST_ENDPOINT).rsplit(":", 1)[1])
        for rank, data_rank in enumerate(RANK_TO_DATA_RANK[:2]):
            receiver = ZMQBatchReceiver(
                tmp_path, data_rank, rank, len(receivers), 1, ZMQTransportConfig(host="127.0.0.1", port=port), candidate
            )
            receiver.socket.setsockopt(zmq.RCVTIMEO, 5000)
            replacements.append(receiver)
        waits = asyncio.gather(*(asyncio.to_thread(receiver.wait) for receiver in replacements), return_exceptions=True)
        while len(sender.handshake.pending) < len(replacements):
            await asyncio.sleep(0.05)
        sender.authorize_recovery(active.attempt_id, uuid4().hex)
        results = await waits
        assert all(
            isinstance(result, RuntimeError) and "another trainer attempt" in str(result) for result in results
        ), results
        assert sender.handshake.candidate is None and not sender.serve_task.done()

    try:
        asyncio.run(run())
    finally:
        if sender.serve_task is not None:
            sender.serve_task.cancel()
        for transport in [sender, *receivers, *replacements]:
            transport.close()


def test_a_newer_attempt_supersedes_a_candidate_that_never_completed(tmp_path):
    """A candidate whose other ranks never said hello is told it was superseded once a newer attempt
    starts, also while its ranks are still connected, and the newer attempt is admitted."""
    sender, receivers = _zmq(tmp_path)
    sender.journal = BatchJournal(tmp_path)
    sender.auto_recover = True
    replacements = []

    async def run():
        await _send_while_ranks_request(sender, receivers)
        checkpoint_path = tmp_path / "checkpoints" / "step_0" / "trainer"
        checkpoint_path.mkdir(parents=True)
        sender.journal.commit_checkpoint(checkpoint_path, 0, f"{sender.journal_session}:0")
        checkpoint = msgspec.json.decode((checkpoint_path / "batch.json").read_bytes(), type=BatchCheckpoint)
        await _send_while_ranks_request(sender, receivers)
        for receiver in receivers:
            receiver.close()
        port = int(sender.socket.getsockopt_string(zmq.LAST_ENDPOINT).rsplit(":", 1)[1])

        def connect(group, ranks):
            attempt = []
            for rank in ranks:
                receiver = ZMQBatchReceiver(
                    tmp_path,
                    RANK_TO_DATA_RANK[rank],
                    rank,
                    len(receivers),
                    1,
                    ZMQTransportConfig(host="127.0.0.1", port=port),
                    group,
                )
                receiver.socket.setsockopt(zmq.RCVTIMEO, 10_000)
                attempt.append(receiver)
            replacements.extend(attempt)
            return asyncio.gather(*(asyncio.to_thread(receiver.wait) for receiver in attempt), return_exceptions=True)

        incomplete = TrainerGroup(uuid4().hex, len(receivers), 1, checkpoint)
        newer = TrainerGroup(uuid4().hex, len(receivers), 1, checkpoint)
        sending = asyncio.create_task(sender.send(MICRO_BATCH_GRID))
        try:
            superseded = connect(incomplete, [0, 1])
            while len(sender.handshake.pending) < 2:
                await asyncio.sleep(0.05)
            admitted = connect(newer, range(len(receivers)))
            results = await superseded
            assert all(
                isinstance(result, RuntimeError) and "newer trainer attempt" in str(result) for result in results
            ), results
            assert await admitted == [None] * len(receivers)
            assert sender.handshake.active == newer
        finally:
            sending.cancel()
            await asyncio.gather(sending, return_exceptions=True)

    try:
        asyncio.run(run())
    finally:
        for transport in [sender, *receivers, *replacements]:
            transport.close()


def test_authorization_while_the_replay_plan_is_read_admits_the_candidate(tmp_path, monkeypatch):
    """``authorize_recovery`` for the candidate itself can arrive while its replay plan is read; the
    candidate is admitted, not turned away as unauthorized with the authorization left behind."""
    import prime_rl.transports.batch.zmq as zmq_transport

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
        loop = asyncio.get_running_loop()
        plan_replay = zmq_transport.plan_replay

        def plan_while_authorized(*args):
            async def authorize():
                sender.authorize_recovery(active.attempt_id, group.attempt_id)

            asyncio.run_coroutine_threadsafe(authorize(), loop).result()
            return plan_replay(*args)

        monkeypatch.setattr(zmq_transport, "plan_replay", plan_while_authorized)
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
            assert results == [None] * len(replacements)
            assert sender.handshake.active == group
        finally:
            sending.cancel()
            await asyncio.gather(sending, return_exceptions=True)

    try:
        asyncio.run(run())
    finally:
        for transport in [sender, *receivers, *replacements]:
            transport.close()


def test_a_rank_lost_during_admission_discards_the_replacement(tmp_path):
    """A replacement whose rank disconnects while it is reconciled is not accepted without that rank;
    the next complete attempt is admitted."""
    sender, receivers = _zmq(tmp_path)
    sender.journal = BatchJournal(tmp_path)
    sender.auto_recover = True
    replacements = []
    dropped = zmq.Context.instance().socket(zmq.DEALER)

    async def run():
        await _send_while_ranks_request(sender, receivers)
        checkpoint_path = tmp_path / "checkpoints" / "step_0" / "trainer"
        checkpoint_path.mkdir(parents=True)
        sender.journal.commit_checkpoint(checkpoint_path, 0, f"{sender.journal_session}:0")
        checkpoint = msgspec.json.decode((checkpoint_path / "batch.json").read_bytes(), type=BatchCheckpoint)
        await _send_while_ranks_request(sender, receivers)
        for receiver in receivers:
            receiver.close()
        broken = TrainerGroup(uuid4().hex, len(receivers), 1, checkpoint)
        complete = TrainerGroup(uuid4().hex, len(receivers), 1, checkpoint)
        port = int(sender.socket.getsockopt_string(zmq.LAST_ENDPOINT).rsplit(":", 1)[1])
        transport = ZMQTransportConfig(host="127.0.0.1", port=port)

        def receiver(group: TrainerGroup, rank: int) -> ZMQBatchReceiver:
            replacement = ZMQBatchReceiver(tmp_path, RANK_TO_DATA_RANK[rank], rank, len(receivers), 1, transport, group)
            replacement.socket.setsockopt(zmq.RCVTIMEO, 10_000)
            replacements.append(replacement)
            return replacement

        async def reconcile(group):
            if group == broken:
                # Rank 1 of the broken attempt goes away while it is reconciled. Its socket belongs
                # to this thread: ZMQ sockets must not be closed while another thread uses them.
                dropped.close(linger=0)
                await asyncio.sleep(0.5)

        sender.on_replacement = reconcile
        sending = asyncio.create_task(sender.send(MICRO_BATCH_GRID))
        try:
            dropped.connect(f"tcp://127.0.0.1:{port}")
            dropped.send(msgspec.msgpack.encode(TrainerHello(broken, 1, RANK_TO_DATA_RANK[1])))
            others = [receiver(broken, rank) for rank in range(len(receivers)) if rank != 1]
            results = await asyncio.gather(*(asyncio.to_thread(rank.wait) for rank in others), return_exceptions=True)
            assert all(isinstance(result, RuntimeError) and "disconnected" in str(result) for result in results)
            assert sender.handshake.active != broken and not sender.serve_task.done()
            attempt = [receiver(complete, rank) for rank in range(len(receivers))]
            assert await asyncio.gather(*(asyncio.to_thread(rank.wait) for rank in attempt)) == [None] * len(attempt)
            assert sender.handshake.active == complete
        finally:
            sending.cancel()
            await asyncio.gather(sending, return_exceptions=True)

    try:
        asyncio.run(run())
    finally:
        dropped.close(linger=0)
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


def test_zmq_trainer_is_asked_to_publish_its_policy_again(tmp_path):
    """With every rank waiting and no batch pending, the orchestrator asks for the policy the trainer
    holds; a request for any other version is dropped and the batch follows as usual."""
    transport = ZMQTransportConfig(host="127.0.0.1", port=_free_port())
    sender = ZMQBatchSender(tmp_path, data_world_size=2, current_step=3, transport=transport)
    group = TrainerGroup(uuid4().hex, len(RANK_TO_DATA_RANK), 3)
    receivers = [
        ZMQBatchReceiver(
            tmp_path, data_rank, rank, len(RANK_TO_DATA_RANK), current_step=3, transport=transport, trainer_group=group
        )
        for rank, data_rank in enumerate(RANK_TO_DATA_RANK)
    ]

    async def run() -> tuple[list[int | None], list[int | None]]:
        sender.start()
        waits = [asyncio.create_task(asyncio.to_thread(receiver.wait)) for receiver in receivers]
        await asyncio.sleep(0.5)
        sender.request_republish(2)
        await asyncio.wait_for(asyncio.gather(*waits), timeout=10)
        asked = [receiver.republish_requested() for receiver in receivers]

        waits = [asyncio.create_task(asyncio.to_thread(receiver.wait)) for receiver in receivers]
        await asyncio.sleep(0.5)
        sender.request_republish(7)  # not the trainer's policy (v2): dropped
        await asyncio.sleep(0.5)
        assert not any(wait.done() for wait in waits)
        await asyncio.wait_for(asyncio.gather(sender.send(MICRO_BATCH_GRID), *waits), timeout=10)
        return asked, [receiver.republish_requested() for receiver in receivers]

    asked, after_batch = asyncio.run(run())
    assert asked == [2, 2, 2, 2]
    assert after_batch == [None] * 4
    assert all(len(receiver.receive()) == 1 for receiver in receivers)
    for transport_end in [sender, *receivers]:
        transport_end.close()


@pytest.mark.parametrize("journal", [True, False], ids=["journal", "no-journal"])
def test_a_rank_lost_before_a_republish_fences_the_attempt(tmp_path, monkeypatch, journal):
    """As for a batch, a republish request that cannot reach every rank fences the attempt: the
    orchestrator waits for a replacement instead of a policy only some ranks were asked for. Without a
    journal no replacement can continue, and the failure ends the serve loop, which fails the run."""
    transport = ZMQTransportConfig(host="127.0.0.1", port=_free_port())
    sender = ZMQBatchSender(tmp_path, data_world_size=2, current_step=3, transport=transport)
    if journal:
        sender.journal = BatchJournal(tmp_path)
    group = TrainerGroup(uuid4().hex, len(RANK_TO_DATA_RANK), 3)
    receivers = [
        ZMQBatchReceiver(
            tmp_path, data_rank, rank, len(RANK_TO_DATA_RANK), current_step=3, transport=transport, trainer_group=group
        )
        for rank, data_rank in enumerate(RANK_TO_DATA_RANK)
    ]
    for receiver in receivers:
        receiver.socket.setsockopt(zmq.RCVTIMEO, 3000)

    async def run() -> None:
        sender.start()
        waits = [asyncio.create_task(asyncio.to_thread(receiver.wait)) for receiver in receivers]
        await asyncio.sleep(0.5)
        lost = sender.handshake.members[1][0]
        reply = sender._reply

        async def unreachable(identity: bytes, frames: list[bytes]) -> bool:
            return False if identity == lost else await reply(identity, frames)

        monkeypatch.setattr(sender, "_reply", unreachable)
        sender.request_republish(2)
        await asyncio.gather(*waits, return_exceptions=True)
        if journal:
            assert sender.failed_attempt == group.attempt_id and not sender.requests
            assert not sender.serve_task.done()
        else:
            await asyncio.wait_for(asyncio.wait({sender.serve_task}), 5)
            assert isinstance(sender.serve_task.exception(), ConnectionError)

    try:
        asyncio.run(run())
    finally:
        for transport_end in [sender, *receivers]:
            transport_end.close()
