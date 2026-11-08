import asyncio
import socket
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from prime_rl.configs.shared import ZMQTransportConfig
from prime_rl.trainer.rl.data import ReplayLog
from prime_rl.transports.batch.base import BatchReceiver, BatchSender
from prime_rl.transports.batch.filesystem import FileSystemBatchReceiver, FileSystemBatchSender
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


def _wait(receiver: BatchReceiver) -> None:
    receiver.request()
    while not receiver.poll(1):
        pass


async def _send_while_ranks_request(
    sender: BatchSender, receivers: list[BatchReceiver]
) -> tuple[bool, list[list[MicroBatch]]]:
    send = asyncio.create_task(sender.send(MICRO_BATCH_GRID))
    # every rank but the last asks, which covers both data ranks
    early_requests = [asyncio.create_task(asyncio.to_thread(_wait, receiver)) for receiver in receivers[:-1]]
    await asyncio.sleep(0.5)
    sent_before_last_request = send.done()

    last_request = asyncio.create_task(asyncio.to_thread(_wait, receivers[-1]))
    await asyncio.wait_for(asyncio.gather(send, *early_requests, last_request), timeout=10)

    return sent_before_last_request, [receiver.receive() for receiver in receivers]


@pytest.mark.parametrize("setup_transport", [_filesystem, _zmq], ids=["filesystem", "zmq"])
def test_step_is_sent_only_after_every_trainer_rank_requests_it(
    tmp_path: Path, setup_transport: Callable[[Path], tuple[BatchSender, list[BatchReceiver]]]
):
    sender, receivers = setup_transport(tmp_path)

    result = asyncio.run(_send_while_ranks_request(sender, receivers))

    sent_before_last_request, received = result
    assert sent_before_last_request is False
    # Streams arrive as raw buffers: compare each rank's tokens with what its data rank was sent
    assert [[np.frombuffer(mb.input_ids, np.int32).tolist() for mb in batches] for batches in received] == [
        [mb.input_ids.tolist() for mb in MICRO_BATCH_GRID[data_rank]] for data_rank in RANK_TO_DATA_RANK
    ]
    for transport in [sender, *receivers]:
        transport.close()


def _fetch(receiver: BatchReceiver) -> list[int]:
    _wait(receiver)
    return [np.frombuffer(mb.input_ids, np.int32).tolist()[0] for mb in receiver.receive()]


def test_zmq_restarted_rank_gets_the_last_step_again(tmp_path: Path):
    async def run():
        transport = ZMQTransportConfig(host="127.0.0.1", port=_free_port())
        sender = ZMQBatchSender(tmp_path, data_world_size=2, current_step=0, transport=transport)
        ranks = [ZMQBatchReceiver(tmp_path, rank, rank, 2, current_step=0, transport=transport) for rank in (0, 1)]
        send = asyncio.create_task(sender.send([[_micro_batch(10)], [_micro_batch(11)]]))
        assert await asyncio.gather(*(asyncio.to_thread(_fetch, rank) for rank in ranks)) == [[10], [11]]
        await send
        # Rank 1 restarts before saving step 0, asks for it again on a new socket, then for step 1
        ranks[1].close()
        ranks[1] = ZMQBatchReceiver(tmp_path, 1, 1, 2, current_step=0, transport=transport)
        send = asyncio.create_task(sender.send([[_micro_batch(20)], [_micro_batch(21)]]))
        assert await asyncio.to_thread(_fetch, ranks[1]) == [11]
        assert await asyncio.gather(*(asyncio.to_thread(_fetch, rank) for rank in ranks)) == [[20], [21]]
        await send
        for transport in [sender, *ranks]:
            transport.close()

    asyncio.run(run())


def test_replay_log_replays_a_step_only_if_every_data_rank_saved_it(tmp_path: Path):
    logs = [ReplayLog(tmp_path, dp_rank=rank, dp_world_size=2, writer=True) for rank in (0, 1)]
    for step in (4, 5):
        logs[0].save(step, [0], [_micro_batch(step)])
    logs[1].save(5, [], [_micro_batch(50)])
    for log in logs:
        log.flush()
    assert logs[0].load(4) is None and logs[1].load(4) is None
    failed, micro_batches = logs[1].load(5)
    assert failed == [] and np.frombuffer(micro_batches[0].input_ids, np.int32).tolist() == [50]
    logs[0].prune(4)
    assert sorted(path.name for path in (tmp_path / "replay").iterdir()) == ["step_5"]
