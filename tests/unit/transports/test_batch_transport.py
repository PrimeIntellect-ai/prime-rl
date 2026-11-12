import asyncio
import socket
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from prime_rl.configs.shared import ZMQTransportConfig
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


def _filesystem_receivers(output_dir: Path, transport: None = None) -> list[BatchReceiver]:
    return [
        FileSystemBatchReceiver(output_dir, data_rank, rank, world_size=len(RANK_TO_DATA_RANK))
        for rank, data_rank in enumerate(RANK_TO_DATA_RANK)
    ]


def _filesystem(output_dir: Path, retain: bool = False) -> tuple[BatchSender, list[BatchReceiver], None]:
    sender = FileSystemBatchSender(output_dir, data_world_size=2, retain=retain)
    return sender, _filesystem_receivers(output_dir), None


def _zmq_receivers(output_dir: Path, transport: ZMQTransportConfig) -> list[BatchReceiver]:
    return [
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


def _zmq(output_dir: Path, retain: bool = False) -> tuple[BatchSender, list[BatchReceiver], ZMQTransportConfig]:
    transport = ZMQTransportConfig(host="127.0.0.1", port=_free_port(), retain=retain)
    sender = ZMQBatchSender(output_dir, data_world_size=2, current_step=0, transport=transport)
    return sender, _zmq_receivers(output_dir, transport), transport


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
def test_step_is_sent_only_after_every_trainer_rank_requests_it(
    tmp_path: Path, setup_transport: Callable[..., tuple[BatchSender, list[BatchReceiver], object]]
):
    sender, receivers, _ = setup_transport(tmp_path)

    result = asyncio.run(_send_while_ranks_request(sender, receivers))

    sent_before_last_request, received = result
    assert sent_before_last_request is False
    # Streams arrive as raw buffers: compare each rank's tokens with what its data rank was sent
    assert [[np.frombuffer(mb.input_ids, np.int32).tolist() for mb in batches] for batches in received] == [
        [mb.input_ids.tolist() for mb in MICRO_BATCH_GRID[data_rank]] for data_rank in RANK_TO_DATA_RANK
    ]
    for transport in [sender, *receivers]:
        transport.close()


def _tokens(step: int, data_rank: int) -> int:
    return 2 * step + data_rank


def _received_tokens(receiver: BatchReceiver) -> list[int]:
    (micro_batch,) = receiver.receive()
    return np.frombuffer(micro_batch.input_ids, np.int32).tolist()


async def _send(sender: BatchSender, receivers: list[BatchReceiver]) -> None:
    """Ship the sender's current step, a grid whose tokens name the step and data rank."""
    step = sender.current_step
    grid = [[_micro_batch(_tokens(step, data_rank))] for data_rank in range(2)]
    waits = [asyncio.to_thread(receiver.wait) for receiver in receivers]
    await asyncio.wait_for(asyncio.gather(sender.send(grid), *waits), timeout=10)
    for receiver in receivers:
        assert _received_tokens(receiver) == [_tokens(step, receiver.data_rank)]


@pytest.mark.parametrize(
    "setup_transport, setup_receivers",
    [(_filesystem, _filesystem_receivers), (_zmq, _zmq_receivers)],
    ids=["filesystem", "zmq"],
)
def test_restarted_ranks_get_the_retained_steps_until_a_checkpoint_covers_them(
    tmp_path: Path, setup_transport: Callable, setup_receivers: Callable
):
    sender, receivers, transport = setup_transport(tmp_path, retain=True)
    for _ in range(2):
        asyncio.run(_send(sender, receivers))

    # A trainer restarted from before step 0 gets both steps while the sender waits on step 2.
    restarted = setup_receivers(tmp_path, transport)
    for step in range(2):
        for receiver in restarted:
            receiver.wait()
            assert _received_tokens(receiver) == [_tokens(step, receiver.data_rank)]
    assert sender.current_step == 2

    for receiver in restarted:
        receiver.committed = 0
    asyncio.run(_send(sender, restarted))
    assert sender.log.read(0, data_rank=0) is None
    assert sender.log.read(1, data_rank=0) is not None
    for transport in [sender, *receivers, *restarted]:
        transport.close()
