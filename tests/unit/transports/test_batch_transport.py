import asyncio
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from prime_rl.configs.shared import ZMQTransportConfig
from prime_rl.transports.batch.base import BatchReceiver, BatchSender
from prime_rl.transports.batch.filesystem import FileSystemBatchReceiver, FileSystemBatchSender
from prime_rl.transports.batch.types import SampleRef, StepBatch
from prime_rl.transports.batch.zmq import ZMQBatchReceiver, ZMQBatchSender
from prime_rl.utils.utils import get_free_port


def _batch(token: int) -> StepBatch:
    """A one-sample step whose sample id names ``token``."""
    sample = SampleRef(
        url="http://h:1", id=str(token), num_tokens=1, env_name="env", trace_id="t", branch_index=0, temperature=1.0
    )
    return StepBatch(samples=[sample], trained_version=0, min_version=0)


BATCH = _batch(0)
WORLD_SIZE = 4


def _filesystem_receivers(output_dir: Path, transport: None = None) -> list[BatchReceiver]:
    return [FileSystemBatchReceiver(output_dir, rank, world_size=WORLD_SIZE) for rank in range(WORLD_SIZE)]


def _filesystem(output_dir: Path, retain: bool = False) -> tuple[BatchSender, list[BatchReceiver], None]:
    sender = FileSystemBatchSender(output_dir, retain=retain)
    return sender, _filesystem_receivers(output_dir), None


def _zmq_receivers(output_dir: Path, transport: ZMQTransportConfig) -> list[BatchReceiver]:
    return [
        ZMQBatchReceiver(output_dir, rank, world_size=WORLD_SIZE, current_step=0, transport=transport)
        for rank in range(WORLD_SIZE)
    ]


def _zmq(output_dir: Path, retain: bool = False) -> tuple[BatchSender, list[BatchReceiver], ZMQTransportConfig]:
    transport = ZMQTransportConfig(host="127.0.0.1", port=get_free_port(), retain=retain)
    sender = ZMQBatchSender(output_dir, current_step=0, transport=transport)
    return sender, _zmq_receivers(output_dir, transport), transport


async def _send_while_ranks_request(
    sender: BatchSender, receivers: list[BatchReceiver]
) -> tuple[bool, list[StepBatch]]:
    send = asyncio.create_task(sender.send(BATCH))
    # every rank but the last asks
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

    started = time.time()
    result = asyncio.run(_send_while_ranks_request(sender, receivers))

    sent_before_last_request, received = result
    assert sent_before_last_request is False
    # Every trainer rank receives the whole step
    assert received == [BATCH] * WORLD_SIZE
    # The sender knows when the last rank asked: the trainer was busy until then (file mtimes are coarse)
    assert sender.last_requested_at is not None
    assert started + 0.4 <= sender.last_requested_at <= time.time() + 1
    for transport in [sender, *receivers]:
        transport.close()


def _received_tokens(receiver: BatchReceiver) -> list[int]:
    return [int(sample.id) for sample in receiver.receive().samples]


async def _send(sender: BatchSender, receivers: list[BatchReceiver]) -> None:
    """Ship the sender's current step, a step whose sample names it."""
    step = sender.current_step
    waits = [asyncio.to_thread(receiver.wait) for receiver in receivers]
    await asyncio.wait_for(asyncio.gather(sender.send(_batch(step)), *waits), timeout=10)
    for receiver in receivers:
        assert _received_tokens(receiver) == [step]


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
            assert _received_tokens(receiver) == [step]
    assert sender.current_step == 2

    for receiver in restarted:
        receiver.committed = 0
    asyncio.run(_send(sender, restarted))
    assert sender.log.read(0) is None
    assert sender.log.read(1) is not None
    for transport in [sender, *receivers, *restarted]:
        transport.close()
