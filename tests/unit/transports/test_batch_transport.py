import asyncio
import socket
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from prime_rl.configs.shared import ZMQTransportConfig
from prime_rl.transports.batch.base import BatchReceiver, BatchSender, dummy_micro_batch, redistribute
from prime_rl.transports.batch.filesystem import FileSystemBatchReceiver, FileSystemBatchSender
from prime_rl.transports.batch.types import MicroBatch, MMRefs
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


def _filesystem_receivers(
    output_dir: Path,
    transport: None = None,
    rank_to_data_rank: list[int] = RANK_TO_DATA_RANK,
    micro_batch_multiple: int = 1,
) -> list[BatchReceiver]:
    return [
        FileSystemBatchReceiver(
            output_dir,
            data_rank,
            len(set(rank_to_data_rank)),
            micro_batch_multiple,
            rank,
            world_size=len(rank_to_data_rank),
        )
        for rank, data_rank in enumerate(rank_to_data_rank)
    ]


def _filesystem(output_dir: Path, retain: bool = False) -> tuple[BatchSender, list[BatchReceiver], None]:
    sender = FileSystemBatchSender(output_dir, data_world_size=2, retain=retain)
    return sender, _filesystem_receivers(output_dir), None


def _zmq_receivers(
    output_dir: Path,
    transport: ZMQTransportConfig,
    rank_to_data_rank: list[int] = RANK_TO_DATA_RANK,
    micro_batch_multiple: int = 1,
) -> list[BatchReceiver]:
    return [
        ZMQBatchReceiver(
            output_dir,
            data_rank,
            len(set(rank_to_data_rank)),
            micro_batch_multiple,
            rank,
            world_size=len(rank_to_data_rank),
            current_step=0,
            transport=transport,
        )
        for rank, data_rank in enumerate(rank_to_data_rank)
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


def _trained_tokens(receivers: list[BatchReceiver]) -> list[list[int]]:
    """Per data rank, the tokens of the micro batches that train, after checking that every
    rank holds the same number of micro batches."""
    received = {receiver.data_rank: receiver.receive() for receiver in receivers}
    assert len({len(micro_batches) for micro_batches in received.values()}) == 1
    return [
        [
            token
            for micro_batch in received[data_rank]
            if np.frombuffer(micro_batch.loss_mask, np.bool_).any()
            for token in np.frombuffer(micro_batch.input_ids, np.int32).tolist()
        ]
        for data_rank in sorted(received)
    ]


@pytest.mark.parametrize("rank_to_data_rank", [[0], [0, 1, 2]], ids=["one-data-rank", "three-data-ranks"])
@pytest.mark.parametrize(
    "setup_transport, setup_receivers",
    [(_filesystem, _filesystem_receivers), (_zmq, _zmq_receivers)],
    ids=["filesystem", "zmq"],
)
def test_a_trainer_restarted_with_another_number_of_data_ranks_gets_every_micro_batch(
    tmp_path: Path, setup_transport: Callable, setup_receivers: Callable, rank_to_data_rank: list[int]
):
    sender, receivers, transport = setup_transport(tmp_path, retain=True)
    asyncio.run(_send(sender, receivers))

    # The retained step was published for two data ranks.
    restarted = setup_receivers(tmp_path, transport, rank_to_data_rank)
    for receiver in restarted:
        receiver.wait()
    assert sorted(token for tokens in _trained_tokens(restarted) for token in tokens) == [_tokens(0, 0), _tokens(0, 1)]

    # So was the step the sender is waiting on, which it deals again for the new trainer.
    async def send_live() -> None:
        grid = [[_micro_batch(_tokens(1, data_rank))] for data_rank in range(2)]
        waits = [asyncio.to_thread(receiver.wait) for receiver in restarted]
        await asyncio.wait_for(asyncio.gather(sender.send(grid), *waits), timeout=10)

    asyncio.run(send_live())
    assert sorted(token for tokens in _trained_tokens(restarted) for token in tokens) == [_tokens(1, 0), _tokens(1, 1)]
    assert sender.data_world_size == len(rank_to_data_rank)
    assert sender.log.data_world_size(1) == len(rank_to_data_rank)
    for transport in [sender, *receivers, *restarted]:
        transport.close()


def test_redistribute_drops_old_dummies_and_keeps_modalities_aligned():
    def micro_batch(token: int, num_tokens: int, multimodal: bool = False) -> MicroBatch:
        return MicroBatch(
            input_ids=np.full(num_tokens, token, np.int32),
            loss_mask=np.ones(num_tokens, np.bool_),
            advantages=np.ones(num_tokens, np.float32),
            inference_logprobs=np.zeros(num_tokens, np.float32),
            position_ids=np.arange(num_tokens, dtype=np.int32),
            sequence_lengths=[num_tokens],
            temperatures=np.ones(num_tokens, np.float32),
            env_names=["env"],
            seq_lens=[num_tokens],
            mm_refs=MMRefs(images=[]) if multimodal else None,
        )

    image, long, short = micro_batch(0, 4, multimodal=True), micro_batch(1, 8), micro_batch(2, 2)
    two_ranks = [[image, long], [dummy_micro_batch(image), short]]

    three_ranks = redistribute(two_ranks, 3)

    assert [len(rank) for rank in three_ranks] == [2, 2, 2]
    # Every rank runs a multimodal micro batch first, then a text one.
    assert [[mb.mm_refs is not None for mb in rank] for rank in three_ranks] == [[True, False]] * 3
    trained = [mb for rank in three_ranks for mb in rank if np.asarray(mb.loss_mask).any()]
    assert sorted(int(mb.input_ids[0]) for mb in trained) == [0, 1, 2]
    assert redistribute(two_ranks, 1) == [[image, long, short]]


@pytest.mark.parametrize(
    "setup_transport, setup_receivers",
    [(_filesystem, _filesystem_receivers), (_zmq, _zmq_receivers)],
    ids=["filesystem", "zmq"],
)
def test_a_trainer_restarted_with_more_context_parallel_ranks_gets_longer_micro_batches(
    tmp_path: Path, setup_transport: Callable, setup_receivers: Callable
):
    sender, receivers, transport = setup_transport(tmp_path, retain=True)
    asyncio.run(_send(sender, receivers))

    # Two data ranks of two context-parallel ranks each become one data rank of four.
    restarted = setup_receivers(tmp_path, transport, rank_to_data_rank=[0, 0, 0, 0], micro_batch_multiple=4)

    async def send_live() -> None:
        grid = [[_micro_batch(_tokens(1, data_rank))] for data_rank in range(2)]
        waits = [asyncio.to_thread(receiver.wait) for receiver in restarted]
        await asyncio.wait_for(asyncio.gather(sender.send(grid), *waits), timeout=10)

    for step, send in ((0, None), (1, send_live)):
        if send is None:
            for receiver in restarted:
                receiver.wait()
        else:
            asyncio.run(send())
        for receiver in restarted:
            micro_batches = receiver.receive()
            assert [mb.sequence_lengths for mb in micro_batches] == [[4], [4]]
            assert [mb.padding for mb in micro_batches] == [3, 3]
            assert [np.frombuffer(mb.loss_mask, np.bool_).tolist() for mb in micro_batches] == [
                [True, False, False, False]
            ] * 2
            assert sorted(int(np.frombuffer(mb.input_ids, np.int32)[0]) for mb in micro_batches) == [
                _tokens(step, 0),
                _tokens(step, 1),
            ]
    assert (sender.data_world_size, sender.micro_batch_multiple) == (1, 4)
    for transport in [sender, *receivers, *restarted]:
        transport.close()
