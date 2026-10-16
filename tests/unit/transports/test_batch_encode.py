from pathlib import Path

import msgspec
import pytest

from prime_rl.transports.batch.base import BatchSender
from prime_rl.transports.batch.types import MicroBatch, RoutedExperts


@pytest.fixture
def sender(tmp_path):
    class Sender(BatchSender):
        async def send(self, micro_batch_grid: list[list[MicroBatch]]) -> None:
            raise NotImplementedError

    return Sender(Path(tmp_path), data_world_size=1)


@pytest.mark.parametrize("count", [0, 1, 2, 15, 16, 255, 65535, 65536])
@pytest.mark.parametrize("with_routing", [False, True])
def test_batch_encode_preserves_wire_bytes(sender, count, with_routing):
    batch = MicroBatch(
        input_ids=[1, 2],
        loss_mask=[False, True],
        advantages=[0.0, 1.0],
        inference_logprobs=[-0.1, -0.2],
        position_ids=[0, 1],
        sequence_lengths=[2],
        temperatures=[1.0, 1.0],
        env_names=["travel"],
        seq_lens=[2],
        routed_experts=RoutedExperts(b"\x00\x01\x02\x03", [2, 1, 2], "uint8") if with_routing else None,
    )
    micro_batches = [batch] * count
    encoded = sender.encode(micro_batches)

    assert isinstance(encoded, bytes)
    assert encoded == msgspec.msgpack.encode(micro_batches)
    assert msgspec.msgpack.decode(encoded, type=list[MicroBatch]) == micro_batches
