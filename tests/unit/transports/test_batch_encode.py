from pathlib import Path

import msgspec
import numpy as np
import pytest

from prime_rl.transports.batch.base import BatchSender
from prime_rl.transports.batch.types import (
    MICRO_BATCH_DTYPES,
    MicroBatch,
    RoutedExperts,
    encode_array,
    micro_batch_stream,
)


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
        input_ids=np.array([1, 2], np.int32),
        loss_mask=np.array([False, True]),
        advantages=np.array([0.0, 1.0], np.float32),
        inference_logprobs=np.array([-0.1, -0.2], np.float32),
        position_ids=np.array([0, 1], np.int32),
        sequence_lengths=[2],
        temperatures=np.array([1.0, 1.0], np.float32),
        env_names=["travel"],
        seq_lens=[2],
        routed_experts=RoutedExperts(b"\x00\x01\x02\x03", [2, 1, 2], "uint8") if with_routing else None,
    )
    micro_batches = [batch] * count
    encoded = sender.encode(micro_batches)

    assert isinstance(encoded, bytes)
    assert encoded == msgspec.msgpack.encode(micro_batches, enc_hook=encode_array)
    decoded = msgspec.msgpack.decode(encoded, type=list[MicroBatch])
    assert len(decoded) == count
    for got in decoded:
        # Streams arrive as raw buffers of their wire dtype
        for name in MICRO_BATCH_DTYPES:
            want = micro_batch_stream(batch, name)
            assert (want is None) == (getattr(got, name) is None)
            if want is not None:
                assert np.array_equal(micro_batch_stream(got, name), want)
        assert got.routed_experts == batch.routed_experts
        assert got.env_names == batch.env_names and got.sequence_lengths == batch.sequence_lengths
