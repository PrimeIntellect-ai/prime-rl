from pathlib import Path

import msgspec
import pytest

from prime_rl.transports.batch.base import BatchSender
from prime_rl.transports.batch.types import MicroBatchPlan, SampleRef


@pytest.fixture
def sender(tmp_path):
    class Sender(BatchSender):
        async def send(self, micro_batch_grid: list[list[MicroBatchPlan]]) -> None:
            raise NotImplementedError

    return Sender(Path(tmp_path), data_world_size=1)


@pytest.mark.parametrize("count", [0, 1, 2, 15, 16, 255, 65535, 65536])
@pytest.mark.parametrize("with_routing", [False, True])
def test_batch_encode_preserves_wire_bytes(sender, count, with_routing):
    plan = MicroBatchPlan(
        samples=[
            SampleRef(
                url="http://10.0.0.1:4000",
                id="trace-0",
                num_tokens=2,
                env_name="travel",
                trace_id="trace",
                branch_index=0,
                temperature=1.0,
                advantage=0.5,
                policy_start=3,
                fields=["routed_experts"] if with_routing else [],
                routed_shape=[1, 2] if with_routing else None,
                routed_dtype="uint8" if with_routing else None,
            )
        ],
        trained_version=4,
        min_version=3,
        padding=2,
    )
    plans = [plan] * count
    encoded = sender.encode(plans)

    assert isinstance(encoded, bytes)
    assert encoded == msgspec.msgpack.encode(plans)
    decoded = msgspec.msgpack.decode(encoded, type=list[MicroBatchPlan])
    assert decoded == plans
