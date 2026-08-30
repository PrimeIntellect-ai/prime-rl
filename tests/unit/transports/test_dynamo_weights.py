from types import SimpleNamespace

import pytest

from prime_rl.transports.weights.base import RECEIVER_READY_MARKER
from prime_rl.transports.weights.dynamo import DynamoWeightReceiver


class FakeAdminPlane:
    def __init__(self) -> None:
        self.workers = (
            SimpleNamespace(
                component="backend",
                instance_id=1,
                world_size=2,
                routes=(
                    "control/pause_generation",
                    "control/resume_generation",
                    "control/is_paused",
                    "control/get_weight_version",
                    "update/update_weight_version",
                    "update/init_weight_transfer_engine",
                    "update/start_weight_update",
                    "update/update_weights",
                    "update/finish_weight_update",
                ),
            ),
        )
        self.collective_clients = [object()]
        self.calls: list[object] = []

    async def pause(self) -> None:
        self.calls.append("pause")

    async def is_paused(self) -> bool:
        self.calls.append("is_paused")
        return True

    async def resume(self) -> None:
        self.calls.append("resume")

    async def update_weight_version(self, version: str) -> None:
        self.calls.append(("update_weight_version", version))

    async def weight_versions(self) -> list[str]:
        self.calls.append("weight_versions")
        return ["3"]

    async def fanout_collective(self, bodies: list[dict]) -> list[dict]:
        self.calls.append(("collective", bodies))
        return [{"results": [None]}]


@pytest.mark.asyncio
async def test_dynamo_native_nccl_receiver_acknowledges_only_after_pause(tmp_path):
    admin = FakeAdminPlane()
    config = SimpleNamespace(type="nccl", timeout=10, inference_world_size=2)
    receiver = DynamoWeightReceiver(tmp_path, config, [], "model", admin)
    receiver.step_dir(3).mkdir(parents=True)

    await receiver.initialize()
    await receiver.receive(3)

    assert (receiver.step_dir(3) / RECEIVER_READY_MARKER).exists()
    assert admin.calls == ["pause", "is_paused", "weight_versions", "resume"]


@pytest.mark.asyncio
async def test_dynamo_filesystem_receiver_uses_admin_collective_rpc(tmp_path):
    admin = FakeAdminPlane()
    config = SimpleNamespace(type="filesystem", timeout=10)
    receiver = DynamoWeightReceiver(tmp_path, config, [], "model", admin)
    step_dir = receiver.step_dir(4)
    step_dir.mkdir(parents=True)
    (step_dir / ".finished").touch()

    await receiver.receive(4)

    assert admin.calls[0:2] == ["pause", "is_paused"]
    assert admin.calls[2] == (
        "collective",
        [
            {
                "method": "reload_weights",
                "timeout": 10,
                "args": [],
                "kwargs": {"weights_path": step_dir.as_posix()},
            }
        ],
    )
    assert ("update_weight_version", "4") in admin.calls
    assert admin.calls[-1] == "resume"
