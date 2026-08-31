import asyncio
from types import SimpleNamespace

import pytest

import prime_rl.transports.weights.dynamo as dynamo_weights
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


def test_dynamo_nixl_receiver_participates_in_orchestrator_handshake(tmp_path, monkeypatch):
    events: list[object] = []

    class OrderedAdminPlane(FakeAdminPlane):
        def __init__(self) -> None:
            super().__init__()
            self.version = "0"

        async def resume(self) -> None:
            events.append("resume")
            await super().resume()

        async def update_weight_version(self, version: str) -> None:
            events.append(("update_weight_version", version))
            self.version = version
            await super().update_weight_version(version)

        async def weight_versions(self) -> list[str]:
            return [self.version]

        async def fanout_collective(self, bodies: list[dict]) -> list[dict]:
            events.append(("collective", bodies[0]["method"]))
            return await super().fanout_collective(bodies)

    class FakeNixlAgent:
        def __init__(self, name: str) -> None:
            events.append(("agent", name))

        def get_metadata(self) -> bytes:
            return b"orchestrator-metadata"

        def add_remote_agent(self, metadata: bytes) -> str:
            events.append(("add_remote_agent", metadata))
            return "trainer-peer"

        def make_connection(self, peer: str) -> None:
            events.append(("make_connection", peer))

        def wait_for_notification(self, peers: list[str], notification: str, timeout: int) -> None:
            events.append(("wait_for_notification", peers, notification, timeout))

        def send_notification(self, peer: str, notification: str) -> None:
            events.append(("send_notification", peer, notification))

    class FakeModelExpressSession:
        def __init__(self, **kwargs) -> None:
            events.append(("session", kwargs))

        def publish(self, *, nixl_metadata: bytes) -> None:
            events.append(("publish", nixl_metadata))

        def wait_for(self, role: str, *, count: int, timeout: int):
            events.append(("wait_for", role, count, timeout))
            return ["trainer-ref"]

        def fetch(self, ref: str):
            events.append(("fetch", ref))
            return SimpleNamespace(nixl_metadata=b"trainer-table")

    class FakeTrainerTensorTable:
        @classmethod
        def decode(cls, payload: bytes):
            events.append(("decode", payload))
            return SimpleNamespace(agents=[SimpleNamespace(metadata=b"trainer-metadata")])

    monkeypatch.setattr(
        dynamo_weights, "set_ucx_env_defaults", lambda device: events.append(("ucx", device)), raising=False
    )
    monkeypatch.setattr(dynamo_weights, "NixlAgent", FakeNixlAgent, raising=False)
    monkeypatch.setattr(dynamo_weights, "make_agent_name", lambda role, rank: f"{role}-{rank}", raising=False)
    monkeypatch.setattr(dynamo_weights, "ModelExpressSession", FakeModelExpressSession, raising=False)
    monkeypatch.setattr(dynamo_weights, "MxClient", lambda server_url: ("client", server_url), raising=False)
    monkeypatch.setattr(dynamo_weights, "TrainerTensorTable", FakeTrainerTensorTable, raising=False)
    monkeypatch.setattr(
        dynamo_weights, "policy_notification", lambda step, state: f"policy-{step}-{state}", raising=False
    )

    admin = OrderedAdminPlane()
    config = SimpleNamespace(
        type="nixl",
        timeout=10,
        inference_world_size=2,
        host="modelexpress",
        port=8001,
        session_id="test-session",
    )
    receiver = DynamoWeightReceiver(tmp_path, config, [], "model", admin)
    receiver.step_dir(3).mkdir(parents=True)
    receiver.step_dir(4).mkdir(parents=True)

    asyncio.run(receiver.initialize())
    asyncio.run(receiver.receive(3))
    asyncio.run(receiver.receive(4))

    critical_events = [
        event
        for event in events
        if event == "resume"
        or (
            isinstance(event, tuple)
            and event[0]
            in {"publish", "wait_for_notification", "collective", "send_notification", "update_weight_version"}
        )
    ]
    assert critical_events == [
        ("collective", "init_broadcaster"),
        ("publish", b"orchestrator-metadata"),
        ("wait_for_notification", ["trainer-peer"], "policy-3-ready", 10),
        ("collective", "update_weights_from_path"),
        ("send_notification", "trainer-peer", "policy-3-complete"),
        ("update_weight_version", "3"),
        "resume",
        ("wait_for_notification", ["trainer-peer"], "policy-4-ready", 10),
        ("collective", "update_weights_from_path"),
        ("send_notification", "trainer-peer", "policy-4-complete"),
        ("update_weight_version", "4"),
        "resume",
    ]
    assert events.count(("wait_for", "trainer", 1, 10)) == 1
    assert events.count(("add_remote_agent", b"trainer-metadata")) == 1
