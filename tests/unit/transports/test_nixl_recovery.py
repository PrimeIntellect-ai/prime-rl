"""NIXL weight transfer across a trainer replacement: stale trainer tables, peer teardown and cancellation."""

import asyncio
import time

import pytest

from prime_rl.configs.shared import NIXLWeightBroadcastConfig
from prime_rl.transports.weights import BroadcastAborted
from prime_rl.transports.weights.nixl import nixl as nixl_transport
from prime_rl.transports.weights.nixl.agent import NixlAgent, NixlPeer, PreparedRead
from prime_rl.transports.weights.nixl.model_express import ModelExpressSession
from prime_rl.transports.weights.nixl.trainer_tensor_table import TrainerAgent, TrainerTensorTable


def make_table(instance_id: str) -> TrainerTensorTable:
    agent = TrainerAgent(name=f"trainer-{instance_id}", metadata=f"trainer-{instance_id}".encode(), device_id=0)
    return TrainerTensorTable(instance_id=instance_id, agents=[agent], staging_buffer_count=1, groups=[])


class FakeSession:
    """ModelExpress holding one trainer entry; ``publish`` overwrites it like a replacement does."""

    def __init__(self, table: TrainerTensorTable | None) -> None:
        self.tables = [table] if table is not None else []

    def trainer_tables(self) -> list[TrainerTensorTable]:
        return list(self.tables)

    def publish(self, table: TrainerTensorTable) -> None:
        self.tables = [table]

    wait_for_trainer_table = ModelExpressSession.wait_for_trainer_table
    session_id = "test"


class FakeAgent:
    def __init__(self, name: str) -> None:
        self.name = name
        self.connected: list[str] = []
        self.removed: list[str] = []
        self.notified: list[tuple[str, str]] = []

    def get_metadata(self) -> bytes:
        return self.name.encode()

    def add_remote_agent(self, metadata: bytes) -> NixlPeer:
        return NixlPeer(local_agent=self, remote_agent_name=metadata.decode())  # type: ignore[arg-type]

    def make_connection(self, peer: NixlPeer) -> None:
        self.connected.append(peer.remote_agent_name)

    def remove_remote_agent(self, peer: NixlPeer) -> None:
        self.removed.append(peer.remote_agent_name)

    def wait_for_notification(self, peers, notification, *, timeout, cancelled=None) -> None:
        # The trainer never announces its policy: only a cancellation ends the wait.
        while cancelled is None or not cancelled():
            time.sleep(0.01)
        raise RuntimeError("NIXL notification wait cancelled")

    def send_notification(self, peer: NixlPeer, notification: str) -> None:
        self.notified.append((peer.remote_agent_name, notification))


class FakeAdminPlane:
    def __init__(self) -> None:
        self.resets: list[list[str]] = []

    async def reset_weight_transfer(self, stale_trainer_instances: list[str]) -> None:
        self.resets.append(stale_trainer_instances)


@pytest.fixture
def receiver(tmp_path, monkeypatch):
    monkeypatch.setattr(nixl_transport, "set_ucx_env_defaults", lambda device: None)
    monkeypatch.setattr(nixl_transport, "NixlAgent", FakeAgent)
    monkeypatch.setattr(nixl_transport, "MxClient", lambda server_url: None)
    receiver = nixl_transport.NIXLWeightReceiver(
        tmp_path / "broadcasts", NIXLWeightBroadcastConfig(timeout=30), FakeAdminPlane(), "model"
    )
    receiver.model_express = FakeSession(make_table("failed"))
    return receiver


def test_wait_for_trainer_table_skips_replaced_instances():
    session = FakeSession(make_table("failed"))
    with pytest.raises(TimeoutError, match="failed"):
        session.wait_for_trainer_table(exclude={"failed"}, timeout=0.05, poll_interval=0.01)
    with pytest.raises(RuntimeError, match="cancelled"):
        session.wait_for_trainer_table(exclude={"failed"}, timeout=5, cancelled=lambda: True)
    session.publish(make_table("replacement"))
    assert session.wait_for_trainer_table(exclude={"failed"}, timeout=1).instance_id == "replacement"


def test_replacement_forgets_the_failed_trainer(receiver):
    receiver.trainer_peer = receiver.nixl_agent.add_remote_agent(b"trainer-failed")
    receiver.clear_offers()
    assert receiver.stale_trainer_instances == {"failed"}
    assert receiver.trainer_peer is None
    assert receiver.nixl_agent.removed == ["trainer-failed"]

    asyncio.run(receiver.reconnect())
    assert receiver.admin_plane.resets == [["failed"]]


def test_abort_ends_a_receive_waiting_on_a_replaced_trainer(receiver):
    receiver.stale_trainer_instances = {"failed"}
    receiver.step_dir(4).mkdir(parents=True)

    async def receive_then_abort() -> None:
        task = asyncio.create_task(receiver.receive(4))
        await asyncio.sleep(0.2)
        assert not task.done()
        receiver.abort()
        await task

    with pytest.raises(BroadcastAborted):
        asyncio.run(receive_then_abort())
    assert receiver.trainer_peer is None


def test_abort_ends_a_wait_for_the_policy_announcement(receiver):
    receiver.model_express.publish(make_table("replacement"))
    receiver.step_dir(5).mkdir(parents=True)

    async def receive_then_abort() -> None:
        task = asyncio.create_task(receiver.receive(5))
        await asyncio.sleep(0.2)
        receiver.abort()
        await task

    with pytest.raises(BroadcastAborted):
        asyncio.run(receive_then_abort())
    assert receiver.nixl_agent.connected == ["trainer-replacement"]


class FakeNixl:
    def __init__(self, check_error: Exception | None = None) -> None:
        self.check_error = check_error
        self.released: list[object] = []

    def check_xfer_state(self, handle):
        if self.check_error is not None:
            raise self.check_error
        return "PROC"

    def release_xfer_handle(self, handle) -> None:
        self.released.append(handle)

    def release_dlist_handle(self, handle) -> None:
        self.released.append(handle)

    def remove_remote_agent(self, name: str) -> None:
        self.released.append(name)


def make_agent(nixl: FakeNixl) -> NixlAgent:
    agent = object.__new__(NixlAgent)
    agent.name = "inference"
    agent._agent = nixl
    agent._notifications = {}
    return agent


def test_wait_releases_a_read_whose_remote_agent_died():
    nixl = FakeNixl(check_error=RuntimeError("NIXL_ERR_REMOTE_DISCONNECT"))
    agent = make_agent(nixl)
    read = PreparedRead(local_descriptors="local", remote_descriptors="remote", handle="handle")
    with pytest.raises(RuntimeError, match="REMOTE_DISCONNECT"):
        agent.wait(read, context="layer.0", timeout=5)
    assert nixl.released == ["handle"]


def test_removing_a_peer_drops_its_unread_notifications():
    nixl = FakeNixl()
    agent = make_agent(nixl)
    peer = NixlPeer(local_agent=agent, remote_agent_name="trainer-failed")
    agent._notifications = {"trainer-failed": [b"00000001:0000000000000000"], "other": [b"x"]}
    agent.release_read(PreparedRead(local_descriptors="local", remote_descriptors="remote", handle="handle"))
    agent.remove_remote_agent(peer)
    assert nixl.released == ["handle", "local", "remote", "trainer-failed"]
    assert agent._notifications == {"other": [b"x"]}
