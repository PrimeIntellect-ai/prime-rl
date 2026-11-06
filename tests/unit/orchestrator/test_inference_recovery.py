import asyncio
import json
import threading
import time
from pathlib import Path

import httpx
import pytest
import torch.nn as nn

import prime_rl.transports.weights.base as weights_base
from prime_rl.configs.shared import ClientConfig, FileSystemWeightBroadcastConfig, InferenceMonitorConfig
from prime_rl.orchestrator.clients import AdminPlane, EngineLost
from prime_rl.orchestrator.inference_monitor import InferenceMonitor
from prime_rl.orchestrator.types import Policy
from prime_rl.orchestrator.watcher import WeightWatcher
from prime_rl.transports.weights import BroadcastAborted
from prime_rl.transports.weights.base import (
    ABORTED_MARKER,
    FINISHED_MARKER,
    RECEIVED_MARKER,
    RECEIVER_READY_MARKER,
    SENDER_READY_MARKER,
    WeightSender,
)
from prime_rl.transports.weights.filesystem import FileSystemWeightReceiver


class Server:
    """A fake inference server: records admin calls; ``hang`` holds /update_weights until released."""

    def __init__(self, name: str):
        self.name = name
        self.calls: list[str] = []
        self.hang: asyncio.Event | None = None
        self.update_started = asyncio.Event()
        self.instance_id = f"{name}-1"
        self.down = False
        self.params: dict[str, str] = {}
        self.engine_dead = False

    async def handle(self, request: httpx.Request) -> httpx.Response:
        if self.down:
            raise httpx.ConnectError("connection refused")
        self.calls.append(request.url.path)
        self.params = dict(request.url.params)
        if request.url.path == "/ready":
            ready = {"instance_id": self.instance_id, "awaiting_policy": False, "engine_dead": self.engine_dead}
            return httpx.Response(503 if self.engine_dead else 200, json=ready)
        if request.url.path == "/update_weights":
            self.update_started.set()
            if self.hang is not None:
                await self.hang.wait()
                return httpx.Response(500)
        return httpx.Response(200)


def admin_plane(*servers: Server) -> AdminPlane:
    admin = AdminPlane(ClientConfig())
    admin.clients = [
        httpx.AsyncClient(base_url=f"http://{server.name}", transport=httpx.MockTransport(server.handle))
        for server in servers
    ]
    return admin


def test_a_lost_server_is_left_out_of_an_update():
    """A filesystem update completes on the other servers; the lost one is not resumed."""

    async def run():
        a, b = Server("a"), Server("b")
        b.hang = asyncio.Event()
        admin = admin_plane(a, b)
        availability = []
        admin.on_availability = availability.append
        update = asyncio.create_task(admin.update_weights(Path("/w"), transport="filesystem", step=3))
        await b.update_started.wait()
        admin.mark_lost(admin.engines[1], "test")
        await update
        assert a.calls == ["/pause", "/update_weights", "/resume"]
        assert b.calls == ["/pause", "/update_weights"]
        assert [engine.status for engine in admin.engines] == ["live", "lost"]
        assert availability == []
        # The next update leaves it out from the start.
        await admin.update_weights(Path("/w"), transport="filesystem", step=4)
        assert b.calls == ["/pause", "/update_weights"]

    asyncio.run(run())


def test_a_lost_server_stops_a_collective_update(tmp_path):
    """With one collective transfer, the others are cancelled through the abort marker and stay paused."""

    async def run():
        a, b = Server("a"), Server("b")
        a.hang, b.hang = asyncio.Event(), asyncio.Event()
        admin = admin_plane(a, b)
        abort_path = tmp_path / ABORTED_MARKER
        update = asyncio.create_task(
            admin.update_weights(None, transport="nccl", step=3, abort_path=abort_path, all_or_nothing=True)
        )
        await a.update_started.wait()
        await b.update_started.wait()
        admin.mark_lost(admin.engines[1], "test")
        assert abort_path.exists()
        a.hang.set()  # the survivor's cancelled transfer fails
        with pytest.raises(EngineLost) as error:
            await update
        assert error.value.engines == [admin.engines[1]]
        assert "/resume" not in a.calls
        assert admin.abort_transfer is None

    asyncio.run(run())


def test_a_rejoining_server_takes_requests_after_its_update():
    async def run():
        a = Server("a")
        admin = admin_plane(a)
        availability = []
        admin.on_availability = availability.append
        admin.mark_lost(admin.engines[0], "test")
        assert availability == [False] and admin.members() == []
        admin.mark_joining(admin.engines[0])
        assert admin.members() == admin.engines and not admin.available
        await admin.update_weights(Path("/w"), transport="filesystem", step=5)
        assert admin.engines[0].status == "live"
        assert availability == [False, True]

    asyncio.run(run())


def test_a_loss_before_the_dispatcher_is_attached_still_holds_scheduling():
    async def run():
        admin = admin_plane(Server("a"))
        admin.mark_lost(admin.engines[0], "test")
        availability = []
        admin.watch_availability(availability.append)
        assert availability == [False]
        admin.mark_joining(admin.engines[0])
        admin.mark_joined(admin.engines)
        assert availability == [False, True]
        # With a server taking requests, attaching says nothing.
        later = []
        admin.watch_availability(later.append)
        assert later == []

    asyncio.run(run())


def test_monitor_tells_lost_replaced_and_returning_servers_apart():
    async def run():
        a = Server("a")
        admin = admin_plane(a)
        monitor = InferenceMonitor(admin, InferenceMonitorConfig(lost_after=2), served_policy=lambda: 7)
        engine = admin.engines[0]
        await monitor.probe(engine)
        assert engine.instance_id == "a-1" and engine.status == "live"
        # Every probe tells the server the policy inference serves and renews its lease.
        assert a.params == {"policy": "7", "lease": str(monitor.lease)}

        # Two failed probes lose it.
        a.down = True
        await monitor.probe(engine)
        assert engine.status == "live"
        await monitor.probe(engine)
        assert engine.status == "lost"

        # The same process answers again; the probe told it the policy to check it is not behind.
        a.down = False
        await monitor.probe(engine)
        assert engine.status == "joining"

        # A replacement process (new instance ID) is lost and rejoins at once.
        engine.status = "live"
        a.instance_id = "a-2"
        await monitor.probe(engine)
        assert engine.status == "joining" and engine.instance_id == "a-2"
        assert a.calls[-1] == "/ready"

        # A dead engine core is lost without waiting out the failed-probe count.
        engine.status = "live"
        a.engine_dead = True
        await monitor.probe(engine)
        assert engine.status == "lost"

    asyncio.run(run())


def test_a_server_replaced_during_an_update_counts_as_lost():
    """A replacement process answers the next probe and rejoins at once; the update its predecessor
    was lost during still leaves it out instead of failing on the cancelled call."""

    async def run():
        a, b = Server("a"), Server("b")
        b.hang = asyncio.Event()
        admin = admin_plane(a, b)
        monitor = InferenceMonitor(admin, InferenceMonitorConfig(), served_policy=lambda: 0)
        for engine in admin.engines:
            await monitor.probe(engine)
        update = asyncio.create_task(admin.update_weights(Path("/w"), transport="filesystem", step=3))
        await b.update_started.wait()
        b.instance_id = "b-2"
        await monitor.probe(admin.engines[1])
        assert admin.engines[1].status == "joining"
        await update
        assert a.calls[-1] == "/resume" and "/resume" not in b.calls
        assert [engine.status for engine in admin.engines] == ["live", "joining"]

    asyncio.run(run())


class RepublishingReceiver:
    """Every server was lost; the trainer offers its version again, and that transfer fails."""

    def __init__(self, admin: AdminPlane, aborted: bool):
        self.admin_plane = admin
        self.aborted = aborted

    def can_catch_up(self, version: int) -> bool:
        return False

    def next_version(self, current: int) -> int:
        return current

    def offered_attempt(self, version: int) -> str:
        return "attempt"

    async def wait_for_new_offer(self, version, attempt, newer=None) -> bool:
        if self.aborted:
            raise BroadcastAborted("Stopped waiting for policy v4 of a replaced trainer")
        return True

    async def receive(self, version: int) -> None:
        raise RuntimeError("the transfer failed")


@pytest.mark.parametrize("aborted", [False, True], ids=["failed", "aborted"])
def test_a_failed_republish_is_handled_like_a_failed_update(aborted):
    """The republish fails, or a replacement trainer's reconciliation aborts the wait for it."""

    async def run():
        admin = admin_plane(Server("a"))
        admin.mark_lost(admin.engines[0], "test")
        admin.mark_joining(admin.engines[0])
        requested = []
        watcher = WeightWatcher(
            RepublishingReceiver(admin, aborted),
            policy=Policy(version=4, model_name="model"),
            observers=[],
            trainer_recovery=True,
            request_republish=requested.append,
        )
        watcher.ckpt_step, watcher.weights_valid = 4, True
        await watcher.catch_up()
        # The request is withdrawn: answered later, nothing would receive the policy.
        assert requested == [4, None]
        if aborted:
            # Nothing reached the servers: inference keeps v4 for the replacement to reconcile.
            assert watcher.weights_valid and watcher.invalid_since is None
        else:
            # The servers may hold part of v4: they stay paused until a complete policy arrives.
            assert not watcher.weights_valid and watcher.invalid_since is not None

    asyncio.run(run())


def test_watcher_catches_up_rejoining_servers_from_disk(tmp_path):
    async def run():
        a, b = Server("a"), Server("b")
        admin = admin_plane(a, b)
        receiver = FileSystemWeightReceiver(tmp_path, FileSystemWeightBroadcastConfig(), admin, "model")
        watcher = WeightWatcher(receiver, policy=Policy(model_name="model"), observers=[])
        watcher.policy.version = 4
        watcher.weights_valid = True
        admin.mark_lost(admin.engines[1], "test")
        admin.mark_joining(admin.engines[1])

        await watcher.catch_up()  # v4 is not on disk: it waits for the next update
        assert admin.engines[1].status == "joining" and b.calls == []

        step_dir = receiver.step_dir(4)
        step_dir.mkdir(parents=True)
        (step_dir / FINISHED_MARKER).touch()
        await watcher.catch_up()
        assert admin.engines[1].status == "live"
        assert b.calls == ["/pause", "/update_weights", "/resume"] and a.calls == []

    asyncio.run(run())


@pytest.mark.parametrize("live_peer", [False, True], ids=["alone", "live-peer"])
def test_a_catch_up_failing_while_no_server_takes_requests_fails_the_run(tmp_path, live_peer):
    """A rejoined server whose catch-up keeps failing still answers probes, so it is never lost.
    Without another server taking requests training would wait forever: the failure is bounded by
    the transfer timeout. With a live server, training goes on and the catch-up is just retried."""

    async def run():
        a, b = Server("a"), Server("b")
        a.hang = asyncio.Event()
        a.hang.set()  # its /update_weights fails at once
        admin = admin_plane(a, b)
        receiver = FileSystemWeightReceiver(tmp_path, FileSystemWeightBroadcastConfig(timeout=1), admin, "model")
        watcher = WeightWatcher(receiver, policy=Policy(model_name="model"), observers=[])
        watcher.policy.version = 4
        watcher.weights_valid = True
        step_dir = receiver.step_dir(4)
        step_dir.mkdir(parents=True)
        (step_dir / FINISHED_MARKER).touch()
        admin.mark_lost(admin.engines[0], "test")
        admin.mark_joining(admin.engines[0])
        if not live_peer:
            admin.mark_lost(admin.engines[1], "test")

        await watcher.catch_up()
        await asyncio.sleep(1.2)
        if live_peer:
            await watcher.catch_up()
            assert watcher.catch_up_failed_since is None and admin.engines[0].status == "joining"
        else:
            with pytest.raises(RuntimeError, match="No inference server takes requests"):
                await watcher.catch_up()

    asyncio.run(run())


class FakeSender(WeightSender):
    confirms_delivery = True

    def __init__(self, output_dir: Path):
        super().__init__(output_dir, timeout=30)
        self.sent: list[dict] = []

    def _broadcast(self, model: nn.Module, step: int, step_dir: Path) -> bool:
        self.sent.append(self.ack)
        return True


def test_sender_offers_a_version_again_after_an_abort(tmp_path, monkeypatch):
    """The consumer aborts the first attempt (a server was lost) and confirms the second."""
    monkeypatch.setattr(weights_base, "wait_for_master", lambda key, timeout: None)
    monkeypatch.setattr(weights_base, "share_from_master", lambda key, value, timeout: value)
    sender = FakeSender(tmp_path)
    step_dir = sender.step_dir(7)

    def consumer():
        for attempt, verdict in enumerate((ABORTED_MARKER, RECEIVED_MARKER)):
            while not (step_dir / SENDER_READY_MARKER).exists() or (step_dir / ABORTED_MARKER).exists():
                time.sleep(0.01)
            attempt_id = (step_dir / SENDER_READY_MARKER).read_text()
            ack = {"attempt": attempt_id, "session": f"g{attempt}", "world_size": 3 - attempt}
            (step_dir / RECEIVER_READY_MARKER).write_text(json.dumps(ack))
            while len(sender.sent) <= attempt:
                time.sleep(0.01)
            (step_dir / verdict).write_text(attempt_id)

    thread = threading.Thread(target=consumer)
    thread.start()
    sender.broadcast(nn.Linear(1, 1), 7)
    thread.join()
    assert [ack["session"] for ack in sender.sent] == ["g0", "g1"]
    assert (step_dir / FINISHED_MARKER).exists() and not (step_dir / ABORTED_MARKER).exists()


def test_sender_rejects_an_acknowledgement_of_another_attempt(tmp_path, monkeypatch):
    monkeypatch.setattr(weights_base, "wait_for_master", lambda key, timeout: None)
    sender = FakeSender(tmp_path)
    step_dir = sender.step_dir(1)

    def consumer():
        while not (step_dir / SENDER_READY_MARKER).exists():
            time.sleep(0.01)
        (step_dir / RECEIVER_READY_MARKER).write_text(json.dumps({"attempt": "stale"}))

    thread = threading.Thread(target=consumer)
    thread.start()
    with pytest.raises(RuntimeError, match="acknowledged attempt stale"):
        sender.broadcast(nn.Linear(1, 1), 1)
    thread.join()


def test_a_dying_servers_failed_call_counts_as_its_loss(tmp_path):
    """A transfer call that fails before the monitor noticed the server died waits for the verdict."""

    async def run():
        a, b = Server("a"), Server("b")
        admin = admin_plane(a, b)
        admin.loss_verdict_s = 5.0

        async def dying(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/update_weights":
                raise httpx.RemoteProtocolError("Server disconnected without sending a response.")
            return httpx.Response(200)

        admin.clients[1]._transport = httpx.MockTransport(dying)
        update = asyncio.create_task(
            admin.update_weights(
                None, transport="nccl", step=3, abort_path=tmp_path / ABORTED_MARKER, all_or_nothing=True
            )
        )
        await asyncio.sleep(0.5)
        assert not update.done()  # waiting for the verdict, not failing the update
        admin.mark_lost(admin.engines[1], "test")
        with pytest.raises(EngineLost):
            await update

    asyncio.run(run())


def test_a_failed_join_stops_the_others_waiting_for_the_group(tmp_path):
    """A server that fails to join the transfer group without being lost would leave the trainer and
    the servers that joined in the group's rendezvous until their deadlines: the transfer is aborted."""

    async def run():
        admin = admin_plane(Server("a"), Server("b"))
        abort_path = tmp_path / ABORTED_MARKER

        async def joins(request: httpx.Request) -> httpx.Response:
            # A recoverable rendezvous waits for every rank until the abort marker appears.
            while request.url.path == "/init_broadcaster" and not abort_path.exists():
                await asyncio.sleep(0.05)
            return httpx.Response(500 if request.url.path == "/init_broadcaster" else 200)

        async def fails(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500 if request.url.path == "/init_broadcaster" else 200)

        admin.clients[0]._transport = httpx.MockTransport(joins)
        admin.clients[1]._transport = httpx.MockTransport(fails)

        async def form_group(engines):
            await admin.form_nccl_group(
                engines, host="h", port=1, timeout=30, gpus_per_server=1, session_id="g1", abort_path=abort_path
            )

        update = admin.update_weights(
            None, transport="nccl", step=3, abort_path=abort_path, before_transfer=form_group, all_or_nothing=True
        )
        with pytest.raises(httpx.HTTPStatusError):
            await asyncio.wait_for(update, 5)
        assert abort_path.exists()
        assert all(engine.status == "live" for engine in admin.engines)

    asyncio.run(run())


def test_a_failed_call_waits_for_its_own_servers_verdict():
    """Two servers dying in one filesystem update are both left out: the first loss must not end the
    wait for the second server's verdict."""

    async def run():
        admin = admin_plane(Server("a"), Server("b"))
        admin.loss_verdict_s = 5.0

        async def dying(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/update_weights":
                raise httpx.RemoteProtocolError("Server disconnected without sending a response.")
            return httpx.Response(200)

        for client in admin.clients:
            client._transport = httpx.MockTransport(dying)
        update = asyncio.create_task(admin.update_weights(Path("/w"), transport="filesystem", step=3))
        await asyncio.sleep(0.5)
        admin.mark_lost(admin.engines[0], "test")
        await asyncio.sleep(1.0)
        assert not update.done()  # b's failure still waits for b's verdict
        admin.mark_lost(admin.engines[1], "test")
        await update
        assert not admin.available

    asyncio.run(run())


def test_the_master_forms_the_elastic_group_before_the_gathers(tmp_path, monkeypatch):
    """Forming the group can outlast the trainer's collective timeout: the other ranks must wait for it
    on the store, before the gathers of a send."""
    import prime_rl.transports.weights.nccl as nccl
    from prime_rl.configs.shared import NCCLWeightBroadcastConfig

    events = []

    class Broadcaster:
        def join(self, ack: dict, cancelled) -> bool:
            events.append(("join", ack["session"]))
            return False

        def send(self, model: nn.Module, joined: bool = True) -> bool:
            events.append(("send", joined))
            return joined

    monkeypatch.setattr(nccl, "NCCLBroadcaster", lambda *args: Broadcaster())
    monkeypatch.setattr(nccl, "wait_for_master", lambda key, timeout: events.append(("barrier", key)))
    config = NCCLWeightBroadcastConfig(recoverable=True, elastic=True)
    sender = nccl.NCCLWeightSender(tmp_path, config, "cpu")
    sender.ack = {"session": "s1", "world_size": 2}
    assert sender._broadcast(nn.Linear(1, 1), 1, tmp_path) is False
    assert events == [("join", "s1"), ("barrier", "weight_broadcast/join/1"), ("send", False)]


def test_a_server_back_from_a_loss_gets_a_new_transfer_group(tmp_path, monkeypatch):
    """A server restarted between two updates takes part with the same servers as before, but its new
    process runs no communicator of the old group: the next update must form a new one."""
    from prime_rl.configs.shared import NCCLWeightBroadcastConfig
    from prime_rl.transports.weights.nccl import NCCLWeightReceiver

    async def run():
        admin = admin_plane(Server("a"), Server("b"))
        config = NCCLWeightBroadcastConfig(recoverable=True, elastic=True, inference_world_size=2)
        receiver = NCCLWeightReceiver(tmp_path, config, admin, "m")
        sessions = []

        async def form_nccl_group(engines, session_id, **kwargs):
            sessions.append(session_id)

        async def update_weights(step_dir, before_transfer, **kwargs):
            await before_transfer(admin.members())

        monkeypatch.setattr(admin, "form_nccl_group", form_nccl_group)
        monkeypatch.setattr(admin, "update_weights", update_weights)
        for step in range(3):
            receiver.step_dir(step).mkdir(parents=True)
            if step == 2:
                admin.mark_lost(admin.engines[0], "replaced")
                admin.mark_joining(admin.engines[0])
            await receiver._receive_elastic(step, "attempt")
        assert sessions == ["g1", "g2"]

    asyncio.run(run())


def test_an_offer_is_not_acknowledged_without_a_server_to_receive_it(tmp_path, monkeypatch):
    """Every server taking part can be lost while pausing. The trainer alone must not form a group:
    it would confirm a version that no server holds. The offer is received once a server is back."""
    from prime_rl.configs.shared import NCCLWeightBroadcastConfig
    from prime_rl.transports.weights.nccl import NCCLWeightReceiver

    async def run():
        admin = admin_plane(Server("a"))
        config = NCCLWeightBroadcastConfig(recoverable=True, elastic=True, inference_world_size=1)
        receiver = NCCLWeightReceiver(tmp_path, config, admin, "m")
        acks, paused = [], []

        async def form_nccl_group(engines, session_id, **kwargs):
            pass

        async def update_weights(step_dir, before_transfer, **kwargs):
            # The first time, the only server is lost while it pauses.
            engines = admin.members() if paused else []
            paused.append(len(engines))
            await before_transfer(engines)

        monkeypatch.setattr(receiver, "_ack", lambda step, **ack: acks.append(ack["world_size"]))
        monkeypatch.setattr(admin, "form_nccl_group", form_nccl_group)
        monkeypatch.setattr(admin, "update_weights", update_weights)
        receiver.step_dir(0).mkdir(parents=True)
        await receiver._receive_elastic(0, "attempt")
        assert paused == [0, 1] and acks == [2]
        assert (receiver.step_dir(0) / RECEIVED_MARKER).exists()

    asyncio.run(run())


def test_ranks_pace_every_broadcast_with_new_keys(tmp_path, monkeypatch):
    """Publishing a version again must not reuse the store keys of its first broadcast: the other ranks
    would pass the barrier and read the old outcome."""
    keys = []
    monkeypatch.setattr(weights_base, "wait_for_master", lambda key, timeout: keys.append(key))
    monkeypatch.setattr(weights_base, "share_from_master", lambda key, value, timeout: keys.append(key) or value)
    sender = FakeSender(tmp_path)
    step_dir = sender.step_dir(5)

    def consumer():
        for _ in range(2):
            while not (step_dir / SENDER_READY_MARKER).exists() or (step_dir / RECEIVED_MARKER).exists():
                time.sleep(0.01)
            attempt_id = (step_dir / SENDER_READY_MARKER).read_text()
            (step_dir / RECEIVER_READY_MARKER).write_text(json.dumps({"attempt": attempt_id}))
            while not (step_dir / RECEIVED_MARKER).exists():
                if len(sender.sent) > _:
                    (step_dir / RECEIVED_MARKER).write_text(attempt_id)
                time.sleep(0.01)
            while (step_dir / RECEIVED_MARKER).exists():
                time.sleep(0.01)

    thread = threading.Thread(target=consumer, daemon=True)
    thread.start()
    for _ in range(2):
        sender.broadcast(nn.Linear(1, 1), 5)
    assert len(keys) == 4 and len(set(keys)) == 4


def test_watcher_gauges_survive_an_update_without_servers(tmp_path):
    """With every server lost, an update takes no time on any server; the pipeline view must still render."""

    async def run():
        admin = admin_plane(Server("a"))
        admin.mark_lost(admin.engines[0], "test")
        await admin.update_weights(Path("/w"), transport="filesystem", step=2)
        receiver = FileSystemWeightReceiver(tmp_path, FileSystemWeightBroadcastConfig(), admin, "model")
        gauges = WeightWatcher(receiver, policy=Policy(model_name="model"), observers=[]).gauges()
        assert "watcher/pause/min" not in gauges and gauges["watcher/policy_version"] == 0.0

    asyncio.run(run())
