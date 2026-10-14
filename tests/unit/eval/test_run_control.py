"""Run control (``prime_rl.eval.control``): a client in another thread reads the
in-flight episodes, steers their agents through the env client and resizes the
concurrency. Runs the real ``Dispatcher``/``EvalSource``/``EvalSink`` and the zmq
endpoint against a fake env and env client."""

import asyncio
import json
import os
import socket
import threading
from types import SimpleNamespace

import pytest
import verifiers.v1 as vf
from verifiers.v1.serve.types import SteerResponse

from prime_rl.configs.orchestrator import ConcurrencyConfig
from prime_rl.eval import control as control_module
from prime_rl.eval import runner as runner_module
from prime_rl.eval.control import CONTROL_FILE, Client, ControlError, ControlServer
from prime_rl.eval.runner import EvalRunner
from prime_rl.orchestrator.concurrency import ConcurrencyController
from prime_rl.orchestrator.dispatcher import Dispatcher
from prime_rl.orchestrator.envs import EvalEnvs
from prime_rl.orchestrator.eval_sink import EvalSink
from prime_rl.orchestrator.eval_source import EvalSource
from prime_rl.orchestrator.task_feed import TaskFeed
from prime_rl.orchestrator.types import Policy

TIMEOUT_S = 30


class KeyedTask(vf.Task[vf.TaskData]):
    @property
    def key(self) -> str:
        return self.data.id


def _episode(task: vf.Task) -> vf.WireEpisode:
    trace_task = vf.TraceTask(
        type="KeyedTask", data=vf.WireTaskData(**task.data.model_dump(mode="json")), key=task.key, hash=task.hash
    )
    trace = vf.Trace(task=trace_task, agent=vf.AgentInfo(config=vf.AgentConfig()), rewards={}, ok=True)
    episode = vf.Episode(task=trace_task, traces=[trace], ok=True)
    return vf.WireEpisode.model_validate(episode.model_dump())


class Stream(vf.Taskset):
    INFINITE = True

    def __init__(self, task_ids: list[str]) -> None:
        super().__init__(vf.TasksetConfig())
        self.task_ids = task_ids
        self.calls: list[tuple[str, str]] = []
        self.reported = threading.Semaphore(0)

    def load(self):
        for task_id in self.task_ids:
            yield KeyedTask(vf.TaskData(id=task_id))
        for _ in self.task_ids:
            assert self.reported.acquire(timeout=TIMEOUT_S)

    def report(self, hook: str, task: vf.Task) -> None:
        self.calls.append((hook, task.data.id))
        self.reported.release()

    def observe(self, task, episode) -> None:
        self.report("observe", task)

    def on_failed(self, task, error) -> None:
        self.report("on_failed", task)

    def on_cancelled(self, task) -> None:
        self.report("on_cancelled", task)


class FakeEnvClient:
    """Accepts a steer for any trace it was told is running."""

    def __init__(self) -> None:
        self.running: set[str] = set()
        self.steers: list[dict] = []

    async def steer(self, text, *, trace_id=None, episode_id=None, seat=None, sender=None, message_id=None):
        self.steers.append({"text": text, "trace_id": trace_id, "sender": sender, "message_id": message_id})
        if trace_id not in self.running:
            return SteerResponse(queued=False, reason=f"no running agent {trace_id}")
        return SteerResponse(queued=True, trace_id=trace_id, delivery="interception")


class SeatedEnv:
    """Each task streams a ``solver`` and a ``critic`` trace (the critic already done),
    then runs until released or cancelled."""

    def __init__(self, taskset: vf.Taskset) -> None:
        self.name = "pipeline"
        self.config = SimpleNamespace(group_size=1)
        self.taskset = taskset
        self.examples: list[vf.Task] = []
        self.feed = TaskFeed(iter(taskset), name=self.name)
        self.env_client = FakeEnvClient()
        self.running = asyncio.Event()
        self.release = asyncio.Event()
        self.aborted: list[str] = []

    async def run(self, *, client, model_name, cache_salt, task_data, on_delta) -> vf.WireEpisode:
        task_id = task_data["id"]
        for seat, timing in (
            ("critic", {"scoring": {"start": 1.0, "end": 2.0}}),
            ("solver", {"agent": {"start": 1.0}}),
        ):
            trace_id = f"{task_id}-{seat}"
            on_delta({"trace": trace_id, "open": {}, "set": {"agent": {"name": seat}, "timing": timing}})
            if seat == "solver":
                self.env_client.running.add(trace_id)
        self.running.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.aborted.append(task_id)
            raise
        return _episode(KeyedTask(vf.TaskData.model_validate(task_data)))


async def _no_sessions(_session_ids) -> None:
    return None


def _runner(tmp_path, env: SeatedEnv, *, max_inflight: int = 4) -> EvalRunner:
    eval_envs = EvalEnvs.__new__(EvalEnvs)
    eval_envs._envs = {env.name: env}
    runner = EvalRunner(SimpleNamespace(), run_dir=tmp_path)
    runner.eval_envs = eval_envs
    runner.eval_source = EvalSource(eval_envs)
    runner.eval_sink = EvalSink(eval_envs=eval_envs)
    runner.policy = Policy(model_name="model")
    runner.dispatcher = Dispatcher(
        train_envs=None,
        eval_envs=eval_envs,
        train_source=None,
        eval_source=runner.eval_source,
        policy_clients=SimpleNamespace(eval_client=None, finish_sessions=_no_sessions),
        policy=runner.policy,
        progress=None,
        initial_max_inflight=max_inflight,
        max_inflight_ceiling=max_inflight,
        tasks_per_minute=None,
        max_off_policy_steps=0,
        run_id="run",
        run_name=None,
        record_eval_tasks=True,
    )
    runner.concurrency = ConcurrencyController(
        ConcurrencyConfig(min_inflight=max_inflight, max_inflight=max_inflight), fallback_cost=1
    )
    runner.concurrency.bind(set_limit=runner.dispatcher.set_limit, get_inflight=lambda: 0)
    return runner


async def _with_control(runner: EvalRunner, during) -> None:
    """Run one epoch with the control endpoint up; ``during(client)`` runs alongside."""
    runner.dispatcher_task = asyncio.create_task(runner.dispatcher.start())
    runner.control = ControlServer(runner)
    await runner.control.start()
    try:
        epoch = asyncio.create_task(runner.run_epoch(runner.eval_source.trigger(0)[0], 0))
        await asyncio.wait_for(during(Client(runner.run_dir, timeout=10)), timeout=TIMEOUT_S)
        await asyncio.wait_for(epoch, timeout=TIMEOUT_S)
    finally:
        await runner.control.stop()
        await runner.dispatcher.stop()


@pytest.fixture(autouse=True)
def fast_poll(monkeypatch):
    monkeypatch.setattr(runner_module, "POLL_INTERVAL_S", 0.05)


def test_status_lists_in_flight_episodes_with_their_seats_and_traces(tmp_path) -> None:
    env = SeatedEnv(Stream(["job/solve/0"]).skip(0))
    runner = _runner(tmp_path, env)
    seen = {}

    async def during(client: Client) -> None:
        await env.running.wait()
        seen.update(await asyncio.to_thread(client.status))
        env.release.set()

    asyncio.run(_with_control(runner, during))

    [episode] = seen["episodes"]
    assert episode["task_key"] == "job/solve/0"
    assert episode["env"] == "pipeline"
    assert episode["seats"] == ["critic", "solver"]
    assert [(t["trace_id"], t["seat"], t["stage"]) for t in episode["traces"]] == [
        ("job/solve/0-critic", "critic", "done"),
        ("job/solve/0-solver", "solver", "running"),
    ]
    assert seen["max_inflight"] == 4
    assert not (tmp_path / CONTROL_FILE).exists()  # removed on stop


def test_steer_by_task_key_reaches_open_traces_and_returns_receipts(tmp_path) -> None:
    env = SeatedEnv(Stream(["job/solve/0"]).skip(0))
    runner = _runner(tmp_path, env)
    out = {}

    async def during(client: Client) -> None:
        await env.running.wait()
        out["all"] = await asyncio.to_thread(client.steer, "look again", task_key="job/solve/0", sender="op")
        out["seat"] = await asyncio.to_thread(client.steer, "x", task_key="job/solve/0", seat="solver")
        out["done_seat"] = await asyncio.to_thread(client.steer, "x", task_key="job/solve/0", seat="critic")
        out["trace"] = await asyncio.to_thread(client.steer, "x", trace_id="job/solve/0-solver")
        out["missing"] = await asyncio.to_thread(client.steer, "x", task_key="other/solve/0")
        env.release.set()

    asyncio.run(_with_control(runner, during))

    # only the open trace (the solver) is steered; the finished critic is not
    [receipt] = out["all"]
    assert receipt["queued"] and receipt["seat"] == "solver" and receipt["trace_id"] == "job/solve/0-solver"
    assert receipt["delivery"] == "interception" and receipt["task_key"] == "job/solve/0"
    assert env.env_client.steers[0]["text"] == "look again" and env.env_client.steers[0]["sender"] == "op"
    assert [r["queued"] for r in out["seat"]] == [True]
    assert out["done_seat"][0]["queued"] is False and "critic" in out["done_seat"][0]["reason"]
    assert [r["trace_id"] for r in out["trace"]] == ["job/solve/0-solver"]
    assert out["missing"] == [{"queued": False, "reason": "no running episode for other/solve/0", "trace_id": None}]


def test_resize_moves_a_pinned_band_live(tmp_path) -> None:
    env = SeatedEnv(Stream(["job/solve/0"]).skip(0))
    runner = _runner(tmp_path, env, max_inflight=2)
    out = {}

    async def during(client: Client) -> None:
        await env.running.wait()
        out["resize"] = await asyncio.to_thread(client.resize, 7)
        out["status"] = await asyncio.to_thread(client.status)
        env.release.set()

    asyncio.run(_with_control(runner, during))

    assert out["resize"] == 7
    assert out["status"]["max_inflight"] == 7 and out["status"]["min_inflight"] == 7
    assert runner.dispatcher.max_inflight == 7


def test_resize_of_an_adaptive_band_moves_its_top() -> None:
    limits: list[int] = []
    controller = ConcurrencyController(
        ConcurrencyConfig(min_inflight=2, initial_inflight=6, max_inflight=8), fallback_cost=1
    )
    controller.bind(set_limit=limits.append, get_inflight=lambda: 0)
    controller.resize(4)
    assert (controller.floor, controller.ceiling, controller.max_inflight) == (2, 4, 4)
    assert limits == [4]
    controller.resize(16)
    assert (controller.floor, controller.ceiling, controller.max_inflight) == (2, 16, 4)  # grows from here
    controller.resize(1)
    assert (controller.floor, controller.max_inflight) == (1, 1)


def test_a_client_without_a_running_eval_says_so(tmp_path) -> None:
    with pytest.raises(ControlError, match="is the eval running"):
        Client(tmp_path)
    with pytest.raises(SystemExit, match="is the eval running"):
        control_module.main(["--run", str(tmp_path), "status"])


def test_a_stale_control_file_fails_fast(tmp_path) -> None:
    """A control.json an eval left behind when it was killed names a dead pid or another
    host: the client refuses it at once rather than wait out its timeout."""
    control_file = tmp_path / CONTROL_FILE
    address = "tcp://127.0.0.1:1"
    control_file.write_text(json.dumps({"address": address, "host": socket.gethostname(), "pid": 2**22 + 1}))
    with pytest.raises(ControlError, match="pid 4194305\\) is gone"):
        Client(tmp_path)
    control_file.write_text(json.dumps({"address": address, "host": "not-" + socket.gethostname(), "pid": os.getpid()}))
    with pytest.raises(ControlError, match="not on this host"):
        Client(tmp_path)
    control_file.write_text("{")
    with pytest.raises(ControlError, match="unreadable"):
        Client(tmp_path)
    control_file.write_text(json.dumps({"address": address, "host": socket.gethostname(), "pid": os.getpid()}))
    assert Client(tmp_path).timeout == control_module.DEFAULT_TIMEOUT_S


def test_the_control_file_appears_whole(tmp_path, monkeypatch) -> None:
    seen: list[str | None] = []
    replace = os.replace

    def watching_replace(src, dst) -> None:
        # what a reader sees right before the file lands: nothing, never a partial write
        seen.append((tmp_path / CONTROL_FILE).read_text() if (tmp_path / CONTROL_FILE).exists() else None)
        replace(src, dst)

    monkeypatch.setattr(control_module.os, "replace", watching_replace)
    server = ControlServer(SimpleNamespace(run_dir=tmp_path))

    async def start_and_stop() -> None:
        await server.start()
        await server.stop()

    asyncio.run(start_and_stop())

    assert seen == [None]
    assert sorted(path.name for path in tmp_path.iterdir()) == []  # stop removed it; no temp file left
