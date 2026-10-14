"""A streamed task runs as a group of ``group_size`` rollouts, each reported to its
taskset once. A streaming taskset cancels a task it handed out (``Taskset.cancel``): the
runner stops every rollout of the task wherever it is (in flight: the env request is
cancelled; not yet dispatched: never run) and reports each through ``on_cancelled``. A
rollout that landed first, even one of the same group, is observed as usual. Runs the
real ``Dispatcher``/``EvalSource``/``EvalSink`` against fake envs."""

import asyncio
import threading
import uuid
from collections import Counter
from types import SimpleNamespace

import pytest
import verifiers.v1 as vf
from verifiers.v1.graph import MessageNode
from verifiers.v1.types import AssistantMessage, UserMessage

from prime_rl.eval import runner as runner_module
from prime_rl.eval.runner import EvalRunner
from prime_rl.orchestrator.dispatcher import Dispatcher
from prime_rl.orchestrator.envs import EvalEnv, EvalEnvs
from prime_rl.orchestrator.eval_sink import EvalSink
from prime_rl.orchestrator.eval_source import EvalSource
from prime_rl.orchestrator.task_feed import TaskFeed
from prime_rl.orchestrator.types import GroupState, Policy

TIMEOUT_S = 30


class KeyedTask(vf.Task[vf.TaskData]):
    """Keyed by id alone, as a pipeline keys its stage tasks."""

    @property
    def key(self) -> str:
        return self.data.id


def _task(task_id: str) -> KeyedTask:
    return KeyedTask(vf.TaskData(id=task_id))


def _episode(task: vf.Task) -> vf.WireEpisode:
    trace_task = vf.TraceTask(
        type="KeyedTask", data=vf.WireTaskData(**task.data.model_dump(mode="json")), key=task.key, hash=task.hash
    )
    nodes = [
        MessageNode(parent=None, message=UserMessage(content="U"), sampled=False, token_ids=[1], mask=[False]),
        MessageNode(
            parent=0, message=AssistantMessage(content="A"), sampled=True, token_ids=[2], mask=[True], logprobs=[-0.1]
        ),
    ]
    trace = vf.Trace(task=trace_task, agent=vf.AgentInfo(config=vf.AgentConfig()), nodes=nodes, rewards={}, ok=True)
    episode = vf.Episode(task=trace_task, traces=[trace], ok=True)
    return vf.WireEpisode.model_validate(episode.model_dump())


class Stream(vf.Taskset):
    """Yields ``task_ids`` in order, then ends once every rollout of every yielded task
    was reported (``group_size`` per task). ``on_yield`` runs on the feed thread right
    after a task is handed out; ``cancel_after`` cancels a task once that many of its
    rollouts were reported."""

    INFINITE = True

    def __init__(
        self, task_ids: list[str], on_yield=None, *, group_size: int = 1, cancel_after: dict[str, int] | None = None
    ) -> None:
        super().__init__(vf.TasksetConfig())
        self.task_ids = task_ids
        self.on_yield = on_yield or (lambda taskset, task_id: None)
        self.group_size = group_size
        self.cancel_after = cancel_after or {}
        self.calls: list[tuple[str, str]] = []
        self.reported = threading.Semaphore(0)

    def load(self):
        for task_id in self.task_ids:
            yield _task(task_id)
            self.on_yield(self, task_id)
        for _ in range(len(self.task_ids) * self.group_size):
            assert self.reported.acquire(timeout=TIMEOUT_S)

    def report(self, hook: str, task: vf.Task) -> None:
        self.calls.append((hook, task.data.id))
        heard = sum(task_id == task.data.id for _, task_id in self.calls)
        if heard == self.cancel_after.get(task.data.id):
            self.cancel(task.key)
        self.reported.release()

    def observe(self, task, episode) -> None:
        self.report("observe", task)

    def on_failed(self, task, error) -> None:
        self.report("on_failed", task)

    def on_cancelled(self, task) -> None:
        self.report("on_cancelled", task)


class HangingEnv:
    """``slow*`` tasks run until cancelled; any other task returns an ok episode."""

    def __init__(self, taskset: vf.Taskset, *, group_size: int = 1) -> None:
        self.name = "pipeline"
        self.config = SimpleNamespace(group_size=group_size)
        self.taskset = taskset
        self.examples: list[vf.Task] = []
        self.feed = TaskFeed(iter(taskset), name=self.name)
        self.started: list[str] = []
        self.aborted: list[str] = []
        self.running = asyncio.Event()

    async def run(self, *, client, model_name, cache_salt, task_data, on_delta) -> vf.WireEpisode:
        task_id = task_data["id"]
        self.started.append(task_id)
        if task_id.startswith("slow"):
            self.running.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.aborted.append(task_id)
                raise
        await asyncio.sleep(0)
        return _episode(KeyedTask(vf.TaskData.model_validate(task_data)))


async def _no_sessions(_session_ids) -> None:
    return None


def _runner(tmp_path, env: HangingEnv, *, max_inflight: int = 4) -> EvalRunner:
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
    return runner


async def _run_epoch(runner: EvalRunner, during=None) -> None:
    runner.dispatcher_task = asyncio.create_task(runner.dispatcher.start())
    try:
        epoch = asyncio.create_task(runner.run_epoch(runner.eval_source.trigger(0)[0], 0))
        if during is not None:
            await asyncio.wait_for(during(), timeout=TIMEOUT_S)
        await asyncio.wait_for(epoch, timeout=TIMEOUT_S)
    finally:
        await runner.dispatcher.stop()


def test_an_in_flight_episode_is_aborted_and_reported_cancelled(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(runner_module, "POLL_INTERVAL_S", 0.05)
    taskset = Stream(["slow-0", "ok-0"])
    env = HangingEnv(taskset.skip(0))
    runner = _runner(tmp_path, env)

    async def cancel_once_running() -> None:
        await env.running.wait()
        taskset.cancel("slow-0")

    asyncio.run(_run_epoch(runner, cancel_once_running))

    assert sorted(taskset.calls) == [("observe", "ok-0"), ("on_cancelled", "slow-0")]
    assert env.aborted == ["slow-0"]
    assert runner.dispatcher.requested_cancels == set()
    assert runner.cancel_tasks == set()


@pytest.mark.parametrize("group_size", [1, 3])
def test_a_task_cancelled_before_dispatch_never_runs(tmp_path, monkeypatch, group_size) -> None:
    monkeypatch.setattr(runner_module, "POLL_INTERVAL_S", 0.05)

    def cancel_waiting(taskset: Stream, task_id: str) -> None:
        # one permit: "ok-1" waits on the feed behind "slow-0" when it is cancelled
        if task_id == "ok-1":
            taskset.cancel("ok-1")

    taskset = Stream(["slow-0", "ok-1"], on_yield=cancel_waiting, group_size=group_size)
    env = HangingEnv(taskset.skip(0), group_size=group_size)
    runner = _runner(tmp_path, env, max_inflight=1)

    async def then_cancel_the_running_one() -> None:
        await env.running.wait()
        while runner.dispatcher.requested_cancels != {("pipeline", "ok-1")}:
            await asyncio.sleep(0.01)
        taskset.cancel("slow-0")

    asyncio.run(_run_epoch(runner, then_cancel_the_running_one))

    # slow-0's first rollout ran under the one permit, its others never started
    assert sorted(taskset.calls) == [("on_cancelled", "ok-1")] * group_size + [("on_cancelled", "slow-0")] * group_size
    assert env.started == ["slow-0"]
    assert runner.dispatcher.requested_cancels == set()


def test_a_cancel_asked_before_the_episode_landed_is_moot(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(runner_module, "POLL_INTERVAL_S", 0.05)

    class CancelsTooLate(Stream):
        def observe(self, task, episode) -> None:
            # the taskset asked before the report; the runner polls it after
            self.cancel(task.key)
            super().observe(task, episode)

    taskset = CancelsTooLate(["ok-0", "ok-1"])
    env = HangingEnv(taskset.skip(0))
    runner = _runner(tmp_path, env)

    asyncio.run(_run_epoch(runner))

    assert sorted(taskset.calls) == [("observe", "ok-0"), ("observe", "ok-1")]
    assert runner.dispatcher.requested_cancels == set()


class RerunEnv(HangingEnv):
    """A key's first run waits until its second has started, then lands; the second
    runs until cancelled."""

    def __init__(self, taskset: vf.Taskset) -> None:
        super().__init__(taskset)
        self.rerun = asyncio.Event()

    async def run(self, *, client, model_name, cache_salt, task_data, on_delta) -> vf.WireEpisode:
        task_id = task_data["id"]
        self.started.append(task_id)
        if self.started.count(task_id) == 1:
            await self.rerun.wait()
            return _episode(KeyedTask(vf.TaskData.model_validate(task_data)))
        self.rerun.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.aborted.append(task_id)
            raise


def test_a_cancel_reaches_a_rerun_of_a_task_reported_since_the_last_poll(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(runner_module, "POLL_INTERVAL_S", 0.05)

    class CancelsTheRerun(Stream):
        def observe(self, task, episode) -> None:
            # the first run is enough: stop the rerun already in flight
            self.cancel(task.key)
            super().observe(task, episode)

    taskset = CancelsTheRerun(["t", "t"])
    env = RerunEnv(taskset.skip(0))
    runner = _runner(tmp_path, env)

    asyncio.run(_run_epoch(runner))

    assert taskset.calls == [("observe", "t"), ("on_cancelled", "t")]
    assert env.aborted == ["t"]
    assert runner.dispatcher.requested_cancels == set()


class GroupEnv(HangingEnv):
    """Each task's rollouts take ``outcomes`` in the order they start: ``ok`` lands an
    episode and ``fail`` raises, both once every rollout of the group has started;
    ``hang`` runs until cancelled."""

    def __init__(self, taskset: vf.Taskset, outcomes: list[str]) -> None:
        super().__init__(taskset, group_size=len(outcomes))
        self.outcomes = outcomes
        self.group_started: dict[str, asyncio.Event] = {}

    async def run(self, *, client, model_name, cache_salt, task_data, on_delta) -> vf.WireEpisode:
        task_id = task_data["id"]
        outcome = self.outcomes[self.started.count(task_id)]
        self.started.append(task_id)
        group_started = self.group_started.setdefault(task_id, asyncio.Event())
        if self.started.count(task_id) == len(self.outcomes):
            group_started.set()
        if outcome == "hang":
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.aborted.append(task_id)
                raise
        await group_started.wait()
        if outcome == "fail":
            raise RuntimeError("env server died")
        return _episode(KeyedTask(vf.TaskData.model_validate(task_data)))


def test_each_rollout_of_a_streamed_group_is_reported(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(runner_module, "POLL_INTERVAL_S", 0.05)
    taskset = Stream(["a", "b"], group_size=3)
    env = GroupEnv(taskset.skip(0), ["ok", "ok", "ok"])
    runner = _runner(tmp_path, env, max_inflight=6)

    asyncio.run(_run_epoch(runner))

    assert Counter(taskset.calls) == {("observe", "a"): 3, ("observe", "b"): 3}
    assert Counter(env.started) == {"a": 3, "b": 3}
    assert runner.dispatcher.groups == {}


def test_a_group_cancelled_before_any_rollout_landed_is_reported_whole(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(runner_module, "POLL_INTERVAL_S", 0.05)
    taskset = Stream(["slow-0"], group_size=3)
    env = HangingEnv(taskset.skip(0), group_size=3)
    runner = _runner(tmp_path, env, max_inflight=2)

    async def cancel_once_running() -> None:
        while len(env.started) < 2:
            await asyncio.sleep(0.01)
        taskset.cancel("slow-0")

    asyncio.run(_run_epoch(runner, cancel_once_running))

    assert taskset.calls == [("on_cancelled", "slow-0")] * 3
    assert len(env.aborted) == 2  # two ran under two permits; the third never started
    assert runner.dispatcher.requested_cancels == set()


def test_a_cancel_after_a_rollout_landed_stops_the_rest_of_its_group(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(runner_module, "POLL_INTERVAL_S", 0.05)
    # the taskset cancels from the first rollout's observe: the key is reported by the
    # time the cancel is polled, yet two rollouts of its group are still in flight
    taskset = Stream(["t"], group_size=3, cancel_after={"t": 1})
    env = GroupEnv(taskset.skip(0), ["ok", "hang", "hang"])
    runner = _runner(tmp_path, env)

    asyncio.run(_run_epoch(runner))

    assert taskset.calls == [("observe", "t"), ("on_cancelled", "t"), ("on_cancelled", "t")]
    assert env.aborted == ["t", "t"]
    assert runner.dispatcher.requested_cancels == set()
    assert runner.dispatcher.groups == {}


def test_a_group_reports_each_rollout_by_its_own_outcome(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(runner_module, "POLL_INTERVAL_S", 0.05)
    taskset = Stream(["t"], group_size=3)
    env = GroupEnv(taskset.skip(0), ["ok", "fail", "ok"])
    runner = _runner(tmp_path, env)

    asyncio.run(_run_epoch(runner))

    assert Counter(taskset.calls) == {("observe", "t"): 2, ("on_failed", "t"): 1}


def test_a_cancel_after_a_success_and_a_failure_stops_only_the_rest(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(runner_module, "POLL_INTERVAL_S", 0.05)
    taskset = Stream(["t"], group_size=3, cancel_after={"t": 2})
    env = GroupEnv(taskset.skip(0), ["ok", "fail", "hang"])
    runner = _runner(tmp_path, env)

    asyncio.run(_run_epoch(runner))

    assert sorted(taskset.calls[:2]) == [("observe", "t"), ("on_failed", "t")]
    assert taskset.calls[2:] == [("on_cancelled", "t")]
    assert env.aborted == ["t"]
    assert runner.dispatcher.requested_cancels == set()


def _infinite_env(*, group_size: int = 1, streams: bool = True) -> tuple[EvalEnv, list[str]]:
    """An eval env over an infinite taskset, and the tasks it pulled."""
    pulled: list[str] = []

    def tasks():
        while True:
            pulled.append("t")
            yield _task("t")

    env = EvalEnv.__new__(EvalEnv)
    env.config = SimpleNamespace(resolved_name="pipeline", group_size=group_size)
    env.streams = streams
    env.num_tasks = None
    env.tasks = tasks()
    env.feed = None
    return env, pulled


def test_a_streaming_source_dispatches_each_task_as_a_group() -> None:
    env, _ = _infinite_env(group_size=3)
    env.examples = []
    env.open_feed()
    eval_envs = EvalEnvs.__new__(EvalEnvs)
    eval_envs._envs = {"pipeline": env}
    source = EvalSource(eval_envs)
    source.trigger(0)

    async def next_request():
        while (request := source.next_task()) is None:
            await asyncio.sleep(0.01)
        return request

    request = asyncio.run(asyncio.wait_for(next_request(), timeout=TIMEOUT_S))
    assert request.task.key == "t" and request.rollouts == 3


def test_an_online_eval_rejects_an_infinite_taskset_before_pulling_it() -> None:
    env, pulled = _infinite_env(streams=False)

    with pytest.raises(ValueError, match="infinite taskset"):
        env.open_feed()
    assert env.feed is None and pulled == []


def test_a_cancel_finds_exactly_the_tasks_eval_groups(tmp_path) -> None:
    runner = _runner(tmp_path, HangingEnv(Stream([]).skip(0)))
    dispatcher = runner.dispatcher

    def group(kind: str, env_name: str, task_id: str) -> uuid.UUID:
        gid = uuid.uuid4()
        dispatcher.groups[gid] = GroupState(
            kind=kind, env_name=env_name, task=_task(task_id), step=0, episodes_to_schedule=1, target_episodes=1
        )
        return gid

    wanted = group("eval", "pipeline", "t")
    group("train", "pipeline", "t")
    group("eval", "other", "t")
    group("eval", "pipeline", "u")

    assert dispatcher.eval_groups("pipeline", "t") == [wanted]
    assert asyncio.run(dispatcher.cancel_eval_task("pipeline", "t")) == 1
    assert dispatcher.eval_groups("pipeline", "t") == []
    assert len(dispatcher.groups) == 3
