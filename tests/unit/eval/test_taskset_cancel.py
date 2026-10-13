"""A streaming taskset cancels a task it handed out (``Taskset.cancel``): the runner
stops the task's rollouts wherever they are (in flight: the env request is cancelled;
not yet dispatched: never run) and reports each through ``on_cancelled``. A rollout
that landed first is observed as usual. Runs the real ``Dispatcher``/``EvalSource``/
``EvalSink`` against fake envs."""

import asyncio
import threading
from types import SimpleNamespace

import verifiers.v1 as vf
from verifiers.v1.graph import MessageNode
from verifiers.v1.types import AssistantMessage, UserMessage

from prime_rl.eval import runner as runner_module
from prime_rl.eval.runner import EvalRunner
from prime_rl.orchestrator.dispatcher import Dispatcher
from prime_rl.orchestrator.envs import EvalEnvs
from prime_rl.orchestrator.eval_sink import EvalSink
from prime_rl.orchestrator.eval_source import EvalSource
from prime_rl.orchestrator.task_feed import TaskFeed
from prime_rl.orchestrator.types import Policy

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
    """Yields ``task_ids`` in order, then ends once every yielded task was reported.
    ``on_yield`` runs on the feed thread right after a task is handed out."""

    INFINITE = True

    def __init__(self, task_ids: list[str], on_yield=None) -> None:
        super().__init__(vf.TasksetConfig())
        self.task_ids = task_ids
        self.on_yield = on_yield or (lambda taskset, task_id: None)
        self.calls: list[tuple[str, str]] = []
        self.reported = threading.Semaphore(0)

    def load(self):
        for task_id in self.task_ids:
            yield _task(task_id)
            self.on_yield(self, task_id)
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


def test_a_task_cancelled_before_dispatch_never_runs(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(runner_module, "POLL_INTERVAL_S", 0.05)

    def cancel_waiting(taskset: Stream, task_id: str) -> None:
        # one permit: "ok-1" waits on the feed behind "slow-0" when it is cancelled
        if task_id == "ok-1":
            taskset.cancel("ok-1")

    taskset = Stream(["slow-0", "ok-1"], on_yield=cancel_waiting)
    env = HangingEnv(taskset.skip(0))
    runner = _runner(tmp_path, env, max_inflight=1)

    async def then_cancel_the_running_one() -> None:
        await env.running.wait()
        while runner.dispatcher.requested_cancels != {("pipeline", "ok-1")}:
            await asyncio.sleep(0.01)
        taskset.cancel("slow-0")

    asyncio.run(_run_epoch(runner, then_cancel_the_running_one))

    assert sorted(taskset.calls) == [("on_cancelled", "ok-1"), ("on_cancelled", "slow-0")]
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


def test_every_unlanded_rollout_of_a_group_is_reported(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(runner_module, "POLL_INTERVAL_S", 0.05)
    taskset = Stream(["slow-0"])
    env = HangingEnv(taskset.skip(0), group_size=3)
    runner = _runner(tmp_path, env, max_inflight=2)

    async def cancel_once_running() -> None:
        while len(env.started) < 2:
            await asyncio.sleep(0.01)
        taskset.cancel("slow-0")

    asyncio.run(_run_epoch(runner, cancel_once_running))

    assert taskset.calls == [("on_cancelled", "slow-0")] * 3
    assert len(env.aborted) == 2  # two ran under two permits; the third never started
