"""The eval runner reports every dispatched rollout to its taskset exactly once:
``observe(task, episode)`` when an episode came back, ``on_failed(task, error)`` when
none did. Runs the real ``Dispatcher``/``EvalSource``/``EvalSink`` against fake envs."""

import asyncio
import threading
from types import SimpleNamespace

import pytest
import verifiers.v1 as vf
from verifiers.v1.graph import MessageNode
from verifiers.v1.types import AssistantMessage, UserMessage

from prime_rl import monitors
from prime_rl.eval import runner as runner_module
from prime_rl.eval.runner import EvalRunner
from prime_rl.orchestrator import eval_sink as eval_sink_module
from prime_rl.orchestrator.dispatcher import Dispatcher
from prime_rl.orchestrator.envs import EvalEnvs
from prime_rl.orchestrator.eval_sink import EvalSink
from prime_rl.orchestrator.eval_source import EvalSource
from prime_rl.orchestrator.task_feed import TaskFeed
from prime_rl.orchestrator.types import Policy

TIMEOUT_S = 30


def _task(task_id: str) -> vf.Task:
    return vf.Task(vf.TaskData(id=task_id))


def _episode(task: vf.Task, *, ok: bool) -> vf.WireEpisode:
    trace_task = vf.TraceTask(
        type="Task", data=vf.WireTaskData(**task.data.model_dump(mode="json")), key=task.key, hash=task.hash
    )
    nodes = [
        MessageNode(parent=None, message=UserMessage(content="U"), sampled=False, token_ids=[1], mask=[False]),
        MessageNode(
            parent=0, message=AssistantMessage(content="A"), sampled=True, token_ids=[2], mask=[True], logprobs=[-0.1]
        ),
    ]
    errors = [] if ok else [vf.Error(type="Boom", message="the agent failed")]
    trace = vf.Trace(
        task=trace_task, agent=vf.AgentInfo(config=vf.AgentConfig()), nodes=nodes, rewards={}, ok=ok, errors=errors
    )
    episode = vf.Episode(task=trace_task, traces=[trace], ok=ok, errors=errors)
    return vf.WireEpisode.model_validate(episode.model_dump())


class RecordingTaskset:
    """Records the hooks the runner calls, in order."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    def observe(self, task: vf.Task, episode: vf.Episode) -> None:
        self.calls.append(("observe", task.data.id, "ok" if episode.ok else episode.last_error.type))

    def on_failed(self, task: vf.Task, error: BaseException) -> None:
        self.calls.append(("on_failed", task.data.id, type(error).__name__))


class FakeEnv:
    """An eval env whose ``run`` answers by task id: ``ok*`` returns an ok episode,
    ``bad*`` a failed one, ``crash*`` raises (the env server died)."""

    def __init__(self, name: str, *, taskset, tasks: list[vf.Task] = (), group_size: int = 1) -> None:
        self.name = name
        self.config = SimpleNamespace(group_size=group_size)
        self.taskset = taskset
        self.examples = list(tasks)
        self.feed: TaskFeed | None = None

    async def run(self, *, client, model_name, cache_salt, task_data, on_delta) -> vf.WireEpisode:
        await asyncio.sleep(0)
        task_id = task_data["id"]
        if task_id.startswith("crash"):
            raise ConnectionError("env server closed the connection")
        return _episode(vf.Task(vf.TaskData.model_validate(task_data)), ok=task_id.startswith("ok"))


async def _no_sessions(_session_ids) -> None:
    return None


def _runner(tmp_path, *envs: FakeEnv) -> EvalRunner:
    eval_envs = EvalEnvs.__new__(EvalEnvs)
    eval_envs._envs = {env.name: env for env in envs}
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
        initial_max_inflight=4,
        max_inflight_ceiling=4,
        tasks_per_minute=None,
        max_off_policy_steps=0,
        run_id="run",
        run_name=None,
        record_eval_tasks=True,
    )
    return runner


async def _run_epoch(runner: EvalRunner) -> None:
    runner.dispatcher_task = asyncio.create_task(runner.dispatcher.start())
    try:
        await asyncio.wait_for(runner.run_epoch(runner.eval_source.trigger(0)[0], 0), timeout=TIMEOUT_S)
    finally:
        await runner.dispatcher.stop()


def test_each_rollout_is_observed_or_failed_once(tmp_path) -> None:
    taskset = RecordingTaskset()
    tasks = [_task("ok-0"), _task("bad-0"), _task("crash-0"), _task("ok-1")]
    runner = _runner(tmp_path, FakeEnv("math", taskset=taskset, tasks=tasks))

    asyncio.run(_run_epoch(runner))

    assert sorted(taskset.calls) == [
        ("observe", "bad-0", "Boom"),
        ("observe", "ok-0", "ok"),
        ("observe", "ok-1", "ok"),
        ("on_failed", "crash-0", "ConnectionError"),
    ]
    assert runner.dispatcher.eval_tasks == {}


def test_every_rollout_of_a_group_is_reported(tmp_path) -> None:
    taskset = RecordingTaskset()
    runner = _runner(tmp_path, FakeEnv("math", taskset=taskset, tasks=[_task("ok-0"), _task("crash-0")], group_size=3))

    asyncio.run(_run_epoch(runner))

    assert sorted(taskset.calls) == [("observe", "ok-0", "ok")] * 3 + [("on_failed", "crash-0", "ConnectionError")] * 3


def test_observe_error_stops_the_epoch(tmp_path) -> None:
    class Broken(RecordingTaskset):
        def observe(self, task, episode) -> None:
            raise RuntimeError("cannot commit")

    runner = _runner(tmp_path, FakeEnv("math", taskset=Broken(), tasks=[_task("ok-0")]))

    with pytest.raises(RuntimeError, match="cannot commit"):
        asyncio.run(_run_epoch(runner))


def test_on_failed_error_stops_the_epoch(tmp_path) -> None:
    class Broken(RecordingTaskset):
        def on_failed(self, task, error) -> None:
            raise RuntimeError("cannot record")

    runner = _runner(tmp_path, FakeEnv("math", taskset=Broken(), tasks=[_task("crash-0")]))

    with pytest.raises(RuntimeError, match="cannot record"):
        asyncio.run(_run_epoch(runner))


def test_a_dead_dispatcher_fails_the_epoch(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(runner_module, "POLL_INTERVAL_S", 0.05)
    env = FakeEnv("math", taskset=RecordingTaskset(), tasks=[_task("ok-0")])

    async def wrong_task(**kwargs) -> vf.WireEpisode:
        return _episode(_task("someone-else"), ok=True)

    env.run = wrong_task
    runner = _runner(tmp_path, env)

    with pytest.raises(ValueError, match="does not match dispatched task"):
        asyncio.run(_run_epoch(runner))


class Chain(vf.Taskset):
    """An infinite taskset whose ``load()`` yields its next task only once the previous
    one was reported, like a pipeline whose next stage waits on the last outcome."""

    INFINITE = True

    def __init__(self, task_ids: list[str]) -> None:
        super().__init__(vf.TasksetConfig())
        self.task_ids = task_ids
        self.reported = threading.Semaphore(0)
        self.calls: list[tuple[str, str]] = []
        self.load_threads: set[int] = set()

    def load(self):
        for task_id in self.task_ids:
            self.load_threads.add(threading.get_ident())
            yield _task(task_id)
            # blocks until the runner reports the task just yielded
            assert self.reported.acquire(timeout=TIMEOUT_S), f"{task_id} was never reported"

    def observe(self, task, episode) -> None:
        self.calls.append(("observe", task.data.id))
        self.reported.release()

    def on_failed(self, task, error) -> None:
        self.calls.append(("on_failed", task.data.id))
        self.reported.release()


def test_a_streaming_taskset_waits_on_observe_without_deadlock(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(runner_module, "POLL_INTERVAL_S", 0.05)
    task_ids = ["ok-0", "crash-0", "bad-0", "ok-1"]
    taskset = Chain(task_ids)
    view = taskset.skip(0)  # the env holds the selected view, as ``Env.start`` does
    env = FakeEnv("pipeline", taskset=view)
    env.feed = TaskFeed(iter(view), name=env.name)
    runner = _runner(tmp_path, env)

    asyncio.run(_run_epoch(runner))

    assert taskset.calls == [("observe", "ok-0"), ("on_failed", "crash-0"), ("observe", "bad-0"), ("observe", "ok-1")]
    assert threading.get_ident() not in taskset.load_threads
    assert env.feed.done


class EpochRecorder:
    """A monitor recording each ``log_eval_epoch`` call and each metrics row."""

    def __init__(self) -> None:
        self.epochs: list[tuple[int, bool]] = []
        self.metrics: list[dict] = []

    async def log(self, data, step, kind="train", subset="effective") -> None:
        if isinstance(data, dict):
            self.metrics.append(data)

    async def log_annotations(self, updates) -> None:
        pass

    async def log_eval_epoch(self, env_name, step, episodes, *, final=True) -> None:
        self.epochs.append((len(episodes), final))


def test_a_streaming_epoch_is_logged_window_by_window(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(runner_module, "POLL_INTERVAL_S", 0.05)
    monkeypatch.setattr(eval_sink_module, "STREAM_WINDOW_ATTEMPTS", 2)
    recorder = EpochRecorder()
    monkeypatch.setattr(monitors, "MONITORS", [recorder])
    taskset = Chain([f"ok-{i}" for i in range(5)])
    view = taskset.skip(0)
    env = FakeEnv("pipeline", taskset=view)
    env.feed = TaskFeed(iter(view), name=env.name)
    runner = _runner(tmp_path, env)
    held: list[int] = []
    add = runner.eval_sink.add

    def add_and_count(episode):
        batch = add(episode)
        held.append(sum(len(episodes) for episodes in runner.eval_sink.pending_batches.values()))
        return batch

    runner.eval_sink.add = add_and_count

    asyncio.run(_run_epoch(runner))

    assert recorder.epochs == [(2, False), (2, False), (1, True)]
    assert [row["eval/pipeline/window"] for row in recorder.metrics] == [0.0, 1.0, 2.0]
    assert max(held) < 2  # a full window is logged and dropped as it closes
    assert runner.eval_sink.pending_batches == {}


def test_a_streaming_window_closes_on_age(monkeypatch) -> None:
    monkeypatch.setattr(eval_sink_module, "STREAM_WINDOW_S", 0.0)
    stream = FakeEnv("pipeline", taskset=RecordingTaskset())
    stream.feed = TaskFeed(iter([]), name=stream.name)
    bounded = FakeEnv("math", taskset=RecordingTaskset(), tasks=[_task("ok-0"), _task("ok-1")])
    eval_envs = EvalEnvs.__new__(EvalEnvs)
    eval_envs._envs = {env.name: env for env in (stream, bounded)}
    sink = EvalSink(eval_envs=eval_envs)

    def episode(env_name: str, task_id: str) -> vf.WireEpisode:
        landed = _episode(_task(task_id), ok=True)
        landed.env.name = env_name
        landed.record_run(vf.TrainRunInfo(id="run", work=vf.EvalWorkInfo(step=0)))
        return landed

    assert sink.add(episode("pipeline", "ok-0")) is None
    assert sink.add(episode("math", "ok-0")) is None
    (window,) = sink.due_windows()  # the bounded epoch has no window
    assert (window.env_name, window.window, window.final, len(window.episodes)) == ("pipeline", 0, False, 1)
    assert sink.due_windows() == []  # nothing arrived since
    assert sorted(sink.batch_progress()) == [("math", 0, 1, 2), ("pipeline", 0, 1, None)]
    last = sink.process_batch(("pipeline", 0))
    assert (last.window, last.final, len(last.episodes)) == (1, True, 0)
    whole = sink.add(episode("math", "ok-1"))
    assert (whole.window, whole.final, len(whole.episodes)) == (None, True, 2)
