"""Durable agent calls end to end: a verifiers env whose ``durable()`` is a
``RunTraceLookup`` over the run, served the way an env server serves it (its deltas
through ``DeltaStreamer`` into the real ``Dispatcher``, the real ``FileMonitor`` writing
the live, partial and finished traces and the labels index), run by the real
``EvalRunner``. An episode that fails after its first agent call is run again and the
call returns its first trace without calling the model, whether the retry is the env's
own (``--env.retries``) or a later attempt of the run (``--resume``) after the episode
was cancelled.

Deterministic, no API key: a scripted OpenAI-compatible endpoint and the null harness
with an in-process program on a subprocess runtime."""

import asyncio
from pathlib import Path
from types import SimpleNamespace

import orjson
import pytest
import verifiers.v1 as vf
from aiohttp import web
from openai import AsyncOpenAI
from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.client import EvalClientConfig
from verifiers.v1.harnesses.null.harness import NullHarness
from verifiers.v1.runtimes import ProgramResult
from verifiers.v1.serve.delta import DeltaStreamer, pack, unpack
from verifiers.v1.utils import retries

from prime_rl import monitors
from prime_rl.configs.monitors import FileMonitorConfig
from prime_rl.eval import resume
from prime_rl.eval.runner import EvalRunner
from prime_rl.monitors.file import FileMonitor
from prime_rl.monitors.file.traces import get_labels_index_path
from prime_rl.monitors.file.traces.live import get_partial_dir
from prime_rl.monitors.file.traces.lookup import RunTraceLookup
from prime_rl.orchestrator.dispatcher import Dispatcher
from prime_rl.orchestrator.envs import EvalEnvs
from prime_rl.orchestrator.eval_sink import EvalSink
from prime_rl.orchestrator.eval_source import EvalSource
from prime_rl.orchestrator.types import Policy

__all__ = ["Jobs"]  # the taskset this module is, by id (`TASKSET_ID`)

TASKSET_ID = __name__
TIMEOUT_S = 60


class Jobs(vf.Taskset[vf.Task, vf.TasksetConfig]):
    def load(self):
        yield vf.Task(vf.TaskData(idx=0, prompt="job-1"))


class ReviewConfig(vf.EnvConfig):
    solver: vf.AgentConfig = vf.AgentConfig(runtime={"type": "subprocess"}, harness={"id": "null"})
    run_dir: str = ""


class ReviewEnv(vf.Env[ReviewConfig]):
    """Drafts, then reviews the draft: two calls of one seat (``solver/0``, ``solver/1``).
    ``crash`` jobs raise between them; ``hang`` jobs wait there until cancelled."""

    crash: dict[str, int] = {}
    hang: set[str] = set()
    drafted: asyncio.Event | None = None
    drafts: list[vf.Trace] = []

    def trace_labels(self, task):
        return {"flow": {"job": task.data.prompt}}

    def durable(self, task):
        if not hasattr(self, "lookup"):
            self.lookup = RunTraceLookup(self.config.run_dir)  # one per process
        return vf.Durable(self.lookup)

    async def run(self, task, agents):
        draft = await agents.solver.run(task)
        ReviewEnv.drafts.append(draft)
        job = task.data.prompt
        if ReviewEnv.crash.get(job):
            ReviewEnv.crash[job] -= 1
            raise RuntimeError("the host died after the draft")
        if job in ReviewEnv.hang:
            ReviewEnv.drafted.set()
            await asyncio.Event().wait()
        await agents.solver.run(vf.Task(vf.TaskData(idx=0, prompt=f"review {draft.last_reply}")))


async def _program(self, ctx, trace, runtime, endpoint, secret, mcp_urls, data, **kwargs) -> ProgramResult:
    """The null harness's program, in process: one chat completion on the prompt."""
    _, prompt = self.resolve_prompt(data)
    client = AsyncOpenAI(base_url=endpoint, api_key=secret, max_retries=0)
    try:
        await client.chat.completions.create(model=ctx.model, messages=[{"role": "user", "content": prompt}])
    finally:
        await client.close()
    return ProgramResult(exit_code=0, stdout="", stderr="")


class ServedEnv:
    """``EnvServer._run`` without the socket: the episode's traces stream as deltas,
    through msgpack, into the dispatcher's ``on_delta``."""

    def __init__(self, env: ReviewEnv) -> None:
        self.env = env
        self.name = "review"
        self.config = SimpleNamespace(group_size=1)
        self.examples = list(Jobs(vf.TasksetConfig(id=TASKSET_ID)))
        self.feed = None
        self.taskset = SimpleNamespace(observe=lambda *a: None, on_failed=lambda *a: None)

    async def run(self, *, client, model_name, cache_salt, task_data, on_delta) -> vf.WireEpisode:
        (slot,) = self.env.slots(vf.Task(vf.TaskData.model_validate(task_data)))

        async def send(delta: dict) -> None:
            on_delta(unpack(pack(delta)))

        ctx = ModelContext(client=client, model=model_name, sampling=vf.SamplingConfig())
        async with DeltaStreamer(lambda: slot.traces, send) as streamer:
            episode = await self.env.run_slot(slot, ctx, on_trace=streamer.watch)
        return vf.WireEpisode.model_validate(episode.model_dump())


@pytest.fixture
def scripted(monkeypatch):
    monkeypatch.setattr(NullHarness, "setup", lambda self, runtime: asyncio.sleep(0))
    monkeypatch.setattr(NullHarness, "launch", _program)
    monkeypatch.setattr(retries, "backoff", lambda attempt: 0.3)
    monkeypatch.setenv("SCRIPTED_KEY", "scripted")
    ReviewEnv.crash, ReviewEnv.hang, ReviewEnv.drafts = {}, set(), []


async def _endpoint(requests: list[dict]) -> tuple[web.AppRunner, str]:
    async def handle(request: web.Request) -> web.Response:
        body = await request.json()
        requests.append(body)
        reply = f"done: {body['messages'][-1]['content']}"
        return web.json_response(
            {
                "id": "scripted",
                "object": "chat.completion",
                "created": 0,
                "model": "scripted",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": reply}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        )

    app = web.Application()
    app.router.add_post("/v1/chat/completions", handle)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    return runner, f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}/v1"


class Run:
    """One attempt of an eval run over ``run_dir``: its monitor, env and runner."""

    def __init__(self, run_dir: Path, url: str, monkeypatch, **env: object) -> None:
        self.monitor = FileMonitor(FileMonitorConfig(compress=False))
        monkeypatch.setattr(monitors, "MONITORS", [self.monitor])
        config = ReviewConfig(taskset=vf.TasksetConfig(id=TASKSET_ID), run_dir=str(run_dir), **env)
        self.served = ServedEnv(ReviewEnv(config))
        eval_envs = EvalEnvs.__new__(EvalEnvs)
        eval_envs._envs = {self.served.name: self.served}
        self.runner = EvalRunner(SimpleNamespace(), run_dir=run_dir)
        self.runner.eval_envs = eval_envs
        self.runner.eval_source = EvalSource(eval_envs)
        self.runner.eval_sink = EvalSink(eval_envs=eval_envs)
        self.runner.policy = Policy(model_name="scripted")
        client = EvalClientConfig(base_url=url, api_key_var="SCRIPTED_KEY")
        self.runner.dispatcher = Dispatcher(
            train_envs=None,
            eval_envs=eval_envs,
            train_source=None,
            eval_source=self.runner.eval_source,
            policy_clients=SimpleNamespace(eval_client=client, finish_sessions=lambda ids: asyncio.sleep(0)),
            policy=self.runner.policy,
            progress=None,
            initial_max_inflight=1,
            max_inflight_ceiling=1,
            tasks_per_minute=None,
            max_off_policy_steps=0,
            run_id="run",
            run_name=None,
            record_eval_tasks=True,
        )
        self.run_dir = run_dir

    async def epoch(self, until: asyncio.Event | None = None) -> None:
        """The epoch to its end, or until ``until`` is set: then the run stops (its
        in-flight episodes cancelled, as at shutdown)."""
        await self.monitor.init(self.run_dir)
        dispatcher = asyncio.create_task(self.runner.dispatcher.start())
        epoch = asyncio.create_task(self.runner.run_epoch(self.runner.eval_source.trigger(0)[0], 0))
        try:
            if until is None:
                await asyncio.wait_for(epoch, timeout=TIMEOUT_S)
            else:
                await asyncio.wait_for(until.wait(), timeout=TIMEOUT_S)
        finally:
            await self.runner.dispatcher.stop()
            epoch.cancel()
            await asyncio.gather(epoch, dispatcher, return_exceptions=True)
            await self.monitor.finalize()

    def landed(self) -> list[dict]:
        stream = self.run_dir / "monitors" / "file" / "traces" / "stream"
        return list(resume.read_records(stream))


def _rows(run_dir: Path) -> list[dict]:
    return [orjson.loads(line) for line in get_labels_index_path(run_dir).read_bytes().splitlines()]


def test_an_env_retry_reuses_the_call_that_finished_before_the_crash(tmp_path, monkeypatch, scripted) -> None:
    ReviewEnv.crash["job-1"] = 1
    requests: list[dict] = []

    async def main() -> Run:
        server, url = await _endpoint(requests)
        try:
            run = Run(tmp_path, url, monkeypatch, retries={"max_retries": 1})
            await run.epoch()
            return run
        finally:
            await server.cleanup()

    run = asyncio.run(main())

    # the draft ran once: the retry found its trace and did not call the model again
    assert [r["messages"][-1]["content"] for r in requests] == ["job-1", "review done: job-1"]
    first, again = ReviewEnv.drafts
    assert again.kept and again.id == first.id and not first.kept
    (episode,) = run.landed()
    assert episode["ok"], episode.get("errors")
    assert [t["id"] for t in episode["traces"]][0] == first.id
    assert [t["info"]["labels"]["call"]["key"] for t in episode["traces"]] == ["solver/0", "solver/1"]
    # the abandoned attempt's trace was kept as a partial, then landed with the retry
    assert (get_partial_dir(tmp_path) / f"{first.id}.jsonl").is_file()
    states = [(row["trace"], row["state"]) for row in _rows(tmp_path)]
    assert (first.id, "partial") in states and states[-2:][0] == (first.id, "finished")
    found = RunTraceLookup(tmp_path).latest({"flow": {"job": "job-1"}, "call": {"key": "solver/0"}})
    assert found.id == first.id and found.is_completed


def test_a_cancelled_episodes_call_is_reused_after_a_resume(tmp_path, monkeypatch, scripted) -> None:
    requests: list[dict] = []

    async def main() -> Run:
        server, url = await _endpoint(requests)
        try:
            ReviewEnv.hang.add("job-1")
            ReviewEnv.drafted = asyncio.Event()
            await Run(tmp_path, url, monkeypatch).epoch(until=ReviewEnv.drafted)
            # cancelled with the draft done and the review not started: nothing landed
            (draft,) = ReviewEnv.drafts
            partial = get_partial_dir(tmp_path) / f"{draft.id}.jsonl"
            assert orjson.loads(partial.read_bytes().splitlines()[-1])["end"] == "cancelled"
            assert _rows(tmp_path)[-1]["state"] == "partial"

            assert resume.take_landed(tmp_path) == []
            ReviewEnv.hang.clear()
            run = Run(tmp_path, url, monkeypatch)
            await run.epoch()
            return run
        finally:
            await server.cleanup()

    run = asyncio.run(main())

    assert [r["messages"][-1]["content"] for r in requests] == ["job-1", "review done: job-1"]
    first, again = ReviewEnv.drafts
    assert again.kept and again.id == first.id
    (episode,) = run.landed()
    assert episode["ok"] and episode["traces"][0]["id"] == first.id
