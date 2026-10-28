"""Each monitor under an orchestrator-sized load: a batch of multi-MB episodes and
thousands of metric rows, logged through ``monitors.log`` as the orchestrator does.
Everything logged must arrive, and the event loop must stay responsive while the
monitor writes or uploads."""

import asyncio
import gzip
import statistics
import time

import numpy as np
import orjson
import pytest
import verifiers.v1 as vf
import wandb
from prime_runs.sinks import RftMetricsSink, TracesSink
from prime_runs.worker import UploadWorker
from prime_traces import TracesClient
from verifiers.v1.graph import MessageNode
from verifiers.v1.trace import ModelCall
from verifiers.v1.types import AssistantMessage, Usage, UserMessage

from prime_rl import monitors
from prime_rl.configs.monitors import FileMonitorConfig, PrimeMonitorConfig, WandbMonitorConfig
from prime_rl.monitors.file.monitor import FileMonitor
from prime_rl.monitors.file.traces import get_trace_stream
from prime_rl.monitors.file.traces.chunks import open_chunk
from prime_rl.monitors.prime import PrimeMonitor
from prime_rl.monitors.wandb import WandbMonitor
from prime_rl.utils.pathing import get_file_monitor_dir

EPISODES = 96
TURNS = 4
TOKENS_PER_MESSAGE = 8_000
METRIC_ROWS = 2_000
METRIC_KEYS = 200
# The orchestrator's own work runs between monitor calls; a stall this long delays
# dispatch and batch assembly visibly.
MAX_LAG_S = 0.25
P99_LAG_S = 0.05
TICK_S = 0.005

pytestmark = pytest.mark.slow


def _trace(rng: np.random.Generator) -> vf.Trace:
    """A multi-turn trace with per-token streams as the env-server wire delivers them."""
    n = TOKENS_PER_MESSAGE
    nodes = []
    for _ in range(TURNS):
        nodes.append(
            MessageNode(
                parent=len(nodes) - 1 if nodes else None,
                message=UserMessage(content="x" * n),
                token_ids=rng.integers(0, 150_000, n, dtype=np.int32),
                mask=np.zeros(n, bool),
            )
        )
        nodes.append(
            MessageNode(
                parent=len(nodes) - 1,
                message=AssistantMessage(content="y" * n),
                sampled=True,
                token_ids=rng.integers(0, 150_000, n, dtype=np.int32),
                mask=np.ones(n, bool),
                is_content=np.ones(n, bool),
                logprobs=-rng.random(n, dtype=np.float32),
                advantages=np.full(n, 0.5, np.float32),
            )
        )
    return vf.Trace(
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt=None)),
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        nodes=nodes,
        ok=True,
        is_completed=True,
        rewards={"correct": vf.Reward(score=1.0)},
        calls=[
            ModelCall(node=2 * i + 1, usage=Usage(prompt_tokens=n, completion_tokens=n), finish_reason="stop")
            for i in range(TURNS)
        ],
    )


@pytest.fixture(scope="module")
def episodes() -> list[vf.Episode]:
    rng = np.random.default_rng(0)
    return [
        vf.Episode(
            env=vf.EnvInfo(id="load", name="load"),
            task=(trace := _trace(rng)).task,
            group=vf.GroupInfo(id=f"g{i // 8}"),
            traces=[trace],
            ok=True,
            run=vf.TrainRunInfo(id="run", work=vf.TrainWorkInfo(step=1, policy=vf.PolicySpan(start=0, end=0))),
        )
        for i in range(EPISODES)
    ]


def _metrics(step: int) -> dict[str, float]:
    return {f"group{key % 10}/metric{key}": step + key / METRIC_KEYS for key in range(METRIC_KEYS)}


async def _drive(episodes: list[vf.Episode]) -> tuple[list[float], list[float]]:
    """Log every episode on arrival and the metric rows between them, as the orchestrator
    does, while a ticker measures how late the event loop wakes it. Returns the event-loop
    lags and the duration of each ``monitors.log`` call."""
    lags: list[float] = []
    stop = asyncio.Event()

    async def ticker() -> None:
        while not stop.is_set():
            start = time.perf_counter()
            await asyncio.sleep(TICK_S)
            lags.append(time.perf_counter() - start - TICK_S)

    tick = asyncio.create_task(ticker())
    calls: list[float] = []

    async def timed(*args, **kwargs) -> None:
        start = time.perf_counter()
        await monitors.log(*args, **kwargs)
        calls.append(time.perf_counter() - start)
        # The orchestrator awaits its next arrival between log calls.
        await asyncio.sleep(0)

    await asyncio.sleep(0)
    every = METRIC_ROWS // len(episodes)
    for step in range(1, METRIC_ROWS + 1):
        if step % every == 0 and step // every <= len(episodes):
            await timed([episodes[step // every - 1]], 1, "train", "all")
        await timed(_metrics(step), step=step)
    stop.set()
    await tick
    return lags, calls


def _assert_responsive(name: str, lags: list[float], calls: list[float]) -> None:
    p99 = statistics.quantiles(lags, n=100, method="inclusive")[98]
    print(
        f"{name}: event-loop lag p50 {statistics.median(lags) * 1e3:.1f} ms, p99 {p99 * 1e3:.1f} ms, "
        f"max {max(lags) * 1e3:.1f} ms; log() p99 {statistics.quantiles(calls, n=100, method='inclusive')[98] * 1e3:.1f} ms, "
        f"max {max(calls) * 1e3:.1f} ms"
    )
    assert p99 < P99_LAG_S, f"{name} stalls the event loop: p99 lag {p99 * 1e3:.0f} ms"
    assert max(lags) < MAX_LAG_S, f"{name} stalls the event loop: max lag {max(lags) * 1e3:.0f} ms"
    assert max(calls) < MAX_LAG_S, f"a {name} log() call took {max(calls) * 1e3:.0f} ms"


@pytest.fixture
def registered():
    yield monitors.MONITORS
    monitors.MONITORS.clear()


def test_file_monitor_under_load(tmp_path, episodes, registered):
    async def run():
        monitor = FileMonitor(FileMonitorConfig())
        await monitor.init(tmp_path)
        registered.append(monitor)
        result = await _drive(episodes)
        await monitors.finalize()
        return result

    lags, calls = asyncio.run(run())

    with open_chunk(get_trace_stream(tmp_path), 0) as chunk:
        records = [orjson.loads(line) for line in chunk.read().splitlines()]
    assert len(records) == EPISODES
    assert len(records[0]["traces"][0]["nodes"]) == 2 * TURNS
    rows = (get_file_monitor_dir(tmp_path) / "metrics.jsonl").read_bytes().splitlines()
    assert len(rows) == METRIC_ROWS
    _assert_responsive("file monitor", lags, calls)


class FakeTracesAPI:
    """The Prime Traces endpoint: gzips each request like the real client and keeps
    the lines it received."""

    def __init__(self) -> None:
        self.lines = 0

    def upload_batch(self, body, idempotency_key, *, line_format, schema_version=1, context=None, compress=True):
        if compress:
            gzip.compress(body, mtime=0)
        self.lines += body.count(b"\n") + (not body.endswith(b"\n"))
        return {"upload_id": idempotency_key.split(":", 1)[1], "status": "committed"}

    def close(self) -> None:
        pass


class FakePlatform:
    """The platform API: keeps every metrics row the run posted."""

    def __init__(self) -> None:
        self.steps: list[int] = []

    def post(self, path, json_body=None, idempotent=False):
        assert path == "/rft/metrics"
        self.steps.append(json_body["metrics"]["step"])
        return {}


def test_prime_monitor_under_load(tmp_path, episodes, registered, monkeypatch):
    """The real SDK run and uploader threads, with fake endpoints behind them."""
    monkeypatch.setenv("PRIME_RUNS_MODE", "disabled")
    for var in ("RUN_ID", "PRL_RUN_FINISHER"):
        monkeypatch.delenv(var, raising=False)
    traces_api, platform = FakeTracesAPI(), FakePlatform()

    async def run():
        monitor = PrimeMonitor(PrimeMonitorConfig(name="load"))
        await monitor.init(config=None, output_dir=tmp_path)
        traces = TracesSink(client=TracesClient(api_client=traces_api))
        metrics = RftMetricsSink(platform)
        for sink in (traces, metrics):
            sink.start(monitor.run.id, {"run_type": "train"})
        monitor.run._worker = UploadWorker([traces])
        monitor.run._metrics_worker = UploadWorker([metrics])
        registered.append(monitor)
        result = await _drive(episodes)
        await monitors.finalize()
        return result, traces

    (lags, calls), traces = asyncio.run(run())

    assert traces.episodes_written == EPISODES
    assert traces_api.lines == EPISODES
    assert sorted(platform.steps) == list(range(1, METRIC_ROWS + 1))
    _assert_responsive("prime monitor", lags, calls)


def test_wandb_monitor_under_load(tmp_path, episodes, registered, monkeypatch):
    """W&B in offline mode: the real SDK writes the run locally, without a server."""
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_SILENT", "true")
    for var in ("WANDB_SHARED_MODE", "WANDB_RUN_ID"):
        monkeypatch.delenv(var, raising=False)
    steps: list[int] = []

    async def run():
        monitor = WandbMonitor(WandbMonitorConfig(project="load", offline=True))
        await monitor.init(output_dir=tmp_path)
        # wandb.init binds the module-level log to the new run.
        log = wandb.log

        def counting_log(data, *args, **kwargs):
            steps.append(data["step"])
            return log(data, *args, **kwargs)

        monkeypatch.setattr(wandb, "log", counting_log)
        registered.append(monitor)
        result = await _drive(episodes)
        await monitors.finalize()
        return result

    lags, calls = asyncio.run(run())

    assert steps == list(range(1, METRIC_ROWS + 1))
    _assert_responsive("W&B monitor", lags, calls)
