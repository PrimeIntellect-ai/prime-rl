"""The index row's ``labels`` (the env's ``trace_labels``, in each trace's
``info["labels"]``) and ``models`` (per-model
calls, tokens, cost and errors), so a reader attributes and prices an episode without
opening the stream."""

import asyncio

import orjson
import verifiers.v1 as vf
from verifiers.v1.durable import call_ref

from prime_rl.configs.monitors import FileMonitorConfig
from prime_rl.dashboard import server
from prime_rl.monitors.file import FileMonitor
from prime_rl.monitors.file.traces import get_index_path, get_trace_stream
from prime_rl.monitors.file.traces.index import index_row, summarize_episode


def _call(model: str | None, prompt: int, completion: int, *, cached: int | None = None, cost=None, error=False):
    usage = vf.Usage(prompt_tokens=prompt, completion_tokens=completion, cached_input_tokens=cached, cost=cost)
    return vf.ModelCall(
        model=model,
        usage=usage,
        error=vf.Error(type="RateLimitError", message="429 Too Many Requests") if error else None,
    )


def _record(labels: dict | None = None) -> dict:
    info = {"labels": labels} if labels is not None else {}
    task = vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="hi"))
    solver = vf.Trace(
        task=task,
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        calls=[_call("big", 100, 10, cached=50, cost=0.5), _call("big", 120, 20, error=True)],
        info={**info, "harness_blob": "x" * 1000},
    )
    judge = vf.Trace(
        task=task,
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        calls=[_call("small", 30, 3), _call(None, 1, 1)],
        info=dict(info),
    )
    return vf.Episode(task=task, traces=[solver, judge], ok=True).to_record()


def test_row_carries_the_trace_labels_not_the_rest_of_info() -> None:
    row = index_row(1, _record({"flow": {"job": "j1", "stage": "solve", "attempt": 2}}), chunk=0, offset=0)
    assert row["labels"] == {"flow": {"job": "j1", "stage": "solve", "attempt": 2}}
    assert "harness_blob" not in str(row)


def test_row_sums_usage_per_requested_model() -> None:
    row = summarize_episode(1, _record())
    assert row["cost"] == 0.5  # the episode's cost is its models' costs summed
    models = row["models"]
    assert models == {
        "big": {
            "calls": 2,
            "errors": 1,
            "prompt_tokens": 220,
            "cached_input_tokens": 50,
            "completion_tokens": 30,
            "cost": 0.5,
        },
        "small": {
            "calls": 1,
            "errors": 0,
            "prompt_tokens": 30,
            "cached_input_tokens": 0,
            "completion_tokens": 3,
            "cost": None,
        },
        "?": {
            "calls": 1,
            "errors": 0,
            "prompt_tokens": 1,
            "cached_input_tokens": 0,
            "completion_tokens": 1,
            "cost": None,
        },
    }


def test_a_record_without_labels_reads_empty() -> None:
    record = _record()
    for trace in record["traces"]:
        trace.pop("calls", None)
    row = summarize_episode(1, record)
    assert row["labels"] == {}
    assert row["models"] == {}
    assert row["cost"] is None


def test_judge_calls_count_under_the_judges_model() -> None:
    record = _record()
    record["traces"][0]["extra_usage"] = [
        {"prompt_tokens": 40, "completion_tokens": 4, "cached_input_tokens": 0, "cost": 0.25}
    ]
    row = summarize_episode(1, record)
    assert row["models"]["judges"] == {
        "calls": 1,
        "errors": 0,
        "prompt_tokens": 40,
        "cached_input_tokens": 0,
        "completion_tokens": 4,
        "cost": 0.25,
    }
    assert row["cost"] == 0.75


# -- a trace that lands twice (a durable call reused it) ------------------------------


JOB = {"flow": {"job": "j1"}}


def _labelled(key: str, *calls: vf.ModelCall) -> vf.Trace:
    return vf.Trace(
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="hi")),
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        calls=list(calls),
        info={"labels": {**JOB, "call": {"key": key, "fingerprint": "f", "ref": call_ref(JOB, key)}}},
    )


def _land(monitor: FileMonitor, *traces: vf.Trace, ok: bool = True) -> None:
    episode = vf.Episode(task=traces[0].task, traces=list(traces), ok=ok)
    asyncio.run(monitor.log_episodes([episode], step=0, kind="eval", subset="all"))


def _index(tmp_path) -> list[dict]:
    index = get_index_path(get_trace_stream(tmp_path))
    return [orjson.loads(line) for line in index.read_bytes().splitlines()]


def _spend(rows: list[dict]) -> dict[str, int | float]:
    """What a rollup over the rows adds up."""
    models = [usage for row in rows for usage in row["models"].values()]
    return {
        "calls": sum(usage["calls"] for usage in models),
        "completion_tokens": sum(usage["completion_tokens"] for usage in models),
        "cost": sum(row["cost"] or 0.0 for row in rows),
    }


def test_a_retried_episode_counts_a_kept_trace_once(tmp_path) -> None:
    draft = _labelled("solver/0", _call("big", 100, 10, cost=0.5))
    review = _labelled("solver/1", _call("big", 50, 5, cost=0.25))
    monitor = FileMonitor(FileMonitorConfig(compress=False))
    asyncio.run(monitor.init(tmp_path))
    # the first attempt failed after the draft and landed with it; the retry reused the
    # draft (the same trace) and ran the review
    _land(monitor, draft, ok=False)
    _land(monitor, draft, review)
    failed, retried = _index(tmp_path)
    assert failed["reused"] == [] and failed["cost"] == 0.5
    assert retried["reused"] == [draft.id]
    assert retried["models"]["big"]["calls"] == 1 and retried["cost"] == 0.25
    assert retried["output_tokens"] == review.num_output_tokens
    assert _spend([failed, retried]) == {"calls": 2, "completion_tokens": 15, "cost": 0.75}
    # a relaunch appending to the same stream still knows the draft landed
    asyncio.run(monitor.finalize())
    relaunched = FileMonitor(FileMonitorConfig(compress=False))
    asyncio.run(relaunched.init(tmp_path))
    _land(relaunched, draft, review)
    assert _index(tmp_path)[-1]["reused"] == [draft.id, review.id]
    assert _index(tmp_path)[-1]["models"] == {} and _index(tmp_path)[-1]["cost"] is None


def test_a_reader_summarising_the_stream_counts_a_kept_trace_once(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(server, "SIDECAR_DIR", tmp_path / "sidecars")
    draft = _labelled("solver/0", _call("big", 100, 10, cost=0.5))
    review = _labelled("solver/1", _call("big", 50, 5, cost=0.25))
    monitor = FileMonitor(FileMonitorConfig(compress=False))
    asyncio.run(monitor.init(tmp_path))
    _land(monitor, draft, ok=False)
    _land(monitor, draft, review)
    asyncio.run(monitor.finalize())
    summaries = server.chunked_summaries(get_trace_stream(tmp_path))
    assert [summary["reused"] for summary in summaries] == [[], [draft.id]]
    assert _spend(summaries) == {"calls": 2, "completion_tokens": 15, "cost": 0.75}
    # summarised on its own, the retried episode counts every trace it holds
    alone = summarize_episode(2, vf.Episode(task=draft.task, traces=[draft, review], ok=True).to_record())
    assert alone["reused"] == [] and alone["models"]["big"]["calls"] == 2


def test_row_labels_leave_out_a_durable_calls_own_labels() -> None:
    job = {"flow": {"job": "j1", "stage": "solve"}}
    call = {"key": "agent/0", "fingerprint": "f", "ref": call_ref(job, "agent/0")}
    record = _record({**job, "call": call, "box": {"ref": "b"}})
    assert summarize_episode(1, record)["labels"] == job

    # a trace labelled only by its durable call does not speak for the episode
    record["traces"][0]["info"]["labels"] = {"call": call}
    assert summarize_episode(1, record)["labels"] == job
