"""The index row's ``labels`` (the env's ``trace_labels``, in each trace's
``info["labels"]``) and ``models`` (per-model
calls, tokens, cost and errors), so a reader attributes and prices an episode without
opening the stream."""

import verifiers.v1 as vf

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
