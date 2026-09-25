from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import orjson
import pytest

import prime_rl.dashboard.platform as platform
from prime_rl.dashboard import server

EVALUATION = {
    "evaluation_id": "ev1",
    "name": "hosted simpleqa",
    "status": "RUNNING",
    "run_id": "r1",
    "environment_names": ["simpleqa"],
    "inference_model": "openai/gpt-4.1-mini",
    "eval_config": {"num_examples": 2, "rollouts_per_example": 2, "sampling_args": {"temperature": 0.3}},
    "viewer_url": "https://app.example/dashboard/evaluations/ev1",
    "created_at": "2026-09-24T23:28:01.038000Z",
    "completed_at": None,
    "updated_at": "2026-09-24T23:31:51.601000Z",
}


def _trace(trace_id: str, reward: float) -> dict:
    return {
        "id": trace_id,
        "nodes": [{"message": {"role": "user", "content": "q"}}, {"parent": 0, "message": {"role": "assistant"}}],
        "rewards": {"correct": {"score": reward, "weight": 1}},
        "run": {"type": "eval", "id": "ev1", "work": {"type": "eval", "step": 0}},
    }


class FakeTraces:
    """Prime Traces as the sync reads it: envelopes hold member ids, traces stand alone."""

    def __init__(self) -> None:
        self.envelopes: dict[str, dict] = {}
        self.traces: dict[str, dict] = {}
        self.created: dict[str, datetime] = {}

    def add(self, episode_id: str, reward: float, created: datetime) -> None:
        trace = _trace(f"{episode_id}-t", reward)
        self.traces[trace["id"]] = trace
        self.envelopes[episode_id] = {
            "id": episode_id,
            "env": {"id": "cooper/simpleqa@0.2.1", "name": "simpleqa"},
            "group": {"id": "g1"},
            "run": trace["run"],
            "ok": True,
            "errors": [],
            "traces": [trace["id"]],
            "num_input_tokens": 10,
            "num_output_tokens": 5,
        }
        self.created[episode_id] = created

    def list_episodes(self, *, run_id, created_after=None, cursor=None, limit=None):
        assert run_id == "ev1"
        after = datetime.fromisoformat(created_after) if created_after else None
        items = [
            SimpleNamespace(episode_id=episode_id, created_at=created)
            for episode_id, created in sorted(self.created.items(), key=lambda item: item[1], reverse=True)
            if after is None or created >= after
        ]
        return SimpleNamespace(items=items, next_cursor=None)

    def get_episode_raw(self, episode_id: str) -> bytes:
        return orjson.dumps(self.envelopes[episode_id])

    def get_raw(self, trace_id: str) -> bytes:
        return orjson.dumps(self.traces[trace_id])


@pytest.fixture
def evaluation(monkeypatch) -> dict:
    current = dict(EVALUATION)
    monkeypatch.setattr(platform, "get_evaluation", lambda config, evaluation_id: dict(current))
    return current


def test_sync_writes_a_run_the_dashboard_reads(tmp_path, evaluation):
    traces = FakeTraces()
    start = datetime(2026, 9, 24, 23, 29, tzinfo=timezone.utc)
    # listed newest first by the service, written oldest first
    traces.add("ep-b", 0.0, start + timedelta(seconds=1))
    traces.add("ep-a", 1.0, start)
    run_dir = tmp_path / "ev1"

    _, written, newest = platform.sync(traces, None, "ev1", run_dir, None)
    assert (written, newest) == (2, start + timedelta(seconds=1))
    assert not server.run_meta(run_dir)["finished"]  # a running evaluation keeps its live chunk

    traces.add("ep-c", 1.0, start + timedelta(seconds=2))
    evaluation.update(status="COMPLETED", completed_at="2026-09-24T23:31:51.601000Z")
    _, written, _ = platform.sync(traces, None, "ev1", run_dir, newest - platform.FOLLOW_MARGIN)
    assert written == 1
    _, written, _ = platform.sync(traces, None, "ev1", run_dir, None)
    assert written == 0  # a rerun fetches nothing it already wrote

    meta = server.run_meta(run_dir)
    assert meta["type"] == "eval"
    assert meta["finished"]
    assert meta["model"] == "openai/gpt-4.1-mini"
    assert meta["eval_totals"] == {"simpleqa": 4}
    assert meta["eval_plan"] == {"simpleqa": {"0": 4}}
    assert meta["platform"]["evaluations"]["simpleqa"]["url"] == EVALUATION["viewer_url"]
    assert meta["updated"] - meta["started"] == pytest.approx(230.563)

    rows = server.episode_rows(run_dir)
    assert [(row["line"], row["id"], row["reward"]) for row in rows] == [
        (1, "ep-a", 1.0),
        (2, "ep-b", 0.0),
        (3, "ep-c", 1.0),
    ]
    assert {row["env"] for row in rows} == {"simpleqa"}
    assert rows[0]["input_tokens"] == 10
    stream = platform.get_trace_stream(run_dir)
    record = server.read_episode_at(stream, 1, (rows[0]["chunk"], rows[0]["offset"]))
    assert record["traces"][0]["id"] == "ep-a-t"


def test_eval_config_without_a_fixed_size():
    evaluation = {**EVALUATION, "eval_config": {"num_examples": -1, "rollouts_per_example": 4}}
    assert platform.expected_episodes(evaluation) is None
    assert platform.eval_config(evaluation, "simpleqa")["source"] == [
        {"name": "simpleqa", "env": {"taskset": {"id": "simpleqa"}}, "group_size": 4}
    ]
