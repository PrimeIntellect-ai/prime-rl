from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import orjson
import pytest
from prime_traces import TransportError

from prime_rl.dashboard import platform, server

EVALUATION = {
    "evaluation_id": "ev1",
    "name": "hosted simpleqa",
    "status": "RUNNING",
    "run_id": "r1",
    "environment_names": ["simpleqa"],
    "model_name": "openai/gpt-4.1-mini",
    "eval_config": {"num_examples": 2, "rollouts_per_example": 2, "sampling_args": {"temperature": 0.3}},
    "viewer_url": "https://app.example/dashboard/evaluations/ev1",
    "created_at": "2026-09-24T23:28:01.038000Z",
    "completed_at": None,
    "updated_at": "2026-09-24T23:31:51.601000Z",
}
START = datetime(2026, 9, 24, 23, 29, tzinfo=timezone.utc)


class FakeTraces:
    """Prime Traces as the sync reads it: envelopes hold member ids, traces stand alone."""

    def __init__(self) -> None:
        self.envelopes: dict[str, dict] = {}
        self.traces: dict[str, dict] = {}
        self.created: dict[str, datetime] = {}
        self.broken: set[str] = set()  # trace ids whose reads drop mid-stream

    def add(self, episode_id: str, reward: float, seconds: int, *, group: str | None = "g1") -> None:
        trace = {
            "id": f"{episode_id}-t",
            "nodes": [{"message": {"role": "user", "content": "q"}}, {"parent": 0, "message": {"role": "assistant"}}],
            "rewards": {"correct": {"score": reward, "weight": 1}},
            "run": {"type": "eval", "id": "ev1", "work": {"type": "eval", "step": 0}},
        }
        self.traces[trace["id"]] = trace
        self.envelopes[episode_id] = {
            "id": episode_id,
            "env": {"id": "cooper/simpleqa@0.2.1", "name": "simpleqa"},
            "task": {"key": "task-a"},
            "run": trace["run"],
            "ok": True,
            "errors": [],
            "traces": [trace["id"]],
            "num_input_tokens": 10,
            "num_output_tokens": 5,
        }
        if group:
            self.envelopes[episode_id]["group"] = {"id": group}
        self.created[episode_id] = START + timedelta(seconds=seconds)

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
        if trace_id in self.broken:
            raise TransportError("peer closed connection")
        return orjson.dumps(self.traces[trace_id])


def test_sync_passes_write_a_run_the_dashboard_reads(tmp_path):
    traces = FakeTraces()
    evaluation = dict(EVALUATION)
    sync = platform.PlatformSync(root=tmp_path)
    sync._platform = SimpleNamespace(traces=traces, evaluation=lambda evaluation_id: dict(evaluation))
    job = platform.SyncJob("ev1")
    run_dir = sync.run_dir("ev1")

    # listed newest first by the service, written oldest first; one trace keeps dropping
    traces.add("ep-b", 0.0, 1)
    traces.add("ep-a", 1.0, 0)
    traces.broken.add("ep-b-t")
    newest = sync.sync_pass(job, None)
    assert (job.state, job.written, list(job.failed)) == ("syncing", 1, ["ep-b"])
    assert newest == START
    assert not server.run_meta(run_dir)["finished"]  # a running evaluation keeps its live chunk

    # the evaluation finishes; the dropped episode is fetched again, and an older
    # producer's episode (no group) is grouped by its task
    traces.broken.clear()
    traces.add("ep-c", 1.0, 2, group=None)
    evaluation.update(status="COMPLETED", completed_at="2026-09-24T23:31:51.601000Z")
    sync.sync_pass(job, newest)
    assert (job.written, job.failed) == (3, {})
    sync.sync_pass(job, None)
    assert job.written == 3  # a later pass fetches nothing it already wrote

    meta = server.run_meta(run_dir)
    assert meta["type"] == "eval"
    assert meta["finished"]
    assert meta["model"] == "openai/gpt-4.1-mini"
    assert meta["eval_totals"] == {"simpleqa": 4}
    assert meta["eval_plan"] == {"simpleqa": {"0": 4}}
    assert meta["platform"]["source"] == "traces"
    assert meta["platform"]["evaluations"]["simpleqa"]["url"] == EVALUATION["viewer_url"]
    assert meta["updated"] - meta["started"] == pytest.approx(230.563)  # created -> completed, not the sync's own time

    rows = server.episode_rows(run_dir)
    assert [(row["line"], row["id"], row["reward"], row["group"]) for row in rows] == [
        (1, "ep-a", 1.0, "g1"),
        (2, "ep-b", 0.0, "g1"),
        (3, "ep-c", 1.0, "task-a"),
    ]
    record = server.read_episode_at(platform.get_trace_stream(run_dir), 1, (rows[0]["chunk"], rows[0]["offset"]))
    assert record["traces"][0]["id"] == "ep-a-t"


def test_eval_config_without_a_fixed_size():
    evaluation = {**EVALUATION, "eval_config": {"num_examples": -1, "rollouts_per_example": 4}}
    assert platform.expected_episodes(evaluation) is None
    assert platform.eval_config(evaluation, "simpleqa")["source"] == [
        {"name": "simpleqa", "env": {"taskset": {"id": "simpleqa"}}, "group_size": 4}
    ]
