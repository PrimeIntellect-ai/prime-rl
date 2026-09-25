"""Published state and explicit execution events drive the read-only projection."""

import json

from verifiers.v1.flow import JobState

from prime_rl.dashboard.flow import flow_etag, project_flow


def publish(job, state):
    job.mkdir(parents=True, exist_ok=True)
    checkpoint = JobState(data_type="verifiers.v1.flow:JobData", stages=[state["stage"]], **state)
    (job / "state.json").write_text(checkpoint.model_dump_json())


def root(tmp_path):
    (tmp_path / "flow.json").write_text("{}")
    publish(tmp_path / "jobs/coordinator", {"stage": "plan", "status": "waiting", "data": {}})
    publish(tmp_path / "jobs/t", {"stage": "evaluate", "status": "held", "data": {}})
    events = [
        {"type": "run_started"},
        {"type": "steer", "revision": 1, "action": {"note": "initial guidance"}},
        {"type": "started", "execution": "first", "error": None},
        {"type": "steer", "revision": 2, "action": {"status": "held", "note": "pause after review"}},
        {"type": "call", "execution": "first", "call": "producer", "status": "started"},
        {"type": "call", "execution": "first", "call": "producer", "status": "succeeded", "trace_id": "trace"},
        {
            "type": "transition",
            "error": None,
            "execution": "first",
            "to": "evaluate",
            "outcome": None,
            "status": "held",
            "reason": "ready for operator",
            "report": "published.md",
            "links": [{"job": "coordinator", "label": "updated"}],
        },
        {"type": "steer", "revision": 4, "action": {"status": "ready", "note": "retry"}},
        {"type": "started", "execution": "second", "error": None},
        {
            "type": "call",
            "execution": "second",
            "call": "attachment",
            "status": "attached",
            "trace_id": "trace",
            "source_call": "producer",
            "source_execution": "first",
        },
        *[
            {"type": "call", "execution": "second", "call": "unkeyed", "key": None, "kind": "fn", "status": status}
            for status in ("started", "cancelled")
        ],
        {"type": "call", "execution": "second", "call": "lost", "status": "started"},
    ]
    # Identical timestamps deliberately cannot identify an execution.
    defaults = {"job": "t", "stage": "evaluate", "at": "2026-01-01T00:00:00+00:00", "key": "solve", "kind": "agent"}
    rows = []
    for event in events:
        row = {**defaults, **event}
        if row["type"] == "call":
            row["invocation"] = {k: row.pop(k) for k in ("job", "stage", "execution", "call", "key", "kind")}
            row["invocation"]["cache"] = None
        rows.append(json.dumps(row) + "\n")
    (tmp_path / "transitions.jsonl").write_text("".join(rows))
    calls = tmp_path / "calls/t"
    calls.mkdir(parents=True)
    (calls / "result.json").write_text(
        json.dumps({"key": "solve", "call": "producer", "execution": "first", "trace_id": "trace"})
    )
    return tmp_path


def test_projection_uses_ids_preserves_provenance_and_shows_incomplete_work(tmp_path):
    run = root(tmp_path)
    result = project_flow(run, {"trace": (0, "episode")})
    assert {u["name"] for u in result["jobs"]} == {"coordinator", "t"}
    assert result["status"] == "incomplete"
    first, second = result["nodes"]
    assert first["status"] == "completed" and first["job_status"] == "held"
    assert first["error"] is None
    assert first["outcome"] is None and second["status"] == "incomplete"
    assert first["links"] == [{"job": "coordinator", "label": "updated"}]
    (producer,) = first["calls"]
    assert producer["id"] == "producer" and producer["episode_line"] == 0
    attachment, unkeyed, lost = second["calls"]
    assert attachment["source_call"] == producer["id"] and attachment["source_execution"] == "first"
    assert unkeyed["status"] == "cancelled" and lost["status"] == "incomplete"
    job = next(u for u in result["jobs"] if u["name"] == "t")
    assert [s["action"]["note"] for s in job["steers"]] == ["initial guidance", "pause after review", "retry"]
    outcome, resume = result["edges"]
    assert outcome["target"] is None
    assert resume["kind"] == "resume" and resume["target"] == second["id"]
    assert '"note":"retry"' in resume["summary"]
    with (run / "transitions.jsonl").open("a") as file:
        for kind, execution in [("stopped", "second"), ("started", "third"), ("cancelled", "third")]:
            file.write(
                json.dumps({"type": kind, "execution": execution, "job": "t", "stage": "evaluate", "error": None})
                + "\n"
            )
        file.write(json.dumps({"type": "run_finished", "reason": "idle", "at": "2026-01-01T00:00:00+00:00"}) + "\n")
    result = project_flow(run)
    assert result["status"] == "idle"
    assert [n["status"] for n in result["nodes"]] == ["completed", "stopped", "cancelled"]


def test_fingerprint_tracks_workflow_and_calls(tmp_path):
    run = root(tmp_path)
    before = flow_etag(run)
    with (run / "transitions.jsonl").open("a") as file:
        file.write('{"type":"drain"}\n')
    changed = flow_etag(run)
    assert changed != before
    (run / "calls/t/new.json").write_text("{}")
    assert flow_etag(run) != changed
    changed = flow_etag(run)
    (run / "live").mkdir()
    (run / "live/trace.jsonl").write_text("{}\n")
    assert flow_etag(run) == changed
    publish(run / "jobs/t", {"stage": "evaluate", "status": "ready", "data": {}})
    assert flow_etag(run) != changed


def test_flow_reports_list_only_published_files(tmp_path, monkeypatch):
    from prime_rl.dashboard import server

    run = root(tmp_path)
    monkeypatch.setattr(server, "_run_registry", {"run": run})
    reports = run / "reports"
    reports.mkdir()
    (reports / "published.md").write_text("Recorded report")
    (reports / "orphan.md").write_text("Stage never published")
    assert [r["file"] for r in server.list_reports("run")["reports"]] == ["published.md"]
    with (run / "transitions.jsonl").open("a") as file:
        file.write(json.dumps({"type": "steer", "job": "t", "revision": 5, "action": {"report": "orphan.md"}}) + "\n")
    assert {r["file"] for r in server.list_reports("run")["reports"]} == {"published.md", "orphan.md"}
    (reports / "published.md").unlink()
    assert [r["file"] for r in server.list_reports("run")["reports"]] == ["orphan.md"]
    (run / "flow.json").unlink()
    assert [r["file"] for r in server.list_reports("run")["reports"]] == ["orphan.md"]
