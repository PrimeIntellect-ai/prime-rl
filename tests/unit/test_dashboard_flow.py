"""Real Git checkpoints and explicit execution events drive the read-only projection."""

import json
import subprocess

from prime_rl.dashboard.flow import flow_etag, project_flow, unit_states


def commit(unit, state):
    unit.mkdir(parents=True, exist_ok=True)
    (unit / "state.json").write_text(json.dumps(state))
    for args in (("init", "-q"), ("add", "state.json"), ("commit", "-qm", "state")):
        subprocess.run(
            ["git", "-C", str(unit), "-c", "user.name=test", "-c", "user.email=test@local", *args], check=True
        )


def root(tmp_path):
    (tmp_path / "flow.json").write_text("{}")
    commit(tmp_path / "units/coordinator", {"stage": "plan", "status": "waiting", "data": {}})
    commit(tmp_path / "units/t", {"stage": "evaluate", "status": "held", "data": {"tree": "artifact"}})
    events = [
        {"type": "run_started"},
        {"type": "started", "execution": "first"},
        {"type": "call", "execution": "first", "call": "producer", "status": "started"},
        *[
            {
                "type": "rollout",
                "execution": "first",
                "call": "producer",
                "rollout": n,
                "trace_id": trace,
                "status": status,
            }
            for n, trace, outcome in [(1, "failed", "failed"), (2, "trace", "succeeded")]
            for status in ("started", outcome)
        ],
        {"type": "call", "execution": "first", "call": "producer", "status": "succeeded", "trace_id": "trace"},
        {
            "type": "transition",
            "execution": "first",
            "to": "evaluate",
            "outcome": "held",
            "status": "held",
            "reason": "provider error",
        },
        {"type": "steer", "sha": "control", "action": {"status": "ready", "note": "retry"}},
        {"type": "started", "execution": "second"},
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
    # Identical timestamps deliberately cannot identify any stage or retry.
    defaults = {"unit": "t", "stage": "evaluate", "at": "2026-01-01T00:00:00", "key": "solve", "kind": "agent"}
    (tmp_path / "transitions.jsonl").write_text("".join(json.dumps({**defaults, **e}) + "\n" for e in events))
    calls = tmp_path / "calls/t"
    calls.mkdir(parents=True)
    (calls / "result.json").write_text(json.dumps({"call": "producer", "execution": "first", "trace_id": "trace"}))
    return tmp_path


def test_projection_uses_ids_preserves_provenance_and_shows_incomplete_work(tmp_path):
    run = root(tmp_path)
    (run / "units/t/state.json").write_text("incomplete operator edit")
    assert unit_states(run)["t"]["status"] == "held"
    result = project_flow(run, {"trace": (0, "episode"), "failed": (1, "failed-episode")})
    assert {u["name"] for u in result["units"]} == {"coordinator", "t"}
    assert result["status"] == "incomplete"
    first, second = result["nodes"]
    assert first["reason"] == "provider error" and second["status"] == "incomplete"
    (producer,) = first["calls"]
    assert producer["id"] == "producer" and producer["episode_line"] == 0
    assert [r["status"] for r in producer["rollouts"]] == ["failed", "succeeded"]
    attachment, unkeyed, lost = second["calls"]
    assert attachment["source_call"] == producer["id"] and attachment["source_execution"] == "first"
    assert unkeyed["status"] == "cancelled" and lost["status"] == "incomplete"
    assert any(edge["summary"] == "retry" for edge in result["edges"])
    with (run / "transitions.jsonl").open("a") as file:
        file.write(json.dumps({"type": "run_finished", "reason": "quiescent", "at": "2026-01-01T00:00:00"}) + "\n")
    assert project_flow(run)["status"] == "quiescent"


def test_fingerprint_tracks_workflow_calls_and_live_changes(tmp_path):
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
    (run / "live/t--call.json").write_text("{}")
    assert flow_etag(run) != changed
    changed = flow_etag(run)
    commit(run / "units/t", {"stage": "evaluate", "status": "ready", "data": {"tree": "artifact"}})
    assert flow_etag(run) != changed
