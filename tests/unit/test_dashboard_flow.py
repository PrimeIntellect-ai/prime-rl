import json

from prime_rl.dashboard import server
from prime_rl.dashboard.flow import project_flow
from prime_rl.dashboard.server import main_config, run_meta, written_index


def write_jsonl(path, rows, tail=""):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows) + tail)


def flow_root(tmp_path, held=False):
    """A campaign whose task ran author then control, with a solve call and a live seat."""
    (tmp_path / "flow.json").write_text('{"model": "m"}')
    for unit, state in (
        ("campaign", {"stage": "plan", "status": "waiting", "reason": "planned 1 tasks"}),
        ("t1", {"stage": "control", "status": "held" if held else "terminal", "reason": "done"}),
    ):
        path = tmp_path / ("campaign" if unit == "campaign" else f"tasks/{unit}")
        path.mkdir(parents=True)
        (path / "state.json").write_text(json.dumps(state))
    at = "2026-01-01T00:00:0{}+00:00"
    events = [
        {"type": "started", "at": at.format(0), "unit": "campaign", "stage": "plan"},
        {
            "type": "transition",
            "at": at.format(1),
            "unit": "campaign",
            "stage": "plan",
            "outcome": "waiting",
            "to": "plan",
            "status": "waiting",
            "reason": "planned 1 tasks",
            "sha": "a",
        },
        {"type": "started", "at": at.format(2), "unit": "t1", "stage": "author"},
        {
            "type": "transition",
            "at": at.format(3),
            "unit": "t1",
            "stage": "author",
            "outcome": "authored",
            "to": "control",
            "status": "ready",
            "reason": "wrote it",
            "sha": "b",
        },
        {"type": "started", "at": at.format(4), "unit": "t1", "stage": "control"},
        {
            "type": "transition",
            "at": at.format(6),
            "unit": "t1",
            "stage": "control",
            "outcome": "held" if held else "proceed",
            "to": "control",
            "status": "held" if held else "terminal",
            "reason": "why",
            "sha": "c",
        },
    ]
    write_jsonl(
        tmp_path / "transitions.jsonl",
        events,
        tail='{"type": "started", "at": "2026-01-01T00:00:07+00:00", "unit": "t1"',
    )
    (tmp_path / "calls" / "t1").mkdir(parents=True)
    (tmp_path / "calls" / "t1" / "x.json").write_text(
        json.dumps(
            {
                "key": "control/abcd/v1",
                "unit": "t1",
                "stage": "control",
                "kind": "agent",
                "trace_id": "trace",
                "started_at": at.format(4),
                "finished_at": at.format(5),
            }
        )
    )
    (tmp_path / "live").mkdir()
    write_jsonl(
        tmp_path / "traces.jsonl",
        [
            {
                "id": "episode",
                "env": {"id": "flow"},
                "task": {},
                "ok": True,
                "traces": [{"id": "trace", "info": {"decision": {"outcome": "proceed", "summary": "scored 1.0"}}}],
            }
        ],
    )
    return tmp_path


def test_flow_projection_builds_lanes_routes_calls_and_ignores_a_torn_tail(tmp_path):
    root = flow_root(tmp_path)
    data = project_flow(root, {"trace": (0, "episode")})
    stages = [n for n in data["nodes"] if n["index"] is None]
    assert [(n["unit"], n["name"], n["status"]) for n in stages] == [
        ("campaign", "plan", "completed"),
        ("t1", "author", "completed"),
        ("t1", "control", "completed"),
    ]
    routes = [e for e in data["edges"] if e["kind"] == "route"]
    assert [(e["outcome"], e["to"], e["target"] is not None) for e in routes] == [
        ("waiting", "plan", False),
        ("authored", "control", True),
        ("proceed", "control", False),
    ]
    assert routes[1]["summary"] == "wrote it"
    (call,) = [n for n in data["nodes"] if n["index"] is not None]
    assert call["trace_id"] == "trace" and call["episode_line"] == 0 and call["name"] == "control/abcd/v1"
    control = stages[-1]
    assert control["trace_id"] == "trace"  # the stage opens its decision trace
    assert data["tasks"] == [
        {
            "id": f"{root.name}/t1",
            "name": "t1",
            "row": root.name,
            "stage": "control",
            "status": "terminal",
            "reason": "done",
            "nodes": 2,
            "traces": 1,
        }
    ]
    assert data["groups"][0]["kind"] == "run" and data["stats"]["routes"] == 3 and data["stats"]["traces"] == 1
    assert data["rows"][0]["status"] == "completed"


def test_a_held_unit_shows_as_failed(tmp_path):
    data = project_flow(flow_root(tmp_path, held=True))
    control = [n for n in data["nodes"] if n["name"] == "control" and n["index"] is None][0]
    assert control["status"] == "failed" and control["error"] == "why"
    assert data["stats"]["held"] == 1 and data["rows"][0]["status"] == "running"


def test_flow_run_detection_and_foreign_trace_index_falls_back(tmp_path):
    (tmp_path / "flow.json").write_text('{"data": "/tmp/tasks"}')
    (tmp_path / "transitions.jsonl").write_text("")
    (tmp_path / "traces.jsonl").write_text("{}\n")
    write_jsonl(tmp_path / "traces.index.jsonl", [{"id": "trace", "offset": 0, "length": 3}])

    assert main_config(tmp_path) == ("flow", {"data": "/tmp/tasks"})
    assert run_meta(tmp_path)["dataset"] is None
    assert written_index(tmp_path) is None


def test_flow_endpoints_etag_detail_and_run_state(tmp_path, monkeypatch):
    root = flow_root(tmp_path)
    monkeypatch.setattr(server, "get_run_dir", lambda run: root)
    first = server.get_flow("run")
    assert first["stats"]["tasks"] == 1 and first["stats"]["routes"] == 3
    assert server.get_flow("run", etag=first["etag"]) == {"etag": first["etag"], "unchanged": True}
    detail = server.get_flow_trace("run", "trace")
    assert detail["decision"] == {"outcome": "proceed", "summary": "scored 1.0"} and detail["episode_id"] == "episode"
    meta = run_meta(root)
    assert meta["type"] == "flow" and meta["finished"] is True
