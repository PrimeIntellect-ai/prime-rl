import json

from prime_rl.dashboard import server
from prime_rl.dashboard.flow import project_flow
from prime_rl.dashboard.server import main_config, run_meta, written_index


def write_jsonl(path, rows, tail=""):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows) + tail)


def test_flow_projection_builds_tasks_spreads_routes_and_ignores_a_torn_tail(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "steps" / "row").mkdir(parents=True)
    events = [
        {"type": "row_started", "at": "2026-01-01T00:00:00+00:00", "row": "row"},
        {"type": "step_started", "at": "2026-01-01T00:00:01+00:00", "row": "row", "path": "task/init#0", "index": None},
        {
            "type": "step_completed",
            "at": "2026-01-01T00:00:02+00:00",
            "row": "row",
            "path": "task/init#0",
            "index": None,
        },
        {
            "type": "route",
            "at": "2026-01-01T00:00:02+00:00",
            "row": "row",
            "path": "task/init#0",
            "outcome": "created",
            "to": "solve",
        },
        {"type": "spread_started", "at": "2026-01-01T00:00:03+00:00", "row": "row", "path": "task/solve#0", "items": 2},
        {"type": "step_started", "at": "2026-01-01T00:00:03+00:00", "row": "row", "path": "task/solve#0", "index": 0},
        {"type": "step_completed", "at": "2026-01-01T00:00:04+00:00", "row": "row", "path": "task/solve#0", "index": 0},
        {
            "type": "spread_finished",
            "at": "2026-01-01T00:00:04+00:00",
            "row": "row",
            "path": "task/solve#0",
            "items": 2,
            "landed": 1,
        },
    ]
    write_jsonl(tmp_path / "events.jsonl", events, '{"type":"step_started"')
    for index, trace_id in [(None, None), (0, "trace-0")]:
        name = "task__init#0.json" if index is None else "task__solve#0.0.json"
        record = {
            "key": "key",
            "row": "row",
            "path": "task/init#0" if index is None else "task/solve#0",
            "index": index,
            "kind": "fn" if index is None else "agent",
            "terminal": "completed",
            "payload": None,
            "trace_id": trace_id,
            "error": None,
            "attempts": 1,
            "started_at": "2026-01-01T00:00:01+00:00",
            "finished_at": "2026-01-01T00:00:04+00:00",
        }
        (tmp_path / "steps" / "row" / name).write_text(json.dumps(record))

    data = project_flow(tmp_path, {"trace-0": (1, "episode-0")})
    assert data["tasks"] == [
        {
            "id": "row/task",
            "name": "task",
            "row": "row",
            "stage": "solve",
            "status": "completed",
            "nodes": 3,
            "traces": 1,
        }
    ]
    assert data["groups"] == [
        {"id": "row/", "name": "row", "row": "row", "kind": "run"},
        {"id": "row/task", "name": "task", "row": "row", "kind": "task"},
    ]
    assert {node["kind"] for node in data["nodes"]} == {"fn", "agent", "spread"}
    solver = next(node for node in data["nodes"] if node["trace_id"] == "trace-0")
    assert (solver["episode_line"], solver["episode_id"]) == (1, "episode-0")
    route = next(edge for edge in data["edges"] if edge["kind"] == "route")
    assert route["outcome"] == "created" and route["target"].endswith("task/solve#0")
    assert data["stats"] == {"rows": 1, "tasks": 1, "steps": 3, "running": 0, "failed": 1, "traces": 1, "routes": 1}


def test_flow_run_detection_and_foreign_trace_index_falls_back(tmp_path):
    (tmp_path / "config.json").write_text('{"data": "/tmp/tasks"}')
    (tmp_path / "events.jsonl").write_text("")
    (tmp_path / "steps").mkdir()
    (tmp_path / "traces.jsonl").write_text("{}\n")
    write_jsonl(tmp_path / "traces.index.jsonl", [{"id": "trace", "offset": 0, "length": 3}])

    assert main_config(tmp_path) == ("flow", {"data": "/tmp/tasks"})
    assert run_meta(tmp_path)["dataset"] is None
    assert written_index(tmp_path) is None


def test_flow_endpoints_etag_detail_and_run_state(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text("{}")
    steps = tmp_path / "steps" / "row"
    steps.mkdir(parents=True)
    events = [
        {"type": "row_started", "at": "2026-01-01T00:00:00+00:00", "row": "row"},
        {
            "type": "step_completed",
            "at": "2026-01-01T00:00:01+00:00",
            "row": "row",
            "path": "task/review#0",
            "index": None,
        },
        {
            "type": "route",
            "at": "2026-01-01T00:00:01+00:00",
            "row": "row",
            "path": "task/review#0",
            "outcome": "revise",
            "to": "author",
        },
        {"type": "row_finished", "at": "2026-01-01T00:00:02+00:00", "row": "row", "state": "ok"},
    ]
    write_jsonl(tmp_path / "events.jsonl", events)
    episode = {
        "id": "episode",
        "env": {"id": "flow"},
        "ok": True,
        "errors": [],
        "traces": [
            {
                "id": "trace",
                "nodes": [],
                "calls": [],
                "rewards": {},
                "metrics": {},
                "timing": {},
                "info": {"decision": {"outcome": "revise", "summary": "Needs stronger evidence."}},
            }
        ],
    }
    raw = json.dumps(episode) + "\n"
    (tmp_path / "traces.jsonl").write_text(raw)
    write_jsonl(tmp_path / "traces.index.jsonl", [{"id": "trace", "offset": 0, "length": len(raw)}])
    record = {
        "key": "key",
        "row": "row",
        "path": "task/review#0",
        "index": None,
        "kind": "agent",
        "terminal": "completed",
        "payload": None,
        "trace_id": "trace",
        "error": None,
        "attempts": 1,
        "started_at": events[0]["at"],
        "finished_at": events[1]["at"],
    }
    (steps / "task__review#0.json").write_text(json.dumps(record))
    monkeypatch.setitem(server._run_registry, "flow-test", tmp_path)

    first = server.get_flow("flow-test")
    assert server.get_flow("flow-test", first["etag"]) == {"etag": first["etag"], "unchanged": True}
    assert server.get_flow_trace("flow-test", "trace")["decision"]["summary"] == "Needs stronger evidence."
    assert run_meta(tmp_path)["finished"] is True

    record["path"] = "task/author#0"
    (steps / "task__author#0.json").write_text(json.dumps(record))
    changed = server.get_flow("flow-test")
    assert changed["etag"] != first["etag"] and changed["stats"]["steps"] == first["stats"]["steps"] + 1

    write_jsonl(tmp_path / "events.jsonl", [*events, {"type": "row_started", "at": events[-1]["at"], "row": "other"}])
    assert run_meta(tmp_path)["finished"] is False


def test_task_identity_includes_the_row(tmp_path):
    (tmp_path / "steps").mkdir()
    write_jsonl(
        tmp_path / "events.jsonl",
        [
            {
                "type": "step_started",
                "at": "2026-01-01T00:00:00+00:00",
                "row": row,
                "path": "same/init#0",
                "index": None,
            }
            for row in ("row-a", "row-b")
        ],
    )
    data = project_flow(tmp_path)
    assert [(task["id"], task["name"]) for task in data["tasks"]] == [
        ("row-a/same", "same"),
        ("row-b/same", "same"),
    ]
