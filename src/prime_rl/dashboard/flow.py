"""Read-only projection of a verifiers flow root for the web dashboard.

A flow root holds units (`campaign/`, `tasks/<id>/`: git repositories whose `state.json` says
where each stands), `transitions.jsonl` (one line per stage start and per transition), `calls/`
(one record per durable call, with the trace it made) and `live/` (a snapshot per seat in
flight). Every stage run is a node in its unit's lane; a transition is a labeled edge to the
stage the unit moved to; the calls a stage made are its children, each opening its trace.
"""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import orjson

TRANSITIONS = "transitions.jsonl"
STATUS = {"ready": "completed", "terminal": "completed", "waiting": "completed", "held": "failed"}
"""A transition's unit status as the node's status: a hold is the failure an operator sees."""


def is_flow_run(run_dir: Path) -> bool:
    return (run_dir / "flow.json").is_file() and (run_dir / TRANSITIONS).is_file()


def read_complete_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read complete JSONL records without touching a live writer's torn tail."""
    if not path.is_file():
        return []
    rows = []
    with path.open("rb") as file:
        for raw in file:
            if not raw.endswith(b"\n"):
                break
            try:
                row = orjson.loads(raw)
            except orjson.JSONDecodeError:
                break
            if isinstance(row, dict):
                rows.append(row)
    return rows


@lru_cache(maxsize=8192)
def _json_file(path: str, size: int, mtime_ns: int) -> dict[str, Any] | None:
    try:
        row = orjson.loads(Path(path).read_bytes())
    except (OSError, orjson.JSONDecodeError):
        return None
    return row if isinstance(row, dict) else None


def _read(path: Path) -> dict[str, Any] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return _json_file(str(path), stat.st_size, stat.st_mtime_ns)


def unit_states(run_dir: Path) -> dict[str, dict[str, Any]]:
    """Every unit's `state.json`: the campaign first, then the tasks by id."""
    states = {}
    if (campaign := _read(run_dir / "campaign" / "state.json")) is not None:
        states["campaign"] = campaign
    for path in sorted((run_dir / "tasks").glob("*/state.json")):
        if (state := _read(path)) is not None:
            states[path.parent.name] = state
    return states


def call_records(run_dir: Path) -> list[dict[str, Any]]:
    return [r for p in sorted((run_dir / "calls").glob("*/*.json")) if (r := _read(p)) is not None]


def live_seats(run_dir: Path) -> dict[str, list[str]]:
    """The seats in flight per unit, from the live snapshot names `<unit>--<key>.json`."""
    out: dict[str, list[str]] = defaultdict(list)
    for path in sorted((run_dir / "live").glob("*.json")):
        unit, _, key = path.stem.partition("--")
        out[unit].append(key.replace("__", "/"))
    return out


def project_flow(run_dir: Path, trace_lines: dict[str, tuple[int, str]] | None = None) -> dict[str, Any]:
    """Fold transitions, unit states and call records into the run/task/step graph the UI draws."""
    events = read_complete_jsonl(run_dir / TRANSITIONS)
    trace_lines = trace_lines or {}
    row = run_dir.name
    states = unit_states(run_dir)
    nodes: list[dict[str, Any]] = []
    open_by_unit: dict[str, dict[str, Any]] = {}
    occurrences: dict[tuple[str, str], int] = defaultdict(int)

    def group_id(unit: str) -> str:
        return f"{row}/" if unit == "campaign" else f"{row}/{unit}"

    for order, event in enumerate(events):
        kind, unit, stage = event.get("type"), event.get("unit"), event.get("stage")
        if not isinstance(unit, str) or not isinstance(stage, str):
            continue
        if kind == "started":
            n = occurrences[(unit, stage)]
            occurrences[(unit, stage)] += 1
            node = {
                "id": f"{row}:{unit}/{stage}#{n}",
                "row": row,
                "unit": unit,
                "path": f"{unit}/{stage}#{n}",
                "scope": [unit],
                "name": stage,
                "occurrence": n,
                "index": None,
                "kind": "stage",
                "status": "running",
                "attached": False,
                "reason": None,
                "error": None,
                "attempts": None,
                "started_at": event.get("at"),
                "finished_at": None,
                "trace_id": None,
                "episode_line": None,
                "episode_id": None,
                "order": order,
                "group": group_id(unit),
                "outcome": None,
                "to": None,
                "live": [],
            }
            nodes.append(node)
            open_by_unit[unit] = node
        elif kind in ("transition", "stopped") and (node := open_by_unit.pop(unit, None)) is not None:
            node["finished_at"] = event.get("at")
            if kind == "stopped":
                node["status"] = "cancelled"
                continue
            status = event.get("status")
            node["status"] = STATUS.get(status, "completed")
            node["outcome"], node["to"], node["reason"] = event.get("outcome"), event.get("to"), event.get("reason")
            if status == "held":
                node["error"] = event.get("reason")

    # The calls a stage made are its children; the last seat's trace is the stage's own (its decision).
    by_unit_stage: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for node in nodes:
        by_unit_stage[(node["unit"], node["name"])].append(node)
    children: list[dict[str, Any]] = []
    for record in call_records(run_dir):
        unit, stage, finished = record.get("unit"), record.get("stage"), record.get("finished_at") or ""
        candidates = by_unit_stage.get((unit, stage)) or []
        parent = next(
            (n for n in candidates if (n["started_at"] or "") <= finished <= (n["finished_at"] or "9")),
            candidates[-1] if candidates else None,
        )
        if parent is None:
            continue
        trace_id = record.get("trace_id")
        episode = trace_lines.get(trace_id) if isinstance(trace_id, str) else None
        index = sum(c["path"] == parent["path"] for c in children)
        children.append(
            {
                **parent,
                "id": f"{parent['id']}:{index}",
                "index": index,
                "name": str(record.get("key") or record.get("kind")),
                "kind": record.get("kind"),
                "status": "completed",
                "reason": None,
                "error": None,
                "outcome": None,
                "to": None,
                "started_at": record.get("started_at"),
                "finished_at": record.get("finished_at"),
                "trace_id": trace_id,
                "episode_line": episode[0] if episode else None,
                "episode_id": episode[1] if episode else None,
            }
        )
        if trace_id is not None:
            parent["trace_id"], parent["episode_line"], parent["episode_id"] = trace_id, *(episode or (None, None))

    edges: list[dict[str, Any]] = []
    by_unit: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for node in nodes:
        by_unit[node["unit"]].append(node)
    for lane in by_unit.values():
        for source, target in zip(lane, lane[1:]):
            edges.append(
                {
                    "id": f"sequence:{source['id']}:{target['id']}",
                    "kind": "sequence",
                    "source": source["id"],
                    "target": target["id"],
                }
            )
        for i, source in enumerate(lane):
            if source["outcome"] is None:
                continue
            target = lane[i + 1] if i + 1 < len(lane) else None
            edges.append(
                {
                    "id": f"route:{source['id']}:{source['outcome']}:{source['to']}",
                    "kind": "route",
                    "source": source["id"],
                    "target": target["id"] if target else None,
                    "outcome": source["outcome"],
                    "to": source["to"],
                    "summary": source["reason"],
                }
            )
    for child in children:
        parent_id = child["id"].rsplit(":", 1)[0]
        edges.append(
            {"id": f"spread:{parent_id}:{child['id']}", "kind": "spread", "source": parent_id, "target": child["id"]}
        )

    live = live_seats(run_dir)
    for unit, node in open_by_unit.items():  # a stage without its transition is running
        node["live"] = live.get(unit, [])

    tasks = []
    for unit, state in states.items():
        if unit == "campaign":
            continue
        lane = by_unit.get(unit, [])
        tasks.append(
            {
                "id": group_id(unit),
                "name": unit,
                "row": row,
                "stage": state.get("stage"),
                "status": state.get("status"),
                "reason": state.get("reason"),
                "nodes": len(lane),
                "traces": sum(c["trace_id"] is not None for c in children if c["unit"] == unit),
            }
        )
    all_nodes = sorted([*nodes, *children], key=lambda n: (n["order"], n["id"]))
    campaign_done = states.get("campaign", {}).get("status") in ("waiting", "terminal")
    finished = campaign_done and bool(tasks) and all(t["status"] == "terminal" for t in tasks) and not open_by_unit
    rows = [
        {
            "id": row,
            "status": "completed" if finished else "running",
            "started_at": events[0]["at"] if events else None,
            "finished_at": events[-1]["at"] if finished and events else None,
        }
    ]
    groups = [{"id": group_id("campaign"), "name": row, "row": row, "kind": "run"}] + [
        {"id": t["id"], "name": t["name"], "row": row, "kind": "task"} for t in tasks
    ]
    return {
        "rows": rows,
        "groups": groups,
        "tasks": tasks,
        "nodes": all_nodes,
        "edges": edges,
        "stats": {
            "rows": 1,
            "tasks": len(tasks),
            "steps": len(nodes),
            "running": sum(n["status"] == "running" for n in nodes),
            "failed": sum(n["status"] == "failed" for n in nodes),
            "traces": sum(c["trace_id"] is not None for c in children),
            "routes": sum(e["kind"] == "route" for e in edges),
            "held": sum(t["status"] == "held" for t in tasks),
            "waiting": sum(t["status"] == "waiting" for t in tasks),
            "live": sum(len(v) for v in live.values()),
        },
    }
