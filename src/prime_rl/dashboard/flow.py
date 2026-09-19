"""Read-only projection of uniform units and explicit stage/call execution events."""

from __future__ import annotations

import fcntl
import hashlib
import subprocess
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import orjson
from verifiers.v1.flow.calls import Record
from verifiers.v1.flow.events import CallEvent, EventRecord, RunEvent, StageEvent, SteerEvent, event_adapter
from verifiers.v1.flow.unit import UnitState

TRANSITIONS = "transitions.jsonl"
STATUS = {"ready": "completed", "terminal": "completed", "waiting": "completed", "held": "failed"}
"""A transition's unit status as the node's status: a hold is the failure an operator sees."""


def flow_etag(run_dir: Path) -> str:
    paths = [run_dir / "transitions.jsonl", run_dir / "traces.jsonl", run_dir / "drain"]
    paths.extend((run_dir / "calls").glob("*/*.json"))
    paths.extend((run_dir / "live").glob("*.json"))
    for unit in sorted((run_dir / "units").glob("*")):
        paths.extend([unit / "state.json", unit / ".git/HEAD", unit / ".git/packed-refs"])
        paths.extend((unit / ".git/refs/heads").rglob("*"))
    parts = [str(_running(run_dir))]
    for path in sorted(paths):
        try:
            stat = path.stat()
        except FileNotFoundError:
            continue
        parts.append(f"{path.relative_to(run_dir)}:{stat.st_size}:{stat.st_mtime_ns}")
    return hashlib.sha256(":".join(parts).encode()).hexdigest()[:16]


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
            row = orjson.loads(raw)
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


def unit_states(run_dir: Path) -> dict[str, UnitState[Any]]:
    """Committed unit state, matching what the scheduler reads."""
    states = {}
    for unit in sorted((run_dir / "units").glob("*")):
        if not (unit / ".git").exists():
            continue
        result = subprocess.run(
            ["git", "-C", str(unit), "show", "HEAD:state.json"],
            capture_output=True,
            check=False,
        )
        if result.returncode == 0:
            states[unit.name] = UnitState[Any].model_validate_json(result.stdout)
    return states


def call_records(run_dir: Path) -> list[Record]:
    return [
        Record.model_validate(r) for p in sorted((run_dir / "calls").glob("*/*.json")) if (r := _read(p)) is not None
    ]


def read_events(run_dir: Path) -> list[EventRecord]:
    return [event_adapter.validate_python(row) for row in read_complete_jsonl(run_dir / TRANSITIONS)]


def _running(run_dir: Path) -> bool:
    try:
        with (run_dir / "flow.lock").open() as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    except FileNotFoundError:
        pass
    return False


def run_status(run_dir: Path, events: list[EventRecord]) -> str:
    boundary = next(
        (e for e in reversed(events) if isinstance(e, RunEvent) and e.type in ("run_started", "run_finished")), None
    )
    if boundary is None:
        return "pending"
    if boundary.type == "run_finished":
        assert boundary.reason is not None
        return boundary.reason
    return "running" if _running(run_dir) else "incomplete"


PAYLOAD_CAP = 65_536


def _payload(value: Any) -> Any:
    """A host call's result as the modal shows it; large ones cut to a preview."""
    if value is None:
        return None
    text = orjson.dumps(value).decode()
    return value if len(text) <= PAYLOAD_CAP else text[:PAYLOAD_CAP] + " …"


def project_flow(run_dir: Path, trace_lines: dict[str, tuple[int, str]] | None = None) -> dict[str, Any]:
    """Fold transitions, unit states and call records into the run/task/step graph the UI draws."""
    events = read_events(run_dir)
    trace_lines = trace_lines or {}
    row = run_dir.name
    states = unit_states(run_dir)
    nodes: list[dict[str, Any]] = []
    executions: dict[str, dict[str, Any]] = {}
    execution_status = run_status(run_dir, events)
    occurrences: dict[tuple[str, str], int] = defaultdict(int)

    def group_id(unit: str) -> str:
        return f"{row}/{unit}"

    steers: list[dict[str, Any]] = []
    for order, event in enumerate(events):
        if isinstance(event, SteerEvent):
            steers.append({**event.model_dump(mode="json"), "order": order})
            continue
        if not isinstance(event, StageEvent):
            continue
        kind, unit, stage = event.type, event.unit, event.stage
        if kind == "started":
            n = occurrences[(unit, stage)]
            occurrences[(unit, stage)] += 1
            node = {
                "id": f"{row}:{event.execution}",
                "execution": event.execution,
                "unit": unit,
                "path": f"{unit}/{stage}#{n}",
                "name": stage,
                "occurrence": n,
                "status": "incomplete",
                "reason": None,
                "started_at": event.at,
                "finished_at": None,
                "order": order,
                "group": group_id(unit),
                "outcome": None,
                "to": None,
                "links": [],
                "done_order": None,
                "calls": [],
                "unit_status": None,
                "report": None,
            }
            nodes.append(node)
            executions[event.execution] = node
        elif kind in ("transition", "stopped", "cancelled") and (node := executions.get(event.execution)) is not None:
            node["finished_at"] = event.at
            node["done_order"] = order
            if kind in ("stopped", "cancelled"):
                node["status"] = "cancelled"
                continue
            status = event.status
            node["status"] = STATUS.get(status, "completed")
            node["unit_status"] = status
            node["outcome"], node["to"], node["reason"] = event.outcome, event.to, event.reason
            node["report"] = event.report
            node["links"] = [link.model_dump() for link in event.links]

    # Invocation IDs, never timestamps, determine attachment to a stage.
    records = {r.call: r for r in call_records(run_dir)}
    calls: dict[str, dict[str, Any]] = {}
    for event in events:
        if not isinstance(event, CallEvent):
            continue
        parent = executions.get(event.execution)
        if parent is None:
            continue
        identity = event.call
        if identity not in calls:
            call = calls[identity] = {
                "id": event.call,
                "execution": event.execution,
                "key": event.key or event.kind,
                "kind": event.kind,
                "status": "incomplete",
                "started_at": event.at,
                "finished_at": None,
                "trace_id": None,
                "payload": None,
                "rollouts": [],
            }
            parent["calls"].append(call)
        call = calls[identity]
        if event.type == "rollout":
            rollouts = call["rollouts"]
            if event.status == "started":
                rollouts.append(
                    {
                        "trace_id": event.trace_id,
                        "status": "incomplete",
                        "rollout": event.rollout,
                        "started_at": event.at,
                    }
                )
            else:
                rollout = next(r for r in rollouts if r["rollout"] == event.rollout)
                rollout.update(
                    status=event.status,
                    finished_at=event.at,
                    error=f"{event.error.type}: {event.error.message}" if event.error else None,
                )
            call["trace_id"] = event.trace_id
            continue
        if event.status != "started":
            call.update(
                status=event.status,
                finished_at=event.at,
                trace_id=event.trace_id or call["trace_id"],
                error=f"{event.error.type}: {event.error.message}" if event.error else None,
                source_call=event.source_call,
                source_execution=event.source_execution,
            )
            record = records.get(event.source_call or event.call)
            if record:
                call["payload"] = _payload(record.payload)
    last_launch = max((i for i, e in enumerate(events) if e.type == "run_started"), default=-1)
    for node in nodes:
        if node["status"] == "incomplete" and execution_status == "running" and node["order"] > last_launch:
            node["status"] = "running"
        for call in node["calls"]:
            for item in [call, *call["rollouts"]]:
                if item["status"] == "incomplete" and node["status"] == "running":
                    item["status"] = "running"
                episode = trace_lines.get(item.get("trace_id"))
                item["episode_line"] = episode[0] if episode else None
                item["episode_id"] = episode[1] if episode else None

    edges: list[dict[str, Any]] = []
    by_unit: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for node in nodes:
        by_unit[node["unit"]].append(node)
    for unit, lane in by_unit.items():
        for i, source in enumerate(lane):
            if source["outcome"] is None:
                continue
            target = lane[i + 1] if i + 1 < len(lane) and source.get("unit_status") == "ready" else None
            edges.append(
                {
                    "id": f"route:{source['id']}:{source['outcome']}:{source['to']}",
                    "kind": "route",
                    "source": source["id"],
                    "target": target["id"] if target else None,
                    "outcome": source["outcome"],
                    "to": source["to"],
                    "summary": source["reason"],
                    "report": source.get("report"),
                }
            )
        for steer in (st for st in steers if st["unit"] == unit):  # an operator moved this unit
            before = [n for n in lane if (n.get("done_order") or n["order"]) < steer["order"]]
            after = [n for n in lane if n["order"] > steer["order"]]
            if not before:
                continue
            edges.append(
                {
                    "id": f"steer:{steer['sha']}",
                    "kind": "route",
                    "source": before[-1]["id"],
                    "target": after[0]["id"] if after else None,
                    "outcome": "steer",
                    "to": after[0]["name"] if after else "operator",
                    "summary": steer["action"].get("note") or orjson.dumps(steer["action"]).decode(),
                }
            )
    for source in nodes:  # between lanes: a stage that created, released or woke another unit
        for link in source["links"]:
            lane = by_unit.get(link["unit"], [])
            after = source.get("done_order", source["order"])
            target = next((n for n in lane if n["order"] > after), None)
            edges.append(
                {
                    "id": f"link:{source['id']}:{link['unit']}:{link['label']}",
                    "kind": "route",
                    "source": source["id"],
                    "target": target["id"] if target else None,
                    "outcome": link["label"],
                    "to": target["name"] if target else link["unit"],
                    "summary": source["reason"],
                }
            )
    units = []
    for unit, state in states.items():
        lane = by_unit.get(unit, [])
        units.append(
            {
                "id": group_id(unit),
                "name": unit,
                "stage": state.stage,
                "status": state.status,
                "nodes": len(lane),
                "traces": sum(c["trace_id"] is not None for n in lane for c in n["calls"]),
            }
        )
    all_nodes = sorted(nodes, key=lambda n: (n["order"], n["id"]))
    return {
        "status": execution_status,
        "units": units,
        "nodes": all_nodes,
        "edges": edges,
        "stats": {
            "units": len(units),
            "steps": len(nodes),
            "running": sum(n["status"] == "running" for n in nodes),
            "failed": sum(n["status"] == "failed" for n in nodes),
            "traces": sum(c["trace_id"] is not None for n in nodes for c in n["calls"]),
            "routes": sum(e["kind"] == "route" for e in edges),
        },
    }
