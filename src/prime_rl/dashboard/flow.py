"""Read-only projection of a verifiers flow ledger for the web dashboard."""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import orjson


def is_flow_run(run_dir: Path) -> bool:
    return (run_dir / "config.json").is_file() and (run_dir / "events.jsonl").is_file() and (run_dir / "steps").is_dir()


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


def _part(part: str) -> tuple[str, int]:
    name, mark, occurrence = part.rpartition("#")
    return (name, int(occurrence)) if mark and occurrence.isdigit() else (part, 0)


def _node_id(row: str, path: str, index: int | None) -> str:
    return f"{row}:{path}" + (f":{index}" if index is not None else "")


@lru_cache(maxsize=4096)
def _record(path: str, size: int, mtime_ns: int) -> dict[str, Any] | None:
    try:
        row = orjson.loads(Path(path).read_bytes())
    except (OSError, orjson.JSONDecodeError):
        return None
    return row if isinstance(row, dict) else None


def _record_rows(run_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in (run_dir / "steps").glob("*/*.json"):
        try:
            stat = path.stat()
        except OSError:
            continue
        row = _record(str(path), stat.st_size, stat.st_mtime_ns)
        if row is not None:
            rows.append(row)
    return rows


def project_flow(run_dir: Path, trace_lines: dict[str, tuple[int, str]] | None = None) -> dict[str, Any]:
    """Fold flow events and step records into a generic run/task/step graph."""
    events = read_complete_jsonl(run_dir / "events.jsonl")
    trace_lines = trace_lines or {}
    nodes: dict[tuple[str, str, int | None], dict[str, Any]] = {}
    row_info: dict[str, dict[str, Any]] = {}
    routes: dict[tuple[str, str, str, str | None], dict[str, Any]] = {}

    def node(row: str, path: str, index: int | None, order: int) -> dict[str, Any]:
        key = (row, path, index)
        if key not in nodes:
            parts = path.split("/")
            name, occurrence = _part(parts[-1])
            nodes[key] = {
                "id": _node_id(row, path, index),
                "row": row,
                "path": path,
                "scope": parts[:-1],
                "name": name,
                "occurrence": occurrence,
                "index": index,
                "kind": None,
                "status": "pending",
                "attached": False,
                "reason": None,
                "error": None,
                "attempts": None,
                "started_at": None,
                "finished_at": None,
                "trace_id": None,
                "episode_line": None,
                "episode_id": None,
                "order": order,
            }
        return nodes[key]

    for order, event in enumerate(events):
        kind = event.get("type")
        row = event.get("row")
        if kind == "row_started" and isinstance(row, str):
            row_info.setdefault(
                row, {"id": row, "status": "running", "started_at": event.get("at"), "finished_at": None}
            )
        elif kind == "row_finished" and isinstance(row, str):
            info = row_info.setdefault(row, {"id": row, "status": "running", "started_at": None, "finished_at": None})
            info.update(status=event.get("state") or "completed", finished_at=event.get("at"))
        if not isinstance(row, str) or not isinstance(event.get("path"), str):
            continue
        path, index = event["path"], event.get("index")
        if kind == "spread_started":
            item = node(row, path, None, order)
            item.update(kind="spread", status="running", started_at=event.get("at"), items=event.get("items"))
        elif kind == "spread_finished":
            item = node(row, path, None, order)
            landed = event.get("landed")
            count = event.get("items")
            status = "failed" if isinstance(landed, int) and isinstance(count, int) and landed < count else "completed"
            item.update(kind="spread", status=status, finished_at=event.get("at"), landed=landed)
        elif kind and kind.startswith("step_"):
            item = node(row, path, index, order)
            if kind == "step_started":
                item.update(status="running", started_at=event.get("at"), reason=event.get("reason"))
            elif kind == "step_attached":
                item.update(status="completed", attached=True)
            elif kind == "step_retrying":
                item.update(status="retrying", error=event.get("error"), attempts=event.get("attempt"))
            elif kind == "step_completed":
                item.update(status="completed", finished_at=event.get("at"))
            elif kind == "step_failed":
                item.update(status="failed", error=event.get("error"), finished_at=event.get("at"))
            elif kind == "step_cancelled":
                item.update(status="cancelled", finished_at=event.get("at"))
        elif kind == "route":
            key = (row, path, str(event.get("outcome") or ""), event.get("to"))
            routes.setdefault(
                key, {"row": row, "path": path, "outcome": event.get("outcome"), "to": event.get("to"), "order": order}
            )

    for record in _record_rows(run_dir):
        row, path = record.get("row"), record.get("path")
        if not isinstance(row, str) or not isinstance(path, str):
            continue
        item = node(row, path, record.get("index"), len(events))
        trace_id = record.get("trace_id")
        episode = trace_lines.get(trace_id) if isinstance(trace_id, str) else None
        item.update(
            kind=record.get("kind"),
            status="completed" if record.get("terminal") == "completed" else "failed",
            error=record.get("error"),
            attempts=record.get("attempts"),
            started_at=record.get("started_at") or item.get("started_at"),
            finished_at=record.get("finished_at") or item.get("finished_at"),
            trace_id=trace_id,
            episode_line=episode[0] if episode else None,
            episode_id=episode[1] if episode else None,
        )

    task_roots = {
        (item["row"], item["scope"][0]) for item in nodes.values() if item["name"] == "init" and item["scope"]
    }

    def group_id(row: str, task: str | None = None) -> str:
        return f"{row}/{task or ''}"

    for item in nodes.values():
        root = item["scope"][0] if item["scope"] else None
        item["group"] = group_id(item["row"], root if (item["row"], root) in task_roots else None)

    edges: list[dict[str, Any]] = []
    by_scope: dict[tuple[str, str, tuple[str, ...]], list[dict[str, Any]]] = defaultdict(list)
    for item in nodes.values():
        if item["index"] is None:
            by_scope[(item["row"], item["group"], tuple(item["scope"]))].append(item)
    for items in by_scope.values():
        ordered = sorted(items, key=lambda item: (item["order"], item["id"]))
        for source, target in zip(ordered, ordered[1:]):
            if source["id"] != target["id"]:
                edges.append(
                    {
                        "id": f"sequence:{source['id']}:{target['id']}",
                        "kind": "sequence",
                        "source": source["id"],
                        "target": target["id"],
                    }
                )

    for item in nodes.values():
        if item["index"] is None:
            continue
        parent = nodes.get((item["row"], item["path"], None))
        if parent and parent.get("kind") == "spread":
            edges.append(
                {
                    "id": f"spread:{parent['id']}:{item['id']}",
                    "kind": "spread",
                    "source": parent["id"],
                    "target": item["id"],
                }
            )

    ordered_nodes = sorted(nodes.values(), key=lambda item: (item["order"], item["id"]))
    for route in routes.values():
        source = nodes.get((route["row"], route["path"], None))
        if source is None:
            continue
        run_group = group_id(route["row"])
        targets = (
            [
                item
                for item in ordered_nodes
                if item["row"] == route["row"]
                and item["name"] == route["to"]
                and item["order"] > route["order"]
                and (source["group"] == run_group or item["group"] == source["group"])
            ]
            if route["to"] is not None
            else []
        )
        if source["group"] == run_group:
            first_by_group = {}
            for item in targets:
                first_by_group.setdefault(item["group"], item)
            targets = list(first_by_group.values())
        else:
            targets = targets[:1]
        if not targets:
            edges.append(
                {
                    "id": f"route:{source['id']}:{route['outcome']}:{route['to']}",
                    "kind": "route",
                    "source": source["id"],
                    "target": None,
                    "outcome": route["outcome"],
                    "to": route["to"],
                }
            )
        for target in targets:
            edges.append(
                {
                    "id": f"route:{source['id']}:{target['id']}:{route['outcome']}:{route['to']}",
                    "kind": "route",
                    "source": source["id"],
                    "target": target["id"],
                    "outcome": route["outcome"],
                    "to": route["to"],
                }
            )

    tasks = []
    for row, task_name in sorted(task_roots):
        task_id = group_id(row, task_name)
        task_nodes = sorted(
            (item for item in nodes.values() if item["group"] == task_id), key=lambda item: (item["order"], item["id"])
        )
        latest = task_nodes[-1] if task_nodes else None
        tasks.append(
            {
                "id": task_id,
                "name": task_name,
                "row": row,
                "stage": latest["name"] if latest else None,
                "status": latest["status"] if latest else "pending",
                "nodes": len(task_nodes),
                "traces": sum(item["trace_id"] is not None for item in task_nodes),
            }
        )

    rows = list(row_info.values())
    for row in {item["row"] for item in nodes.values()}:
        if row not in row_info:
            rows.append({"id": row, "status": "unknown", "started_at": None, "finished_at": None})
    groups = [
        {"id": group_id(row["id"]), "name": row["id"], "row": row["id"], "kind": "run"}
        for row in sorted(rows, key=lambda item: item["id"])
    ] + [{"id": task["id"], "name": task["name"], "row": task["row"], "kind": "task"} for task in tasks]
    result_nodes = sorted(nodes.values(), key=lambda item: (item["order"], item["id"]))
    return {
        "rows": sorted(rows, key=lambda item: item["id"]),
        "groups": groups,
        "tasks": tasks,
        "nodes": result_nodes,
        "edges": edges,
        "stats": {
            "rows": len(rows),
            "tasks": len(tasks),
            "steps": len(result_nodes),
            "running": sum(item["status"] in {"running", "retrying"} for item in result_nodes),
            "failed": sum(item["status"] == "failed" for item in result_nodes),
            "traces": sum(item["trace_id"] is not None for item in result_nodes),
            "routes": len(routes),
        },
    }
