"""Live traces: the in-flight rollouts, one file of deltas per trace.

Env servers stream every rollout as it happens (``verifiers.v1.serve.delta``): the
trace's header when it is minted, then each committed turn and each phase change. The
file monitor appends those deltas to ``traces/live/<trace_id>.jsonl`` — the first line
also carries the ``dispatch`` identity (kind, env, group, task, step) — and deletes the
file when the episode lands in the finished stream, so the directory only ever holds
in-flight work. Reading one live trace is folding one small file; ``ls`` lists what is
in flight; finished traces are untouched (the stream and its index).

    uv run python -m prime_rl.monitors.file.traces <run_dir>            # a table of live rollouts
    uv run python -m prime_rl.monitors.file.traces <run_dir> <trace_id> # one assembled trace
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import orjson
from verifiers.v1.serve import EpisodeAssembly

from prime_rl.monitors.file.traces import get_trace_dir

STAGES = ("pending", "boot", "setup", "running", "finalize", "scoring", "done", "error")

SNIPPET_CHARS = 160


def get_live_dir(output_dir: Path) -> Path:
    return get_trace_dir(output_dir) / "live"


def live_path(output_dir: Path, trace_id: str) -> Path:
    return get_live_dir(output_dir) / f"{trace_id}.jsonl"


def get_pending_dir(output_dir: Path) -> Path:
    """Dispatched episodes whose first trace has not streamed yet, one JSON of dispatch
    identity each; the file goes away once a trace streams or the episode ends."""
    return get_live_dir(output_dir) / "pending"


def list_pending(output_dir: Path) -> list[dict[str, Any]]:
    pending_dir = get_pending_dir(output_dir)
    if not pending_dir.is_dir():
        return []
    rows = []
    for path in pending_dir.glob("*.json"):
        try:
            rows.append(orjson.loads(path.read_bytes()))
        except (FileNotFoundError, orjson.JSONDecodeError):
            continue  # gone or mid-write: it is not pending anymore, or not yet
    return rows


def read_live(path: Path) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """``(dispatch, trace)`` folded from one live file, None when the file vanished
    (its episode finished) or nothing has landed in it yet."""
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return None
    assembly = EpisodeAssembly()
    dispatch: dict[str, Any] = {}
    for line in data.splitlines():
        if not line.strip():
            continue
        try:
            delta = orjson.loads(line)
        except orjson.JSONDecodeError:
            break  # the last line, caught mid-append
        dispatch = delta.pop("dispatch", dispatch)
        assembly.apply(delta)
    if not assembly.traces:
        return None
    (trace,) = assembly.traces.values()
    return dispatch, trace


def list_live(output_dir: Path) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Every in-flight trace with its dispatch identity, oldest dispatch first."""
    live_dir = get_live_dir(output_dir)
    if not live_dir.is_dir():
        return []
    folded = [read_live(path) for path in sorted(live_dir.glob("*.jsonl"))]  # not the pending/ subdir
    return sorted((item for item in folded if item is not None), key=lambda item: item[0].get("started") or 0)


def stage(trace: dict[str, Any]) -> str:
    """The phase a trace is in, read off its timing spans the way the rollout sets them."""
    if trace.get("errors"):
        return "error"
    timing = trace.get("timing") or {}
    if (timing.get("scoring") or {}).get("end"):
        return "done"
    for span, name in (
        ("scoring", "scoring"),
        ("finalize", "finalize"),
        ("agent", "running"),
        ("setup", "setup"),
        ("boot", "boot"),
    ):
        if (timing.get(span) or {}).get("start"):
            return name
    return "pending"


def message_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(part.get("text", "") for part in content if isinstance(part, dict))
    return ""


def trace_row(trace: dict[str, Any]) -> dict[str, Any]:
    """The live table's view of one trace: phase, turns, tokens, cost, last message."""
    calls = trace.get("calls") or []
    usage = [call.get("usage") or {} for call in calls]
    # The newest node is usually a tool result; the assistant's latest words say more.
    last = ""
    for node in reversed(trace.get("nodes") or []):
        message = node.get("message") or {}
        if message.get("role") == "assistant" and (last := " ".join(message_text(message).split())):
            break
    costs = [u["cost"] for u in usage if u.get("cost") is not None]
    nodes = trace.get("nodes") or []
    parents = {node.get("parent") for node in nodes if node.get("parent") is not None}
    return {
        "trace": trace.get("id"),
        "agent": (trace.get("agent") or {}).get("name", "agent"),
        "stage": stage(trace),
        "turns": len(calls),
        "branches": sum(1 for index in range(len(nodes)) if index not in parents),
        "input_tokens": sum(u.get("prompt_tokens") or 0 for u in usage),
        "output_tokens": sum(u.get("completion_tokens") or 0 for u in usage),
        "cost": sum(costs) if costs else None,
        "stop_condition": trace.get("stop_condition"),
        "errors": len(trace.get("errors") or []),
        "last": last[:SNIPPET_CHARS],
    }


def live_row(dispatch: dict[str, Any], trace: dict[str, Any]) -> dict[str, Any]:
    started = dispatch.get("started")
    return {**dispatch, "elapsed": time.time() - started if started else None, **trace_row(trace)}


def pending_row(dispatch: dict[str, Any]) -> dict[str, Any]:
    started = dispatch.get("started")
    return {
        **dispatch,
        "elapsed": time.time() - started if started else None,
        "trace": None,
        "agent": None,
        "stage": "pending",
        "turns": 0,
        "branches": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cost": None,
        "stop_condition": None,
        "errors": 0,
        "last": "",
    }


def live_rows(output_dir: Path) -> list[dict[str, Any]]:
    """Every in-flight rollout: dispatched-but-not-yet-streaming placeholders and the
    streaming traces, oldest dispatch first."""
    rows = [pending_row(dispatch) for dispatch in list_pending(output_dir)]
    rows.extend(live_row(dispatch, trace) for dispatch, trace in list_live(output_dir))
    return sorted(rows, key=lambda row: row.get("started") or 0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Read a run's live (in-flight) traces.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("trace_id", nargs="?", help="print this trace assembled, as JSON")
    args = parser.parse_args()
    if args.trace_id:
        folded = read_live(live_path(args.run_dir, args.trace_id))
        if folded is None:
            sys.exit(f"no live trace {args.trace_id} under {get_live_dir(args.run_dir)}")
        dispatch, trace = folded
        print(json.dumps({"dispatch": dispatch, "trace": trace}, indent=2, default=str))
        return
    rows = live_rows(args.run_dir)
    if not rows:
        print(f"no live traces under {get_live_dir(args.run_dir)}")
        return
    for row in rows:
        elapsed = f"{row['elapsed']:.0f}s" if row.get("elapsed") is not None else "-"
        dispatched = time.strftime("%H:%M:%S", time.localtime(row["started"])) if row.get("started") else "--:--:--"
        print(
            f"{dispatched}  {(row['trace'] or '-')[:8]:8s}  {row.get('kind', ''):5s} {row.get('env', ''):20s} {str(row.get('task', '')):24s} "
            f"{row['stage']:9s} turns {row['turns']:3d}  in {row['input_tokens']:>7d} out {row['output_tokens']:>7d}  "
            f"{elapsed:>6s}  {row['last'][:60]}"
        )


if __name__ == "__main__":
    main()
