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
import hashlib
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any

import orjson
from verifiers.v1.serve import EpisodeAssembly

from prime_rl.monitors.file.traces import get_trace_dir

STAGES = ("pending", "boot", "setup", "running", "finalize", "scoring", "done", "error")

SNIPPET_CHARS = 160


def get_live_dir(output_dir: Path) -> Path:
    if (output_dir / "flow.json").is_file():
        return output_dir / "live"
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


def fold_lines(data: bytes, dispatch: dict[str, Any], assembly: EpisodeAssembly) -> tuple[int, dict[str, Any]]:
    """Apply the whole lines in ``data``; returns how many bytes were consumed (a line
    torn by an append in progress waits for the next read) and the dispatch identity."""
    end = data.rfind(b"\n") + 1
    for line in data[:end].splitlines():
        if not line.strip():
            continue
        delta = orjson.loads(line)
        dispatch = delta.pop("dispatch", dispatch)
        assembly.apply(delta)  # KeyError("open") when a file lost its header line
    return end, dispatch


def read_live(path: Path) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """``(dispatch, trace)`` folded from one live file, None when the file vanished
    (its episode finished), nothing has landed in it yet, or it does not fold."""
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return None
    assembly = EpisodeAssembly()
    try:
        _, dispatch = fold_lines(data, {}, assembly)
    except (orjson.JSONDecodeError, KeyError):
        return None
    if not assembly.traces:
        return None
    (trace,) = assembly.traces.values()
    return dispatch, trace


class LiveFolds:
    """Incremental folds of a run's live files: a poll reads only the bytes a file gained
    since the last one and applies them to the assembly it already holds. Kept per process
    by a long-lived reader (the dashboard); the CLI folds from scratch."""

    def __init__(self) -> None:
        self._folds: dict[Path, tuple[int, dict[str, Any], EpisodeAssembly]] = {}
        # the list and the single-trace endpoints serve from one cache on a thread pool:
        # a delta must fold once, and a caller serializes a copy, never the growing fold
        self._lock = threading.Lock()

    def read(self, path: Path) -> tuple[dict[str, Any], dict[str, Any]] | None:
        with self._lock:
            try:
                size = path.stat().st_size
            except FileNotFoundError:
                self._folds.pop(path, None)
                return None
            consumed, dispatch, assembly = self._folds.get(path) or (0, {}, EpisodeAssembly())
            if size < consumed:  # replaced by a new attempt's file
                consumed, dispatch, assembly = 0, {}, EpisodeAssembly()
            if size > consumed:
                with path.open("rb") as f:
                    f.seek(consumed)
                    data = f.read()
                try:
                    read, dispatch = fold_lines(data, dispatch, assembly)
                except (orjson.JSONDecodeError, KeyError):
                    self._folds.pop(path, None)
                    return None
                consumed += read
                self._folds[path] = (consumed, dispatch, assembly)
            if not assembly.traces:
                return None
            (trace,) = assembly.traces.values()
            snapshot = orjson.dumps((dispatch, trace), option=orjson.OPT_NON_STR_KEYS)
        return tuple(orjson.loads(snapshot))

    def forget_missing(self, live_dir: Path) -> None:
        present = set(live_dir.glob("*.jsonl")) if live_dir.is_dir() else set()
        with self._lock:
            for path in [path for path in self._folds if path not in present]:
                del self._folds[path]


def list_live(output_dir: Path, folds: LiveFolds | None = None) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Every in-flight trace with its dispatch identity, oldest dispatch first."""
    live_dir = get_live_dir(output_dir)
    if not live_dir.is_dir():
        return []
    if folds is not None:
        folds.forget_missing(live_dir)
    read = folds.read if folds is not None else read_live
    folded = [read(path) for path in sorted(live_dir.glob("*.jsonl"))]  # not the pending/ subdir
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


def trace_row(record: dict[str, Any]) -> dict[str, Any]:
    """The live table's view of one trace: phase, turns, tokens, cost, last message."""
    calls = record.get("calls") or []
    usage = [call.get("usage") or {} for call in calls]
    # The newest node is usually a tool result; the assistant's latest words say more.
    last = ""
    for node in reversed(record.get("nodes") or []):
        message = node.get("message") or {}
        if message.get("role") == "assistant" and (last := " ".join(message_text(message).split())):
            break
    costs = [u["cost"] for u in usage if u.get("cost") is not None]
    nodes = record.get("nodes") or []
    parents = {node.get("parent") for node in nodes if node.get("parent") is not None}
    return {
        "trace": record.get("id"),
        "agent": (record.get("agent") or {}).get("name", "agent"),
        "stage": stage(record),
        "turns": len(calls),
        "branches": sum(1 for index in range(len(nodes)) if index not in parents),
        "input_tokens": record.get("num_input_tokens"),
        "output_tokens": record.get("num_output_tokens"),
        "cost": sum(costs) if costs else None,
        "stop_condition": record.get("stop_condition"),
        "errors": len(record.get("errors") or []),
        "pending": len(record.get("pending") or []),
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
        "pending": 0,
        "last": "",
    }


def live_rows(output_dir: Path, folds: LiveFolds | None = None) -> list[dict[str, Any]]:
    """Every in-flight rollout: dispatched-but-not-yet-streaming placeholders and the
    streaming traces, oldest dispatch first. A placeholder whose episode already streams
    a trace is on its way out and is not listed twice."""
    live = list_live(output_dir, folds)
    streaming = {dispatch.get("id") for dispatch, _ in live}
    rows = [pending_row(dispatch) for dispatch in list_pending(output_dir) if dispatch.get("id") not in streaming]
    rows.extend(live_row(dispatch, trace) for dispatch, trace in live)
    return sorted(rows, key=lambda row: row.get("started") or 0)


def live_etag(output_dir: Path) -> str:
    """A fingerprint of the live directory that changes whenever any in-flight rollout
    does: the set of files and their sizes. Cheap enough to answer a poll a second."""
    live_dir = get_live_dir(output_dir)
    if not live_dir.is_dir():
        return "0"
    entries = []
    for directory, pattern in ((live_dir, "*.jsonl"), (get_pending_dir(output_dir), "*.json")):
        if not directory.is_dir():
            continue
        for entry in directory.glob(pattern):
            try:
                entries.append(f"{entry.name}:{entry.stat().st_size}")
            except FileNotFoundError:
                continue  # unlinked between listing and stat: its episode finished
    return hashlib.md5("\n".join(sorted(entries)).encode()).hexdigest()[:16]


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
