"""Durable-call refs: where a retried durable call finds its earlier trace.

Every trace of a verifiers durable call or durable box carries a ref, a hash of the
call's identity (``vf.durable.call_ref``, ``box_ref``), in its labels. The file monitor
keeps one pointer per ref, ``traces/refs/<ref>.json``, at the trace carrying it that
moved last (its latest checkpoint, else its start), wherever that trace is:

    {"trace": id, "state": "live" | "partial" | "finished", "recency": t,
     "dispatch": id, "end": ending, "episode": id, "chunk": n, "offset": b}

``dispatch`` goes with a live or partial trace, ``end`` with a partial one, and
``episode``/``chunk``/``offset`` address a finished trace's episode record. A pointer
moves as its trace streams, ends without landing (``traces/partial/``) or lands. It moves
to another trace only when that one moved as late or later, and a finished trace is
final: the same id streaming again (a durable call reusing it) leaves it in place.
``RunTraceLookup`` reads one pointer to answer ``vf.TraceLookup.get``.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import orjson

State = Literal["live", "partial", "finished"]


def trace_labels(trace: Mapping[str, Any]) -> dict[str, Any]:
    labels = (trace.get("info") or {}).get("labels")
    return labels if isinstance(labels, dict) else {}


def trace_recency(trace: Mapping[str, Any]) -> float:
    """When a dumped trace last moved: its boot start or its latest checkpoint, whichever
    is later (``vf.durable.recency``)."""
    timing = trace.get("timing") or {}
    times = [((timing.get("boot") or {}).get("start") if isinstance(timing, Mapping) else None) or 0.0]
    times += [c["time"] for c in trace.get("checkpoints") or [] if isinstance(c, Mapping) and c.get("time")]
    return max(times)


def read_ref(refs_dir: Path, ref: str) -> dict[str, Any] | None:
    try:
        held = orjson.loads((refs_dir / f"{ref}.json").read_bytes())
    except (OSError, orjson.JSONDecodeError):
        return None
    return held if isinstance(held, dict) and isinstance(held.get("trace"), str) else None


def point(refs_dir: Path, ref: str, trace_id: str, state: State, recency: float, **where: Any) -> None:
    """Point ``ref`` at ``trace_id`` (see the module docs), atomically. Callers serialize
    their writes to one directory."""
    held = read_ref(refs_dir, ref)
    if held is not None:
        if held["trace"] == trace_id:
            if held.get("state") == "finished" and state != "finished":
                return  # a landed trace is final
        elif (held.get("recency") or 0.0) > recency:
            return  # it points at a trace that moved later
    pointer = {"trace": trace_id, "state": state, "recency": recency}
    pointer.update((key, value) for key, value in where.items() if value is not None)
    refs_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=refs_dir, suffix=".tmp")
    with os.fdopen(fd, "wb") as f:
        f.write(orjson.dumps(pointer, default=str))
    os.replace(tmp, refs_dir / f"{ref}.json")
