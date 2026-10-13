"""The rows of the labels index (``traces/labels.index.jsonl``): one per labelled trace,
live, partial or finished, written by the file monitor and read by ``lookup``."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import orjson

State = Literal["live", "partial", "finished"]


def trace_moments(trace: Mapping[str, Any]) -> tuple[float | None, float | None]:
    """A dumped trace's ``(start, latest checkpoint time)``: when its boot began and
    when it last saved, the two times ``vf.durable.recency`` orders traces by."""
    timing = trace.get("timing") or {}
    start = (timing.get("boot") or {}).get("start") if isinstance(timing, Mapping) else None
    times = [c["time"] for c in trace.get("checkpoints") or [] if isinstance(c, Mapping) and c.get("time")]
    return start, max(times, default=None)


def trace_labels(trace: Mapping[str, Any]) -> dict[str, Any]:
    labels = (trace.get("info") or {}).get("labels")
    return labels if isinstance(labels, dict) else {}


def label_row(
    trace_id: str,
    labels: Mapping[str, Any],
    state: State,
    *,
    start: float | None,
    checkpoint: float | None,
    **where: Any,
) -> dict[str, Any]:
    """One row of the labels index (see the module docs); ``where`` holds ``dispatch``
    (live, partial), ``end`` (partial) or ``episode``/``chunk``/``offset`` (finished)."""
    return {
        "trace": trace_id,
        "labels": labels,
        "state": state,
        "start": start,
        "checkpoint": checkpoint,
        **{key: value for key, value in where.items() if value is not None},
    }


def landed_traces(path: Path) -> set[str]:
    """The ids of the traces the labels index at ``path`` holds as landed (a
    ``finished`` row); empty when there is no index yet."""
    landed: set[str] = set()
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return landed
    for line in data.splitlines():
        try:
            row = orjson.loads(line)
        except orjson.JSONDecodeError:
            continue  # a row torn by a crash mid-append
        if isinstance(row, dict) and row.get("state") == "finished" and isinstance(row.get("trace"), str):
            landed.add(row["trace"])
    return landed


def recency(row: Mapping[str, Any]) -> float:
    """When a row's trace last moved: its latest checkpoint or its start, whichever is later."""
    return max(row.get("start") or 0.0, row.get("checkpoint") or 0.0)
