"""Finding a durable call's trace by its ref: the run's side of durable agent calls.

A verifiers env that installs ``vf.Durable(lookup)`` makes every agent call of its
episodes durable: before a call runs, the latest earlier trace of the call is looked up
by the call's ref (``vf.TraceLookup.get``), and a finished one is reused, a partial one
with checkpoints continued, anything else run fresh. The file monitor keeps one pointer
per ref, ``traces/refs/<ref>.json``, at the trace carrying it that moved last, wherever
that trace is: in flight (``live/<id>.jsonl``), ended without landing
(``partial/<id>.jsonl``) or landed (the finished ``stream``, by chunk and offset); see
``refs``.

``RunTraceLookup(run_dir)`` reads the pointer from the run's file monitor directory, and
from the archived attempts' (``monitors/file.attempt_N``, which ``--resume`` moves the
directory to) when the current one has none, newest first: a later attempt's trace of a
call moved after any earlier attempt's. It reads from any process on the run's host
while the run is live, the env servers' included:

    class MyEnv(vf.Env[MyConfig]):
        def durable(self, task):
            return vf.Durable(RunTraceLookup(run_dir))
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import orjson
import verifiers.v1 as vf

from prime_rl.eval.resume import archives
from prime_rl.monitors.file.traces import get_refs_dir, get_trace_dir, get_trace_stream
from prime_rl.monitors.file.traces.chunks import open_chunk
from prime_rl.monitors.file.traces.live import get_live_dir, get_partial_dir, read_live
from prime_rl.monitors.file.traces.refs import read_ref
from prime_rl.utils.pathing import get_file_monitor_dir


class RunTraceLookup:
    """A ``vf.TraceLookup`` over a prime-rl run's traces (``run_dir``, the run's output
    directory): every attempt's live, partial and finished traces, found by the ref
    pointers the file monitor keeps. Holds no state; safe to share across threads."""

    def __init__(self, run_dir: Path | str) -> None:
        self.run_dir = Path(run_dir)

    def get(self, ref: str) -> vf.Trace | None:
        """The trace ``ref`` points at, as of its latest save; None when there is none."""
        current = get_file_monitor_dir(self.run_dir)
        for directory in [current, *reversed(archives(self.run_dir))]:
            # an archived attempt keeps the layout of the directory it was
            traces = directory / get_trace_dir(Path()).name
            held = read_ref(traces / get_refs_dir(Path()).name, ref)
            if held is not None and (trace := _load(traces, held)) is not None:
                return trace
        return None


def _load(traces: Path, held: dict[str, Any]) -> vf.Trace | None:
    trace_id = held["trace"]
    if held.get("state") == "finished":
        chunk, offset = held.get("chunk"), held.get("offset")
        if not isinstance(chunk, int) or not isinstance(offset, int):
            return None
        try:
            with open_chunk(traces / get_trace_stream(Path()).name, chunk) as f:
                f.seek(offset)
                record = orjson.loads(f.readline())
        except (FileNotFoundError, orjson.JSONDecodeError):
            return None
        for trace in record.get("traces") or []:
            if trace.get("id") == trace_id:
                try:
                    return vf.WireTrace.model_validate(trace)
                except ValueError:
                    return None  # a record this verifiers cannot read: run the call fresh
        return None
    live = traces / get_live_dir(Path()).name / f"{trace_id}.jsonl"
    partial = traces / get_partial_dir(Path()).name / f"{trace_id}.jsonl"
    # the pointer may be behind its file: a live file moves to partial/ atomically
    for path in (partial, live) if held.get("state") == "partial" else (live, partial):
        folded = read_live(path)
        if folded is None:
            continue
        _, record = folded
        record = {key: value for key, value in record.items() if key != "pending"}
        try:
            return vf.WireTrace.model_validate(record)
        except ValueError:
            return None  # not foldable into a trace yet (no agent header)
    return None
