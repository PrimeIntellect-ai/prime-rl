"""Finding a run's traces by their labels: the run's side of durable agent calls.

A verifiers env that installs ``vf.Durable(lookup)`` makes every agent call of its
episodes durable: before a call runs, the latest earlier trace carrying the call's
labels is looked up (``vf.TraceLookup.latest``), and a finished one is reused, a
partial one with checkpoints continued, anything else run fresh. The traces live in
three places under ``traces/``: in flight (``live/<id>.jsonl``), ended without landing
(``partial/<id>.jsonl``) and landed (the finished ``stream``). The file monitor writes
one row per labelled trace to ``labels.index.jsonl`` wherever it is, so a lookup never
reads a trace it will not return:

    {"trace": id, "labels": {...}, "state": "live" | "partial" | "finished",
     "start": t, "checkpoint": t, "dispatch": id, "episode": id, "chunk": n, "offset": b}

A row goes out when a live trace's labels arrive, on each checkpoint it takes, when its
file moves to ``partial/`` (with its ``end``), and for each labelled trace of a landed
episode (``state`` finished, ``chunk``/``offset`` addressing its episode record).
``start`` is when the trace's boot began, ``checkpoint`` its latest checkpoint's time:
together, when it last moved. The last row of a trace wins, except that a finished row
is never displaced: a trace landed is final, and the same id streaming again is the
same trace (a durable call reusing it).

``RunTraceLookup(run_dir)`` tails that index (a lookup reads only the rows written
since the last one) across the run's attempts: a ``--resume`` moves the file monitor
directory to ``monitors/file.attempt_N``, and the archived attempts' traces stay
findable where they are. It reads from any process on the run's host while the run
is live, the env servers' included:

    class MyEnv(vf.Env[MyConfig]):
        def durable(self, task):
            return vf.Durable(RunTraceLookup(run_dir))
"""

from __future__ import annotations

import os
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import orjson
import verifiers.v1 as vf
from verifiers.v1.durable import labels_match

from prime_rl.eval.resume import archives
from prime_rl.monitors.file.traces import get_labels_index_path, get_trace_dir, get_trace_stream
from prime_rl.monitors.file.traces.chunks import open_chunk
from prime_rl.monitors.file.traces.labels import recency
from prime_rl.monitors.file.traces.live import get_live_dir, get_partial_dir, read_live
from prime_rl.utils.pathing import get_file_monitor_dir


class _Attempt:
    """One file monitor directory of the run and how far its labels index was read."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        # an archived attempt keeps the layout of the directory it was (get_trace_dir & co)
        self.traces = directory / get_trace_dir(Path()).name
        self.index = self.traces / get_labels_index_path(Path()).name
        self.live = self.traces / get_live_dir(Path()).name
        self.partial = self.traces / get_partial_dir(Path()).name
        self.stream = self.traces / get_trace_stream(Path()).name
        self.offset = 0
        self.inode: int | None = None


class RunTraceLookup:
    """A ``vf.TraceLookup`` over a prime-rl run's traces (``run_dir``, the run's output
    directory): every attempt's live, partial and finished traces, found through the
    labels index. ``latest(labels)`` first reads the index rows written since the last
    call, then matches in memory and loads only the trace it returns. Safe to share
    across threads; one per process is enough."""

    def __init__(self, run_dir: Path | str) -> None:
        self.run_dir = Path(run_dir)
        self.rows_read = 0
        """Index rows read so far, over every attempt (a reread after a resume counts again)."""
        self._lock = threading.Lock()
        self._reset()

    def _reset(self) -> None:
        current = get_file_monitor_dir(self.run_dir)
        self._attempts = [_Attempt(directory) for directory in [*archives(self.run_dir), current]]
        self._rows: dict[str, tuple[_Attempt, dict[str, Any]]] = {}

    def poll(self) -> int:
        """Read the index rows written since the last poll; returns how many. A resume
        (the current directory became an archive) rereads every attempt."""
        with self._lock:
            return self._poll()

    def _poll(self) -> int:
        current = self._attempts[-1]
        try:
            stat = current.index.stat()
            replaced = current.inode is not None and (stat.st_ino != current.inode or stat.st_size < current.offset)
        except FileNotFoundError:
            replaced = current.inode is not None
        if replaced:
            self._reset()
        read = 0
        for attempt in self._attempts:
            read += self._tail(attempt)
        self.rows_read += read
        return read

    def _tail(self, attempt: _Attempt) -> int:
        try:
            with attempt.index.open("rb") as f:
                attempt.inode = os.fstat(f.fileno()).st_ino
                f.seek(attempt.offset)
                data = f.read()
        except FileNotFoundError:
            return 0
        end = data.rfind(b"\n") + 1  # a row torn by an append in progress waits
        attempt.offset += end
        read = 0
        for line in data[:end].splitlines():
            try:
                row = orjson.loads(line)
            except orjson.JSONDecodeError:
                continue
            if not isinstance(row, dict) or not isinstance(row.get("trace"), str):
                continue
            read += 1
            held = self._rows.get(row["trace"])
            if held is not None and held[1].get("state") == "finished" and row.get("state") != "finished":
                continue  # a landed trace is final
            self._rows[row["trace"]] = (attempt, row)
        return read

    def latest(self, labels: Mapping[str, Any]) -> vf.Trace | None:
        """Of the run's traces whose labels contain ``labels`` (a nested subset), the one
        that moved last, as of its latest save; None when there is none."""
        with self._lock:
            for _ in range(2):
                self._poll()
                matches = [held for held in self._rows.values() if labels_match(held[1].get("labels") or {}, labels)]
                vanished = False
                for attempt, row in sorted(matches, key=lambda held: recency(held[1]), reverse=True):
                    trace = self._load(attempt, row)
                    if trace is None:
                        vanished = True  # landed or moved since its row: the next poll has it
                        continue
                    if labels_match(trace.labels, labels):
                        return trace
                if not vanished:
                    break
        return None

    def _load(self, attempt: _Attempt, row: Mapping[str, Any]) -> vf.Trace | None:
        trace_id = row["trace"]
        if row.get("state") == "finished":
            return self._load_finished(attempt, trace_id, row.get("chunk"), row.get("offset"))
        live, partial = attempt.live / f"{trace_id}.jsonl", attempt.partial / f"{trace_id}.jsonl"
        # the row may be behind its file: a live file moves to partial/ atomically
        for path in (partial, live) if row.get("state") == "partial" else (live, partial):
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

    def _load_finished(self, attempt: _Attempt, trace_id: str, chunk: Any, offset: Any) -> vf.Trace | None:
        if not isinstance(chunk, int) or not isinstance(offset, int):
            return None
        try:
            with open_chunk(attempt.stream, chunk) as f:
                f.seek(offset)
                record = orjson.loads(f.readline())
        except (FileNotFoundError, orjson.JSONDecodeError):
            return None
        for trace in record.get("traces") or []:
            if trace.get("id") == trace_id:
                return vf.WireTrace.model_validate(trace)
        return None
