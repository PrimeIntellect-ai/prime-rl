from __future__ import annotations

import asyncio
import json
import os
import shutil
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO, TextIO

import orjson
from verifiers.v1.durable import refs_of

from prime_rl.configs.monitors import FileMonitorConfig
from prime_rl.monitors.base import Kind, Monitor, Subset
from prime_rl.monitors.file.traces import get_annotations_dir, get_index_path, get_refs_dir, get_trace_stream
from prime_rl.monitors.file.traces.chunks import ChunkedJsonl
from prime_rl.monitors.file.traces.index import index_row
from prime_rl.monitors.file.traces.live import (
    end_line,
    get_live_dir,
    get_partial_dir,
    get_pending_dir,
    read_live,
    trim_torn_line,
)
from prime_rl.monitors.file.traces.refs import State, point, trace_labels, trace_recency
from prime_rl.monitors.file.traces.update import update_index_row
from prime_rl.utils.pathing import get_eval_plan_path, get_file_monitor_dir
from prime_rl.utils.utils import sanitize

if TYPE_CHECKING:
    import verifiers.v1 as vf

OPTS = orjson.OPT_APPEND_NEWLINE | orjson.OPT_SERIALIZE_NUMPY

ENDED_CAP = 10_000
"""How many ended traces' ids the monitor remembers, to drop deltas that straggle in
after their trace's file moved to ``partial/``."""


class LiveRefs:
    """What a live trace's ref pointers need, gathered from its deltas: its refs, when it
    last moved, and its dispatch."""

    __slots__ = ("refs", "recency", "dispatch")

    def __init__(self, dispatch: str | None) -> None:
        self.refs: list[str] = []
        self.recency = 0.0
        self.dispatch = dispatch

    def apply(self, delta: dict[str, Any]) -> bool:
        """Fold one delta in; whether anything a pointer holds changed."""
        before = (self.refs, self.recency)
        changed = delta.get("set") or {}
        if isinstance(changed.get("info"), dict):
            self.refs = refs_of(trace_labels({"info": changed["info"]}))
        moved = trace_recency({"timing": changed.get("timing") or {}, "checkpoints": delta.get("checkpoints") or []})
        self.recency = max(self.recency, moved)
        return (self.refs, self.recency) != before

    @classmethod
    def of(cls, dispatch: dict[str, Any], trace: dict[str, Any]) -> LiveRefs:
        """From a folded live file (a trace the process restarted under)."""
        held = cls(dispatch.get("id"))
        held.refs = refs_of(trace_labels(trace))
        held.recency = trace_recency(trace)
        return held


class FileMonitor(Monitor):
    """Logs metrics and episodes to local JSONL files."""

    config: FileMonitorConfig
    file: TextIO

    async def init(self, output_dir: Path, producer: str | None = None) -> None:
        self.output_dir = output_dir
        self.producer = producer
        # a relaunch appends to the stream it finds, so the numbering carries on, and so
        # does the set of traces it landed
        self._logged = 0
        self._landed: set[str] = set()
        """The ids of the traces the stream's episodes landed that can land again (a
        durable call's, carrying a ref; the index rows of a relaunch's stream list them
        all): a reused trace's usage counts with the episode that landed it first, as
        the dashboard's summaries count it."""
        index = get_index_path(get_trace_stream(output_dir))
        if index.is_file():
            with index.open("rb") as f:
                for line in f:
                    self._logged += 1
                    try:
                        row = orjson.loads(line)
                    except orjson.JSONDecodeError:
                        continue  # torn by a crash mid-append
                    self._landed.update(row.get("trace_ids") or [])
        self._refs_dir = get_refs_dir(output_dir)
        self.path = get_file_monitor_dir(output_dir) / self.config.path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._live_cleared = False
        # Line-buffered append so a concurrently-running dashboard can tail the file.
        self.file = open(self.path, "a", buffering=1)  # noqa: SIM115
        self._streams: dict[Path, tuple[ChunkedJsonl, BinaryIO]] = {}
        # the ref pointers: moved from log_live and log_episodes, which run on threads
        self._refs_lock = threading.Lock()
        self._live_refs: dict[str, LiveRefs] = {}
        # one batch at a time: a flush cancelled mid-write keeps writing on its thread
        self._live_lock = threading.Lock()
        self._ended: OrderedDict[str, None] = OrderedDict()
        self.logger.info(f"Logging metrics and episodes to the local filesystem ({output_dir})")

    async def log_metrics(self, metrics: dict[str, Any], step: int | None) -> None:
        """``step=None`` logs a time-keyed row (e.g. inference metrics, which are
        sampled on wall time rather than the training step)."""
        sanitized, dropped = sanitize(metrics)
        if dropped:
            self.logger.warning(
                f"Dropping {len(dropped)} non-finite value(s) from {self.config.path}: {', '.join(dropped[:5])}"
            )

        row = {"step": step, "time": time.time(), **sanitized}
        if self.producer is not None:
            row["producer"] = self.producer
        self.file.write(json.dumps(row) + "\n")

    async def log_eval_plan(self, env_name: str, step: int, expected: int) -> None:
        """Merge the epoch's expected count into ``plan.json`` (atomic replace)."""
        path = get_eval_plan_path(self.output_dir)
        plan = orjson.loads(path.read_bytes()) if path.is_file() else {}
        plan.setdefault(env_name, {})[str(step)] = expected
        tmp = path.with_suffix(".json.tmp")
        tmp.write_bytes(orjson.dumps(plan))
        tmp.replace(path)

    async def log_live(self, events: list[dict[str, Any]]) -> None:
        """Append each delta to its trace's file under ``traces/live/`` (the first line
        carrying the dispatch identity). A trace whose episode landed loses its file (the
        finished stream has it); one that ended otherwise (``done`` with an ``end``, or a
        ``discard``) keeps it, moved to ``traces/partial/`` with its ending as the last
        line. The pointers of each trace's durable refs follow it (``refs``). A
        dispatched episode holds a placeholder under ``traces/live/pending/`` until its
        first trace streams."""
        live_dir = get_live_dir(self.output_dir)
        pending_dir = get_pending_dir(self.output_dir)
        # A previous process's live traces are over: nothing streams into them again. Only
        # the process that streams recovers them - every process of a run shares this monitor.
        recover = not self._live_cleared
        self._live_cleared = True

        def write() -> None:
            with self._live_lock:
                write_batch()

        def write_batch() -> None:
            if recover:
                self._recover_live(live_dir, pending_dir)
            pending_dir.mkdir(parents=True, exist_ok=True)
            appends: dict[Path, list[bytes]] = {}
            touched: dict[str, None] = {}
            for event in events:
                if "done" in event:
                    trace_id = event["done"]
                    path = live_dir / f"{trace_id}.jsonl"
                    touched.pop(trace_id, None)
                    end = event.get("end", "landed")
                    if end == "landed":
                        appends.pop(path, None)
                        path.unlink(missing_ok=True)
                        self._live_refs.pop(trace_id, None)
                    else:
                        self._keep_partial(path, appends.pop(path, []), end)
                    continue
                if "pending" in event:
                    path = pending_dir / f"{event['pending']}.json"
                    tmp = path.with_suffix(".json.tmp")
                    tmp.write_bytes(orjson.dumps(event["dispatch"]))
                    tmp.replace(path)
                    continue
                if "dispatched" in event:
                    (pending_dir / f"{event['dispatched']}.json").unlink(missing_ok=True)
                    continue
                delta = event["delta"]
                trace_id = delta["trace"]
                path = live_dir / f"{trace_id}.jsonl"
                if delta.get("discard"):
                    touched.pop(trace_id, None)
                    self._keep_partial(path, appends.pop(path, []), "discarded")
                    continue
                if trace_id in self._ended:
                    if "open" not in delta:
                        continue  # straggling in after its trace ended: its file has moved
                    del self._ended[trace_id]  # streaming again (a durable call reusing it)
                held = self._live_refs.get(trace_id)
                if held is None:
                    held = self._live_refs[trace_id] = LiveRefs((event.get("dispatch") or {}).get("id"))
                if held.apply(delta) and held.refs:
                    touched[trace_id] = None
                # Training arrays stay on the wire; live records only need display data.
                line = {key: value for key, value in delta.items() if key != "routing_repairs"}
                if "nodes" in line:
                    line["nodes"] = [
                        {key: value for key, value in node.items() if key not in ("routed_experts", "sampling_mask")}
                        for node in line["nodes"]
                    ]
                if "open" in delta:
                    line["dispatch"] = event["dispatch"]
                # deltas key semantic links by node index (int)
                appends.setdefault(path, []).append(
                    orjson.dumps(line, default=str, option=OPTS | orjson.OPT_NON_STR_KEYS)
                )
            # one open per file per batch: a trace's deltas of the last half second land together
            for path, lines in appends.items():
                with path.open("ab") as f:
                    f.write(b"".join(line + b"\n" for line in lines))
            # after the files: a pointer a reader sees points at what it says
            for trace_id in touched:
                self._point(trace_id, self._live_refs[trace_id], "live")

        await asyncio.to_thread(write)

    def _keep_partial(self, path: Path, lines: list[bytes], end: str) -> None:
        """Move a durable live trace's file to ``partial/``, its unwritten deltas and its
        ending appended first, and its ref pointers with it. A trace with no durable ref
        (an RL rollout cut or dropped stale, most eval failures) is only ever looked up
        live, so its file goes, as a landed one's does."""
        trace_id = path.stem
        held = self._live_refs.pop(trace_id, None)
        self._ended[trace_id] = None
        while len(self._ended) > ENDED_CAP:
            self._ended.popitem(last=False)
        if held is None or not held.refs:
            path.unlink(missing_ok=True)
            return
        if path.exists():
            trim_torn_line(path)  # a process killed mid-append left a torn last line
        elif not lines:
            return  # nothing of it ever streamed
        with path.open("ab") as f:
            f.write(b"".join(line + b"\n" for line in lines) + end_line(end))
        partial_dir = get_partial_dir(self.output_dir)
        partial_dir.mkdir(parents=True, exist_ok=True)
        os.replace(path, partial_dir / path.name)
        self._point(trace_id, held, "partial", end=end)

    def _recover_live(self, live_dir: Path, pending_dir: Path) -> None:
        """Keep the durable live traces a dead process left as ``crashed`` partials, and
        drop the rest; its pending placeholders are gone with it."""
        shutil.rmtree(pending_dir, ignore_errors=True)
        for path in sorted(live_dir.glob("*.jsonl")) if live_dir.is_dir() else []:
            folded = read_live(path)
            if folded is not None:
                self._live_refs[path.stem] = LiveRefs.of(*folded)
            self._keep_partial(path, [], "crashed")

    def _point(self, trace_id: str, held: LiveRefs, state: State, **where: Any) -> None:
        with self._refs_lock:
            for ref in held.refs:
                point(self._refs_dir, ref, trace_id, state, held.recency, dispatch=held.dispatch, **where)

    def _stream(self, directory: Path) -> tuple[ChunkedJsonl, BinaryIO]:
        """A stream and its index, opened on first use and kept open. Writers flush the
        stream before the index: a row a reader sees points at a record it can read."""
        if directory not in self._streams:
            stream = ChunkedJsonl(directory, self.config.chunk_bytes, self.config.compress)
            self._streams[directory] = (stream, open(get_index_path(directory), "ab"))
        return self._streams[directory]

    async def log_episodes(self, episodes: list[vf.Episode], step: int, kind: Kind, subset: Subset) -> None:
        """Append each episode to the trace stream as it completes — every episode is
        serialized exactly once, in arrival order, whatever its kind, so an in-progress
        run can be tailed. Episode-level failures are preserved even when no trace was
        produced. The shipped cohort writes no second copy: what it learns arrives as
        annotations that readers fold back onto the stream."""
        if subset == "effective":
            return

        def write() -> None:
            stream, index = self._stream(get_trace_stream(self.output_dir))
            # An index row goes out with every episode: summarising the record here,
            # while it is already in hand, saves every reader from parsing a stream
            # that outgrows memory long before the run does.
            # a durable call can land a trace again; its usage counts with the episode
            # that landed it first (``index_row``'s ``landed``)
            moves: list[tuple[str, LiveRefs, dict[str, Any]]] = []
            for episode in episodes:
                record = episode.to_record(float_decimals=self.config.float_decimals)
                chunk, offset = stream.append(orjson.dumps(record, default=str, option=OPTS))
                self._logged += 1
                traces = [trace for trace in record.get("traces") or [] if trace.get("id")]
                row = index_row(self._logged, record, chunk, offset, self._landed)
                index.write(orjson.dumps(row, default=str, option=OPTS))
                for trace in traces:
                    held = LiveRefs.of({}, trace)
                    if held.refs:
                        # only a durable call's trace can land again: an RL run adds nothing
                        self._landed.add(trace["id"])
                        where = {"episode": record.get("id"), "chunk": chunk, "offset": offset}
                        moves.append((trace["id"], held, where))
            stream.flush()
            index.flush()
            # after the stream: a pointer a reader sees points at a record it can read
            for trace_id, held, where in moves:
                self._point(trace_id, held, "finished", **where)

        # Record serialization is heavy pure-Python work; keep it off the event loop.
        # Awaited (not fire-and-forget) so appends to one stream never interleave.
        await asyncio.to_thread(write)

    async def log_annotations(self, updates: list[dict[str, Any]]) -> None:
        """Append trace updates to this producer's annotation stream — one writer per
        stream, so producers never interleave."""
        if not updates:
            return

        def write() -> None:
            stream, index = self._stream(get_annotations_dir(self.output_dir) / (self.producer or "unknown"))
            # the scalars go to a sibling index so a reader can answer "which cohort,
            # what credit" without touching the token streams
            for update in updates:
                chunk, offset = stream.append(orjson.dumps(update, option=OPTS))
                index.write(orjson.dumps(update_index_row(update, chunk, offset), option=OPTS))
            stream.flush()
            index.flush()

        await asyncio.to_thread(write)

    async def finalize(self) -> None:
        for stream, index in self._streams.values():
            stream.close()
            index.close()
        self.logger.info(f"Finalized metrics at {self.path}")
