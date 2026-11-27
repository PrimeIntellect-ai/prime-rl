from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO, TextIO

import numpy as np
import orjson
from verifiers.v1 import decode_ndarray

from prime_rl.configs.monitors import FileMonitorConfig
from prime_rl.monitors.base import Kind, Monitor
from prime_rl.monitors.file.traces import get_annotations_dir, get_index_path, get_trace_stream
from prime_rl.monitors.file.traces.chunks import ChunkedJsonl
from prime_rl.monitors.file.traces.index import index_row
from prime_rl.monitors.file.traces.live import get_live_dir, get_pending_dir
from prime_rl.monitors.file.traces.update import STREAM_FIELDS, update_index_row
from prime_rl.utils.pathing import get_eval_plan_path, get_file_monitor_dir
from prime_rl.utils.utils import sanitize

if TYPE_CHECKING:
    from prime_rl.env.heads import EpisodeHead, EpisodeRecord

OPTS = orjson.OPT_APPEND_NEWLINE | orjson.OPT_SERIALIZE_NUMPY

LIVE_NODE_EXCLUDE = frozenset({"routed_experts", "sampler_support"})
"""Node fields that stay on the wire but out of live trace files: expensive per-token information
only the trainer needs."""


def _live_value(value: Any) -> Any:
    """A node field for a live trace file: per-token streams arrive as raw-bytes `__nd__` dicts
    (a dict of them for `loss_weights`), which orjson writes as lists once decoded."""
    if isinstance(value, dict):
        if value.get("__nd__"):
            return decode_ndarray(value)
        return {key: _live_value(item) for key, item in value.items()}
    return value


def _group_id(episode: EpisodeHead) -> str | None:
    return episode.group.id if episode.group is not None else None


def _forget(written: set[str] | None, key: str) -> bool:
    """Whether a file for ``key`` may exist, dropping it from ``written``: always when
    nothing is tracked (every trace is written), else only for a file this process made."""
    if written is None:
        return True
    if key in written:
        written.discard(key)
        return True
    return False


def in_trace_sample(group_id: str, fraction: float) -> bool:
    """Whether a group's traces are written at ``fraction``: a stable hash of its id, so the
    episode, live and annotation writers - and every process - make the same call."""
    if fraction >= 1:
        return True
    digest = hashlib.blake2b(group_id.encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") < fraction * 2**64


def _write_streams(branch: dict[str, Any], value_stream: ChunkedJsonl) -> dict[str, Any]:
    """The branch with each per-token stream replaced by its ``{bin, offset, count}`` ref,
    the float32 values appended to ``value_stream``."""
    line = dict(branch)
    for field in STREAM_FIELDS:
        values = line.get(field)
        if values is None:
            continue
        count = len(values)
        bin_chunk, offset = value_stream.append(np.ascontiguousarray(values, dtype=np.float32).tobytes())
        line[field] = {"bin": bin_chunk, "offset": offset, "count": count}
    return line


class FileMonitor(Monitor):
    """Logs metrics and episodes to local JSONL files."""

    config: FileMonitorConfig
    file: TextIO

    async def init(self, output_dir: Path, producer: str | None = None) -> None:
        self.output_dir = output_dir
        self.producer = producer
        index = get_index_path(get_trace_stream(output_dir))
        # a relaunch appends to the stream it finds, so the numbering carries on
        self._logged = sum(1 for _ in index.open("rb")) if index.is_file() else 0
        self.path = get_file_monitor_dir(output_dir) / self.config.path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._live_cleared = False
        # Line-buffered append so a concurrently-running dashboard can tail the file.
        self.file = open(self.path, "a", buffering=1)  # noqa: SIM115
        self._streams: dict[Path, tuple[ChunkedJsonl, BinaryIO]] = {}
        self._value_streams: dict[Path, ChunkedJsonl] = {}
        # Under a trace sample, what this process wrote: annotations follow its episodes, and
        # only files it created are removed. Unused (None) when every trace is written.
        sampled = self.config.trace_sample < 1
        self._written_traces: set[str] | None = set() if sampled else None
        self._live_traces: set[str] | None = set() if sampled else None
        self._pending: set[str] | None = set() if sampled else None
        self._logs_episodes = False
        self.logger.info(f"Logging metrics and episodes to the local filesystem ({output_dir})")

    def _samples(self, kind: str | None, group_id: Any) -> bool:
        """Whether a train group's traces are written; eval work and groupless episodes always are."""
        if kind != "train" or group_id is None:
            return True
        return in_trace_sample(str(group_id), self.config.trace_sample)

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
        carrying the dispatch identity); a finished or discarded trace's file goes away.
        A dispatched episode holds a placeholder under ``traces/live/pending/`` until its
        first trace streams."""
        live_dir = get_live_dir(self.output_dir)
        pending_dir = get_pending_dir(self.output_dir)
        # A previous attempt's live traces are stale: nothing streams into them again. Only
        # the process that streams clears them - every process of a run shares this monitor.
        clear = not self._live_cleared
        self._live_cleared = True
        if not self.config.live:
            # nothing streams to disk; only a previous attempt's live traces are cleared
            if clear:
                await asyncio.to_thread(lambda: shutil.rmtree(live_dir, ignore_errors=True))
            return

        def write() -> None:
            if clear:
                shutil.rmtree(live_dir, ignore_errors=True)
            pending_dir.mkdir(parents=True, exist_ok=True)
            appends: dict[Path, list[bytes]] = {}
            for event in events:
                if "done" in event:
                    appends.pop(live_dir / f"{event['done']}.jsonl", None)
                    if _forget(self._live_traces, event["done"]):
                        (live_dir / f"{event['done']}.jsonl").unlink(missing_ok=True)
                    continue
                if "pending" in event:
                    dispatch = event["dispatch"]
                    if not self._samples(dispatch.get("kind"), dispatch.get("group")):
                        continue
                    if self._pending is not None:
                        self._pending.add(event["pending"])
                    path = pending_dir / f"{event['pending']}.json"
                    tmp = path.with_suffix(".json.tmp")
                    tmp.write_bytes(orjson.dumps(dispatch))
                    tmp.replace(path)
                    continue
                if "dispatched" in event:
                    if _forget(self._pending, event["dispatched"]):
                        (pending_dir / f"{event['dispatched']}.json").unlink(missing_ok=True)
                    continue
                delta = event["delta"]
                dispatch = event.get("dispatch") or {}
                if not self._samples(dispatch.get("kind"), dispatch.get("group")):
                    continue
                path = live_dir / f"{delta['trace']}.jsonl"
                if delta.get("discard"):
                    appends.pop(path, None)
                    if _forget(self._live_traces, delta["trace"]):
                        path.unlink(missing_ok=True)
                    continue
                if self._live_traces is not None:
                    self._live_traces.add(delta["trace"])
                line = dict(delta)
                if "nodes" in line:
                    line["nodes"] = [
                        {key: _live_value(value) for key, value in node.items() if key not in LIVE_NODE_EXCLUDE}
                        for node in line["nodes"]
                    ]
                if "open" in delta:
                    line["dispatch"] = event["dispatch"]
                # env-defined dicts (`info`, `metrics`, task data) may be keyed by int
                appends.setdefault(path, []).append(
                    orjson.dumps(line, default=str, option=OPTS | orjson.OPT_NON_STR_KEYS)
                )
            # one open per file per batch: a trace's deltas of the last half second land together
            for path, lines in appends.items():
                with path.open("ab") as f:
                    f.write(b"".join(line + b"\n" for line in lines))

        await asyncio.to_thread(write)

    def _stream(self, directory: Path) -> tuple[ChunkedJsonl, BinaryIO]:
        """A stream and its index, opened on first use and kept open. Writers flush the
        stream before the index: a row a reader sees points at a record it can read."""
        if directory not in self._streams:
            stream = ChunkedJsonl(directory, self.config.chunk_bytes, self.config.compress)
            self._streams[directory] = (stream, open(get_index_path(directory), "ab"))
        return self._streams[directory]

    def _value_stream(self, directory: Path) -> ChunkedJsonl:
        """The ``.bin`` chunks holding an annotation stream's float32 values, opened on
        first use and kept open."""
        if directory not in self._value_streams:
            self._value_streams[directory] = ChunkedJsonl(
                directory, self.config.chunk_bytes, self.config.compress, suffix=".bin"
            )
        return self._value_streams[directory]

    def wants_record(self, kind: Kind, group_id: str | None) -> bool:
        # Only a producer that logs episodes asks, before any of its annotations: from here it
        # annotates only the traces it writes, also before its first sampled record lands.
        self._logs_episodes = True
        return self._samples(kind, group_id)

    def wants_live(self, kind: Kind, group_id: str | None) -> bool:
        return self.config.live and self._samples(kind, group_id)

    async def log_records(self, records: list[tuple[EpisodeHead, EpisodeRecord]], step: int, kind: Kind) -> None:
        """Append each episode's record to the trace stream as it completes — every episode is
        written exactly once, in arrival order, whatever its kind, so an in-progress run can
        be tailed. Episode-level failures are preserved even when no trace was produced. The
        shipped cohort writes no second copy: what it learns arrives as annotations that
        readers fold back onto the stream."""
        self._logs_episodes = True
        if self._written_traces is not None:
            records = [(episode, record) for episode, record in records if self._samples(kind, _group_id(episode))]
            self._written_traces.update(trace.id for episode, _ in records for trace in episode.traces)
        if not records:
            return

        def write() -> None:
            stream, index = self._stream(get_trace_stream(self.output_dir))
            # An index row goes out with every episode, so no reader has to parse a stream
            # that outgrows memory long before the run does.
            for _, record in records:
                chunk, offset = stream.append(record.json, b"\n")
                self._logged += 1
                row = index_row(self._logged, record.index, chunk, offset)
                index.write(orjson.dumps(row, default=str, option=OPTS))
            stream.flush()
            index.flush()

        # Awaited (not fire-and-forget) so appends to one stream never interleave.
        await asyncio.to_thread(write)

    async def log_annotations(self, updates: list[dict[str, Any]]) -> None:
        """Append trace updates to this producer's annotation stream — one writer per
        stream, so producers never interleave. Under a trace sample, a producer that logs
        episodes annotates only the traces it wrote."""
        if self._written_traces is not None and self._logs_episodes:
            updates = [update for update in updates if update["trace_id"] in self._written_traces]
        if not updates:
            return

        def write() -> None:
            directory = get_annotations_dir(self.output_dir) / (self.producer or "unknown")
            stream, index = self._stream(directory)
            value_stream = self._value_stream(directory)
            # the scalars go to a sibling index so a reader can answer "which cohort,
            # what credit" without touching the token streams
            for update in updates:
                line = dict(update)
                line["branches"] = [_write_streams(branch, value_stream) for branch in update.get("branches") or []]
                value_stream.flush()
                chunk, offset = stream.append(orjson.dumps(line, option=OPTS))
                index.write(orjson.dumps(update_index_row(update, chunk, offset), option=OPTS))
            stream.flush()
            index.flush()

        await asyncio.to_thread(write)

    async def finalize(self, paused: bool = False) -> None:
        for stream, index in self._streams.values():
            stream.close()
            index.close()
        for value_stream in self._value_streams.values():
            value_stream.close()
        self.logger.info(f"Finalized metrics at {self.path}")
