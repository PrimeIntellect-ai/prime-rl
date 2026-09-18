from __future__ import annotations

import asyncio
import json
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO, TextIO

import orjson

from prime_rl.configs.monitors import FileMonitorConfig
from prime_rl.monitors.base import Kind, Monitor, Subset
from prime_rl.monitors.file.traces import get_annotations_dir, get_index_path, get_trace_stream
from prime_rl.monitors.file.traces.chunks import ChunkedJsonl
from prime_rl.monitors.file.traces.index import index_row
from prime_rl.monitors.file.traces.live import get_live_dir, get_pending_dir
from prime_rl.monitors.file.traces.update import update_index_row
from prime_rl.utils.pathing import get_eval_plan_path, get_file_monitor_dir, get_file_monitor_run_path
from prime_rl.utils.utils import sanitize

if TYPE_CHECKING:
    import verifiers.v1 as vf

OPTS = orjson.OPT_APPEND_NEWLINE | orjson.OPT_SERIALIZE_NUMPY


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
        if self.config.project is not None:
            # every process of a run writes the same project, so the last writer wins harmlessly
            run_path = get_file_monitor_run_path(output_dir)
            tmp = run_path.with_suffix(".json.tmp")
            tmp.write_bytes(orjson.dumps({"project": self.config.project}))
            tmp.replace(run_path)
        self._live_cleared = False
        # Line-buffered append so a concurrently-running dashboard can tail the file.
        self.file = open(self.path, "a", buffering=1)  # noqa: SIM115
        self._streams: dict[Path, tuple[ChunkedJsonl, BinaryIO]] = {}
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
        carrying the dispatch identity); a finished or discarded trace's file goes away.
        A dispatched episode holds a placeholder under ``traces/live/pending/`` until its
        first trace streams."""
        live_dir = get_live_dir(self.output_dir)
        pending_dir = get_pending_dir(self.output_dir)
        # A previous attempt's live traces are stale: nothing streams into them again. Only
        # the process that streams clears them - every process of a run shares this monitor.
        clear = not self._live_cleared
        self._live_cleared = True

        def write() -> None:
            if clear:
                shutil.rmtree(live_dir, ignore_errors=True)
            pending_dir.mkdir(parents=True, exist_ok=True)
            appends: dict[Path, list[bytes]] = {}
            for event in events:
                if "done" in event:
                    appends.pop(live_dir / f"{event['done']}.jsonl", None)
                    (live_dir / f"{event['done']}.jsonl").unlink(missing_ok=True)
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
                path = live_dir / f"{delta['trace']}.jsonl"
                if delta.get("discard"):
                    appends.pop(path, None)
                    path.unlink(missing_ok=True)
                    continue
                line = {**delta, "dispatch": event["dispatch"]} if "open" in delta else delta
                # deltas key semantic links by node index (int)
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
            for episode in episodes:
                record = episode.to_record(float_decimals=self.config.float_decimals)
                chunk, offset = stream.append(orjson.dumps(record, default=str, option=OPTS))
                self._logged += 1
                index.write(orjson.dumps(index_row(self._logged, record, chunk, offset), default=str, option=OPTS))
            stream.flush()
            index.flush()

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
