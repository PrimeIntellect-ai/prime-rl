from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO

import orjson

from prime_rl.configs.monitors import FileMonitorConfig
from prime_rl.monitors.base import Kind, Monitor, Subset
from prime_rl.monitors.file.traces import get_annotations_dir, get_index_path, get_trace_stream
from prime_rl.monitors.file.traces.index import index_row
from prime_rl.monitors.file.traces.update import update_index_row
from prime_rl.utils.pathing import get_file_monitor_dir
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
        # Line-buffered append so a concurrently-running dashboard can tail the file.
        self.file = open(self.path, "a", buffering=1)  # noqa: SIM115
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

    async def log_episodes(self, episodes: list[vf.Episode], step: int, kind: Kind, subset: Subset) -> None:
        """Append each episode to the trace stream as it completes — every episode is
        serialized exactly once, in arrival order, whatever its kind, so an in-progress
        run can be tailed. Episode-level failures are preserved even when no trace was
        produced. The shipped cohort writes no second copy: what it learns arrives as
        annotations that readers fold back onto the stream."""
        if subset == "effective":
            return

        def write() -> None:
            path = get_trace_stream(self.output_dir)
            path.parent.mkdir(parents=True, exist_ok=True)
            # An index row goes out with every episode: summarising the record here,
            # while it is already in hand, saves every reader from parsing a stream
            # that outgrows memory long before the run does.
            with open(path, "ab") as f, open(get_index_path(path), "ab") as index:
                offset = f.tell()
                for episode in episodes:
                    record = episode.to_record()
                    line = orjson.dumps(record, default=str, option=OPTS)
                    f.write(line)
                    self._logged += 1
                    index.write(orjson.dumps(index_row(self._logged, record, offset), default=str, option=OPTS))
                    offset += len(line)

        # Record serialization is heavy pure-Python work; keep it off the event loop.
        # Awaited (not fire-and-forget) so appends to one file never interleave.
        await asyncio.to_thread(write)

    async def log_annotations(self, updates: list[dict[str, Any]]) -> None:
        """Append trace updates to this producer's annotation file — one writer per
        file, so producers never interleave."""
        if not updates:
            return

        def write() -> None:
            path = get_annotations_dir(self.output_dir) / f"{self.producer or 'unknown'}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            # the scalars go to a sibling index so a reader can answer "which cohort,
            # what credit" without touching the token streams
            with open(path, "ab") as f, open(get_index_path(path), "ab") as index:
                offset = f.tell()
                for update in updates:
                    line = orjson.dumps(update, option=OPTS)
                    f.write(line)
                    index.write(orjson.dumps(update_index_row(update, offset), option=OPTS))
                    offset += len(line)

        await asyncio.to_thread(write)

    async def finalize(self) -> None:
        self.logger.info(f"Finalized metrics at {self.path}")
