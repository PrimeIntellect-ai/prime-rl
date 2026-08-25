from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO

import orjson

from prime_rl.configs.monitors import FileMonitorConfig
from prime_rl.monitors.base import Kind, Monitor, Subset
from prime_rl.utils.utils import sanitize

if TYPE_CHECKING:
    import verifiers.v1 as vf


class FileMonitor(Monitor):
    """Logs metrics and episodes to local JSONL files."""

    config: FileMonitorConfig
    file: TextIO

    async def init(self, output_dir: Path, producer: str | None = None) -> None:
        self.output_dir = output_dir
        self.producer = producer
        self.path = output_dir / self.config.path
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
        """Append the cohort to its per-step episode file. ``all`` grows one
        episode at a time as they complete, ``effective`` one batch at a time on finalize,
        so an in-progress run can be inspected live. Episode-level failures are
        preserved even when no trace was produced."""

        def write() -> None:
            path = self.output_dir / "rollouts" / f"step_{step}" / kind / subset / "traces.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            opts = orjson.OPT_APPEND_NEWLINE | orjson.OPT_SERIALIZE_NUMPY
            with open(path, "ab") as f:
                for episode in episodes:
                    f.write(orjson.dumps(episode.to_record(), default=str, option=opts))

        # Record serialization is heavy pure-Python work; keep it off the event loop.
        # Awaited (not fire-and-forget) so appends to one file never interleave.
        await asyncio.to_thread(write)

    async def finalize(self) -> None:
        self.logger.info(f"Finalized metrics at {self.path}")
