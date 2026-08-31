from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO

import orjson

from prime_rl.configs.monitors import FileMonitorConfig
from prime_rl.monitors.base import Kind, Monitor, Subset
from prime_rl.utils.pathing import get_step_path, get_traces_dir
from prime_rl.utils.utils import sanitize

if TYPE_CHECKING:
    import verifiers.v1 as vf


def _effective_update(trace: vf.Trace, kind: Kind, step: int) -> dict[str, Any]:
    """The post-hoc facts the ship-time cohort adds over the arrival record, as a
    TraceUpdate: cohort membership, the scalar advantage, the per-token advantage
    streams, and (for train) the step the trace was trained at."""
    kind_info: dict[str, Any] = {"effective": True}
    if kind == "train":
        kind_info["trained_at_step"] = step
    info: dict[str, Any] = {kind: kind_info}
    if (advantage := trace.info.get("advantage")) is not None:
        info["advantage"] = advantage
    branches = [
        {"index": branch.index, "advantages": advantages}
        for branch in trace.branches
        if (advantages := branch.advantages) is not None
    ]
    return {"version": 1, "trace_id": trace.id, "info": info, "branches": branches}


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
        """``all`` appends each episode to ``traces/step_<n>/<kind>.jsonl`` as it completes,
        so an in-progress run can be inspected live; every episode is serialized exactly
        once, at arrival. ``effective`` writes no second copy: it appends TraceUpdate
        records to ``traces/step_<n>/annotations/orchestrator.jsonl`` next to each trace's
        arrival file, carrying what the ship-time cohort learned (see
        ``_effective_update``); readers fold them onto the arrival records. Episode-level
        failures are preserved even when no trace was produced."""

        def write() -> None:
            opts = orjson.OPT_APPEND_NEWLINE | orjson.OPT_SERIALIZE_NUMPY
            if subset == "all":
                path = get_step_path(get_traces_dir(self.output_dir), step) / f"{kind}.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "ab") as f:
                    for episode in episodes:
                        if kind == "train":
                            # Annotations key back to this file; the ship site reads the stamp
                            # to place them, since arrival step != ship step under lag.
                            for trace in episode.traces:
                                trace.info.setdefault("train", {})["logged_at_step"] = step
                        f.write(orjson.dumps(episode.to_record(), default=str, option=opts))
                return
            updates_by_step: dict[int, list[dict[str, Any]]] = {}
            for episode in episodes:
                for trace in episode.traces:
                    logged_step = step
                    if kind == "train":
                        logged_step = (trace.info.get("train") or {}).get("logged_at_step", step)
                    updates_by_step.setdefault(logged_step, []).append(_effective_update(trace, kind, step))
            for logged_step, updates in updates_by_step.items():
                path = (
                    get_step_path(get_traces_dir(self.output_dir), logged_step) / "annotations" / "orchestrator.jsonl"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "ab") as f:
                    for update in updates:
                        f.write(orjson.dumps(update, option=opts))

        # Record serialization is heavy pure-Python work; keep it off the event loop.
        # Awaited (not fire-and-forget) so appends to one file never interleave.
        await asyncio.to_thread(write)

    async def finalize(self) -> None:
        self.logger.info(f"Finalized metrics at {self.path}")
