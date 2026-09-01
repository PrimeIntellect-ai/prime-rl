from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO

import orjson

from prime_rl.configs.monitors import FileMonitorConfig
from prime_rl.monitors.base import Kind, Monitor, Subset
from prime_rl.utils.pathing import get_file_monitor_dir
from prime_rl.utils.trace_updates import make_update
from prime_rl.utils.utils import sanitize

if TYPE_CHECKING:
    import verifiers.v1 as vf

OPTS = orjson.OPT_APPEND_NEWLINE | orjson.OPT_SERIALIZE_NUMPY


def _ship_step(episode: vf.Episode, step: int) -> int:
    """The step an episode's cohort ties to: the orchestrator step whose batch shipped
    it for train work, the policy version it measured for eval work."""
    work = getattr(episode.run, "work", None)
    if work is not None and work.type == "eval" and work.policy is not None:
        return work.policy.start
    return step


def _effective_update(trace: vf.Trace, step: int, now: float) -> dict[str, Any]:
    """What the ship-time cohort adds over the arrival record: membership, the step
    the cohort ties to, the scalar advantage, and the per-token advantage streams."""
    info: dict[str, Any] = {"effective": True, "ship": {"step": step, "time": now}}
    if (advantage := trace.info.get("advantage")) is not None:
        info["advantage"] = advantage
    branches = {
        branch.index: {"advantages": advantages}
        for branch in trace.branches
        if (advantages := branch.advantages) is not None
    }
    return make_update(trace.id, info=info, branches=branches)


class FileMonitor(Monitor):
    """Logs metrics and episodes to local JSONL files."""

    config: FileMonitorConfig
    file: TextIO

    async def init(self, output_dir: Path, producer: str | None = None) -> None:
        self.output_dir = output_dir
        self.producer = producer
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
        """``all`` appends each episode to the trace stream as it completes — every
        episode is serialized exactly once, in arrival order, whatever its kind, so an
        in-progress run can be tailed. ``effective`` writes no second copy: it records
        what the ship-time cohort learned (see ``_effective_update``) as annotations
        that readers fold back onto the stream. Episode-level failures are preserved
        even when no trace was produced."""
        if subset == "effective":
            now = time.time()
            await self.log_annotations(
                [
                    _effective_update(trace, _ship_step(episode, step), now)
                    for episode in episodes
                    for trace in episode.traces
                ]
            )
            return

        def write() -> None:
            now = time.time()
            path = get_file_monitor_dir(self.output_dir) / "traces.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "ab") as f:
                for episode in episodes:
                    # Steps are facts about events, not about the trace: dispatch rides
                    # ``run.work``, this is when the orchestrator saw the episode come back.
                    for trace in episode.traces:
                        trace.info["arrival"] = {"step": step, "time": now}
                    f.write(orjson.dumps(episode.to_record(), default=str, option=OPTS))

        # Record serialization is heavy pure-Python work; keep it off the event loop.
        # Awaited (not fire-and-forget) so appends to one file never interleave.
        await asyncio.to_thread(write)

    async def log_annotations(self, updates: list[dict[str, Any]]) -> None:
        """Append trace updates to this producer's annotation file — one writer per
        file, so producers never interleave."""
        if not updates:
            return

        def write() -> None:
            path = get_file_monitor_dir(self.output_dir) / "annotations" / f"{self.producer or 'unknown'}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "ab") as f:
                for update in updates:
                    f.write(orjson.dumps(update, option=OPTS))

        await asyncio.to_thread(write)

    async def finalize(self) -> None:
        self.logger.info(f"Finalized metrics at {self.path}")
