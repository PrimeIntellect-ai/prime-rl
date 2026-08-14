from __future__ import annotations

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
    """Logs metrics and episodes to a local JSONL files."""

    config: FileMonitorConfig
    file: TextIO | None = None

    def init(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        path = output_dir / self.config.path
        path.parent.mkdir(parents=True, exist_ok=True)
        # Line-buffered append so a concurrently-running dashboard can tail the file.
        self.file = open(path, "a", buffering=1)  # noqa: SIM115
        self.logger.info(f"Logging metrics to {path}")

    def log_metrics(self, metrics: dict[str, Any], step: int) -> None:
        if self.file is None:
            return

        sanitized, dropped = sanitize(metrics)
        if dropped:
            self.logger.warning(
                f"Dropping {len(dropped)} non-finite value(s) from {self.config.path}: {', '.join(dropped[:5])}"
            )

        row = {"step": step, "time": time.time(), **sanitized}
        self.file.write(json.dumps(row) + "\n")

    def log_episodes(self, episodes: list[vf.Episode], step: int, kind: Kind, subset: Subset) -> None:
        """Append the cohort's traces to its per-step trace file. ``all`` grows one
        episode at a time as they complete, ``effective`` one batch at a time on finalize,
        so an in-progress run's traces can be inspected live."""
        path = self.output_dir / "rollouts" / f"step_{step}" / kind / subset / "traces.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        opts = orjson.OPT_APPEND_NEWLINE | orjson.OPT_SERIALIZE_NUMPY
        with open(path, "ab") as f:
            for episode in episodes:
                for trace in episode.traces:
                    f.write(orjson.dumps(trace.to_record(), default=str, option=opts))
