from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, TextIO

from prime_rl.configs.shared import FileMonitorConfig
from prime_rl.monitors.base import Monitor, drop_non_finite_json_values
from prime_rl.utils.logger import get_logger


class FileMonitor(Monitor):
    """Appends logged metrics to a local ``metrics.jsonl`` file.

    A self-hosted, dependency-free mirror of what ``WandbMonitor.log`` sees: one
    JSON object per line, ``{"step": step, "time": <wall>, **metrics}``. The file is
    flushed after every write so an in-progress run can be read (e.g. to build a
    static dashboard snapshot mid-run). Scalars only.
    """

    def __init__(self, config: FileMonitorConfig, output_dir: Path):
        self.config = config
        self.output_dir = output_dir
        self.logger = get_logger()
        self._file: TextIO | None = None

    def init(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / self.config.filename
        # Line-buffered append so a concurrently-running dashboard can tail the file.
        self._file = open(path, "a", buffering=1)  # noqa: SIM115
        self.logger.info(f"Logging metrics to {path}")

    def log(self, metrics: dict[str, Any], step: int) -> None:
        if self._file is None:
            return

        dropped_paths: list[str] = []
        sanitized = drop_non_finite_json_values(metrics, dropped_paths)
        if dropped_paths:
            preview = ", ".join(dropped_paths[:5])
            suffix = " ..." if len(dropped_paths) > 5 else ""
            self.logger.debug(
                f"Dropping {len(dropped_paths)} non-finite value(s) from {self.config.filename}: {preview}{suffix}"
            )

        row = {"step": step, "time": time.time(), **sanitized}
        self._file.write(json.dumps(row) + "\n")
