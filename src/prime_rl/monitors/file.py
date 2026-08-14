from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, TextIO

from prime_rl.configs.monitors import FileMonitorConfig
from prime_rl.monitors.base import Monitor, sanitize


class FileMonitor(Monitor):
    """Logs metrics to a local ``metrics.jsonl`` file."""

    config: FileMonitorConfig
    file: TextIO | None = None

    def init(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / self.config.filename
        # Line-buffered append so a concurrently-running dashboard can tail the file.
        self.file = open(path, "a", buffering=1)  # noqa: SIM115
        self.logger.info(f"Logging metrics to {path}")

    def log(self, metrics: dict[str, Any], step: int) -> None:
        if self.file is None:
            return

        sanitized, dropped = sanitize(metrics)
        if dropped:
            self.logger.warning(
                f"Dropping {len(dropped)} non-finite value(s) from {self.config.filename}: {', '.join(dropped[:5])}"
            )

        row = {"step": step, "time": time.time(), **sanitized}
        self.file.write(json.dumps(row) + "\n")
