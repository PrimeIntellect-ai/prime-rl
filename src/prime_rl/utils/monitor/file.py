from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from prime_rl.configs.shared import FileMonitorConfig
from prime_rl.utils.config import BaseConfig
from prime_rl.utils.logger import get_logger
from prime_rl.utils.monitor.base import Monitor, drop_non_finite_json_values

if TYPE_CHECKING:
    from prime_rl.orchestrator.types import Rollout


class FileMonitor(Monitor):
    """Appends logged metrics to a local ``metrics.jsonl`` file.

    A self-hosted, dependency-free mirror of what ``WandbMonitor.log`` sees: one
    JSON object per line, ``{"step": step, "time": <wall>, **metrics}``. The file is
    flushed after every write so an in-progress run can be read (e.g. to build a
    static dashboard snapshot mid-run). Only rank 0 writes. Samples and
    distributions are not persisted here (scalars only).
    """

    def __init__(
        self,
        config: FileMonitorConfig | None,
        output_dir: Path | None = None,
        run_config: BaseConfig | None = None,
        keep_full_history: bool = True,
    ):
        self.config = config
        self.logger = get_logger()
        self.history: list[dict[str, Any]] = []
        self._keep_full_history = keep_full_history
        self.output_dir = output_dir
        self._file = None

        rank = int(os.environ.get("RANK", os.environ.get("DP_RANK", "0")))
        self.enabled = self.config is not None
        self.is_master = rank == 0
        if not self.enabled or not self.is_master:
            if not self.is_master:
                self.logger.warning(f"Skipping {self.__class__.__name__} initialization from non-master rank ({rank})")
            return

        if output_dir is None:
            self.logger.warning("FileMonitor requires an output_dir; disabling.")
            self.enabled = False
            return

        output_dir.mkdir(parents=True, exist_ok=True)
        self._path = output_dir / config.filename
        # Line-buffered append so a concurrently-running dashboard can tail the file.
        self._file = open(self._path, "a", buffering=1)  # noqa: SIM115
        self.logger.info(f"Logging metrics to {self._path}")

    def log(self, metrics: dict[str, Any], step: int) -> None:
        if self._keep_full_history:
            self.history.append(metrics)
        else:
            self.history = [metrics]
        if not self.is_master or not self.enabled or self._file is None:
            return

        dropped_paths: list[str] = []
        sanitized = drop_non_finite_json_values(metrics, dropped_paths)
        if dropped_paths:
            preview = ", ".join(dropped_paths[:5])
            suffix = " ..." if len(dropped_paths) > 5 else ""
            self.logger.debug(
                f"Dropping {len(dropped_paths)} non-finite value(s) from metrics.jsonl: {preview}{suffix}"
            )

        row = {"step": step, "time": time.time(), **sanitized}
        self._file.write(json.dumps(row))
        self._file.write("\n")

    def log_samples(self, rollouts: list[Rollout], step: int) -> None:
        pass

    def log_eval_samples(self, rollouts: list[Rollout], env_name: str, step: int) -> None:
        pass

    def log_distributions(self, distributions: dict[str, list[float]], step: int) -> None:
        pass

    def save_final_summary(self, filename: str = "final_summary.json") -> None:
        if not self.is_master or not self.enabled or self.output_dir is None:
            return
        summary = self.history[-1] if self.history else {}
        dropped_paths: list[str] = []
        sanitized = drop_non_finite_json_values(summary, dropped_paths)
        with open(self.output_dir / filename, "w") as f:
            json.dump(sanitized, f)

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
