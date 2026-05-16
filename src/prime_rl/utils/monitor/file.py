"""Monitor that streams step-indexed metrics to a local JSONL file.

Used by the sweep controller to tail intermediate metrics while a trial is
running (Phase 5b pruning) and to read the final objective from a single
canonical location (Phase 4 early stopping). Activation is opt-in: the sweep
launcher sets ``PRIME_RL_SWEEP_METRICS_JSONL`` to the destination path, the
trial subprocess inherits it, and ``setup_monitor`` adds this monitor only
when the env var is present. Non-sweep runs see no behavioral change.

Each ``log(metrics, step)`` call appends one JSON line of the form
``{"step": <int>, ...metrics}``. The file is opened in append mode and
flushed after every line so a polling reader sees writes promptly. Only the
master rank writes to avoid interleaved lines from different ranks.
"""

import json
import math
import operator
import os
from pathlib import Path
from typing import Any

import verifiers as vf

from prime_rl.utils.logger import get_logger
from prime_rl.utils.monitor.base import Monitor


def _coerce_finite(value: Any) -> Any:
    """Replace non-finite floats with None so JSONL stays parseable.

    json.dumps writes NaN/Infinity as non-standard tokens; the sweep reader
    treats None as a missing value, which matches its existing behavior for
    absent keys and keeps the sidecar consumable by strict JSON readers.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _coerce_finite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_coerce_finite(v) for v in value]
    if isinstance(value, tuple):
        return [_coerce_finite(v) for v in value]
    return value


def _coerce_step(step: int) -> int:
    if isinstance(step, bool):
        raise TypeError("FileMonitor step must be an integer, not bool")
    try:
        value = operator.index(step)
    except TypeError as exc:
        raise TypeError("FileMonitor step must be an integer") from exc
    if value < 0:
        raise ValueError("FileMonitor step must be non-negative")
    return value


class FileMonitor(Monitor):
    """Append step-indexed metrics to ``output_path`` as one JSON line per call."""

    required: bool = True

    def __init__(self, output_path: Path, keep_full_history: bool = True):
        self.output_path = Path(output_path)
        self.history: list[dict[str, Any]] = []
        self._keep_full_history = keep_full_history
        self.logger = get_logger()

        rank = int(os.environ.get("RANK", os.environ.get("DP_RANK", "0")))
        self.is_master = rank == 0

        if self.is_master:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(self.output_path, "a", buffering=1, encoding="utf-8")
        else:
            self._file = None

    def log(self, metrics: dict[str, Any], step: int) -> None:
        step = _coerce_step(step)
        if self._keep_full_history:
            self.history.append(metrics)
        else:
            self.history = [metrics]
        if self._file is None:
            return
        record = {**_coerce_finite(metrics), "step": step}
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()

    def log_samples(self, rollouts: list[vf.RolloutOutput], step: int) -> None:
        # Per-step metrics are the only signal pruning needs; rollout text and
        # eval rollouts live in the higher-volume cloud monitors.
        return

    def log_eval_samples(self, rollouts: list[vf.RolloutOutput], env_name: str, step: int) -> None:
        return

    def log_distributions(self, distributions: dict[str, list[float]], step: int) -> None:
        return

    def save_final_summary(self, filename: str = "final_summary.json") -> None:
        # The last line of metrics.jsonl IS the final summary for sweep
        # purposes; cloud monitors still own final_summary.json.
        return

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
