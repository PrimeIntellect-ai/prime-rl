from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TOKEN_EXPORT_METRIC_FIELDS = ("mismatch_kl", "entropy")


@dataclass
class TokenExportMetricsResult:
    step: int
    metrics: dict[str, float | int]


@dataclass
class _Stats:
    total: float = 0.0
    count: int = 0
    maximum: float = -math.inf

    def add(self, value: Any) -> None:
        if not isinstance(value, int | float):
            return
        value = float(value)
        if not math.isfinite(value):
            return
        self.total += value
        self.count += 1
        self.maximum = max(self.maximum, value)

    def as_metrics(self, prefix: str) -> dict[str, float]:
        if self.count == 0:
            return {}
        return {
            f"{prefix}/mean": self.total / self.count,
            f"{prefix}/max": self.maximum,
        }


def collect_next_token_export_metrics(
    output_dir: Path,
    *,
    last_logged_step: int,
    max_step: int,
) -> TokenExportMetricsResult | None:
    stable_steps = [step for step in _stable_token_export_steps(output_dir) if last_logged_step < step <= max_step]
    if not stable_steps:
        return None

    step = min(stable_steps)
    metrics = collect_token_export_metrics(output_dir, step)
    return TokenExportMetricsResult(step=step, metrics=metrics)


def collect_token_export_metrics(output_dir: Path, step: int) -> dict[str, float | int]:
    step_dir = output_dir / "token_exports" / f"step_{step}"
    if not (step_dir / "STABLE").exists():
        return {}

    stats_by_key: dict[str, _Stats] = {}
    sequence_count = 0

    for export_file in sorted(step_dir.glob("rank_*.jsonl")):
        with export_file.open(encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                record = json.loads(line)
                sequence_count += 1
                for field in TOKEN_EXPORT_METRIC_FIELDS:
                    for value in _loss_token_values(record, field):
                        stats_by_key.setdefault(field, _Stats()).add(value)

    metrics: dict[str, float | int] = {}
    if sequence_count == 0:
        return metrics
    for key, stats in sorted(stats_by_key.items()):
        metrics.update(stats.as_metrics(key))
    return metrics


def _stable_token_export_steps(output_dir: Path) -> list[int]:
    token_exports_dir = output_dir / "token_exports"
    if not token_exports_dir.exists():
        return []

    steps = []
    for step_dir in token_exports_dir.glob("step_*"):
        if not (step_dir / "STABLE").exists():
            continue
        try:
            steps.append(int(step_dir.name.removeprefix("step_")))
        except ValueError:
            continue
    return sorted(steps)


def _loss_token_values(record: dict[str, Any], field: str) -> list[Any]:
    loss_mask = record.get("loss_mask", [])
    values = record.get(field, [])
    if len(loss_mask) != len(values):
        raise ValueError(
            f"Token export record has mismatched loss_mask and {field} lengths: {len(loss_mask)} != {len(values)}"
        )
    return [value for value, keep in zip(values, loss_mask) if keep]
