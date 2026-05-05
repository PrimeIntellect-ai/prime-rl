from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True)
class ModelSpec:
    model: str
    short_name: str
    family: str
    tp: int
    dp: int
    max_concurrency: int
    sampling: dict[str, Any] | None = None
    sampling_drop: tuple[str, ...] = ()
    vllm_extra: dict[str, Any] | None = None
    prompt_pack: str | None = None
    max_model_len: int | None = None
    max_completion_tokens: int | None = None
    requires_multinode: bool = False
    multinode_strategy: Literal["dp_ep", "tp"] = "dp_ep"
    launch_vllm_extra: dict[str, Any] | None = None
    launch_enforce_eager: bool | None = None
    launch_use_deep_gemm: bool = False
    launch_srun_network: str | None = None
    launch_srun_cpus_per_task: int | None = None
    blocked_reason: str | None = None


def slug(value: str) -> str:
    value = value.split("/")[-1]
    value = value.replace(".", "").replace("_", "-").lower()
    value = re.sub(r"[^a-z0-9-]+", "-", value)
    value = re.sub(r"-+", "-", value)
    return value.strip("-")


def expected_rollouts(num_examples: int, record_ids: list[str], rollouts_per_example: int) -> int:
    return (len(record_ids) if record_ids else num_examples) * rollouts_per_example


def artifact_complete(path: Path, expected_rollout_count: int) -> bool:
    summary_path = path / "summary.json"
    records_path = path / "records.jsonl"
    if not summary_path.exists() or not records_path.exists():
        return False
    try:
        summary = json.loads(summary_path.read_text())
    except json.JSONDecodeError:
        return False
    if "num_rollouts" not in summary or "error_rate" not in summary:
        return False
    try:
        num_rollouts = int(summary["num_rollouts"])
        error_rate = float(summary["error_rate"])
    except (TypeError, ValueError):
        return False
    return (
        num_rollouts == expected_rollout_count
        and error_rate == 0.0
        and _records_complete(records_path, expected_rollout_count)
    )


def _records_complete(records_path: Path, expected_rollout_count: int) -> bool:
    count = 0
    try:
        with records_path.open() as records_file:
            for line in records_file:
                if not line.strip():
                    return False
                json.loads(line)
                count += 1
                if count > expected_rollout_count:
                    return False
    except (OSError, json.JSONDecodeError):
        return False
    return count == expected_rollout_count


def blocked_summaries(specs: list[ModelSpec]) -> dict[str, dict[str, Any]]:
    return {
        spec.short_name: {"skipped": True, "blocked": True, "blocked_reason": spec.blocked_reason}
        for spec in specs
        if spec.blocked_reason is not None
    }


def filter_blocked_specs(
    specs: list[ModelSpec],
    *,
    explicitly_requested: bool,
    include_blocked: bool,
) -> tuple[list[ModelSpec], dict[str, dict[str, Any]]]:
    blocked = [spec for spec in specs if spec.blocked_reason is not None]
    if not blocked or include_blocked:
        return specs, {}

    if explicitly_requested:
        details = "\n".join(f"- {spec.short_name}: {spec.blocked_reason}" for spec in blocked)
        raise ValueError(f"Explicitly selected blocked model(s); rerun with --include-blocked to force:\n{details}")

    return [spec for spec in specs if spec.blocked_reason is None], blocked_summaries(blocked)


def select_specs(specs: list[ModelSpec], requested: set[str]) -> list[ModelSpec]:
    if not requested:
        return specs
    selected = []
    matched: set[str] = set()
    for spec in specs:
        aliases = {spec.model, spec.short_name, slug(spec.model)}
        hits = aliases & requested
        if hits:
            selected.append(spec)
            matched.update(hits)

    unmatched = requested - matched
    if unmatched:
        names = ", ".join(sorted(unmatched))
        raise ValueError(f"Unknown requested model(s): {names}")
    return selected
