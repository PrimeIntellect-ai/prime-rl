#!/usr/bin/env python3
"""Compare hosted and local repeated-rollout General Agent screens."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--hosted", type=Path)
    parser.add_argument("--local", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--max-preferred-truncation-rate", type=float, default=0.25)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def reward(row: dict[str, Any]) -> float:
    return float((row.get("rewards") or {}).get("solved", 0.0))


def is_truncated(row: dict[str, Any]) -> bool:
    if row.get("stop_condition") in {"timeout", "harness_timeout", "max_tokens"}:
        return True
    return any(node.get("finish_reason") == "length" for node in row.get("nodes") or [])


def start_time(row: dict[str, Any]) -> float:
    return float((row.get("timing") or {}).get("start", 0.0))


def group_outcome(values: list[float]) -> str:
    unique = {round(value, 12) for value in values}
    if unique == {0.0}:
        return "all_fail"
    if unique == {1.0}:
        return "all_pass"
    if len(unique) == 1:
        return "uniform_partial"
    return "mixed"


def summarize_backend(
    rows: list[dict[str, Any]],
    manifest: dict[str, Any],
    max_preferred_truncation_rate: float = 0.25,
) -> dict[str, Any]:
    expected = int(manifest["rollouts_per_task"])
    group_size = int(manifest["group_size"])
    task_meta = {entry["task"]: entry for entry in manifest["tasks"]}
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    unknown_tasks: set[str] = set()
    for row in rows:
        name = row["task"]["name"]
        by_task[name].append(row)
        if name not in task_meta:
            unknown_tasks.add(name)

    task_results = []
    for name, meta in task_meta.items():
        samples = sorted(by_task.get(name, []), key=start_time)
        values = [reward(row) for row in samples]
        groups = [values[i : i + group_size] for i in range(0, len(values), group_size)]
        complete_groups = [group for group in groups if len(group) == group_size]
        outcomes = Counter(group_outcome(group) for group in complete_groups)
        errors = sum(bool(row.get("errors")) for row in samples)
        truncations = sum(is_truncated(row) for row in samples)
        truncation_rate = truncations / len(samples) if samples else None
        success_count = sum(value >= 1.0 for value in values)
        mixed_groups = outcomes["mixed"]
        eligible = (
            len(samples) == expected
            and errors == 0
            and len(complete_groups) == expected // group_size
            and mixed_groups >= 2
        )
        task_results.append(
            {
                **meta,
                "rollouts": len(samples),
                "success_count": success_count,
                "pass_rate": success_count / len(samples) if samples else None,
                "reward_mean": sum(values) / len(values) if values else None,
                "reward_histogram": dict(sorted(Counter(values).items())),
                "complete_groups": len(complete_groups),
                "all_fail_groups": outcomes["all_fail"],
                "mixed_groups": mixed_groups,
                "all_pass_groups": outcomes["all_pass"],
                "uniform_partial_groups": outcomes["uniform_partial"],
                "group_outcomes": [group_outcome(group) for group in complete_groups],
                "error_rollouts": errors,
                "truncated_rollouts": truncations,
                "truncation_rate": truncation_rate,
                "eligible_for_group_relative_rl": eligible,
                "preferred_for_training": (
                    eligible
                    and truncation_rate is not None
                    and truncation_rate <= max_preferred_truncation_rate
                ),
            }
        )

    band_results = []
    for band in [entry["name"] for entry in manifest["bands"]]:
        members = [entry for entry in task_results if entry["band"] == band]
        rollouts = sum(entry["rollouts"] for entry in members)
        successes = sum(entry["success_count"] for entry in members)
        band_results.append(
            {
                "band": band,
                "tasks": len(members),
                "rollouts": rollouts,
                "pass_rate": successes / rollouts if rollouts else None,
                "all_fail_groups": sum(entry["all_fail_groups"] for entry in members),
                "mixed_groups": sum(entry["mixed_groups"] for entry in members),
                "all_pass_groups": sum(entry["all_pass_groups"] for entry in members),
                "uniform_partial_groups": sum(
                    entry["uniform_partial_groups"] for entry in members
                ),
                "eligible_tasks": sum(entry["eligible_for_group_relative_rl"] for entry in members),
                "preferred_tasks": sum(entry["preferred_for_training"] for entry in members),
                "error_rollouts": sum(entry["error_rollouts"] for entry in members),
                "truncated_rollouts": sum(entry["truncated_rollouts"] for entry in members),
            }
        )

    expected_total = expected * len(task_meta)
    return {
        "rollouts": len(rows),
        "expected_rollouts": expected_total,
        "complete": len(rows) == expected_total
        and not unknown_tasks
        and all(entry["rollouts"] == expected for entry in task_results),
        "unknown_tasks": sorted(unknown_tasks),
        "eligible_tasks": [
            entry["task"] for entry in task_results if entry["eligible_for_group_relative_rl"]
        ],
        "preferred_tasks": [entry["task"] for entry in task_results if entry["preferred_for_training"]],
        "bands": band_results,
        "tasks": task_results,
    }


def compare_backends(backends: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    if set(backends) != {"hosted", "local"}:
        return []
    hosted = {entry["task"]: entry for entry in backends["hosted"]["tasks"]}
    local = {entry["task"]: entry for entry in backends["local"]["tasks"]}
    comparisons = []
    for task in hosted.keys() & local.keys():
        hosted_rate = hosted[task]["pass_rate"]
        local_rate = local[task]["pass_rate"]
        comparisons.append(
            {
                "task": task,
                "band": hosted[task]["band"],
                "hosted_pass_rate": hosted_rate,
                "local_pass_rate": local_rate,
                "local_minus_hosted_pass_rate": (
                    local_rate - hosted_rate
                    if hosted_rate is not None and local_rate is not None
                    else None
                ),
                "hosted_mixed_groups": hosted[task]["mixed_groups"],
                "local_mixed_groups": local[task]["mixed_groups"],
            }
        )
    return sorted(comparisons, key=lambda entry: entry["task"])


def write_csv(path: Path, backends: dict[str, dict[str, Any]]) -> None:
    fields = [
        "backend",
        "task",
        "band",
        "tier",
        "metadata_pass_rate",
        "rollouts",
        "success_count",
        "pass_rate",
        "all_fail_groups",
        "mixed_groups",
        "all_pass_groups",
        "uniform_partial_groups",
        "error_rollouts",
        "truncated_rollouts",
        "eligible_for_group_relative_rl",
        "preferred_for_training",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for backend, summary in backends.items():
            for task in summary["tasks"]:
                writer.writerow({"backend": backend, **{field: task.get(field) for field in fields[1:]}})


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text())
    paths = {"hosted": args.hosted, "local": args.local}
    backends = {
        name: summarize_backend(
            read_jsonl(path), manifest, args.max_preferred_truncation_rate
        )
        for name, path in paths.items()
        if path is not None
    }
    payload = {
        "manifest": str(args.manifest),
        "group_definition": (
            "For each task, rollouts are ordered by timing.start and split into consecutive "
            f"groups of {manifest['group_size']}."
        ),
        "eligibility": (
            "Exact rollout count, zero errors, all groups complete, and at least two observed "
            "mixed-reward groups. Truncation is reported separately and only gates the preferred "
            f"training subset at <= {args.max_preferred_truncation_rate:.1%}."
        ),
        "backends": backends,
        "comparison": compare_backends(backends),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if args.csv:
        write_csv(args.csv, backends)


if __name__ == "__main__":
    main()
