#!/usr/bin/env python3
"""Freeze a 100-task, train-disjoint General Agent evaluation panel."""

from __future__ import annotations

import hashlib
import json
import os
import random
import tomllib
from pathlib import Path

import tomli_w

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "opd-gap"
SOURCE_MANIFEST = CONFIG_DIR / "genagent-band000060-qwen35-opsd-hints-r41-manifest.json"
BASE_CONFIG = CONFIG_DIR / "posthoc" / "genagent-heldout-r28.toml"
OUTPUT_CONFIG = CONFIG_DIR / "posthoc" / "genagent-heldout100-r41.toml"
OUTPUT_MANIFEST = CONFIG_DIR / "posthoc" / "genagent-heldout100-r41-manifest.json"
SEED = 20260715
STRATUM_COUNTS = {
    "metadata_0.6_0.8": 41,
    "metadata_0.8_1.0": 41,
}


def metadata_pass_rate(metadata: dict) -> float | None:
    for entry in metadata.get("pass_rates", []):
        if entry.get("solver") == "local" and entry.get("model") == "openai/gpt-5-mini":
            return float(entry["value"])
    return None


def main() -> None:
    source = json.loads(SOURCE_MANIFEST.read_text())
    cache_default = Path.home() / ".cache" / "verifiers" / "general_agent_verified"
    cache_root = Path(os.environ.get("GENERAL_AGENT_CACHE_DIR", cache_default))
    task_root = cache_root / source["taskset_ref"] / "tasks"
    existing = list(source["heldout_tasks"])
    excluded = set(source["train_tasks"])
    excluded.update(source["gepa_train_tasks"])
    excluded.update(source["gepa_eval_tasks"])
    excluded.update(existing)

    original_entries: list[dict] = []
    drifted_original: list[dict] = []
    for name in existing:
        task_toml = task_root / name / "task.toml"
        gold_path = task_toml.parent / "gold.json"
        raw = task_toml.read_bytes()
        metadata = tomllib.loads(raw.decode()).get("metadata", {})
        if metadata.get("name") != name or not gold_path.is_file():
            drifted_original.append(
                {"directory": name, "metadata_name": metadata.get("name"), "reason": "identity_drift"}
            )
            continue
        original_entries.append(
            {
                "task": name,
                "tier": int(metadata["tier"]),
                "metadata_pass_rate": metadata_pass_rate(metadata),
                "task_toml_sha256": hashlib.sha256(raw).hexdigest(),
                "gold_sha256": hashlib.sha256(gold_path.read_bytes()).hexdigest(),
                "stratum": "original18",
            }
        )

    strata: dict[str, list[dict]] = {name: [] for name in STRATUM_COUNTS}
    for task_toml in sorted(task_root.glob("*/task.toml")):
        raw = task_toml.read_bytes()
        metadata = tomllib.loads(raw.decode()).get("metadata", {})
        name = task_toml.parent.name
        gold_path = task_toml.parent / "gold.json"
        rate = metadata_pass_rate(metadata)
        if (
            name in excluded
            or metadata.get("name") != name
            or metadata.get("tier") not in {3, 4}
            or rate is None
            or not gold_path.is_file()
        ):
            continue
        stratum = (
            "metadata_0.6_0.8"
            if 0.6 <= rate < 0.8
            else "metadata_0.8_1.0"
            if 0.8 <= rate <= 1.0
            else None
        )
        if stratum is None:
            continue
        strata[stratum].append(
            {
                "task": name,
                "tier": int(metadata["tier"]),
                "metadata_pass_rate": rate,
                "task_toml_sha256": hashlib.sha256(raw).hexdigest(),
                "gold_sha256": hashlib.sha256(gold_path.read_bytes()).hexdigest(),
                "stratum": stratum,
            }
        )

    rng = random.Random(SEED)
    selected: list[dict] = []
    for name, candidates in strata.items():
        count = STRATUM_COUNTS[name]
        rng.shuffle(candidates)
        if len(candidates) < count:
            raise RuntimeError(f"{name} has only {len(candidates)} candidates")
        selected.extend(candidates[:count])

    tasks = [row["task"] for row in original_entries] + [row["task"] for row in selected]
    if len(tasks) != 100 or len(set(tasks)) != 100:
        raise RuntimeError("heldout panel is not exactly 100 unique tasks")
    if set(tasks) & set(source["train_tasks"]):
        raise RuntimeError("heldout panel overlaps training")

    config = tomllib.loads(BASE_CONFIG.read_text())
    config["num_tasks"] = 100
    config["taskset"]["tasks"] = tasks
    config["taskset"]["min_pass_rate"] = 0.0
    config["taskset"]["max_pass_rate"] = 1.01
    config["taskset"]["shuffle_seed"] = SEED
    config["taskset"]["start_idx"] = 0
    config["taskset"]["limit"] = 100
    OUTPUT_CONFIG.write_text(tomli_w.dumps(config))

    manifest = {
        "schema_version": 1,
        "purpose": "fixed post-hoc General Agent evaluation for r41 OPSD/GRPO",
        "taskset_id": source["taskset_id"],
        "taskset_ref": source["taskset_ref"],
        "selection_seed": SEED,
        "task_count": 100,
        "preserved_original_heldout": [row["task"] for row in original_entries],
        "excluded_drifted_original_heldout": drifted_original,
        "additional_selection": STRATUM_COUNTS,
        "tasks": tasks,
        "task_entries": original_entries + selected,
        "additional_task_entries": selected,
        "train_overlap": [],
        "gepa_overlap": [],
        "sampling": {"rollouts_per_task": 1, "temperature": 1.0, "max_tokens": 4096},
        "reporting_rule": "report overall and separately for original18, metadata_0.6_0.8, metadata_0.8_1.0",
    }
    OUTPUT_MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {OUTPUT_CONFIG} and {OUTPUT_MANIFEST}")


if __name__ == "__main__":
    main()
