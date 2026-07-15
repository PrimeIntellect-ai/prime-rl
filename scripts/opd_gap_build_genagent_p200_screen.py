#!/usr/bin/env python3
"""Build the frozen, difficulty-stratified General Agent 200-task screen."""

from __future__ import annotations

import hashlib
import json
import random
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASKSET_REF = "a2b76f6ac3469f7f50171760c0d0dba38360edc4"
TASK_ROOT = Path.home() / ".cache/verifiers/general_agent" / TASKSET_REF / "tasks"
SEED = 20260715
PER_BAND = 50
BANDS = [
    ("very_hard", 0.0, 0.1),
    ("hard", 0.1, 0.3),
    ("medium", 0.3, 0.6),
    ("easy", 0.6, 1.0001),
]


def existing_exclusions() -> set[str]:
    prior = json.loads(
        (ROOT / "configs/opd-gap/qualification/genagent-base-band40-k32-manifest.json").read_text()
    )
    hints = json.loads(
        (ROOT / "configs/opd-gap/genagent-p14mix-qwen35-opsd-hints-r38-manifest.json").read_text()
    )
    return {entry["task"] for entry in prior["tasks"]} | set(hints["heldout_tasks"])


def metadata_pass_rate(metadata: dict) -> float | None:
    for entry in metadata.get("pass_rates", []):
        if entry.get("solver") == "local" and entry.get("model") == "openai/gpt-5-mini":
            return float(entry["value"])
    return None


def main() -> None:
    excluded = existing_exclusions()
    candidates: dict[str, list[dict]] = {name: [] for name, _, _ in BANDS}
    for task_toml in sorted(TASK_ROOT.glob("*/task.toml")):
        raw = task_toml.read_bytes()
        metadata = tomllib.loads(raw.decode())["metadata"]
        if "tier" not in metadata:
            continue
        tier = int(metadata["tier"])
        rate = metadata_pass_rate(metadata)
        if metadata["name"] in excluded or tier not in {3, 4} or rate is None:
            continue
        for name, lower, upper in BANDS:
            if lower <= rate < upper:
                candidates[name].append(
                    {
                        "task": metadata["name"],
                        "tier": tier,
                        "metadata_pass_rate": rate,
                        "band": name,
                        "task_toml_sha256": hashlib.sha256(raw).hexdigest(),
                    }
                )
                break

    rng = random.Random(SEED)
    selected = []
    for name, _, _ in BANDS:
        pool = list({entry["task"]: entry for entry in candidates[name]}.values())
        if len(pool) < PER_BAND:
            raise RuntimeError(f"{name} has only {len(pool)} eligible tasks")
        selected.extend(rng.sample(pool, PER_BAND))

    manifest = {
        "revision": "r01",
        "taskset_id": "general-agent-v1",
        "taskset_ref": TASKSET_REF,
        "selection_seed": SEED,
        "selection": "50 random tier-3/4 tasks per metadata band; prior 40-task panel and fixed heldout excluded",
        "excluded_task_count": len(excluded),
        "group_size": 8,
        "rollouts_per_task": 8,
        "minimum_mixed_groups": 1,
        "bands": [
            {
                "name": name,
                "min_metadata_pass_rate": lower,
                "max_metadata_pass_rate_exclusive": upper,
                "available_after_exclusions": len({entry["task"] for entry in candidates[name]}),
            }
            for name, lower, upper in BANDS
        ],
        "tasks": selected,
    }
    manifest_path = ROOT / "configs/opd-gap/qualification/genagent-base-p200-k8-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    tasks = ", ".join(json.dumps(entry["task"]) for entry in selected)
    config = f'''max_turns = 40
max_input_tokens = 56000
max_output_tokens = 24000
max_total_tokens = 60000
multiplex = 32
model = "qwen/qwen3.5-35b-a3b"
num_tasks = 200
num_rollouts = 8
shuffle = false
max_concurrent = 32
verbose = false
dry_run = false
rich = false
server = false
output_dir = "evals/genagent-base-p200-k8-hosted-r01-20260715"

[taskset]
id = "general-agent-v1"
ref = "{TASKSET_REF}"
tasks = [{tasks}]
min_tier = 3
max_tier = 4
pass_rate_model = "openai/gpt-5-mini"
pass_rate_solver = "local"
min_pass_rate = 0.0
max_pass_rate = 1.0
shuffle_seed = {SEED}
start_idx = 0
limit = 200

[taskset.tools]
colocated = false
shared = false
fork = false

[taskset.tools.runtime]
type = "subprocess"

[harness]
id = "null"
forward_env = []

[harness.runtime]
type = "subprocess"

[harness.env]

[timeout]
setup = 120.0
rollout = 600.0
scoring = 60.0

[retries.rollout]
max_retries = 2
include = ["ProviderError"]
exclude = []

[args]

[extra_env_kwargs]

[pool]
type = "elastic"
multiplex = 128

[client]
base_url = "https://api.pinference.ai/api/v1"
api_key_var = "PRIME_API_KEY"
type = "eval"

[client.headers]
X-Prime-Team-ID = "cmlr3u2er002zhr01tj8f48ts"

[sampling]
temperature = 1.0
max_tokens = 4096
'''
    config_path = ROOT / "configs/opd-gap/qualification/genagent-base-p200-k8-hosted-r01.toml"
    config_path.write_text(config)
    print(json.dumps({"manifest": str(manifest_path), "config": str(config_path), "tasks": len(selected)}))


if __name__ == "__main__":
    main()
