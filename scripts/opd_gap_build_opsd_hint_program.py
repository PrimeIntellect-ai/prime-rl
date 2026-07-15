#!/usr/bin/env python3

from __future__ import annotations

import copy
import hashlib
import json
import random
import tomllib
from pathlib import Path
from typing import Any

import tomli_w

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "opd-gap"
BASE_CONFIG = CONFIG_DIR / "genagent-p8var8lt-qwen35-opsd-d64-r37-full100.toml"
GRPO_BASE_CONFIG = CONFIG_DIR / "genagent-p8var8lt-qwen35-grpo-r37-full100.toml"
REVISION = "r40"
DATE = "20260715"
SEED = 20260715
TASKSET_REF = "a2b76f6ac3469f7f50171760c0d0dba38360edc4"
TASK_ROOT = Path.home() / ".cache/verifiers/general_agent" / TASKSET_REF / "tasks"
MIN_METADATA_PASS_RATE = 0.0
MAX_METADATA_PASS_RATE_EXCLUSIVE = 0.6
GEPA_TRAIN_COUNT = 8
GEPA_EVAL_COUNT = 8

HELDOUT_TASKS = [
    "soap_making_t4",
    "incense_crafting_t3",
    "car_show_t3",
    "timber_t3",
    "bbq_competition_t3",
    "textile_mill_t4",
    "vertical_farm_t3",
    "fireworks_show_t4",
    "party_rental_t4",
    "wine_blend_t3",
    "space_debris_t4",
    "freelance_platform_t3",
    "sewing_workshop_t3",
    "chocolate_boutique_t3",
    "science_fair_t4",
    "fulfillment_center_t4",
    "film_archive_t4",
    "fermentation_lab_t3",
    "coffee_cupping_t3",
    "cable_car_t4",
]

FULL_ANSWER_TEMPLATE = """Use the validated reference tool-call chain below as privileged information when scoring the original trajectory. Preserve the original task and trajectory exactly.
<validated_tool_calls>
{demonstration}
</validated_tool_calls>"""

ANSWER_PLAN_TEMPLATE = """Use the validated structural plan below to reason about the original task. The plan gives only ordered tool names and argument names; infer every concrete argument value from the original task and tool observations. Preserve the original trajectory exactly.
<answer_plan>
{demonstration}
</answer_plan>"""

ARMS = {
    "fullanswer": {"transform": "identity", "template": FULL_ANSWER_TEMPLATE},
    "answerplan": {"transform": "tool_sequence_plan", "template": ANSWER_PLAN_TEMPLATE},
    # Replaced with the selected GEPA prompt before this arm is launched.
    "gepaplan": {"transform": "tool_sequence_plan", "template": ANSWER_PLAN_TEMPLATE},
}


def metadata_pass_rate(metadata: dict[str, Any]) -> float | None:
    for entry in metadata.get("pass_rates", []):
        if entry.get("solver") == "local" and entry.get("model") == "openai/gpt-5-mini":
            return float(entry["value"])
    return None


def select_tasks() -> tuple[list[dict[str, Any]], list[str], list[str], list[str]]:
    candidates: dict[str, dict[str, Any]] = {}
    for task_toml in sorted(TASK_ROOT.glob("*/task.toml")):
        raw = task_toml.read_bytes()
        metadata = tomllib.loads(raw.decode()).get("metadata", {})
        tier = metadata.get("tier")
        rate = metadata_pass_rate(metadata)
        name = metadata.get("name")
        if (
            name is None
            or name in HELDOUT_TASKS
            or tier not in {3, 4}
            or rate is None
            or not MIN_METADATA_PASS_RATE <= rate < MAX_METADATA_PASS_RATE_EXCLUSIVE
        ):
            continue
        candidates[name] = {
            "task": name,
            "tier": int(tier),
            "metadata_pass_rate": rate,
            "task_toml_sha256": hashlib.sha256(raw).hexdigest(),
            "gold_sha256": hashlib.sha256((task_toml.parent / "gold.json").read_bytes()).hexdigest(),
        }

    entries = sorted(candidates.values(), key=lambda entry: entry["task"])
    random.Random(SEED).shuffle(entries)
    required = GEPA_TRAIN_COUNT + GEPA_EVAL_COUNT + 1
    if len(entries) < required:
        raise RuntimeError(f"only {len(entries)} eligible tasks; need at least {required}")
    gepa_train = [entry["task"] for entry in entries[:GEPA_TRAIN_COUNT]]
    gepa_eval = [entry["task"] for entry in entries[GEPA_TRAIN_COUNT : GEPA_TRAIN_COUNT + GEPA_EVAL_COUNT]]
    train = [entry["task"] for entry in entries[GEPA_TRAIN_COUNT + GEPA_EVAL_COUNT :]]
    return entries, train, gepa_train, gepa_eval


def set_algo(config: dict[str, Any], arm: str) -> None:
    spec = ARMS[arm]
    for algo in (
        config["orchestrator"]["algo"],
        config["orchestrator"]["train"]["env"][0]["algo"],
    ):
        algo.update(
            type="opsd",
            ref_logprob_granularity="single_token",
            ref_top_k=64,
            diag_top_k=64,
            demo_key="gold_tool_calls",
            demo_transform=spec["transform"],
            template=spec["template"],
        )


def configure(arm: str, steps: int, placement: str, train_tasks: list[str]) -> dict[str, Any]:
    config = copy.deepcopy(tomllib.loads(BASE_CONFIG.read_text()))
    phase = "smoke2" if steps == 2 else "full100"
    descriptor = f"opsd-1lp-d64-{arm}-band000060-k8-tp4-{placement}-{REVISION}-{phase}-{DATE}"
    output = f"outputs-genagent-{descriptor}"

    config["output_dir"] = output
    config["max_steps"] = steps
    config["trainer"]["max_steps"] = steps
    config["trainer"]["ckpt"]["interval"] = 1 if steps == 2 else 10
    config["trainer"]["wandb"]["name"] = descriptor
    config["orchestrator"]["max_steps"] = steps
    config["orchestrator"]["ckpt"]["interval"] = 1 if steps == 2 else 10
    config["orchestrator"]["wandb"]["name"] = descriptor
    config["orchestrator"]["prime_monitor"]["run_name"] = descriptor
    config["wandb"]["name"] = descriptor
    config["orchestrator"]["pre_batch_filters"] = [
        {"type": "gibberish", "enforce": False, "token_id_threshold": 100000, "logprob_offset": 2.0},
        {"type": "repetition", "enforce": False, "window": 3000, "prob_threshold": 0.99},
        {"type": "zero_advantage", "enforce": False},
    ]
    config["orchestrator"]["post_batch_filters"] = copy.deepcopy(
        config["orchestrator"]["pre_batch_filters"]
    )

    train_taskset = config["orchestrator"]["train"]["env"][0]["taskset"]
    train_taskset["tasks"] = train_tasks
    train_taskset["limit"] = len(train_tasks)
    train_taskset["min_pass_rate"] = 0.0
    train_taskset["max_pass_rate"] = MAX_METADATA_PASS_RATE_EXCLUSIVE
    train_taskset["shuffle_seed"] = SEED
    eval_taskset = config["orchestrator"]["eval"]["env"][0]["taskset"]
    eval_taskset["tasks"] = HELDOUT_TASKS
    eval_taskset["limit"] = len(HELDOUT_TASKS)
    config["orchestrator"]["eval"]["env"][0]["num_examples"] = len(HELDOUT_TASKS)
    config["orchestrator"]["eval"]["env"][0]["interval"] = 10

    set_algo(config, arm)
    config["inference"].setdefault("env_vars", {})["VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS"] = "1800"

    if placement == "pod":
        config.pop("slurm", None)
    else:
        config["slurm"]["job_name"] = f"genagent-{arm}-band000060-{REVISION}-{phase}"
    return config


def configure_grpo(steps: int, placement: str, train_tasks: list[str]) -> dict[str, Any]:
    config = copy.deepcopy(tomllib.loads(GRPO_BASE_CONFIG.read_text()))
    phase = "smoke2" if steps == 2 else "full100"
    descriptor = f"grpo-band000060-k8-tp4-{placement}-{REVISION}-{phase}-{DATE}"
    config["output_dir"] = f"outputs-genagent-{descriptor}"
    config["max_steps"] = steps
    config["trainer"]["max_steps"] = steps
    config["trainer"]["ckpt"]["interval"] = 1 if steps == 2 else 10
    config["trainer"]["wandb"]["name"] = descriptor
    config["orchestrator"]["max_steps"] = steps
    config["orchestrator"]["ckpt"]["interval"] = 1 if steps == 2 else 10
    config["orchestrator"]["wandb"]["name"] = descriptor
    config["orchestrator"]["prime_monitor"]["run_name"] = descriptor
    config["wandb"]["name"] = descriptor
    filters = [
        {"type": "gibberish", "enforce": False, "token_id_threshold": 100000, "logprob_offset": 2.0},
        {"type": "repetition", "enforce": False, "window": 3000, "prob_threshold": 0.99},
        {"type": "zero_advantage", "enforce": True},
    ]
    config["orchestrator"]["pre_batch_filters"] = filters
    config["orchestrator"]["post_batch_filters"] = copy.deepcopy(filters)
    train_taskset = config["orchestrator"]["train"]["env"][0]["taskset"]
    train_taskset["tasks"] = train_tasks
    train_taskset["limit"] = len(train_tasks)
    train_taskset["min_pass_rate"] = MIN_METADATA_PASS_RATE
    train_taskset["max_pass_rate"] = MAX_METADATA_PASS_RATE_EXCLUSIVE
    train_taskset["shuffle_seed"] = SEED
    eval_env = config["orchestrator"]["eval"]["env"][0]
    eval_env["taskset"]["tasks"] = HELDOUT_TASKS
    eval_env["taskset"]["limit"] = len(HELDOUT_TASKS)
    eval_env["num_examples"] = len(HELDOUT_TASKS)
    eval_env["interval"] = 10
    config["inference"].setdefault("env_vars", {})["VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS"] = "1800"
    if placement == "pod":
        config.pop("slurm", None)
    else:
        config["slurm"]["job_name"] = f"genagent-grpo-band000060-{REVISION}-{phase}"
    return config


def main() -> None:
    task_entries, train_tasks, gepa_train_tasks, gepa_eval_tasks = select_tasks()
    configs: dict[str, Any] = {}
    for arm in ARMS:
        for steps in (2, 100):
            phase = "smoke2" if steps == 2 else "full100"
            for placement in ("ar", "pod"):
                config = configure(arm, steps, placement, train_tasks)
                path = CONFIG_DIR / f"genagent-band000060-qwen35-opsd-{arm}-{placement}-{REVISION}-{phase}.toml"
                path.write_bytes(tomli_w.dumps(config, multiline_strings=True).encode())
                configs[f"{arm}-{placement}-{phase}"] = {
                    "path": str(path.relative_to(ROOT)),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "output_dir": config["output_dir"],
                }
                print(path.relative_to(ROOT))

    for steps in (2, 100):
        phase = "smoke2" if steps == 2 else "full100"
        for placement in ("ar", "pod"):
            config = configure_grpo(steps, placement, train_tasks)
            path = CONFIG_DIR / f"genagent-band000060-qwen35-grpo-{placement}-{REVISION}-{phase}.toml"
            path.write_bytes(tomli_w.dumps(config, multiline_strings=True).encode())
            configs[f"grpo-{placement}-{phase}"] = {
                "path": str(path.relative_to(ROOT)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "output_dir": config["output_dir"],
            }
            print(path.relative_to(ROOT))

    manifest = {
        "revision": REVISION,
        "taskset_id": "general-agent-v1",
        "taskset_ref": TASKSET_REF,
        "selection": (
            "all unique tier-3/4 tasks with local openai/gpt-5-mini metadata pass rate "
            ">=0.0 and <0.6; no Qwen-success filtering"
        ),
        "selection_seed": SEED,
        "minimum_metadata_pass_rate": MIN_METADATA_PASS_RATE,
        "maximum_metadata_pass_rate_exclusive": MAX_METADATA_PASS_RATE_EXCLUSIVE,
        "task_entries": task_entries,
        "train_tasks": train_tasks,
        "gepa_train_tasks": gepa_train_tasks,
        "gepa_eval_tasks": gepa_eval_tasks,
        "heldout_tasks": HELDOUT_TASKS,
        "group_size": 8,
        "objective": "single_token_ref_kl",
        "diagnostic_top_k": 64,
        "gepa_selected_prompt": None,
        "configs": configs,
    }
    manifest_path = CONFIG_DIR / f"genagent-band000060-qwen35-opsd-hints-{REVISION}-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(manifest_path.relative_to(ROOT))


if __name__ == "__main__":
    main()
