#!/usr/bin/env python3

from __future__ import annotations

import copy
import hashlib
import json
import tomllib
from pathlib import Path
from typing import Any

import tomli_w

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "opd-gap"
BASE_CONFIG = CONFIG_DIR / "genagent-p8var8lt-qwen35-opsd-d64-r37-full100.toml"
REVISION = "r38"
DATE = "20260715"

TRAIN_TASKS = [
    "cocktail_competition_t4",
    "lighthouse_mgmt_t4",
    "semiconductor_fab_t4",
    "plant_clinic_t4",
    "immigration_office_t3",
    "museum_curation_t3",
    "transplant_registry_t3",
    "game_night_t3",
    "leather_shop_t3",
    "pest_control_t3",
    "shipwreck_salvage_t3",
    "factory_floor_t4",
    "toy_lending_t3",
    "trade_show_t4",
]

GEPA_TASKS = [
    "poetry_slam_t4",
    "mine_rescue_t4",
    "translation_agency_t4",
    "distillery_t3",
]

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


def configure(arm: str, steps: int, placement: str) -> dict[str, Any]:
    config = copy.deepcopy(tomllib.loads(BASE_CONFIG.read_text()))
    phase = "smoke2" if steps == 2 else "qual20"
    descriptor = f"opsd-1lp-d64-{arm}-p14mix-k8-tp4-{placement}-{REVISION}-{phase}-{DATE}"
    output = f"outputs-genagent-{descriptor}"

    config["output_dir"] = output
    config["max_steps"] = steps
    config["trainer"]["max_steps"] = steps
    config["trainer"]["ckpt"]["interval"] = 1 if steps == 2 else 5
    config["trainer"]["wandb"]["name"] = descriptor
    config["orchestrator"]["max_steps"] = steps
    config["orchestrator"]["ckpt"]["interval"] = 1 if steps == 2 else 5
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
    train_taskset["tasks"] = TRAIN_TASKS
    train_taskset["limit"] = len(TRAIN_TASKS)
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
        config["slurm"]["job_name"] = f"genagent-{arm}-p14mix-{REVISION}-{phase}"
    return config


def main() -> None:
    configs: dict[str, Any] = {}
    for arm in ARMS:
        for steps in (2, 20):
            phase = "smoke2" if steps == 2 else "qual20"
            for placement in ("ar", "pod"):
                config = configure(arm, steps, placement)
                path = CONFIG_DIR / f"genagent-p14mix-qwen35-opsd-{arm}-{placement}-{REVISION}-{phase}.toml"
                path.write_bytes(tomli_w.dumps(config, multiline_strings=True).encode())
                configs[f"{arm}-{placement}-{phase}"] = {
                    "path": str(path.relative_to(ROOT)),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "output_dir": config["output_dir"],
                }
                print(path.relative_to(ROOT))

    manifest = {
        "revision": REVISION,
        "taskset_id": "general-agent-v1",
        "taskset_ref": "a2b76f6ac3469f7f50171760c0d0dba38360edc4",
        "qualification_source": "evals/genagent-base-band40-k32-hosted-r01-20260715/analysis.json",
        "qualification_rule": "at least two mixed-reward groups of k=8, zero errors, truncation <= 0.25",
        "train_tasks": TRAIN_TASKS,
        "gepa_only_tasks": GEPA_TASKS,
        "heldout_tasks": HELDOUT_TASKS,
        "group_size": 8,
        "objective": "single_token_ref_kl",
        "diagnostic_top_k": 64,
        "gepa_selected_prompt": None,
        "configs": configs,
    }
    manifest_path = CONFIG_DIR / f"genagent-p14mix-qwen35-opsd-hints-{REVISION}-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(manifest_path.relative_to(ROOT))


if __name__ == "__main__":
    main()
