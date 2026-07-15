#!/usr/bin/env python3

from __future__ import annotations

import json
import shutil
from pathlib import Path

from prime_rl.orchestrator.algo.opsd import OPSDAlgorithm

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path.home() / ".cache/verifiers/general_agent/a2b76f6ac3469f7f50171760c0d0dba38360edc4/tasks"
OUTPUT = ROOT / "evals/genagent-opsd-gepaplan-r40-20260715"
MANIFEST = ROOT / "configs/opd-gap/genagent-band000060-qwen35-opsd-hints-r40-manifest.json"


def materialize(split: str, tasks: list[str]) -> None:
    split_dir = OUTPUT / split
    if split_dir.exists():
        shutil.rmtree(split_dir)
    split_dir.mkdir(parents=True)
    for task in tasks:
        source = SOURCE / task
        destination = split_dir / task
        shutil.copytree(source, destination)
        plan = OPSDAlgorithm._tool_sequence_plan((source / "gold.json").read_text())
        instruction = (source / "instruction.md").read_text().strip()
        (destination / "instruction.md").write_text(
            f"{instruction}\n\nValidated structural answer plan (argument values intentionally withheld):\n{plan}\n"
        )


def main() -> None:
    program = json.loads(MANIFEST.read_text())
    train_tasks = program["gepa_train_tasks"]
    eval_tasks = program["gepa_eval_tasks"]
    materialize("train_tasks", train_tasks)
    materialize("eval_tasks", eval_tasks)
    manifest = {
        "source": str(SOURCE),
        "taskset_ref": "a2b76f6ac3469f7f50171760c0d0dba38360edc4",
        "plan_transform": "tool_sequence_plan",
        "train_tasks": train_tasks,
        "eval_tasks": eval_tasks,
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
