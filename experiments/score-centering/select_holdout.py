"""Select unseen Lego tasks after both training runs have stopped."""

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

REQUIRED_FILES = ("task.toml", "instruction.md", "tests/test.sh", "tests/test_outputs.py")
DISPATCH = re.compile(r"rollout start: id=\S+ task=(\d+) harness=")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=128)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    dataset = json.loads((Path(__file__).parent / "dataset.json").read_text())
    root = Path(dataset["path"])
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if revision != dataset["revision"]:
        raise ValueError(f"Dataset revision changed: {revision}")
    tasks = [
        path.name
        for path in sorted(root.iterdir())
        if path.is_dir() and path.name.startswith("task_") and all((path / name).is_file() for name in REQUIRED_FILES)
    ]
    if not tasks:
        raise ValueError("No valid tasks in the pinned dataset")
    seen = set()
    run_records = []
    for run in args.runs:
        logs = sorted((run / "logs").glob("attempt_*/envs/train/terminal-lego.log"))
        if not logs:
            raise ValueError(f"No training dispatch logs: {run}")
        indices = set()
        for path in logs:
            indices.update(int(match[1]) for match in DISPATCH.finditer(path.read_text(errors="replace")))
        if not indices or max(indices) >= len(tasks):
            raise ValueError(f"Invalid or missing task indices: {run}")
        config_records = []
        for path in sorted((run / "configs").glob("attempt_*/resolved/orchestrator.json")):
            config = json.loads(path.read_text())
            sources = [source for source in config["train"]["source"] if source["name"] == "terminal-lego"]
            if len(sources) != 1 or sources[0]["env"]["taskset"].get("tasks") is not None:
                raise ValueError(f"Task indices require the complete Lego source: {path}")
            config_records.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        if not config_records:
            raise ValueError(f"Missing resolved configs: {run}")
        seen.update(tasks[index] for index in indices)
        run_records.append({"path": str(run.resolve()), "dispatched_tasks": len(indices), "configs": config_records})
    eligible = sorted(set(tasks) - seen)
    selected = sorted(
        eligible, key=lambda name: hashlib.sha256(f"score-centering-heldout-v1:{name}".encode()).hexdigest()
    )[: args.count]
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset": dataset,
        "runs": run_records,
        "source_order_tasks": tasks,
        "excluded_tasks": sorted(seen),
        "eligible_tasks": eligible,
        "selected_tasks": selected,
        "requested_count": args.count,
        "meets_minimum_64": len(selected) >= 64,
        "rule": "First tasks sorted by SHA256(score-centering-heldout-v1: + task_name), excluding all dispatched tasks.",
        "note": "Use only after both training runs stop. Selection uses exposure, never evaluation scores.",
    }
    (args.output / "heldout.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output / "lego-tasks.toml").write_text(
        '[[source]]\nname = "terminal-lego-unseen"\nenv.taskset.id = "terminal-lego"\n'
        f"env.taskset.tasks = {json.dumps(selected)}\n"
    )
    print(
        json.dumps(
            {"source_tasks": len(tasks), "excluded": len(seen), "eligible": len(eligible), "selected": len(selected)}
        )
    )


if __name__ == "__main__":
    main()
