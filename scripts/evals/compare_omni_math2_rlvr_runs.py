from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any

STEP_RE = re.compile(r"step_(\d+)$")
DEFAULT_KS = (1, 2, 3, 4, 5, 6, 8)
STEP0_BASELINE = {
    "p@1": 0.367,
    "p@2": 0.458,
    "p@3": 0.506,
    "p@4": 0.539,
    "p@5": 0.562,
    "p@6": 0.581,
    "p@8": 0.610,
}


def _parse_mapping(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected NAME=PATH")
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("NAME must be non-empty")
    return name, Path(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _maybe_mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def _first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _step_from_dir(path: Path) -> int | None:
    match = STEP_RE.fullmatch(path.name)
    return int(match.group(1)) if match else None


def summarize_train_rollouts(run_root: Path, steps: set[int] | None) -> list[dict[str, Any]]:
    rollout_root = run_root / "rollouts"
    if not rollout_root.exists():
        return []

    rows = []
    for step_dir in sorted(rollout_root.glob("step_*")):
        step = _step_from_dir(step_dir)
        if step is None or steps is not None and step not in steps:
            continue
        rollouts_path = step_dir / "train_rollouts.jsonl"
        if not rollouts_path.exists():
            continue
        rollouts = _read_jsonl(rollouts_path)
        rewards = [float(r["reward"]) for r in rollouts if r.get("reward") is not None]
        accepted = [
            float(r["reward"])
            for r in rollouts
            if r.get("reward") is not None and not bool(r.get("is_filtered", False))
        ]
        filtered = [
            float(r["reward"]) for r in rollouts if r.get("reward") is not None and bool(r.get("is_filtered", False))
        ]
        sidecar = step_dir / "train_filter_metrics.json"
        sidecar_metrics = json.loads(sidecar.read_text()) if sidecar.exists() else {}
        reward_unconditioned = _first_not_none(
            sidecar_metrics.get("train_batch_refill/reward_unconditioned_on_filtering/mean"),
            sidecar_metrics.get("train_batch/reward_unconditioned_on_filtering/mean"),
            _maybe_mean(rewards),
        )
        reward_conditioned = _first_not_none(
            sidecar_metrics.get("train_batch_refill/reward_conditioned_on_filtering/mean"),
            sidecar_metrics.get("train_batch/reward_conditioned_on_filtering/mean"),
            _maybe_mean(accepted),
        )
        reward_filtered = _first_not_none(
            sidecar_metrics.get("train_batch_refill/reward_filtered_out/mean"),
            sidecar_metrics.get("train_batch/reward_filtered_out/mean"),
            _maybe_mean(filtered),
        )
        candidate_rollouts = sidecar_metrics.get("train_batch_refill/candidate_rollouts")
        accepted_sidecar = sidecar_metrics.get("train_batch_refill/accepted_rollouts")
        filtered_sidecar = (
            candidate_rollouts - accepted_sidecar
            if candidate_rollouts is not None and accepted_sidecar is not None
            else None
        )
        rows.append(
            {
                "step": step,
                "rollouts": candidate_rollouts if candidate_rollouts is not None else len(rollouts),
                "accepted_rollouts": accepted_sidecar if accepted_sidecar is not None else len(accepted),
                "filtered_rollouts": filtered_sidecar if filtered_sidecar is not None else len(filtered),
                "reward_unconditioned_on_filtering": reward_unconditioned,
                "reward_conditioned_on_filtering": reward_conditioned,
                "reward_filtered_out": reward_filtered,
                "refill_rounds": sidecar_metrics.get("train_batch_refill/refill_rounds"),
                "candidate_groups": sidecar_metrics.get("train_batch_refill/candidate_groups"),
                "accepted_groups": sidecar_metrics.get("train_batch_refill/accepted_groups"),
                "prompts_consumed_per_accepted_group": sidecar_metrics.get(
                    "train_batch_refill/prompts_consumed_per_accepted_group"
                ),
                "metrics_sidecar": sidecar_metrics,
            }
        )
    return rows


def summarize_eval(eval_root: Path, arm: str, steps: set[int] | None, ks: tuple[int, ...]) -> list[dict[str, Any]]:
    arm_root = eval_root / arm
    if not arm_root.exists():
        return []

    rows = []
    for step_dir in sorted(arm_root.glob("step_*")):
        step = _step_from_dir(step_dir)
        if step is None or steps is not None and step not in steps:
            continue
        summary_path = step_dir / "summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text())
        row = {
            "step": step,
            "num_examples": summary.get("num_examples"),
            "num_rollouts": summary.get("num_rollouts"),
            "mean_sample_accuracy": summary.get("mean_sample_accuracy"),
            "single_shot_accuracy": summary.get("single_shot_accuracy"),
            "truncation_rate": summary.get("truncation_rate"),
            "error_rate": summary.get("error_rate"),
        }
        pass_summary = summary.get("pass", {})
        for k in ks:
            k_summary = pass_summary.get(str(k), {})
            row[f"p@{k}"] = k_summary.get("pass_at_k")
        rows.append(row)
    return rows


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = ["# OmniMath2 RLVR Comparison", ""]
    lines.append("Step 0 baseline: " + ", ".join(f"{k}={v:.3f}" for k, v in STEP0_BASELINE.items()))
    lines.append("")

    for name, run in report["runs"].items():
        lines.append(f"## {name}")
        lines.append("")
        eval_rows = {row["step"]: row for row in run["eval"]}
        train_rows = {row["step"]: row for row in run["train"]}
        steps = sorted(set(eval_rows) | set(train_rows))
        header = [
            "step",
            "p@1",
            "p@2",
            "p@4",
            "p@8",
            "sample_acc",
            "train_reward_all",
            "train_reward_accepted",
            "train_reward_filtered",
            "accepted",
            "filtered",
        ]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")
        for step in steps:
            erow = eval_rows.get(step, {})
            trow = train_rows.get(step, {})
            values = [
                step,
                erow.get("p@1"),
                erow.get("p@2"),
                erow.get("p@4"),
                erow.get("p@8"),
                erow.get("mean_sample_accuracy"),
                trow.get("reward_unconditioned_on_filtering"),
                trow.get("reward_conditioned_on_filtering"),
                trow.get("reward_filtered_out"),
                trow.get("accepted_rollouts"),
                trow.get("filtered_rollouts"),
            ]
            lines.append("| " + " | ".join(_fmt(value) for value in values) + " |")
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=_parse_mapping, required=True, help="NAME=RUN_ROOT")
    parser.add_argument(
        "--eval",
        action="append",
        type=_parse_mapping,
        default=[],
        help="NAME=EVAL_ROOT. The eval arm directory must match NAME unless --eval-arm is provided.",
    )
    parser.add_argument(
        "--eval-arm",
        action="append",
        type=_parse_mapping,
        default=[],
        help="NAME=ARM_DIR_NAME for eval roots whose arm directory differs from NAME.",
    )
    parser.add_argument("--steps", type=str, default=None, help="Comma-separated steps to include")
    parser.add_argument("--ks", type=str, default=",".join(str(k) for k in DEFAULT_KS))
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--md-out", type=Path, default=None)
    args = parser.parse_args()

    steps = {int(part) for part in args.steps.split(",") if part.strip()} if args.steps else None
    ks = tuple(int(part) for part in args.ks.split(",") if part.strip())
    run_roots = dict(args.run)
    eval_roots = dict(args.eval)
    eval_arms = {name: str(path) for name, path in args.eval_arm}

    report = {"step0_baseline": STEP0_BASELINE, "runs": {}}
    for name, run_root in run_roots.items():
        eval_root = eval_roots.get(name)
        eval_arm = eval_arms.get(name, name)
        report["runs"][name] = {
            "run_root": str(run_root),
            "eval_root": str(eval_root) if eval_root is not None else None,
            "eval_arm": eval_arm,
            "train": summarize_train_rollouts(run_root, steps),
            "eval": summarize_eval(eval_root, eval_arm, steps, ks) if eval_root is not None else [],
        }

    output = json.dumps(report, indent=2, sort_keys=True)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(output + "\n")
    else:
        print(output)
    if args.md_out is not None:
        write_markdown(report, args.md_out)


if __name__ == "__main__":
    main()
