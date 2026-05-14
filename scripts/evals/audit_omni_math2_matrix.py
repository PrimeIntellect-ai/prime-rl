from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evals.analyze_baseline_matrix import load_jsonl, safe_float
from scripts.evals.run_omni_math2_model_bench import REQUESTED_MODELS
from prime_rl.baselines.benchmark import filter_blocked_specs

EXPECTED_KS = ("1", "3", "5", "8", "16")


def expected_run_dirs(
    matrix_dir: Path,
    run_prefix: str,
    rollouts_per_example: int,
    *,
    multinode_nodes: int,
) -> dict[str, Path]:
    specs, _ = filter_blocked_specs(
        REQUESTED_MODELS,
        explicitly_requested=False,
        include_blocked=False,
    )
    if multinode_nodes <= 1:
        specs = [spec for spec in specs if not spec.requires_multinode]
    return {
        spec.short_name: matrix_dir / f"{run_prefix}-{spec.short_name}-omni_math2-k{rollouts_per_example}"
        for spec in specs
    }


def audit_run(
    short_name: str,
    run_dir: Path,
    *,
    expected_examples: int,
    expected_rollouts: int,
    rollouts_per_example: int,
    require_zero_errors: bool,
) -> dict[str, Any]:
    problems: list[str] = []
    summary_path = run_dir / "summary.json"
    records_path = run_dir / "records.jsonl"
    if not summary_path.exists():
        problems.append("missing summary.json")
    if not records_path.exists():
        problems.append("missing records.jsonl")
    if problems:
        return {"model_short": short_name, "run_dir": str(run_dir), "ok": False, "problems": problems}

    summary = json.loads(summary_path.read_text())
    records = load_jsonl(records_path)
    by_example: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        by_example.setdefault(str(row.get("example_id")), []).append(row)

    if int(summary.get("num_examples", -1)) != expected_examples:
        problems.append(f"summary num_examples={summary.get('num_examples')} expected {expected_examples}")
    if int(summary.get("num_rollouts", -1)) != expected_rollouts:
        problems.append(f"summary num_rollouts={summary.get('num_rollouts')} expected {expected_rollouts}")
    if len(records) != expected_rollouts:
        problems.append(f"records count={len(records)} expected {expected_rollouts}")
    if len(by_example) != expected_examples:
        problems.append(f"distinct examples={len(by_example)} expected {expected_examples}")

    bad_group_sizes = sorted(
        (example_id, len(rows)) for example_id, rows in by_example.items() if len(rows) != rollouts_per_example
    )
    if bad_group_sizes:
        preview = ", ".join(f"{example_id}:{count}" for example_id, count in bad_group_sizes[:8])
        problems.append(f"{len(bad_group_sizes)} examples do not have {rollouts_per_example} rollouts ({preview})")

    missing_ks = [k for k in EXPECTED_KS if k not in summary.get("pass", {})]
    if missing_ks:
        problems.append(f"missing pass@K entries: {', '.join(missing_ks)}")

    record_errors = sum(1 for row in records if row.get("error") not in (None, ""))
    summary_error_rate = safe_float(summary.get("error_rate"))
    if require_zero_errors and (record_errors or summary_error_rate):
        problems.append(f"errors present: record_errors={record_errors}, summary_error_rate={summary_error_rate}")

    return {
        "model_short": short_name,
        "run_dir": str(run_dir),
        "ok": not problems,
        "problems": problems,
        "num_examples": summary.get("num_examples"),
        "num_rollouts": summary.get("num_rollouts"),
        "record_count": len(records),
        "record_errors": record_errors,
        "summary_error_rate": summary_error_rate,
        "truncation_rate": summary.get("truncation_rate"),
        "single_shot_accuracy": summary.get("single_shot_accuracy"),
        "pass_at_16": (summary.get("pass", {}).get("16") or {}).get("pass_at_k"),
    }


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# Omni-MATH-2 Artifact Audit",
        "",
        f"Matrix dir: `{result['matrix_dir']}`",
        f"Expected examples/model: {result['expected_examples']}",
        f"Expected rollouts/model: {result['expected_rollouts']}",
        f"Overall status: {'PASS' if result['ok'] else 'FAIL'}",
        "",
        "| model | ok | records | examples | errors | trunc | single-shot | pass@16 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in result["runs"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(run["model_short"]),
                    "yes" if run["ok"] else "no",
                    str(run.get("record_count", "")),
                    str(run.get("num_examples", "")),
                    str(run.get("record_errors", "")),
                    f"{safe_float(run.get('truncation_rate')):.3f}",
                    f"{safe_float(run.get('single_shot_accuracy')):.3f}",
                    f"{safe_float(run.get('pass_at_16')):.3f}",
                ]
            )
            + " |"
        )
    failures = [run for run in result["runs"] if not run["ok"]]
    if failures:
        lines.extend(["", "## Problems", ""])
        for run in failures:
            lines.append(f"### {run['model_short']}")
            for problem in run["problems"]:
                lines.append(f"- {problem}")
            lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit completed Omni-MATH-2 matrix artifacts.")
    parser.add_argument("--matrix-dir", type=Path, required=True)
    parser.add_argument("--run-prefix", default="k16-32k-v5")
    parser.add_argument("--num-examples", type=int, default=600)
    parser.add_argument("--rollouts-per-example", type=int, default=16)
    parser.add_argument("--multinode-nodes", type=int, default=1)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--allow-errors", action="store_true")
    args = parser.parse_args()

    expected_rollouts = args.num_examples * args.rollouts_per_example
    runs = [
        audit_run(
            short_name,
            run_dir,
            expected_examples=args.num_examples,
            expected_rollouts=expected_rollouts,
            rollouts_per_example=args.rollouts_per_example,
            require_zero_errors=not args.allow_errors,
        )
        for short_name, run_dir in expected_run_dirs(
            args.matrix_dir,
            args.run_prefix,
            args.rollouts_per_example,
            multinode_nodes=args.multinode_nodes,
        ).items()
    ]
    result = {
        "ok": all(run["ok"] for run in runs),
        "matrix_dir": str(args.matrix_dir),
        "expected_examples": args.num_examples,
        "expected_rollouts": expected_rollouts,
        "runs": runs,
    }
    output_dir = args.output_dir or args.matrix_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "artifact_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    (output_dir / "artifact_audit.md").write_text(render_report(result))
    if not result["ok"]:
        print(f"artifact audit failed; see {output_dir / 'artifact_audit.md'}", file=sys.stderr)
        raise SystemExit(1)
    print(f"artifact audit passed; wrote {output_dir / 'artifact_audit.md'}")


if __name__ == "__main__":
    main()
