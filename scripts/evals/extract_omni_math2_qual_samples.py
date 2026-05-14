from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evals.analyze_baseline_matrix import extract_text, load_jsonl, model_short, safe_float


def run_dirs(matrix_dir: Path) -> list[Path]:
    return sorted(path.parent for path in matrix_dir.glob("*/summary.json") if (path.parent / "records.jsonl").exists())


def category(row: dict[str, Any]) -> str:
    metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
    if row.get("error") not in (None, ""):
        return "error"
    if row.get("correct") is True:
        if safe_float(metrics.get("math_verify_score")) >= 1.0:
            return "correct_math_verify"
        if safe_float(metrics.get("choice_alias_score")) >= 1.0:
            return "correct_choice_alias"
        if safe_float(metrics.get("text_alias_score")) >= 1.0:
            return "correct_text_alias"
        if safe_float(metrics.get("judge_score")) >= 1.0:
            return "correct_judge"
        return "correct"
    if row.get("is_truncated") is True:
        return "truncated"
    if not str(row.get("parsed_answer") or "").strip():
        return "format_or_parse_failure"
    if row.get("judge_response") not in (None, "") or row.get("judge_decision_last") is not None:
        return "judge_rejected"
    return "wrong_answer"


def unique_append(selected: list[dict[str, Any]], row: dict[str, Any] | None) -> None:
    if row is None:
        return
    key = (row.get("example_id"), row.get("trial_index"))
    if any((item.get("example_id"), item.get("trial_index")) == key for item in selected):
        return
    selected.append(row)


def pick_correct(rows: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    correct = sorted((row for row in rows if row.get("correct") is True), key=lambda row: safe_float(row.get("output_tokens")))
    if len(correct) <= n:
        return correct
    selected: list[dict[str, Any]] = []
    unique_append(selected, correct[0])
    unique_append(selected, correct[len(correct) // 2])
    unique_append(selected, correct[-1])
    for row in correct:
        if len(selected) >= n:
            break
        unique_append(selected, row)
    return selected[:n]


def pick_incorrect(rows: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    incorrect = [row for row in rows if row.get("correct") is not True]
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in incorrect:
        buckets.setdefault(category(row), []).append(row)
    for bucket in buckets.values():
        bucket.sort(key=lambda row: safe_float(row.get("output_tokens")), reverse=True)

    priority = ["error", "truncated", "format_or_parse_failure", "judge_rejected", "wrong_answer"]
    selected: list[dict[str, Any]] = []
    for name in priority:
        unique_append(selected, buckets.get(name, [None])[0])
        if len(selected) >= n:
            return selected[:n]
    for row in sorted(incorrect, key=lambda item: safe_float(item.get("output_tokens")), reverse=True):
        if len(selected) >= n:
            break
        unique_append(selected, row)
    return selected[:n]


def preview(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = limit // 2
    tail = limit - head
    return text[:head].rstrip() + "\n\n[...snip...]\n\n" + text[-tail:].lstrip()


def sample_record(row: dict[str, Any], *, response_chars: int) -> dict[str, Any]:
    info = row.get("info") if isinstance(row.get("info"), dict) else {}
    judge = row.get("judge_decision_last")
    return {
        "example_id": row.get("example_id"),
        "trial_index": row.get("trial_index"),
        "category": category(row),
        "correct": row.get("correct"),
        "target": row.get("target"),
        "parsed_answer": row.get("parsed_answer"),
        "is_truncated": row.get("is_truncated"),
        "input_tokens": row.get("input_tokens"),
        "output_tokens": row.get("output_tokens"),
        "generation_ms": row.get("generation_ms"),
        "domain": info.get("primary_domain") or info.get("domain"),
        "difficulty": info.get("difficulty_label") or info.get("difficulty"),
        "source": info.get("source"),
        "metrics": row.get("metrics") or {},
        "judge_decision_last": judge if isinstance(judge, dict) else None,
        "response_preview": preview(extract_text(row), response_chars),
    }


def render_markdown(samples: list[dict[str, Any]]) -> str:
    lines = [
        "# Omni-MATH-2 Qualitative Samples",
        "",
        "Each model gets deterministic correct and incorrect samples. Correct samples are short/median/long by output tokens. Incorrect samples prioritize errors, truncation, parse/format failures, judge rejections, then ordinary wrong answers.",
        "",
    ]
    for run in samples:
        summary = run["summary"]
        lines.extend(
            [
                f"## {run['model_short']}",
                "",
                f"- run: `{run['run_dir']}`",
                f"- rollouts: {summary.get('num_rollouts')} / examples: {summary.get('num_examples')}",
                f"- sample accuracy: {safe_float(summary.get('mean_sample_accuracy')):.3f}",
                f"- single-shot accuracy: {safe_float(summary.get('single_shot_accuracy')):.3f}",
                f"- truncation rate: {safe_float(summary.get('truncation_rate')):.3f}",
                "",
            ]
        )
        for label in ("correct_samples", "incorrect_samples"):
            lines.extend([f"### {label.replace('_', ' ').title()}", ""])
            for sample in run[label]:
                lines.extend(
                    [
                        f"#### {sample['category']} · example `{sample['example_id']}` trial `{sample['trial_index']}`",
                        "",
                        f"- target: `{sample['target']}`",
                        f"- parsed: `{sample['parsed_answer']}`",
                        f"- tokens: in={sample['input_tokens']} out={sample['output_tokens']}",
                        f"- truncated: `{sample['is_truncated']}`",
                        f"- domain/source: `{sample['domain']}` / `{sample['source']}`",
                        "",
                        "```text",
                        sample["response_preview"],
                        "```",
                        "",
                    ]
                )
    return "\n".join(lines)


def build_samples(matrix_dir: Path, *, n: int, response_chars: int) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for run_dir in run_dirs(matrix_dir):
        summary = json.loads((run_dir / "summary.json").read_text())
        rows = load_jsonl(run_dir / "records.jsonl")
        model = str(summary.get("model") or rows[0].get("model"))
        samples.append(
            {
                "run_dir": str(run_dir),
                "model": model,
                "model_short": model_short(model),
                "summary": summary,
                "correct_samples": [sample_record(row, response_chars=response_chars) for row in pick_correct(rows, n)],
                "incorrect_samples": [
                    sample_record(row, response_chars=response_chars) for row in pick_incorrect(rows, n)
                ],
            }
        )
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract qualitative Omni-MATH-2 samples from completed runs.")
    parser.add_argument("--matrix-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--samples-per-split", type=int, default=3)
    parser.add_argument("--response-chars", type=int, default=2400)
    args = parser.parse_args()

    output_dir = args.output_dir or args.matrix_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = build_samples(args.matrix_dir, n=args.samples_per_split, response_chars=args.response_chars)
    if not samples:
        raise SystemExit(f"No completed runs found under {args.matrix_dir}")
    (output_dir / "qualitative_samples.json").write_text(json.dumps(samples, indent=2) + "\n")
    (output_dir / "qualitative_samples.md").write_text(render_markdown(samples) + "\n")
    print(f"wrote qualitative samples for {len(samples)} runs to {output_dir}")


if __name__ == "__main__":
    main()
