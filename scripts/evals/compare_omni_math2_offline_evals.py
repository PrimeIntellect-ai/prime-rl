from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "outputs/omni_math2_rlvr_canary/offline_eval_comparison_20260512.md"
KS = (1, 2, 3, 4, 5, 6, 8)

RAW_BASELINE = {
    "family": "raw_baseline",
    "arm": "OLMo Inst DPO",
    "step": "raw",
    "source": "user-provided raw model reference",
    "num_rollouts": "",
    "trunc": "",
    "p1": 0.367,
    "p2": 0.458,
    "p3": 0.506,
    "p4": 0.539,
    "p5": 0.562,
    "p6": 0.581,
    "p8": 0.610,
}


@dataclass(frozen=True)
class SummarySet:
    family: str
    pattern: str


SUMMARY_SETS = (
    SummarySet(
        "nonfilter_14i2t_longrun",
        "outputs/omni_math2_rlvr_canary/20260509_1825/offline_eval_600x8/default/step_*/summary.json",
    ),
    SummarySet(
        "nonfilter_28i4t_short",
        "outputs/omni_math2_rlvr_canary/default_8node_28i4t_compile_fsasync4_20260512_1430/"
        "offline_eval_600x8_7node_clean/default_28i4t_compile_fsasync4/step_*/summary.json",
    ),
    SummarySet(
        "dapo_refill_v1_lr1e6_28i4t",
        "outputs/omni_math2_rlvr_canary/default_8node_28i4t_compile_fsasync4_refill_20260512_1745/"
        "offline_eval_600x8_7node_clean/refill_lr1e6_28i4t/step_*/summary.json",
    ),
    SummarySet(
        "dapo_refill_v1_lr1e6_28i4t_router",
        "outputs/omni_math2_rlvr_canary/default_8node_28i4t_compile_fsasync4_refill_20260512_1745/"
        "offline_eval_600x8_7node_router/refill_lr1e6_28i4t/step_*/summary.json",
    ),
    SummarySet(
        "dapo_refill_v1_lr1e6_28i4t_router_8node",
        "outputs/omni_math2_rlvr_canary/default_8node_28i4t_compile_fsasync4_refill_20260512_1745/"
        "offline_eval_600x8_8node_router/refill_lr1e6_28i4t/step_*/summary.json",
    ),
    SummarySet(
        "dapo_refill_lr3e6_28i4t",
        "outputs/omni_math2_rlvr_canary/lr3e6_28i4t_refill_shared_submit_*/"
        "offline_eval_600x8_7node_clean/refill_lr3e6_28i4t/step_*/summary.json",
    ),
    SummarySet(
        "dapo_refill_lr3e6_28i4t_router",
        "outputs/omni_math2_rlvr_canary/lr3e6_28i4t_refill_shared_submit_*/"
        "offline_eval_600x8_7node_router/refill_lr3e6_28i4t/step_*/summary.json",
    ),
    SummarySet(
        "dapo_refill_lr3e6_28i4t_router_8node",
        "outputs/omni_math2_rlvr_canary/lr3e6_28i4t_refill_shared_submit_*/"
        "offline_eval_600x8_8node_router/refill_lr3e6_28i4t/step_*/summary.json",
    ),
    SummarySet(
        "dapo_refill_lr3e6_28i4t",
        "outputs/omni_math2_rlvr_canary/lr3e6_supervisor_20260512/retry_*/"
        "offline_eval_600x8_7node_clean/refill_lr3e6_28i4t/step_*/summary.json",
    ),
)


def _pass_at(summary: dict[str, Any], k: int) -> float | None:
    data = (summary.get("pass") or {}).get(str(k)) or {}
    value = data.get("pass_at_k")
    return float(value) if value is not None else None


def _row_from_summary(family: str, path: Path) -> dict[str, Any]:
    summary = json.loads(path.read_text())
    row: dict[str, Any] = {
        "family": family,
        "arm": summary.get("arm", path.parents[1].name),
        "step": int(summary.get("step", path.parent.name.removeprefix("step_"))),
        "source": str(path.relative_to(REPO_ROOT)),
        "num_rollouts": summary.get("num_rollouts", ""),
        "trunc": summary.get("truncation_rate", ""),
    }
    for k in KS:
        row[f"p{k}"] = _pass_at(summary, k)
    return row


def collect_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [dict(RAW_BASELINE)]
    seen: set[Path] = set()
    for summary_set in SUMMARY_SETS:
        for path in sorted(REPO_ROOT.glob(summary_set.pattern)):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            rows.append(_row_from_summary(summary_set.family, path))
    rows[1:] = sorted(rows[1:], key=lambda row: (str(row["family"]), str(row["arm"]), int(row["step"])))
    return rows


def _fmt(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _delta(value: Any, baseline: float | None) -> str:
    if value is None or value == "" or baseline is None:
        return ""
    return f"{float(value) - baseline:+.6f}"


def write_markdown(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    raw = rows[0]
    headers = [
        "family",
        "arm",
        "step",
        "p@1",
        "p@2",
        "p@3",
        "p@4",
        "p@5",
        "p@6",
        "p@8",
        "Δp@1 raw",
        "Δp@8 raw",
        "trunc",
        "rollouts",
    ]

    lines = [
        "# Omni-MATH-2 Offline Eval Comparison",
        "",
        "Unbiased pass@k values over the available `600x8` offline summaries. "
        "Raw baseline is the user-provided OLMo Inst DPO reference row.",
        "",
        "Note: `dapo_refill_v1_lr1e6_28i4t` used the first refill "
        "implementation, which generated full 256-rollout candidate batches "
        "for each refill round and emitted a misleading old retry warning. "
        "`dapo_refill_lr3e6_28i4t` used the later candidate-batched refill "
        "implementation.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        cells = [
            _fmt(row.get("family")),
            _fmt(row.get("arm")),
            _fmt(row.get("step")),
            _fmt(row.get("p1")),
            _fmt(row.get("p2")),
            _fmt(row.get("p3")),
            _fmt(row.get("p4")),
            _fmt(row.get("p5")),
            _fmt(row.get("p6")),
            _fmt(row.get("p8")),
            _delta(row.get("p1"), raw.get("p1")),
            _delta(row.get("p8"), raw.get("p8")),
            _fmt(row.get("trunc")),
            _fmt(row.get("num_rollouts")),
        ]
        lines.append("| " + " | ".join(cells) + " |")

    lines.extend(["", "## Sources", ""])
    for row in rows:
        lines.append(f"- `{row['family']}` step `{row['step']}`: {row['source']}")

    output.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    rows = collect_rows()
    write_markdown(rows, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
