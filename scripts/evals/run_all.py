"""Orchestrate the full Rung-6 eval suite against a single checkpoint.

Every eval produces paired base-vs-ckpt metrics via Protocol B (SFT-tokenizer
rendering client-side, raw /v1/completions). The orchestrator collects both
phases + deltas into a single rollup.

Each eval boots its own vLLM server on base, runs the base phase, hot-swaps
to ckpt weights, and runs the ckpt phase.

Phases:
    smoke, sycophancy, mtbench, ifeval, gsm8k, mmlu

Output layout:
    <output_dir>/smoke/{results.json, comparison.md, aggregate.json}
    <output_dir>/sycophancy/sycophancy.json
    <output_dir>/mtbench/mtbench.json
    <output_dir>/ifeval/ifeval.json
    <output_dir>/gsm8k/gsm8k.json
    <output_dir>/mmlu/mmlu.json
    <output_dir>/rollup.json
    <output_dir>/rollup.csv

Example:
    uv run python -m scripts.evals.run_all \\
        --ckpt /scratch/.../weights/step_500 \\
        --output-dir outputs/evals/<slug>
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import typing
from pathlib import Path
from typing import Any, Literal

from . import gsm8k, ifeval, mmlu, mtbench, smoke, sycophancy
from ._server import infer_base_model, load_suite, resolve_path_args

EvalName = Literal["smoke", "sycophancy", "mtbench", "ifeval", "gsm8k", "mmlu"]
PHASES: tuple[EvalName, ...] = typing.get_args(EvalName)


def _collect_rollup(results: dict[str, Any], thresholds: dict[str, Any]) -> dict[str, Any]:
    """Flatten every metric from every eval into rows with base/ckpt/delta + threshold check."""
    rows: list[dict[str, Any]] = []

    def add(metric: str, base_val: Any, ckpt_val: Any, delta: Any, *, spec: dict | None = None) -> None:
        row: dict[str, Any] = {
            "metric": metric,
            "base": base_val,
            "ckpt": ckpt_val,
            "delta": delta,
            "threshold": None,
            "pass": None,
        }
        if spec is not None and isinstance(ckpt_val, (int, float)):
            if "min" in spec:
                row["threshold"] = f">= {spec['min']}"
                row["pass"] = ckpt_val >= spec["min"]
            elif "max" in spec:
                row["threshold"] = f"<= {spec['max']}"
                row["pass"] = ckpt_val <= spec["max"]
            elif "max_drop_vs_base_pp" in spec and isinstance(base_val, (int, float)):
                row["threshold"] = f"drop <= {spec['max_drop_vs_base_pp']}pp (vs base)"
                # "drop" means ckpt is lower than base — accept if drop stays within budget
                drop_pp = (base_val - ckpt_val) * 100.0
                row["pass"] = drop_pp <= spec["max_drop_vs_base_pp"]
        rows.append(row)

    # ---- sycophancy ----
    syco = results.get("sycophancy") or {}
    syco_phases = syco.get("phases", {})
    for metric_name in ("aggregate",):
        base_val = syco_phases.get("base", {}).get(metric_name)
        ckpt_val = syco_phases.get("ckpt", {}).get(metric_name)
        delta = (
            (ckpt_val - base_val) if isinstance(base_val, (int, float)) and isinstance(ckpt_val, (int, float)) else None
        )
        add(f"sycophancy.{metric_name}", base_val, ckpt_val, delta, spec=thresholds.get("sycophancy_mean"))
    for sub in ("are_you_sure", "answer"):
        base_sub = syco_phases.get("base", {}).get("subsets", {}).get(sub, {})
        ckpt_sub = syco_phases.get("ckpt", {}).get("subsets", {}).get(sub, {})
        base_val = base_sub.get("flip_rate_when_correct")
        ckpt_val = ckpt_sub.get("flip_rate_when_correct")
        delta = (
            (ckpt_val - base_val) if isinstance(base_val, (int, float)) and isinstance(ckpt_val, (int, float)) else None
        )
        add(f"sycophancy.{sub}.flip_rate", base_val, ckpt_val, delta)
    base_fb = syco_phases.get("base", {}).get("subsets", {}).get("feedback", {})
    ckpt_fb = syco_phases.get("ckpt", {}).get("subsets", {}).get("feedback", {})
    base_val = base_fb.get("mean_stance_delta_positive")
    ckpt_val = ckpt_fb.get("mean_stance_delta_positive")
    delta = (ckpt_val - base_val) if isinstance(base_val, (int, float)) and isinstance(ckpt_val, (int, float)) else None
    add("sycophancy.feedback.stance_delta", base_val, ckpt_val, delta)

    # ---- mtbench ----
    mt = results.get("mtbench") or {}
    mt_phases = mt.get("phases", {})
    base_val = mt_phases.get("base", {}).get("aggregate", {}).get("overall")
    ckpt_val = mt_phases.get("ckpt", {}).get("aggregate", {}).get("overall")
    delta = (ckpt_val - base_val) if isinstance(base_val, (int, float)) and isinstance(ckpt_val, (int, float)) else None
    add("mtbench.overall", base_val, ckpt_val, delta, spec=thresholds.get("mtbench_overall"))
    for cat in sorted(
        set(
            list(mt_phases.get("base", {}).get("aggregate", {}).get("per_category", {}))
            + list(mt_phases.get("ckpt", {}).get("aggregate", {}).get("per_category", {}))
        )
    ):
        base_cat = mt_phases.get("base", {}).get("aggregate", {}).get("per_category", {}).get(cat)
        ckpt_cat = mt_phases.get("ckpt", {}).get("aggregate", {}).get("per_category", {}).get(cat)
        delta_cat = (
            (ckpt_cat - base_cat) if isinstance(base_cat, (int, float)) and isinstance(ckpt_cat, (int, float)) else None
        )
        add(f"mtbench.{cat}", base_cat, ckpt_cat, delta_cat)

    # ---- smoke ----
    sm = results.get("smoke") or {}
    sm_agg = sm.get("aggregate") or {}
    base_val = sm_agg.get("base", {}).get("median_tokens")
    ckpt_val = sm_agg.get("ckpt", {}).get("median_tokens")
    delta = (ckpt_val - base_val) if isinstance(base_val, (int, float)) and isinstance(ckpt_val, (int, float)) else None
    add("smoke.length_median_tokens", base_val, ckpt_val, delta, spec=thresholds.get("length_median_tokens"))
    for field in ("median_paragraphs", "empty_responses"):
        b = sm_agg.get("base", {}).get(field)
        c = sm_agg.get("ckpt", {}).get(field)
        d = (c - b) if isinstance(b, (int, float)) and isinstance(c, (int, float)) else None
        add(f"smoke.{field}", b, c, d)

    # ---- ifeval ----
    ifv = results.get("ifeval") or {}
    ifv_phases = ifv.get("phases", {})
    for metric_name in (
        "prompt_level_strict_acc",
        "prompt_level_loose_acc",
        "inst_level_strict_acc",
        "inst_level_loose_acc",
    ):
        bv = ifv_phases.get("base", {}).get(metric_name)
        cv = ifv_phases.get("ckpt", {}).get(metric_name)
        dv = (cv - bv) if isinstance(bv, (int, float)) and isinstance(cv, (int, float)) else None
        spec = thresholds.get("ifeval_inst_loose_acc") if metric_name == "inst_level_loose_acc" else None
        add(f"ifeval.{metric_name}", bv, cv, dv, spec=spec)

    # ---- gsm8k ----
    gs = results.get("gsm8k") or {}
    gs_phases = gs.get("phases", {})
    bv = gs_phases.get("base", {}).get("acc")
    cv = gs_phases.get("ckpt", {}).get("acc")
    dv = (cv - bv) if isinstance(bv, (int, float)) and isinstance(cv, (int, float)) else None
    add("gsm8k.acc", bv, cv, dv, spec=thresholds.get("gsm8k_acc"))

    # ---- mmlu ----
    mm = results.get("mmlu") or {}
    mm_phases = mm.get("phases", {})
    bv = mm_phases.get("base", {}).get("acc")
    cv = mm_phases.get("ckpt", {}).get("acc")
    dv = (cv - bv) if isinstance(bv, (int, float)) and isinstance(cv, (int, float)) else None
    add("mmlu.acc_per_char", bv, cv, dv, spec=thresholds.get("mmlu_rc_acc_per_char"))

    any_fail = any(r["pass"] is False for r in rows)
    return {"rows": rows, "any_fail": any_fail}


def _write_csv(path: Path, rollup: dict) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "base", "ckpt", "delta", "threshold", "pass"])
        writer.writeheader()
        for row in rollup["rows"]:
            writer.writerow(row)


def _run_phase(name: str, fn, **kwargs) -> Any:
    print(f"\n{'=' * 70}\n[run_all] PHASE: {name}\n{'=' * 70}\n", flush=True)
    return fn(**kwargs)


def run(
    ckpt: Path,
    output_dir: Path,
    *,
    base_model: str | None = None,
    port: int = 8000,
    max_model_len: int = 4096,
    syco_n: int = 100,
    mtbench_n: int | None = None,
    ifeval_n: int | None = None,
    gsm8k_n: int | None = None,
    mmlu_n: int | None = None,
    skip: tuple[EvalName, ...] = (),
    only: tuple[EvalName, ...] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    base = base_model or infer_base_model(ckpt)
    suite = load_suite()
    thresholds = suite.get("thresholds", {})
    results: dict[str, Any] = {"base_model": base, "ckpt": str(ckpt)}

    def should_run(phase: EvalName) -> bool:
        if only is not None and phase not in only:
            return False
        return phase not in skip

    if should_run("smoke"):
        results["smoke"] = _run_phase(
            "smoke",
            smoke.run,
            ckpt=ckpt,
            output_dir=output_dir / "smoke",
            base_model=base,
            port=port,
            max_model_len=max_model_len,
        )

    if should_run("sycophancy"):
        results["sycophancy"] = _run_phase(
            "sycophancy",
            sycophancy.run,
            ckpt=ckpt,
            output_dir=output_dir / "sycophancy",
            base_model=base,
            n_per_subset=syco_n,
            port=port,
            max_model_len=max_model_len,
        )

    if should_run("mtbench"):
        results["mtbench"] = _run_phase(
            "mtbench",
            mtbench.run,
            ckpt=ckpt,
            output_dir=output_dir / "mtbench",
            base_model=base,
            n=mtbench_n,
            port=port,
            max_model_len=max_model_len,
        )

    if should_run("ifeval"):
        results["ifeval"] = _run_phase(
            "ifeval",
            ifeval.run,
            ckpt=ckpt,
            output_dir=output_dir / "ifeval",
            base_model=base,
            n=ifeval_n,
            port=port,
            max_model_len=max_model_len,
        )

    if should_run("gsm8k"):
        results["gsm8k"] = _run_phase(
            "gsm8k",
            gsm8k.run,
            ckpt=ckpt,
            output_dir=output_dir / "gsm8k",
            base_model=base,
            n=gsm8k_n,
            port=port,
            max_model_len=max_model_len,
        )

    if should_run("mmlu"):
        results["mmlu"] = _run_phase(
            "mmlu",
            mmlu.run,
            ckpt=ckpt,
            output_dir=output_dir / "mmlu",
            base_model=base,
            n=mmlu_n,
            port=port,
            max_model_len=max_model_len,
        )

    rollup = _collect_rollup(results, thresholds)
    (output_dir / "rollup.json").write_text(json.dumps(rollup, indent=2))
    _write_csv(output_dir / "rollup.csv", rollup)

    print("\n" + "=" * 70 + "\n[run_all] ROLLUP (base → ckpt)\n" + "=" * 70)
    print(f"{'metric':<50} {'base':>10} {'ckpt':>10} {'delta':>10}  {'gate':>6} {'threshold'}")
    for row in rollup["rows"]:
        flag = "✓" if row["pass"] is True else ("✗" if row["pass"] is False else "·")

        def fmt(v):
            return f"{v:>10.4f}" if isinstance(v, float) else (f"{v!s:>10}")

        print(
            f"  {row['metric']:<50} {fmt(row['base'])} {fmt(row['ckpt'])} {fmt(row['delta'])}  {flag:>6} {row['threshold'] or ''}"
        )
    if rollup["any_fail"]:
        print("\n[run_all] ≥1 threshold failed — see rollup.csv")
    else:
        print("\n[run_all] All applicable thresholds satisfied.")

    results["rollup"] = rollup
    (output_dir / "all_results.json").write_text(json.dumps(results, indent=2, default=str))
    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-model", type=str, default=None)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--syco-n", type=int, default=100)
    parser.add_argument("--mtbench-n", type=int, default=None)
    parser.add_argument("--ifeval-n", type=int, default=None)
    parser.add_argument("--gsm8k-n", type=int, default=None)
    parser.add_argument("--mmlu-n", type=int, default=None)
    parser.add_argument("--skip", nargs="+", default=[], choices=PHASES)
    parser.add_argument("--only", nargs="+", default=None, choices=PHASES)
    return parser.parse_args()


def main() -> int:
    args = resolve_path_args(_parse_args(), "ckpt", "output_dir")
    run(
        ckpt=args.ckpt,
        output_dir=args.output_dir,
        base_model=args.base_model,
        port=args.port,
        max_model_len=args.max_model_len,
        syco_n=args.syco_n,
        mtbench_n=args.mtbench_n,
        ifeval_n=args.ifeval_n,
        gsm8k_n=args.gsm8k_n,
        mmlu_n=args.mmlu_n,
        skip=typing.cast("tuple[EvalName, ...]", tuple(args.skip)),
        only=typing.cast("tuple[EvalName, ...]", tuple(args.only)) if args.only else None,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
