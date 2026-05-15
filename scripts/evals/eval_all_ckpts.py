"""Sweep the full eval harness over every ckpt in a training run.

For each `<ckpt-root>/step_<N>/` dir, launch `scripts.evals.run_all` against
that ckpt, writing per-step results under `<output-dir>/step_<N>/`. Designed
to run on a single GPU; one ckpt at a time (each run_all phase already boots
its own vLLM server).

Parallelism (if you need it): wrap multiple invocations in a SLURM array,
one array task per ckpt step. This script stays single-process.

Usage:
    uv run --env-file .env python -m scripts.evals.eval_all_ckpts \\
        --ckpt-root /lus/.../outputs/sft-baseline-marin/weights \\
        --output-dir outputs/evals/marin-sweep \\
        --base-model marin-community/marin-8b-base \\
        --only ifeval gsm8k mmlu   # skip judge-eval phases by default

The script skips ckpts that already have a `rollup.json` so it's idempotent
across re-runs.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import typing
from pathlib import Path
from typing import Any

from . import run_all
from ._server import resolve_path_args
from .run_all import EvalName

_STEP_RE = re.compile(r"step_(\d+)$")


def _find_ckpts(ckpt_root: Path, pattern: str = "step_*") -> list[tuple[int, Path]]:
    """Return list of (step, ckpt_dir) sorted by step."""
    out: list[tuple[int, Path]] = []
    for p in sorted(ckpt_root.glob(pattern)):
        if not p.is_dir():
            continue
        m = _STEP_RE.search(p.name)
        if m is None:
            continue
        out.append((int(m.group(1)), p))
    return sorted(out)


def run(
    ckpt_root: Path,
    output_dir: Path,
    *,
    base_model: str | None = None,
    only: tuple[EvalName, ...] | None = None,
    skip: tuple[EvalName, ...] = (),
    force: bool = False,
    syco_n: int = 100,
    mtbench_n: int | None = None,
    ifeval_n: int | None = None,
    gsm8k_n: int | None = None,
    mmlu_n: int | None = None,
    port: int = 8000,
    max_model_len: int = 4096,
) -> dict[str, Any]:
    ckpts = _find_ckpts(ckpt_root)
    if not ckpts:
        raise ValueError(f"No step_N ckpts under {ckpt_root}")
    print(f"[eval_all_ckpts] found {len(ckpts)} ckpts: steps={[s for s, _ in ckpts]}", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"ckpt_root": str(ckpt_root), "steps": {}}
    for step, ckpt_dir in ckpts:
        step_dir = output_dir / f"step_{step}"
        rollup_path = step_dir / "rollup.json"
        if rollup_path.exists() and not force:
            print(f"\n[eval_all_ckpts] skip step_{step} (rollup.json exists)", flush=True)
            summary["steps"][step] = json.loads(rollup_path.read_text())
            continue

        print(
            f"\n{'=' * 70}\n[eval_all_ckpts] EVALUATING step_{step}\n"
            f"  ckpt: {ckpt_dir}\n  out:  {step_dir}\n{'=' * 70}\n",
            flush=True,
        )
        t0 = time.time()
        try:
            result = run_all.run(
                ckpt=ckpt_dir,
                output_dir=step_dir,
                base_model=base_model,
                port=port,
                max_model_len=max_model_len,
                syco_n=syco_n,
                mtbench_n=mtbench_n,
                ifeval_n=ifeval_n,
                gsm8k_n=gsm8k_n,
                mmlu_n=mmlu_n,
                skip=skip,
                only=only,
            )
        except Exception as exc:
            print(f"[eval_all_ckpts] step_{step} FAILED: {exc}", flush=True)
            summary["steps"][step] = {"error": str(exc)}
            continue
        elapsed = time.time() - t0
        print(f"\n[eval_all_ckpts] step_{step} done in {elapsed / 60:.1f}min", flush=True)
        summary["steps"][step] = result.get("rollup", {})

    (output_dir / "sweep_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n[eval_all_ckpts] wrote {output_dir / 'sweep_summary.json'}", flush=True)

    total = len(summary["steps"])
    failed = [str(s) for s, r in summary["steps"].items() if isinstance(r, dict) and "error" in r]
    succeeded = total - len(failed)
    print(
        f"\n[eval_all_ckpts] {succeeded}/{total} steps succeeded, {len(failed)} failed: {failed}",
        flush=True,
    )
    summary["n_total"] = total
    summary["n_succeeded"] = succeeded
    summary["n_failed"] = len(failed)
    summary["failed_steps"] = failed
    return summary


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt-root", type=Path, required=True, help="Directory containing step_N subdirs.")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--base-model", type=str, default=None)
    p.add_argument("--only", nargs="+", default=None, choices=run_all.PHASES)
    p.add_argument("--skip", nargs="+", default=[], choices=run_all.PHASES)
    p.add_argument("--force", action="store_true", help="Re-run even if rollup.json exists.")
    p.add_argument("--syco-n", type=int, default=100)
    p.add_argument("--mtbench-n", type=int, default=None)
    p.add_argument("--ifeval-n", type=int, default=None)
    p.add_argument("--gsm8k-n", type=int, default=None)
    p.add_argument("--mmlu-n", type=int, default=None)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--max-model-len", type=int, default=4096)
    return p.parse_args()


def main() -> int:
    args = resolve_path_args(_parse_args(), "ckpt_root", "output_dir")
    summary = run(
        ckpt_root=args.ckpt_root,
        output_dir=args.output_dir,
        base_model=args.base_model,
        only=typing.cast("tuple[EvalName, ...]", tuple(args.only)) if args.only else None,
        skip=typing.cast("tuple[EvalName, ...]", tuple(args.skip)),
        force=args.force,
        syco_n=args.syco_n,
        mtbench_n=args.mtbench_n,
        ifeval_n=args.ifeval_n,
        gsm8k_n=args.gsm8k_n,
        mmlu_n=args.mmlu_n,
        port=args.port,
        max_model_len=args.max_model_len,
    )
    return 1 if summary.get("n_failed", 0) > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
