"""Invoke olmes (Ai2's eval harness) against a local checkpoint.

olmes at https://github.com/allenai/olmes pins transformers<5, torch<2.9,
vllm==0.11.0. Those pins conflict with prime-rl's (transformers 5.5, torch 2.9,
vllm 0.19) but the runtime code is compatible with our newer versions — we
install olmes + leaf deps via `scripts/install_evals.sh` (uses `uv pip install
--no-deps` so prime-rl's core pins stay intact) and invoke olmes directly in
the shared venv. See `skills/installation/SKILL.md` for the install recipe.

olmes loads the checkpoint from disk, runs the task suite, and writes
`metrics-all.jsonl` + per-task dirs to `--output-dir`.

Usage:
    uv run python -m scripts.evals.run_olmes \\
        --ckpt /scratch/.../weights/step_500 \\
        --output-dir outputs/evals/<ckpt_slug>/olmes \\
        --gpus 4

Or programmatic: `from scripts.evals.run_olmes import run` → returns dict of metrics.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from ._server import load_suite, resolve_path_args


def build_olmes_cmd(
    *,
    ckpt: Path,
    output_dir: Path,
    tasks: list[str],
    gpus: int,
    batch_size: str | None,
    limit: float | None,
    model_args: str | None,
    model_type: str = "hf",
) -> list[str]:
    """Construct the `olmes ...` command. Runs in the current uv-managed venv."""
    olmes_bin = shutil.which("olmes")
    if olmes_bin is None:
        raise RuntimeError(
            "`olmes` not found on PATH. Install with `bash scripts/install_evals.sh` "
            "(uses uv pip install --no-deps to coexist with prime-rl's core pins)."
        )
    # Use `uv run --no-sync` to prevent uv from reconciling the venv back to
    # the prime-rl lockfile (which doesn't know about the olmes-side pins we
    # installed via install_evals.sh — e.g. antlr4-python3-runtime==4.11,
    # lm_eval==0.4.11, etc.).
    cmd = [
        "uv",
        "run",
        "--no-sync",
        olmes_bin,
        "--model",
        str(ckpt),
        "--model-type",
        model_type,
        "--task",
        *tasks,
        "--output-dir",
        str(output_dir),
        "--gpus",
        str(gpus),
    ]
    if batch_size is not None:
        cmd.extend(["--batch-size", batch_size])
    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    if model_args is not None:
        cmd.extend(["--model-args", model_args])
    return cmd


def parse_metrics(output_dir: Path) -> dict[str, dict]:
    """Read `metrics-all.jsonl` and return a dict keyed by task name."""
    metrics_file = output_dir / "metrics-all.jsonl"
    if not metrics_file.exists():
        raise FileNotFoundError(f"olmes did not produce {metrics_file}. Something went wrong during the run.")
    out: dict[str, dict] = {}
    with open(metrics_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            task_name = record.get("task_name") or record.get("task", "unknown")
            out[task_name] = record
    return out


def run(
    ckpt: Path,
    output_dir: Path,
    *,
    tasks: list[str] | None = None,
    gpus: int = 1,
    batch_size: str | None = None,
    limit: float | None = None,
    model_args: str | None = None,
    model_type: str = "hf",
    dry_run: bool = False,
) -> dict[str, dict]:
    """Run olmes and return parsed metrics keyed by task name.

    If `tasks` is None, the list is sourced from the locked Rung-6 suite at
    `configs/evals/rung6_suite.toml`.
    """
    suite = load_suite()
    resolved_tasks = tasks if tasks is not None else list(suite["olmes"]["tasks"])

    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_olmes_cmd(
        ckpt=ckpt,
        output_dir=output_dir,
        tasks=resolved_tasks,
        gpus=gpus,
        batch_size=batch_size,
        limit=limit,
        model_args=model_args,
        model_type=model_type,
    )

    print(f"[run_olmes] cmd: {' '.join(cmd)}", flush=True)
    if dry_run:
        return {}

    log_path = output_dir / "olmes.log"
    env = os.environ.copy()
    with open(log_path, "w") as log_file:
        proc = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"olmes exited with code {proc.returncode}. Log: {log_path}")

    metrics = parse_metrics(output_dir)
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[run_olmes] wrote summary to {summary_path}", flush=True)
    return metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True, help="HF-compatible checkpoint dir.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory.")
    parser.add_argument("--gpus", type=int, default=1, help="GPUs olmes's internal vLLM uses.")
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=None,
        help="Override task list (default: Rung-6 suite).",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="hf",
        choices=["hf", "vllm", "litellm"],
        help="olmes backend. hf = transformers path (always works). "
        "vllm = faster inference via olmes's internal vLLM. "
        "litellm = proxy through our prime-rl inference server.",
    )
    parser.add_argument("--batch-size", type=str, default=None, help="Passed to olmes --batch-size.")
    parser.add_argument(
        "--limit",
        type=float,
        default=None,
        help="Subsample N items per task (or fraction < 1). For smoke/debug only.",
    )
    parser.add_argument("--model-args", type=str, default=None, help="Passed to olmes --model-args.")
    parser.add_argument("--dry-run", action="store_true", help="Print the command; don't run.")
    return parser.parse_args()


def main() -> int:
    args = resolve_path_args(_parse_args(), "ckpt", "output_dir")
    metrics = run(
        ckpt=args.ckpt,
        output_dir=args.output_dir,
        tasks=args.tasks,
        gpus=args.gpus,
        batch_size=args.batch_size,
        limit=args.limit,
        model_args=args.model_args,
        model_type=args.model_type,
        dry_run=args.dry_run,
    )
    print(json.dumps({task: m.get("metrics", m) for task, m in metrics.items()}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
