#!/usr/bin/env python3
"""Overnight SFT profiling campaign.

Sweeps 2 models x 2 dtypes x 16 sequence lengths (8192..131072, step 8192), runs a short
traced SFT run for each combination (EP=8, flash attention, 8 GPUs) and post-processes
each rank-0 trace into per-step perfetto traces + an averaged-step flamegraph trace.

Resume-safe: combinations whose processed summary already exists are skipped.

Usage:
    uv run python benchmarks/profiling/run_campaign.py [--runs-dir benchmarks/profiling/runs]
        [--models qwen3-30b glm-4.5] [--dtypes bf16 mxfp8] [--seq-lens 8192 16384 ...]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = Path(__file__).resolve().parent / "configs"

MODELS = ["qwen3-30b", "glm-4.5"]
DTYPES = ["bf16", "mxfp8"]
ALL_DTYPES = ["bf16", "mxfp8", "fp8"]  # fp8 (blockwise) is the Hopper alternative to mxfp8 (SM100-only)
SEQ_LENS = list(range(8192, 128 * 1024 + 1, 8192))

RUN_TIMEOUT_S = 100 * 60
MIN_FREE_DISK_GB = 25


def is_known_bad(model: str, dtype: str, seq_len: int) -> str | None:
    """Return a reason string for combinations known to fail, or None."""
    # torchao's mxfp8 CUDA dim1 cast kernel raises "CUDA error: invalid configuration
    # argument" in the wgrad path once the routed token count gets large enough
    # (observed: qwen3-30b top_k=8 at seq_len 98304; 90112 still passes).
    if model == "qwen3-30b" and dtype == "mxfp8" and seq_len >= 98304:
        return "torchao mxfp8 CUDA dim1 cast kernel launch-config limit"
    # torchao's mxfp8 a2a dispatch (a2a_dispatch_mxfp8_fwd_hp_bwd) hits an illegal memory
    # access at large token counts (observed: glm-4.5 at seq_len 114688; 106496 still passes).
    if model == "glm-4.5" and dtype == "mxfp8" and seq_len >= 114688:
        return "torchao mxfp8 a2a dispatch illegal memory access at large token counts"
    return None


def free_disk_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1024**3


def run_name(model: str, dtype: str, seq_len: int) -> str:
    return f"{model}/{dtype}/seq_{seq_len}"


def launch_run(
    model: str, dtype: str, seq_len: int, run_dir: Path, extra_args: list[str] | None = None
) -> tuple[bool, float, str]:
    """Launch one traced SFT run. Returns (success, duration_s, error_tail)."""
    cmd = [
        "uv",
        "run",
        "sft",
        "@",
        str(CONFIG_DIR / f"{model}.toml"),
        "--output-dir",
        str(run_dir / "output"),
        "--trace-path",
        str(run_dir / "trace"),
        "--data.seq-len",
        str(seq_len),
    ]
    if dtype in ("mxfp8", "fp8"):
        cmd += ["--model.quantization.type", dtype]
    if extra_args:
        cmd += extra_args

    log_path = run_dir / "launcher.log"
    run_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with open(log_path, "w") as log_file:
        proc = subprocess.Popen(
            cmd, cwd=REPO_ROOT, stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            returncode = proc.wait(timeout=RUN_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            try:
                proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                proc.wait()
            return False, time.perf_counter() - start, f"timeout after {RUN_TIMEOUT_S}s"
    duration = time.perf_counter() - start
    if returncode != 0:
        tail = "".join(log_path.read_text(errors="replace").splitlines(keepends=True)[-30:])
        return False, duration, f"exit code {returncode}\n{tail}"
    return True, duration, ""


def post_process(run_dir: Path) -> dict:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from trace_tools import process_trace

    trace_file = run_dir / "trace" / "trace_0.json.gz"
    if not trace_file.exists():
        raise FileNotFoundError(f"Trace not found: {trace_file}")
    return process_trace(trace_file, run_dir / "processed")


def append_manifest(manifest_path: Path, record: dict) -> None:
    with open(manifest_path, "a") as f:
        f.write(json.dumps(record) + "\n")


def write_rollup(runs_dir: Path, combos: list[tuple[str, str, int]]) -> None:
    rollup_path = runs_dir / "rollup.csv"
    rows = []
    for model, dtype, seq_len in combos:
        summary_path = runs_dir / model / dtype / f"seq_{seq_len}" / "processed" / "summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text())
        annotations = summary.get("avg_annotation_ms", {})
        rows.append(
            {
                "model": model,
                "dtype": dtype,
                "seq_len": seq_len,
                "steps_averaged": summary["num_steps"],
                "avg_step_wall_ms": summary["avg_wall_ms"],
                "avg_gpu_busy_ms": summary["avg_gpu_busy_ms"],
                "avg_forward_ms": annotations.get("forward", ""),
                "avg_backward_ms": annotations.get("backward", ""),
            }
        )
    if not rows:
        return
    with open(rollup_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote rollup: {rollup_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=Path(__file__).resolve().parent / "runs")
    parser.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    parser.add_argument("--dtypes", nargs="+", default=DTYPES, choices=ALL_DTYPES)
    parser.add_argument("--seq-lens", nargs="+", type=int, default=SEQ_LENS)
    parser.add_argument("--keep-raw-trace", action="store_true", help="Keep the raw multi-step trace after processing")
    parser.add_argument(
        "--extra-args",
        type=str,
        default="",
        help='Extra CLI overrides appended to every sft invocation, e.g. "--model.optim-cpu-offload false"',
    )
    args = parser.parse_args()
    extra_args = args.extra_args.split()

    runs_dir = args.runs_dir
    runs_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = runs_dir / "manifest.jsonl"

    # seq_len-major order: if the campaign is cut short, every model/dtype still has a
    # complete cross-section up to some sequence length.
    combos = [(m, d, s) for s in args.seq_lens for m in args.models for d in args.dtypes]
    print(f"Campaign: {len(combos)} runs -> {runs_dir}")

    for i, (model, dtype, seq_len) in enumerate(combos, 1):
        name = run_name(model, dtype, seq_len)
        run_dir = runs_dir / model / dtype / f"seq_{seq_len}"
        summary_path = run_dir / "processed" / "summary.json"
        if summary_path.exists():
            print(f"[{i}/{len(combos)}] {name}: already done, skipping")
            continue

        skip_reason = is_known_bad(model, dtype, seq_len)
        if skip_reason is not None:
            print(f"[{i}/{len(combos)}] {name}: skipping ({skip_reason})")
            continue

        if free_disk_gb(runs_dir) < MIN_FREE_DISK_GB:
            print(f"ABORT: less than {MIN_FREE_DISK_GB}GB free disk, stopping campaign")
            break

        print(f"[{i}/{len(combos)}] {name}: launching", flush=True)
        record = {
            "name": name,
            "model": model,
            "dtype": dtype,
            "seq_len": seq_len,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        ok, duration, error = launch_run(model, dtype, seq_len, run_dir, extra_args=extra_args)
        record["duration_s"] = round(duration, 1)
        if not ok:
            record["status"] = "failed"
            record["error"] = error
            append_manifest(manifest_path, record)
            print(f"[{i}/{len(combos)}] {name}: FAILED after {duration:.0f}s: {error.splitlines()[0]}", flush=True)
            continue

        try:
            summary = post_process(run_dir)
            record["status"] = "ok"
            record["avg_step_wall_ms"] = summary["avg_wall_ms"]
            record["avg_gpu_busy_ms"] = summary["avg_gpu_busy_ms"]
            if not args.keep_raw_trace:
                # Raw multi-step trace is redundant with the per-step splits; drop it to save disk.
                raw = run_dir / "trace" / "trace_0.json.gz"
                processed_ok = (run_dir / "processed" / "average.json.gz").exists()
                if processed_ok and raw.exists():
                    raw.unlink()
            print(
                f"[{i}/{len(combos)}] {name}: OK in {duration:.0f}s, avg step {summary['avg_wall_ms']:.0f}ms",
                flush=True,
            )
        except Exception as e:  # noqa: BLE001 — keep the campaign alive overnight; failure is recorded
            record["status"] = "postprocess_failed"
            record["error"] = str(e)
            print(f"[{i}/{len(combos)}] {name}: post-processing FAILED: {e}", flush=True)
        append_manifest(manifest_path, record)

    write_rollup(runs_dir, combos)
    print("Campaign finished")


if __name__ == "__main__":
    main()
