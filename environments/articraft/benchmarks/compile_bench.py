"""
Articraft compile 性能基准测试。

测试内容:
  0a. 单次 compile 耗时分布（in-process vs subprocess）
  0b. 多 worker 并发压力测试（4/8/16 workers）
  0c. 空 scaffold compile 基线

用法:
  cd /path/to/articraft
  uv run python /path/to/prime-rl/environments/articraft/benchmarks/compile_bench.py \
      --data-root ./data/records \
      --sample-size 20 \
      --concurrency 1,4,8,16

环境变量:
  URDF_COMPILE_TIMEOUT_SECONDS=0   禁用子进程超时（用于 in-process 基线）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import statistics
import sys
import tempfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class CompileResult:
    record_id: str
    mode: Literal["in_process", "subprocess"]
    elapsed_s: float
    success: bool
    status: str = ""
    n_signals: int = 0
    n_blocking: int = 0
    n_warnings: int = 0
    error: str = ""
    model_py_lines: int = 0


@dataclass
class ConcurrencyResult:
    n_workers: int
    n_tasks: int
    wall_time_s: float
    results: list[CompileResult] = field(default_factory=list)


SCAFFOLD_CODE = """\
from __future__ import annotations
from sdk import ArticulatedObject, TestContext, TestReport

def build_object_model() -> ArticulatedObject:
    model = ArticulatedObject(name="draft_model")
    return model

def run_tests() -> TestReport:
    ctx = TestContext(object_model)
    return ctx.report()

object_model = build_object_model()
"""


def discover_records(data_root: Path, sample_size: int, seed: int = 42) -> list[Path]:
    """Find records with model.py, sample randomly."""
    candidates = []
    for rec_dir in sorted(data_root.iterdir()):
        if not rec_dir.is_dir() or not rec_dir.name.startswith("rec_"):
            continue
        record_json = rec_dir / "record.json"
        if not record_json.exists():
            continue
        try:
            meta = json.loads(record_json.read_text())
        except Exception:
            continue
        model_py_rel = (meta.get("artifacts") or {}).get("model_py")
        if not model_py_rel:
            continue
        model_py = rec_dir / model_py_rel
        if model_py.exists():
            candidates.append(model_py)

    print(f"Found {len(candidates)} records with model.py")
    rng = random.Random(seed)
    if len(candidates) > sample_size:
        candidates = rng.sample(candidates, sample_size)
    return candidates


def prepare_work_copy(model_py: Path, work_dir: Path) -> Path:
    """Copy model.py into an isolated work directory for safe compilation."""
    rec_name = model_py.parent.parent.parent.name
    dest_dir = work_dir / rec_name
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "model.py"
    shutil.copy2(model_py, dest)
    return dest


def _compile_one_in_process(script_path: str, sdk_package: str) -> dict:
    """Worker function: compile in-process (no subprocess timeout wrapper)."""
    from agent.compiler import compile_urdf_report

    path = Path(script_path)
    t0 = time.perf_counter()
    try:
        report = compile_urdf_report(path, sdk_package=sdk_package)
        elapsed = time.perf_counter() - t0
        bundle = report.signal_bundle
        signals = list(bundle.signals)
        return {
            "success": True,
            "elapsed_s": elapsed,
            "status": bundle.status,
            "n_signals": len(signals),
            "n_blocking": sum(1 for s in signals if s.blocking),
            "n_warnings": sum(1 for s in signals if s.severity == "warning"),
            "error": "",
        }
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        return {
            "success": False,
            "elapsed_s": elapsed,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "n_signals": 0,
            "n_blocking": 0,
            "n_warnings": 0,
        }


def _compile_one_subprocess(script_path: str, sdk_package: str) -> dict:
    """Worker function: compile via subprocess timeout wrapper."""
    from agent.compiler import compile_urdf_report_maybe_timeout

    path = Path(script_path)
    t0 = time.perf_counter()
    try:
        report = compile_urdf_report_maybe_timeout(path, sdk_package=sdk_package)
        elapsed = time.perf_counter() - t0
        bundle = report.signal_bundle
        signals = list(bundle.signals)
        return {
            "success": True,
            "elapsed_s": elapsed,
            "status": bundle.status,
            "n_signals": len(signals),
            "n_blocking": sum(1 for s in signals if s.blocking),
            "n_warnings": sum(1 for s in signals if s.severity == "warning"),
            "error": "",
        }
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        return {
            "success": False,
            "elapsed_s": elapsed,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "n_signals": 0,
            "n_blocking": 0,
            "n_warnings": 0,
        }


def run_single_benchmarks(
    scripts: list[tuple[str, Path]],
    sdk_package: str,
) -> tuple[list[CompileResult], list[CompileResult]]:
    """Run each script once in-process and once via subprocess, sequentially."""
    in_process_results: list[CompileResult] = []
    subprocess_results: list[CompileResult] = []

    for record_id, script_path in scripts:
        lines = script_path.read_text().count("\n")

        # --- in-process ---
        print(f"  [in_process] {record_id} ({lines} lines)...", end=" ", flush=True)
        old_env = os.environ.get("URDF_COMPILE_TIMEOUT_SECONDS")
        os.environ["URDF_COMPILE_TIMEOUT_SECONDS"] = "0"
        try:
            r = _compile_one_in_process(str(script_path), sdk_package)
        finally:
            if old_env is None:
                os.environ.pop("URDF_COMPILE_TIMEOUT_SECONDS", None)
            else:
                os.environ["URDF_COMPILE_TIMEOUT_SECONDS"] = old_env

        res = CompileResult(
            record_id=record_id,
            mode="in_process",
            elapsed_s=r["elapsed_s"],
            success=r["success"],
            status=r["status"],
            n_signals=r["n_signals"],
            n_blocking=r["n_blocking"],
            n_warnings=r["n_warnings"],
            error=r["error"],
            model_py_lines=lines,
        )
        in_process_results.append(res)
        status = "OK" if res.success else f"FAIL({res.error[:60]})"
        print(f"{res.elapsed_s:.2f}s {status}")

        # --- subprocess ---
        print(f"  [subprocess] {record_id} ({lines} lines)...", end=" ", flush=True)
        old_env = os.environ.get("URDF_COMPILE_TIMEOUT_SECONDS")
        os.environ["URDF_COMPILE_TIMEOUT_SECONDS"] = "120"
        try:
            r = _compile_one_subprocess(str(script_path), sdk_package)
        finally:
            if old_env is None:
                os.environ.pop("URDF_COMPILE_TIMEOUT_SECONDS", None)
            else:
                os.environ["URDF_COMPILE_TIMEOUT_SECONDS"] = old_env

        res = CompileResult(
            record_id=record_id,
            mode="subprocess",
            elapsed_s=r["elapsed_s"],
            success=r["success"],
            status=r["status"],
            n_signals=r["n_signals"],
            n_blocking=r["n_blocking"],
            n_warnings=r["n_warnings"],
            error=r["error"],
            model_py_lines=lines,
        )
        subprocess_results.append(res)
        status = "OK" if res.success else f"FAIL({res.error[:60]})"
        print(f"{res.elapsed_s:.2f}s {status}")

    return in_process_results, subprocess_results


def _pool_worker(args: tuple[str, str, str]) -> dict:
    """Top-level function for ProcessPoolExecutor (must be picklable)."""
    script_path, sdk_package, mode = args
    if mode == "in_process":
        os.environ["URDF_COMPILE_TIMEOUT_SECONDS"] = "0"
        return _compile_one_in_process(script_path, sdk_package)
    else:
        os.environ["URDF_COMPILE_TIMEOUT_SECONDS"] = "120"
        return _compile_one_subprocess(script_path, sdk_package)


def run_concurrency_benchmark(
    scripts: list[tuple[str, Path]],
    sdk_package: str,
    n_workers: int,
    mode: Literal["in_process", "subprocess"] = "subprocess",
) -> ConcurrencyResult:
    """Run all scripts with N parallel workers."""
    tasks = [(str(sp), sdk_package, mode) for _, sp in scripts]

    wall_start = time.perf_counter()
    results: list[CompileResult] = []

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(_pool_worker, task): scripts[i][0]
            for i, task in enumerate(tasks)
        }
        for future in as_completed(futures):
            record_id = futures[future]
            try:
                r = future.result()
            except Exception as exc:
                r = {
                    "success": False,
                    "elapsed_s": 0.0,
                    "status": "pool_error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "n_signals": 0,
                    "n_blocking": 0,
                    "n_warnings": 0,
                }
            lines = 0
            for rid, sp in scripts:
                if rid == record_id:
                    lines = sp.read_text().count("\n")
                    break
            results.append(CompileResult(
                record_id=record_id,
                mode=mode,
                elapsed_s=r["elapsed_s"],
                success=r["success"],
                status=r["status"],
                n_signals=r["n_signals"],
                n_blocking=r["n_blocking"],
                n_warnings=r["n_warnings"],
                error=r["error"],
                model_py_lines=lines,
            ))

    wall_time = time.perf_counter() - wall_start
    return ConcurrencyResult(
        n_workers=n_workers,
        n_tasks=len(tasks),
        wall_time_s=wall_time,
        results=results,
    )


def run_scaffold_benchmark(work_dir: Path, sdk_package: str, n_runs: int = 5) -> list[float]:
    """Compile the empty scaffold multiple times to measure baseline latency."""
    scaffold_dir = work_dir / "_scaffold"
    scaffold_dir.mkdir(parents=True, exist_ok=True)
    scaffold_path = scaffold_dir / "model.py"
    scaffold_path.write_text(SCAFFOLD_CODE)

    timings: list[float] = []
    for i in range(n_runs):
        print(f"  [scaffold] run {i+1}/{n_runs}...", end=" ", flush=True)
        os.environ["URDF_COMPILE_TIMEOUT_SECONDS"] = "0"
        r = _compile_one_in_process(str(scaffold_path), sdk_package)
        timings.append(r["elapsed_s"])
        status = "OK" if r["success"] else f"FAIL({r['error'][:60]})"
        print(f"{r['elapsed_s']:.2f}s {status}")
    return timings


def print_summary(label: str, results: list[CompileResult]) -> None:
    times = [r.elapsed_s for r in results]
    successes = [r for r in results if r.success]
    failures = [r for r in results if not r.success]

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Total: {len(results)}  Success: {len(successes)}  Fail: {len(failures)}")
    if times:
        print(f"  Latency (s): min={min(times):.2f}  median={statistics.median(times):.2f}  "
              f"mean={statistics.mean(times):.2f}  max={max(times):.2f}  "
              f"stdev={statistics.stdev(times):.2f}" if len(times) > 1 else
              f"  Latency (s): {times[0]:.2f}")
        p90 = sorted(times)[int(len(times) * 0.9)] if len(times) >= 10 else max(times)
        print(f"  P90={p90:.2f}s")
    if failures:
        print(f"  Failure samples:")
        for f in failures[:5]:
            print(f"    {f.record_id}: {f.error[:80]}")


def print_concurrency_summary(cr: ConcurrencyResult) -> None:
    times = [r.elapsed_s for r in cr.results]
    successes = sum(1 for r in cr.results if r.success)
    sum_compute = sum(times)

    print(f"\n{'='*60}")
    print(f"  Concurrency: {cr.n_workers} workers × {cr.n_tasks} tasks")
    print(f"{'='*60}")
    print(f"  Wall time:     {cr.wall_time_s:.2f}s")
    print(f"  Sum(compute):  {sum_compute:.2f}s")
    print(f"  Parallelism:   {sum_compute / cr.wall_time_s:.2f}x" if cr.wall_time_s > 0 else "")
    print(f"  Throughput:    {cr.n_tasks / cr.wall_time_s:.2f} compiles/s" if cr.wall_time_s > 0 else "")
    print(f"  Success rate:  {successes}/{cr.n_tasks}")
    if times:
        print(f"  Per-task (s):  min={min(times):.2f}  median={statistics.median(times):.2f}  "
              f"max={max(times):.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Articraft compile benchmark")
    parser.add_argument("--data-root", type=Path, default=Path("data/records"),
                        help="Path to articraft records directory")
    parser.add_argument("--sample-size", type=int, default=20,
                        help="Number of records to sample")
    parser.add_argument("--concurrency", type=str, default="1,4,8",
                        help="Comma-separated concurrency levels for stress test")
    parser.add_argument("--sdk-package", type=str, default="sdk")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-single", action="store_true",
                        help="Skip single-threaded benchmark")
    parser.add_argument("--skip-concurrent", action="store_true",
                        help="Skip concurrency benchmark")
    parser.add_argument("--subprocess-mode", choices=["in_process", "subprocess"],
                        default="subprocess",
                        help="Mode for concurrency benchmark")
    args = parser.parse_args()

    concurrency_levels = [int(x.strip()) for x in args.concurrency.split(",")]

    print("=" * 60)
    print("  Articraft Compile Benchmark")
    print("=" * 60)

    # --- Discover & prepare ---
    model_pys = discover_records(args.data_root, args.sample_size, args.seed)
    if not model_pys:
        print("ERROR: No records found. Check --data-root.")
        sys.exit(1)

    work_dir = Path(tempfile.mkdtemp(prefix="articraft_bench_"))
    print(f"Work directory: {work_dir}")

    scripts: list[tuple[str, Path]] = []
    for mp in model_pys:
        rec_id = mp.parent.parent.parent.name
        work_copy = prepare_work_copy(mp, work_dir)
        scripts.append((rec_id, work_copy))

    # --- 0c: Scaffold baseline ---
    print("\n--- Step 0c: Scaffold compile baseline ---")
    scaffold_times = run_scaffold_benchmark(work_dir, args.sdk_package)
    print(f"\n  Scaffold latency: {[f'{t:.2f}s' for t in scaffold_times]}")
    if scaffold_times:
        print(f"  Mean: {statistics.mean(scaffold_times):.2f}s")

    # --- 0a: Single-threaded benchmarks ---
    if not args.skip_single:
        print("\n--- Step 0a: Single compile benchmarks ---")
        in_proc, subproc = run_single_benchmarks(scripts, args.sdk_package)
        print_summary("In-process (URDF_COMPILE_TIMEOUT_SECONDS=0)", in_proc)
        print_summary("Subprocess (URDF_COMPILE_TIMEOUT_SECONDS=120)", subproc)

        # Overhead analysis
        paired = []
        for ip in in_proc:
            for sp in subproc:
                if ip.record_id == sp.record_id and ip.success and sp.success:
                    paired.append((ip.elapsed_s, sp.elapsed_s))
                    break
        if paired:
            overheads = [sp - ip for ip, sp in paired]
            ratios = [sp / ip if ip > 0 else float("inf") for ip, sp in paired]
            print(f"\n  Subprocess overhead (paired, n={len(paired)}):")
            print(f"    Absolute: min={min(overheads):.2f}s  median={statistics.median(overheads):.2f}s  max={max(overheads):.2f}s")
            finite_ratios = [r for r in ratios if r != float("inf")]
            if finite_ratios:
                print(f"    Ratio:    min={min(finite_ratios):.2f}x  median={statistics.median(finite_ratios):.2f}x  max={max(finite_ratios):.2f}x")

    # --- 0b: Concurrency benchmarks ---
    if not args.skip_concurrent:
        print("\n--- Step 0b: Concurrency benchmarks ---")
        for n in concurrency_levels:
            print(f"\n  Running {len(scripts)} compiles with {n} workers ({args.subprocess_mode})...")
            cr = run_concurrency_benchmark(scripts, args.sdk_package, n, args.subprocess_mode)
            print_concurrency_summary(cr)

    # --- Throughput estimation for RL training ---
    print("\n" + "=" * 60)
    print("  RL Training Throughput Estimation")
    print("=" * 60)
    if not args.skip_single:
        median_compile = statistics.median([r.elapsed_s for r in subproc]) if subproc else 10.0
    else:
        median_compile = 10.0
    for compiles_per_rollout in [3, 5, 8]:
        for group_size in [4, 8]:
            serial_time = compiles_per_rollout * median_compile * group_size
            for n_cpu_workers in [4, 8, 16, 32]:
                parallel_time = serial_time / min(n_cpu_workers, group_size * compiles_per_rollout)
                print(f"  {compiles_per_rollout} compiles/rollout × {group_size} rollouts/group "
                      f"× {median_compile:.1f}s/compile → "
                      f"serial={serial_time:.0f}s  "
                      f"w/{n_cpu_workers} workers={parallel_time:.1f}s")

    # --- Cleanup ---
    print(f"\nWork directory retained at: {work_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
