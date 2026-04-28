"""Micro-benchmark for one Blender Cycles render.

Runs ``run_blender`` N times for a single BlenderGym task at the requested
samples/denoiser/compute-device and reports duration percentiles.

Used to (a) verify the OptiX speed-up on a fresh Blender build, (b) decide
between 16 / 32 spp via reward-variance comparison, and (c) form a baseline
for future Persistent-Worker performance regressions.

Example::

    uv run python -m blendergym.scripts.bench_render \\
        --task placement1 --samples 16 --denoiser OPENIMAGEDENOISE \\
        --compute-device OPTIX -n 5
"""

from __future__ import annotations

import argparse
import os
import shutil
import statistics
import sys
import tempfile
from pathlib import Path

from blendergym.render import DEFAULT_BLENDER_BIN, run_blender


def _percentile(values: list[float], p: float) -> float:
    """Linear-interp percentile (p in [0, 100]). Handles 1-element lists."""
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m blendergym.scripts.bench_render",
        description="Render a single BlenderGym task N times and report timings.",
    )
    parser.add_argument(
        "--task",
        required=True,
        help="Task id (e.g. placement1, placement25).",
    )
    parser.add_argument(
        "--data-root",
        default="data/blendergym",
        help="BlenderGym data root (default: data/blendergym).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=16,
        help="Cycles samples per pixel (default 16).",
    )
    parser.add_argument(
        "--denoiser",
        default="OPENIMAGEDENOISE",
        help="Cycles denoiser (default OPENIMAGEDENOISE).",
    )
    parser.add_argument(
        "--compute-device",
        dest="compute_device",
        default="OPTIX",
        help="Cycles compute device (default OPTIX, falls back to CUDA).",
    )
    parser.add_argument(
        "-n",
        "--num-runs",
        type=int,
        default=5,
        help="Number of repeat renders to time (default 5).",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU id (CUDA_VISIBLE_DEVICES inside child).",
    )
    parser.add_argument(
        "--blender-bin",
        default=str(DEFAULT_BLENDER_BIN),
        help="Path to Blender binary.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Per-render timeout in seconds (default 300 — placement25 needs ~10s+ headroom).",
    )
    parser.add_argument(
        "--keep-output",
        action="store_true",
        help="Don't auto-clean the temporary output directory after the bench.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_argparser()
    args = parser.parse_args(argv)

    task_dir = Path(args.data_root).expanduser().resolve() / args.task
    if not task_dir.is_dir():
        sys.stderr.write(f"task dir not found: {task_dir}\n")
        return 2
    blend_file = task_dir / "blender_file.blend"
    start_code_path = task_dir / "start.py"
    if not blend_file.is_file() or not start_code_path.is_file():
        sys.stderr.write(f"task missing blend / start.py: {task_dir}\n")
        return 2
    code_text = start_code_path.read_text(encoding="utf-8")

    os.environ["BLENDERGYM_CYCLES_SAMPLES"] = str(args.samples)
    os.environ["BLENDERGYM_CYCLES_DENOISER"] = args.denoiser
    os.environ["BLENDERGYM_CYCLES_COMPUTE_DEVICE"] = args.compute_device

    print(
        f"[bench] task={args.task} samples={args.samples} "
        f"denoiser={args.denoiser} compute={args.compute_device} "
        f"n={args.num_runs} gpu={args.gpu}"
    )

    durations: list[float] = []
    successes = 0
    timed_outs = 0

    with tempfile.TemporaryDirectory(prefix="bench_render_", suffix=f"_{args.task}") as tmp:
        tmp_path = Path(tmp)
        for i in range(args.num_runs):
            run_dir = tmp_path / f"run_{i}"
            result = run_blender(
                blend_file=blend_file,
                code=code_text,
                output_dir=run_dir,
                blender_bin=args.blender_bin,
                gpu_id=args.gpu,
                timeout=args.timeout,
            )
            durations.append(result.duration_s)
            successes += int(result.success)
            timed_outs += int(result.timed_out)
            print(
                f"  run {i}: success={result.success} "
                f"timed_out={result.timed_out} "
                f"duration={result.duration_s:.2f}s "
                f"returncode={result.returncode}"
            )

        if args.keep_output:
            persist = Path(f"outputs/bench_render_{args.task}").resolve()
            persist.mkdir(parents=True, exist_ok=True)
            for child in tmp_path.iterdir():
                target = persist / child.name
                if target.exists():
                    shutil.rmtree(target, ignore_errors=True)
                # ``shutil.move`` falls back to copy+delete across filesystems
                # (``/tmp`` on tmpfs vs ``/data`` on disk).
                shutil.move(str(child), str(target))
            print(f"[bench] kept output: {persist}")

    print(
        f"[bench] {args.num_runs} runs: "
        f"mean={statistics.fmean(durations):.2f}s "
        f"stdev={statistics.stdev(durations) if len(durations) > 1 else 0.0:.2f}s "
        f"p50={_percentile(durations, 50):.2f}s "
        f"p99={_percentile(durations, 99):.2f}s "
        f"success={successes}/{args.num_runs} "
        f"timeouts={timed_outs}"
    )

    return 0 if successes == args.num_runs else 1


if __name__ == "__main__":
    raise SystemExit(main())
