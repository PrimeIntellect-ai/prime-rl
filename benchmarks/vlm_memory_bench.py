"""Benchmark memory usage of VLM rollout processing with and without disk offloading.

Generates rollouts with triangle-duplicated screenshots (same pattern as multi-turn
VLM environments) and measures RSS at each stage. Simulates ZMQ msgpack roundtrip
to create real string copies (matching hosted-rl behavior).

Usage:
    uv run python benchmarks/vlm_memory_bench.py [--num-rollouts 32] [--num-turns 10]
"""

import argparse
import base64
import ctypes
import gc
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "tic_tac_toe"))


def get_rss_mb() -> float:
    return psutil.Process().memory_info().rss / 1024 / 1024


def malloc_trim():
    """Call glibc malloc_trim to return freed memory to OS."""
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def measure_rss() -> float:
    """Force GC + malloc_trim, then measure RSS."""
    gc.collect()
    malloc_trim()
    return get_rss_mb()


def _render_screenshot(seed, screen_width, screen_height):
    """Render a realistic-sized screenshot with random content.

    Uses numpy for speed. Produces ~500KB-1MB base64 at 512x384,
    matching typical browser screenshot sizes.
    """
    import io

    import numpy as np
    from PIL import Image

    rng = np.random.RandomState(seed)
    arr = rng.randint(0, 256, (screen_height, screen_width, 3), dtype=np.uint8)
    img = Image.fromarray(arr)

    buf = io.BytesIO()
    img.save(buf, format="PNG", compress_level=9)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def generate_rollouts(num_rollouts, num_turns, screen_width, screen_height):
    """Generate synthetic rollouts with triangle-duplicated screenshots.

    Each rollout has num_turns steps, each step's prompt contains the FULL
    cumulative conversation (triangle duplication pattern). Screenshots are
    realistically-sized noisy images.
    """

    # Pre-render a pool of unique screenshots
    print(f"  Pre-rendering screenshots ({screen_width}x{screen_height})...", end=" ", flush=True)
    total_unique = num_rollouts * num_turns
    sample = _render_screenshot(0, screen_width, screen_height)
    sample_kb = len(sample) / 1024
    print(f"~{sample_kb:.0f}KB each, {total_unique} total needed")
    screenshot_pool = [sample]
    for idx in range(1, total_unique):
        screenshot_pool.append(_render_screenshot(idx, screen_width, screen_height))

    rollouts = []
    for i in range(num_rollouts):
        trajectory = []
        conversation_history = []

        for turn in range(num_turns):
            img_url = screenshot_pool[i * num_turns + turn]
            obs_msg = {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Turn {turn + 1}"},
                    {"type": "image_url", "image_url": {"url": img_url}},
                ],
            }
            conversation_history.append(obs_msg)
            asst_msg = {"role": "assistant", "content": f"Turn {turn + 1} done."}
            conversation_history.append(asst_msg)

            step = {
                "prompt": list(conversation_history),
                "completion": [asst_msg],
                "tokens": None,
            }
            trajectory.append(step)

        rollouts.append(
            {
                "trajectory": trajectory,
                "example_id": i,
                "reward": 1.0,
                "error": None,
                "task": "tic_tac_toe",
                "sampling_args": {"temperature": 1.0},
            }
        )

    del screenshot_pool
    return rollouts


def simulate_zmq(rollouts):
    """Simulate ZMQ msgpack roundtrip — creates fresh string objects for everything."""
    import msgpack

    packed = msgpack.packb(rollouts, use_bin_type=True)
    payload_mb = len(packed) / 1024 / 1024
    result = msgpack.unpackb(packed, raw=False)
    del packed
    return result, payload_mb


def count_images(rollouts):
    count = 0
    total_bytes = 0
    for r in rollouts:
        for step in r.get("trajectory", []):
            prompt = step.get("prompt", [])
            if not isinstance(prompt, list):
                continue
            for msg in prompt:
                content = msg.get("content", [])
                if not isinstance(content, list):
                    continue
                for item in content:
                    if item.get("type") == "image_url":
                        url = item.get("image_url", {}).get("url", "")
                        if url.startswith("data:image"):
                            count += 1
                            total_bytes += len(url)
                        elif url.startswith("file://"):
                            count += 1
    return count, total_bytes


def run_test(label, num_rollouts, num_turns, screen_width, screen_height, use_offload, simulate_zmq_roundtrip):
    """Run a single test: generate rollouts, optionally offload, measure memory."""
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")

    # Pre-import heavy deps so their memory is in the baseline
    if use_offload:
        from prime_rl.orchestrator.trajectories import offload_images_to_disk  # noqa: F811

    baseline = measure_rss()
    print(f"  Baseline RSS: {baseline:.1f} MB")

    # Generate
    t0 = time.perf_counter()
    rollouts = generate_rollouts(num_rollouts, num_turns, screen_width, screen_height)
    gen_time = time.perf_counter() - t0

    img_count, img_bytes = count_images(rollouts)
    avg_turns = sum(len(r["trajectory"]) for r in rollouts) / len(rollouts)
    print(f"  Generated {num_rollouts} rollouts in {gen_time:.1f}s (avg {avg_turns:.0f} turns)")
    print(f"  Base64 image refs: {img_count} ({img_bytes / 1024 / 1024:.1f} MB)")

    after_gen = measure_rss()
    print(f"  RSS after generate: {after_gen:.1f} MB (+{after_gen - baseline:.1f})")

    # ZMQ roundtrip
    if simulate_zmq_roundtrip:
        rollouts, payload_mb = simulate_zmq(rollouts)
        after_zmq = measure_rss()
        print(f"  ZMQ payload: {payload_mb:.1f} MB")
        print(f"  RSS after ZMQ: {after_zmq:.1f} MB (+{after_zmq - baseline:.1f})")

        img_count, img_bytes = count_images(rollouts)
        print(f"  Base64 refs after ZMQ: {img_count} ({img_bytes / 1024 / 1024:.1f} MB)")

    # Offload (if enabled)
    offload_time = 0
    tmpdir = None
    if use_offload:
        tmpdir = Path(tempfile.mkdtemp(prefix="vlm_bench_"))
        t0 = time.perf_counter()
        num_written = offload_images_to_disk(rollouts, tmpdir)
        offload_time = time.perf_counter() - t0

        after_offload = measure_rss()

        images_dir = tmpdir / "assets" / "images"
        disk_files = list(images_dir.glob("*.png"))
        disk_mb = sum(f.stat().st_size for f in disk_files) / 1024 / 1024

        img_count, _ = count_images(rollouts)
        print(f"  Offloaded {num_written} unique images in {offload_time:.3f}s")
        print(f"  Disk usage: {len(disk_files)} files ({disk_mb:.1f} MB)")
        print(f"  RSS after offload: {after_offload:.1f} MB (+{after_offload - baseline:.1f})")

    final_rss = measure_rss()
    delta = final_rss - baseline

    print(f"\n  RESULT: {final_rss:.1f} MB total, +{delta:.1f} MB from baseline")
    print(f"  Per-rollout: {delta / num_rollouts:.2f} MB")

    # Cleanup
    del rollouts
    if tmpdir:
        shutil.rmtree(tmpdir, ignore_errors=True)
    gc.collect()
    malloc_trim()

    return delta


def main():
    parser = argparse.ArgumentParser(description="VLM memory benchmark")
    parser.add_argument("--num-rollouts", type=int, default=32)
    parser.add_argument("--num-turns", type=int, default=10)
    parser.add_argument("--screen-width", type=int, default=1024)
    parser.add_argument("--screen-height", type=int, default=768)
    parser.add_argument("--no-zmq", action="store_true", help="Skip ZMQ roundtrip simulation")
    parser.add_argument("--_run-single", choices=["offload", "no-offload"], help=argparse.SUPPRESS)
    args = parser.parse_args()

    use_zmq = not args.no_zmq

    # Single-test mode (called by subprocess)
    if args._run_single:
        use_offload = args._run_single == "offload"
        label = "WITH disk offloading" if use_offload else "WITHOUT disk offloading (current behavior)"
        run_test(label, args.num_rollouts, args.num_turns, args.screen_width, args.screen_height, use_offload, use_zmq)
        return

    print("=== VLM Memory Benchmark ===")
    print(
        f"Config: {args.num_rollouts} rollouts, {args.num_turns} turns, "
        f"{args.screen_width}x{args.screen_height}, ZMQ={'off' if args.no_zmq else 'on'}"
    )

    # Run each test in a subprocess for clean memory isolation
    def run_isolated(label, use_offload):
        env = dict(os.environ)
        result = subprocess.run(
            [
                sys.executable,
                __file__,
                "--num-rollouts",
                str(args.num_rollouts),
                "--num-turns",
                str(args.num_turns),
                "--screen-width",
                str(args.screen_width),
                "--screen-height",
                str(args.screen_height),
                *(["--no-zmq"] if args.no_zmq else []),
                "--_run-single",
                "offload" if use_offload else "no-offload",
            ],
            capture_output=True,
            text=True,
            env=env,
        )
        print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr)
        # Extract delta from output
        for line in result.stdout.splitlines():
            if "RESULT:" in line:
                # Parse "+123.4 MB from baseline"
                parts = line.split("+")[1].split(" MB")[0]
                return float(parts)
        return 0.0

    no_offload = run_isolated("WITHOUT", False)
    with_offload = run_isolated("WITH", True)

    # Summary
    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    print(f"  WITHOUT offload: +{no_offload:.1f} MB")
    print(f"  WITH offload:    +{with_offload:.1f} MB")
    if no_offload > 0:
        reduction = (no_offload - with_offload) / no_offload * 100
        print(f"  Reduction:       {reduction:.1f}%")
        print(f"  Ratio:           {no_offload / max(with_offload, 0.1):.1f}x less memory")


if __name__ == "__main__":
    main()
