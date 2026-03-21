"""Benchmark VLM image preprocessing: sequential vs overlapped with interleave_rollout.

Generates synthetic rollout data with images and measures:
1. Sequential: build_vlm_image_cache → interleave_rollout (current main)
2. Overlapped: extract → (preprocess || interleave_rollout) → attach (PR6)

Usage:
    uv run python scripts/bench_vlm_image_processing.py
"""

import base64
import io
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from PIL import Image
from transformers import AutoProcessor

from prime_rl.orchestrator.trajectories import (
    attach_vlm_images,
    build_vlm_image_cache,
    extract_vlm_images,
    interleave_rollout,
    preprocess_vlm_images,
)


def make_color_image(width: int, height: int, color: tuple[int, int, int] = (255, 0, 0)) -> str:
    """Create a solid-color image as a base64 data URL."""
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"


def make_fake_rollout(
    example_id: int,
    num_turns: int = 1,
    images_per_turn: int = 1,
    image_size: tuple[int, int] = (100, 100),
    prompt_tokens: int = 50,
    completion_tokens: int = 20,
    unique_images: bool = True,
) -> dict:
    """Create a fake RolloutOutput-like dict with embedded images."""
    # Pre-generate images (unique or shared)
    if unique_images:
        colors = [
            ((example_id * 37 + t * 13 + i * 7) % 256, (t * 41 + i * 23) % 256, (i * 67) % 256)
            for t in range(num_turns)
            for i in range(images_per_turn)
        ]
        image_urls = [make_color_image(image_size[0], image_size[1], c) for c in colors]
    else:
        shared_url = make_color_image(image_size[0], image_size[1], (128, 128, 128))
        image_urls = [shared_url] * (num_turns * images_per_turn)

    trajectory = []
    img_idx = 0
    for turn in range(num_turns):
        # Build prompt messages with images
        image_items = []
        for _ in range(images_per_turn):
            image_items.append({"type": "image_url", "image_url": {"url": image_urls[img_idx]}})
            img_idx += 1
        image_items.append({"type": "text", "text": f"Turn {turn + 1}"})

        messages = [{"role": "user", "content": image_items}]
        if turn > 0:
            messages = trajectory[turn - 1]["prompt"] + [{"role": "assistant", "content": "ok"}] + messages

        # Fake token data (prompt extends across turns for extension property)
        base_prompt_len = prompt_tokens + turn * (prompt_tokens + completion_tokens)
        step = {
            "prompt": messages,
            "completion": [{"role": "assistant", "content": "response"}],
            "response": {"choices": [{"message": {"content": "response"}}]},
            "tokens": {
                "prompt_ids": list(range(base_prompt_len)),
                "prompt_mask": [False] * base_prompt_len,
                "completion_ids": list(range(1000, 1000 + completion_tokens)),
                "completion_mask": [True] * completion_tokens,
                "completion_logprobs": [-0.1] * completion_tokens,
                "overlong_prompt": False,
                "is_truncated": False,
                "routed_experts": None,
            },
            "reward": 1.0,
            "advantage": 0.5,
            "is_truncated": False,
            "trajectory_id": f"traj_{example_id}",
            "extras": {},
        }
        trajectory.append(step)

    return {
        "example_id": example_id,
        "task": "bench",
        "prompt": trajectory[-1]["prompt"],
        "completion": trajectory[-1]["completion"],
        "reward": 1.0,
        "timing": {"start": 0.0, "end": 1.0, "turns": []},
        "is_completed": True,
        "is_truncated": False,
        "metrics": {},
        "answer": "response",
        "info": {},
        "error": None,
        "stop_condition": None,
        "trajectory": trajectory,
        "tool_defs": [],
        "token_usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "sampling_args": {"temperature": 1.0, "max_tokens": 64},
    }


@dataclass
class BenchResult:
    name: str
    num_rollouts: int
    total_images: int
    sequential_s: float
    overlapped_s: float

    @property
    def speedup(self) -> float:
        return self.sequential_s / self.overlapped_s if self.overlapped_s > 0 else float("inf")

    def __str__(self) -> str:
        return (
            f"  {self.name:<45s} | "
            f"rollouts={self.num_rollouts:>4d} | "
            f"images={self.total_images:>5d} | "
            f"seq={self.sequential_s:>6.2f}s | "
            f"overlap={self.overlapped_s:>6.2f}s | "
            f"speedup={self.speedup:>5.2f}x"
        )


def _simulate_advantage_computation(rollouts: list[dict]) -> list[float]:
    """Simulate the work done between extract and interleave in production."""
    rewards = [r["reward"] for r in rollouts]
    # Mimics compute_advantages: per-example normalization
    n = len(rewards)
    mean = sum(rewards) / max(n, 1)
    return [(r - mean) for r in rewards]


def bench_sequential(rollouts: list[dict], processor) -> float:
    """Time the sequential path: build_vlm_image_cache → advantages → interleave_rollout."""
    start = time.perf_counter()
    vlm_cache = build_vlm_image_cache(rollouts, processor)
    _simulate_advantage_computation(rollouts)
    for idx, rollout in enumerate(rollouts):
        interleave_rollout(rollout, vlm_cache=vlm_cache, cache_key=idx)
    return time.perf_counter() - start


def bench_overlapped(rollouts: list[dict], processor) -> float:
    """Time the overlapped path: extract → start preprocess → advantages + interleave → await + attach.

    This mirrors the production orchestrator where preprocessing runs in
    a background thread while advantage computation and interleave_rollout
    proceed on the main thread.
    """
    executor = ThreadPoolExecutor(max_workers=4)

    start = time.perf_counter()

    # Phase 1: extract (must happen first, modifies rollouts)
    extracted = extract_vlm_images(rollouts)

    # Phase 2: kick off preprocessing in background
    preprocess_future = executor.submit(preprocess_vlm_images, extracted, processor)

    # Phase 3: do other work while preprocessing runs (advantages + interleave)
    _simulate_advantage_computation(rollouts)
    all_results = []
    for idx, rollout in enumerate(rollouts):
        samples = interleave_rollout(rollout, vlm_cache=None, cache_key=idx)
        all_results.append((idx, rollout, samples))

    # Phase 4: wait for preprocess (likely already done) and attach images
    vlm_cache = preprocess_future.result()
    for idx, rollout, samples in all_results:
        if samples is not None:
            attach_vlm_images(samples, rollout, vlm_cache, cache_key=idx)

    elapsed = time.perf_counter() - start
    executor.shutdown(wait=False)
    return elapsed


def run_scenario(name: str, rollouts: list[dict], processor, warmup: int = 1, repeats: int = 3) -> BenchResult:
    """Run a benchmark scenario with warmup and averaging."""
    total_images = sum(
        len(
            [
                item
                for step in r["trajectory"]
                for msg in step["prompt"]
                for item in (msg.get("content") if isinstance(msg.get("content"), list) else [])
                if isinstance(item, dict) and item.get("type") == "image_url"
            ]
        )
        for r in rollouts
    )

    # Warmup
    for _ in range(warmup):
        # Deep copy rollouts since extraction modifies them in-place
        import copy

        bench_sequential(copy.deepcopy(rollouts), processor)
        bench_overlapped(copy.deepcopy(rollouts), processor)

    # Benchmark
    import copy

    seq_times = []
    ovl_times = []
    for _ in range(repeats):
        seq_times.append(bench_sequential(copy.deepcopy(rollouts), processor))
        ovl_times.append(bench_overlapped(copy.deepcopy(rollouts), processor))

    return BenchResult(
        name=name,
        num_rollouts=len(rollouts),
        total_images=total_images,
        sequential_s=sum(seq_times) / len(seq_times),
        overlapped_s=sum(ovl_times) / len(ovl_times),
    )


def main():
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct", trust_remote_code=True)
    print("Processor loaded.\n")

    results: list[BenchResult] = []

    N = 128

    # --- Single small image ---
    rollouts = [make_fake_rollout(i, num_turns=1, images_per_turn=1, image_size=(100, 100)) for i in range(N)]
    results.append(run_scenario("1img, 1 turn, 100x100", rollouts, processor))

    # --- Multi-image (3 per turn) ---
    rollouts = [make_fake_rollout(i, num_turns=1, images_per_turn=3, image_size=(100, 100)) for i in range(N)]
    results.append(run_scenario("3img, 1 turn, 100x100", rollouts, processor))

    # --- Multi-turn (3 turns, 1 image each) ---
    rollouts = [make_fake_rollout(i, num_turns=3, images_per_turn=1, image_size=(100, 100)) for i in range(N)]
    results.append(run_scenario("1img, 3 turns, 100x100", rollouts, processor))

    # --- Large images ---
    rollouts = [make_fake_rollout(i, num_turns=1, images_per_turn=1, image_size=(400, 400)) for i in range(N)]
    results.append(run_scenario("1img, 1 turn, 400x400", rollouts, processor))

    # --- Heterogeneous image sizes ---
    sizes = [(100, 100), (200, 200), (400, 400), (100, 400)]
    rollouts = [
        make_fake_rollout(i, num_turns=1, images_per_turn=1, image_size=sizes[i % len(sizes)]) for i in range(N)
    ]
    results.append(run_scenario("1img, 1 turn, mixed sizes", rollouts, processor))

    # --- Same image, multi-turn (dedup) ---
    rollouts = [
        make_fake_rollout(i, num_turns=3, images_per_turn=1, image_size=(100, 100), unique_images=False)
        for i in range(N)
    ]
    results.append(run_scenario("shared img, 3 turns, 100x100", rollouts, processor))

    # --- Heavy: 3 images x 3 turns x medium ---
    rollouts = [make_fake_rollout(i, num_turns=3, images_per_turn=3, image_size=(200, 200)) for i in range(N)]
    results.append(run_scenario("3img, 3 turns, 200x200", rollouts, processor))

    # --- Production-like: multi-turn, realistic token lengths ---
    rollouts = [
        make_fake_rollout(
            i, num_turns=3, images_per_turn=1, image_size=(100, 100), prompt_tokens=1024, completion_tokens=256
        )
        for i in range(N)
    ]
    results.append(run_scenario("prod: 1img 3t 1k+256tok", rollouts, processor))

    # --- Production-like: multi-image, large, long tokens ---
    rollouts = [
        make_fake_rollout(
            i, num_turns=2, images_per_turn=2, image_size=(400, 400), prompt_tokens=2048, completion_tokens=512
        )
        for i in range(N)
    ]
    results.append(run_scenario("prod: 2img 2t 400px 2k+512tok", rollouts, processor))

    # --- Max stress: 5 images x 3 turns x 400px x 2k tokens ---
    rollouts = [
        make_fake_rollout(
            i, num_turns=3, images_per_turn=5, image_size=(400, 400), prompt_tokens=2048, completion_tokens=512
        )
        for i in range(N)
    ]
    results.append(run_scenario("stress: 5img 3t 400px 2k+512tok", rollouts, processor))

    # Print results
    print("\n" + "=" * 120)
    print("VLM Image Processing Benchmark: Sequential vs Overlapped")
    print("=" * 120)
    print(
        f"  {'Scenario':<45s} | {'Rollouts':>8s} | {'Images':>7s} | {'Sequential':>10s} | {'Overlapped':>10s} | {'Speedup':>8s}"
    )
    print("-" * 120)
    for r in results:
        print(r)
    print("=" * 120)


if __name__ == "__main__":
    main()
