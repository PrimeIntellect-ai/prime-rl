"""Prove that VLM image preprocessing pipelines with inference.

Simulates the orchestrator loop with a fake inference delay to show
that image preprocessing runs DURING inference, not after it.

Sequential (before PR6):
  inference(5s) → image_process(Xs) → interleave → pack → send
  Total per step: 5 + X seconds

Pipelined (PR6):
  Step N: inference(5s) → extract → [kick off preprocess in BG]
  Step N+1: inference(5s) ← preprocess runs during this
            → await preprocess (already done) → interleave → attach → pack → send
  Total per step: max(5, X) seconds ≈ 5s (since X < 5)

Usage:
    uv run python scripts/bench_pipeline_overlap.py
"""

import asyncio
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from transformers import AutoProcessor

from prime_rl.orchestrator.trajectories import (
    attach_vlm_images,
    build_vlm_image_cache,
    extract_vlm_images,
    interleave_rollout,
    preprocess_vlm_images,
)

sys.path.insert(0, str(Path(__file__).parent))
from bench_vlm_image_processing import make_fake_rollout

INFERENCE_DELAY = 5.0  # Simulated inference time per batch
NUM_STEPS = 4
N_ROLLOUTS = 128
NUM_TURNS = 3
IMAGES_PER_TURN = 3
IMAGE_SIZE = (400, 400)
PROMPT_TOKENS = 2048
COMPLETION_TOKENS = 512


def make_batch():
    return [
        make_fake_rollout(
            i,
            num_turns=NUM_TURNS,
            images_per_turn=IMAGES_PER_TURN,
            image_size=IMAGE_SIZE,
            prompt_tokens=PROMPT_TOKENS,
            completion_tokens=COMPLETION_TOKENS,
        )
        for i in range(N_ROLLOUTS)
    ]


async def fake_inference():
    """Simulate vLLM inference taking INFERENCE_DELAY seconds."""
    await asyncio.sleep(INFERENCE_DELAY)
    return make_batch()


def process_batch_sequential(rollouts, processor):
    """Sequential: build cache (blocking) → interleave."""
    vlm_cache = build_vlm_image_cache(rollouts, processor)
    for idx, rollout in enumerate(rollouts):
        interleave_rollout(rollout, vlm_cache=vlm_cache, cache_key=idx)


async def run_sequential(processor):
    """Simulate sequential orchestrator: inference → process → repeat."""
    print(f"\n{'=' * 60}")
    print(f"SEQUENTIAL: inference({INFERENCE_DELAY}s) → image_process → interleave")
    print(f"{'=' * 60}")

    total_start = time.perf_counter()
    for step in range(NUM_STEPS):
        step_start = time.perf_counter()

        # Inference
        inf_start = time.perf_counter()
        rollouts = await fake_inference()
        inf_time = time.perf_counter() - inf_start

        # Process (blocking)
        proc_start = time.perf_counter()
        process_batch_sequential(rollouts, processor)
        proc_time = time.perf_counter() - proc_start

        step_time = time.perf_counter() - step_start
        print(f"  Step {step}: inference={inf_time:.2f}s, process={proc_time:.2f}s, total={step_time:.2f}s")

    total = time.perf_counter() - total_start
    print(f"  TOTAL: {total:.2f}s ({total / NUM_STEPS:.2f}s/step)")
    return total


async def run_pipelined(processor):
    """Simulate pipelined orchestrator: overlap image preprocessing with next inference."""
    print(f"\n{'=' * 60}")
    print("PIPELINED: inference overlaps with prev step's image_process")
    print(f"{'=' * 60}")

    executor = ThreadPoolExecutor(max_workers=4)
    loop = asyncio.get_event_loop()

    total_start = time.perf_counter()
    pending_preprocess = None  # Future from previous step's image preprocessing
    pending_results = None  # Interleave results from previous step
    pending_rollouts = None

    for step in range(NUM_STEPS + 1):  # +1 to flush the last batch
        step_start = time.perf_counter()

        if step < NUM_STEPS:
            # Kick off inference (runs concurrently with any pending preprocessing)
            inf_start = time.perf_counter()
            rollouts = await fake_inference()
            inf_time = time.perf_counter() - inf_start
        else:
            inf_time = 0
            rollouts = None

        # Finish previous step: await preprocess + attach images + "send to trainer"
        if pending_preprocess is not None:
            attach_start = time.perf_counter()
            vlm_cache = await pending_preprocess
            for idx, (rollout, samples) in enumerate(zip(pending_rollouts, pending_results)):
                if samples is not None:
                    attach_vlm_images(samples, rollout, vlm_cache, cache_key=idx)
            attach_time = time.perf_counter() - attach_start
        else:
            attach_time = 0

        if rollouts is None:
            step_time = time.perf_counter() - step_start
            print(f"  Step {step} (flush): attach={attach_time:.2f}s, total={step_time:.2f}s")
            break

        # Start processing current batch: extract → preprocess (BG) → interleave
        proc_start = time.perf_counter()

        extracted = extract_vlm_images(rollouts)

        # Kick off preprocessing in background — will run during NEXT step's inference
        pending_preprocess = loop.run_in_executor(executor, preprocess_vlm_images, extracted, processor)

        # Interleave rollouts (fast, no images needed)
        pending_results = []
        pending_rollouts = rollouts
        for idx, rollout in enumerate(rollouts):
            samples = interleave_rollout(rollout, vlm_cache=None, cache_key=idx)
            pending_results.append(samples)

        proc_time = time.perf_counter() - proc_start

        step_time = time.perf_counter() - step_start
        print(
            f"  Step {step}: inference={inf_time:.2f}s, extract+interleave={proc_time:.2f}s, "
            f"prev_attach={attach_time:.2f}s, total={step_time:.2f}s"
        )

    total = time.perf_counter() - total_start
    effective_steps = NUM_STEPS
    print(f"  TOTAL: {total:.2f}s ({total / effective_steps:.2f}s/step)")
    executor.shutdown(wait=False)
    return total


async def main():
    print(
        f"Config: {N_ROLLOUTS} rollouts, {NUM_TURNS} turns, {IMAGES_PER_TURN} img/turn, "
        f"{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}, {PROMPT_TOKENS}+{COMPLETION_TOKENS} tokens"
    )
    print(f"Simulated inference delay: {INFERENCE_DELAY}s per batch")
    print(f"Steps: {NUM_STEPS}")

    print("\nLoading processor...")
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct", trust_remote_code=True)
    print("Processor loaded.")

    # Warmup
    print("\nWarmup...")
    warmup_rollouts = make_batch()
    build_vlm_image_cache(warmup_rollouts, processor)
    print("Warmup done.")

    seq_total = await run_sequential(processor)
    pipe_total = await run_pipelined(processor)

    print(f"\n{'=' * 60}")
    print(f"RESULT: Sequential={seq_total:.2f}s, Pipelined={pipe_total:.2f}s")
    print(f"Speedup: {seq_total / pipe_total:.2f}x")
    print(f"Image processing cost effectively hidden: {seq_total - pipe_total:.2f}s saved over {NUM_STEPS} steps")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    asyncio.run(main())
