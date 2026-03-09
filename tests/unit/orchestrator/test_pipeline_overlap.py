"""Test pipelined postprocessing overlap with generate_batch."""

import asyncio
import time

import pytest


def busy_wait(duration: float):
    """Synchronous CPU work that blocks the event loop."""
    end = time.perf_counter() + duration
    while time.perf_counter() < end:
        pass


async def simulate_generate_batch(step: int, duration: float) -> list[dict]:
    await asyncio.sleep(duration)
    return [{"step": step, "reward": 0.5, "example_id": i} for i in range(4)]


async def simulate_postprocess(step: int, sync_duration: float, async_duration: float) -> bool:
    busy_wait(sync_duration)
    if async_duration > 0:
        await asyncio.sleep(async_duration)
    return True


async def run_sequential(num_steps: int, generate_time: float, sync_post: float, async_post: float) -> float:
    start = time.perf_counter()
    for step in range(num_steps):
        await simulate_generate_batch(step, generate_time)
        await simulate_postprocess(step, sync_post, async_post)
    return time.perf_counter() - start


async def run_pipelined(num_steps: int, generate_time: float, sync_post: float, async_post: float) -> float:
    start = time.perf_counter()
    postprocess_task = None

    for step in range(num_steps):
        train_task = asyncio.create_task(simulate_generate_batch(step, generate_time))
        await train_task

        if postprocess_task is not None:
            await postprocess_task

        postprocess_task = asyncio.create_task(simulate_postprocess(step, sync_post, async_post))

    if postprocess_task is not None:
        await postprocess_task

    return time.perf_counter() - start


@pytest.mark.parametrize(
    "generate_time,sync_post,async_post,expected_speedup_min",
    [
        (0.5, 0.02, 0.005, 0.95),  # text-only: no teacher, negligible async
        (0.1, 0.015, 0.3, 1.18),  # text + teacher: teacher logprobs dominates async
        (0.3, 0.05, 0.01, 0.95),  # VLM, no teacher: VLM cache is sync, no overlap
        (0.3, 0.05, 0.35, 1.30),  # VLM + teacher: teacher overlaps despite sync VLM cache
    ],
    ids=["text_only", "text_teacher", "vlm_no_teacher", "vlm_teacher"],
)
def test_realistic_pipeline_overlap(generate_time, sync_post, async_post, expected_speedup_min):
    num_steps = 5
    sequential_time = asyncio.run(run_sequential(num_steps, generate_time, sync_post, async_post))
    pipelined_time = asyncio.run(run_pipelined(num_steps, generate_time, sync_post, async_post))
    speedup = sequential_time / pipelined_time

    assert speedup >= expected_speedup_min, (
        f"Expected >= {expected_speedup_min}x, got {speedup:.2f}x "
        f"(seq={sequential_time:.3f}s, pipe={pipelined_time:.3f}s)"
    )


def test_pipeline_handles_empty_batch():
    async def run():
        postprocess_task = None
        steps_generated = []
        empty_count = 0

        async def make_empty():
            return False

        step = 0
        while step < 3:
            await simulate_generate_batch(step, 0.01)

            if postprocess_task is not None:
                success = await postprocess_task
                if not success:
                    empty_count += 1
                    postprocess_task = asyncio.create_task(simulate_postprocess(step, 0.0, 0.01))
                    continue

            steps_generated.append(step)
            if step == 1 and empty_count == 0:
                postprocess_task = asyncio.create_task(make_empty())
            else:
                postprocess_task = asyncio.create_task(simulate_postprocess(step, 0.0, 0.01))
            step += 1

        if postprocess_task is not None:
            await postprocess_task
        return steps_generated, empty_count

    steps, empty_count = asyncio.run(run())
    assert steps == [0, 1, 2]
    assert empty_count == 1


def test_pipeline_sync_points():
    async def run():
        postprocess_task = None
        sync_order = []

        for step in range(4):
            need_sync = step == 2

            if postprocess_task is not None and need_sync:
                await postprocess_task
                sync_order.append(f"awaited_{step - 1}_before_sync")
                postprocess_task = None

            if need_sync:
                sync_order.append(f"sync_{step}")

            await simulate_generate_batch(step, 0.05)

            if postprocess_task is not None:
                await postprocess_task
                postprocess_task = None

            postprocess_task = asyncio.create_task(simulate_postprocess(step, 0.0, 0.05))

        if postprocess_task is not None:
            await postprocess_task
        return sync_order

    order = asyncio.run(run())
    assert order == ["awaited_1_before_sync", "sync_2"]


def test_sync_work_blocks_overlap():
    """All-sync postprocess should give no speedup."""

    async def run():
        start = time.perf_counter()
        for step in range(3):
            await simulate_generate_batch(step, 0.1)
            await simulate_postprocess(step, 0.1, 0.0)
        seq = time.perf_counter() - start

        start = time.perf_counter()
        postprocess_task = None
        for step in range(3):
            train_task = asyncio.create_task(simulate_generate_batch(step, 0.1))
            await train_task
            if postprocess_task is not None:
                await postprocess_task
            postprocess_task = asyncio.create_task(simulate_postprocess(step, 0.1, 0.0))
        if postprocess_task is not None:
            await postprocess_task
        pipe = time.perf_counter() - start

        return seq, pipe

    seq, pipe = asyncio.run(run())
    speedup = seq / pipe
    assert speedup < 1.10, f"Expected no speedup with all-sync work, got {speedup:.2f}x"
