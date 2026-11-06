from prime_rl.orchestrator.live import reuse
from prime_rl.orchestrator.types import LiveTrace


def usage(prompt: int, cached: int | None) -> dict:
    return {"prompt_tokens": prompt, "completion_tokens": 50, "cached_input_tokens": cached}


def test_reuse_counts_the_uncached_part_of_the_previous_prompt():
    trace = LiveTrace()
    assert reuse(trace, usage(1000, 0)) == (0, 0)  # first call: nothing computed before
    assert reuse(trace, usage(600, 1008)) == (1000, 0)  # prefix hit, 600 new tokens
    assert reuse(trace, usage(2200, 400)) == (1608, 1208)  # partly evicted
    assert reuse(trace, usage(100, 0)) == (0, 0)  # history shrank: rewritten, not recompute
    assert reuse(trace, usage(300, None)) == (0, 0)  # no cache usage reported
    assert trace.prev_input is None
    assert reuse(trace, usage(500, 0)) == (0, 0)  # chain restarts
