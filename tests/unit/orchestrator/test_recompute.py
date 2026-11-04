from prime_rl.configs.orchestrator import ConcurrencyConfig, RecomputeConfig
from prime_rl.orchestrator.concurrency import ConcurrencyController, EngineLoadSample
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


def sample(kv_usage: float) -> list[EngineLoadSample]:
    return [
        EngineLoadSample(
            engine_id="e0",
            role=None,
            kv_capacity_tokens=1_000_000,
            max_model_len=32_768,
            kv_usage=kv_usage,
            running=4,
            waiting=0,
            waiting_capacity=0,
            preemptions_delta=0,
        )
    ]


def controller(inflight: int) -> tuple[ConcurrencyController, list[int], list[int]]:
    config = ConcurrencyConfig(initial_inflight=20, recompute=RecomputeConfig(grow_below=0.4, trim_above=0.7))
    controller = ConcurrencyController(config, fallback_cost=1)
    limits, cancelled = [], []
    controller.bind(set_limit=limits.append, get_inflight=lambda: inflight, on_overload=cancelled.append)
    return controller, limits, cancelled


def test_kv_soft_trim_applies_until_recompute_is_measured():
    ctl, limits, cancelled = controller(inflight=20)
    ctl.record_calls(50_000, 0)  # below the minimum prefix: ratio unknown
    ctl.observe(sample(0.85))
    assert ctl.recompute_ratio is None
    assert limits == [16] and cancelled == []


def test_recompute_replaces_the_kv_gauge():
    ctl, limits, cancelled = controller(inflight=20)
    ctl.record_calls(200_000, 20_000)
    ctl.observe(sample(0.85))  # a per-engine KV spike no longer trims or blocks growth
    assert ctl.recompute_ratio == 0.1
    assert ctl.signal == "clear" and ctl.can_grow and limits == []
    assert ctl.growth_multiplier == 1.0 + 0.2 * 0.75

    ctl.record_calls(800_000, 780_000)  # windowed: 800k of 1M
    ctl.observe(sample(0.3))
    assert ctl.recompute_ratio == 0.8
    assert ctl.signal == "soft" and not ctl.can_grow
    assert limits == [18] and cancelled == []  # trimmed without cancelling

    ctl.observe(sample(0.3))  # no repeat trim within the window
    assert limits == [18]


def queued_sample(running: int, waiting: int) -> list[EngineLoadSample]:
    return [
        EngineLoadSample(
            engine_id="e0",
            role=None,
            kv_capacity_tokens=1_000_000,
            max_model_len=32_768,
            kv_usage=0.3,
            running=running,
            waiting=waiting,
            waiting_capacity=waiting,
            preemptions_delta=0,
        )
    ]


def test_recompute_control_grows_per_call_and_tolerates_a_small_queue():
    ctl, limits, _ = controller(inflight=20)
    ctl.record_calls(1_000_000, 0)
    ctl.observe(queued_sample(running=40, waiting=2))  # 5% queued: turns arriving together
    assert ctl.signal == "clear" and ctl.can_grow
    ctl.record_episode(10_000)  # episodes no longer pace growth
    assert limits == []
    for _ in range(20):  # one turn turnover of calls: up to CALL_GROWTH_MAX
        ctl.record_calls(10_000, 0)
    assert limits == [21, 22]  # x1.1 over one turnover

    ctl.observe(queued_sample(running=40, waiting=8))  # 20% queued once: still clear, but no growth
    assert ctl.signal == "clear" and not ctl.can_grow
    ctl.observe(queued_sample(running=40, waiting=8))  # persisting: soft
    assert ctl.signal == "soft"
