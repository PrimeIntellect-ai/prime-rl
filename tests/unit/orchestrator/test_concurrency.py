import pytest

from prime_rl.configs.orchestrator import ConcurrencyConfig
from prime_rl.orchestrator import concurrency
from prime_rl.orchestrator.concurrency import (
    LOAD_WINDOW_POLLS,
    RESUME_SKIP_POLLS,
    THROUGHPUT_WINDOW_POLLS,
    ConcurrencyController,
    EngineLoadSample,
    Poll,
    RoleLoad,
    per_second,
    quantile,
)

POLL_SECONDS = 5.0


class Clock:
    """Stands in for the controller module's ``time``: every poll reads one poll interval later."""

    def __init__(self):
        self.now = 0.0

    def monotonic(self) -> float:
        self.now += POLL_SECONDS
        return self.now


@pytest.fixture(autouse=True)
def clock(monkeypatch):
    monkeypatch.setattr(concurrency, "time", Clock())


def engine(
    engine_id: str = "e0",
    role: str | None = None,
    *,
    running: int = 10,
    waiting: int = 0,
    kv_usage: float = 0.3,
    preemptions: int = 0,
    finished: int = 0,
    generated: int = 0,
    latency: float = 0.0,
    host_kv_usage: float | None = None,
) -> EngineLoadSample:
    return EngineLoadSample(
        engine_id=engine_id,
        role=role,
        kv_capacity_tokens=1_000_000,
        max_model_len=32_768,
        kv_usage=kv_usage,
        running=running,
        waiting=waiting,
        waiting_capacity=None,
        preemptions_delta=preemptions,
        finished_delta=finished,
        generation_tokens_delta=generated,
        latency_delta=latency,
        host_kv_usage=host_kv_usage,
    )


class Pool:
    """The dispatcher side: units in flight, and what the controller asked for."""

    def __init__(self, inflight: int):
        self.inflight = inflight
        self.limits: list[int] = []
        self.cancelled: list[int] = []


def controller(inflight: int = 100, measured: bool = True) -> tuple[ConcurrencyController, Pool]:
    ctl = ConcurrencyController(ConcurrencyConfig(initial_inflight=100), fallback_cost=1)
    pool = Pool(inflight)
    ctl.bind(set_limit=pool.limits.append, get_inflight=lambda: pool.inflight, on_overload=pool.cancelled.append)
    if measured:
        ctl.record_calls(1_000_000, 0, 2)
    return ctl, pool


def turnover(ctl: ConcurrencyController, pool: Pool) -> None:
    """Every unit in flight completes (and is replaced) a little over once."""
    for _ in range(pool.inflight + 1):
        ctl.record_episode(10_000, 16)


def run(ctl: ConcurrencyController, pool: Pool, throughput, polls: int) -> None:
    """The pool fills the cap, turns over every 50 polls, and the engines generate ``throughput(inflight)`` tokens
    per second."""
    for _ in range(polls):
        pool.inflight = ctl.max_inflight
        for _ in range(max(1, pool.inflight // 50)):
            ctl.record_episode(10_000, 16)
        ctl.record_calls(1_000_000, 0, 2)
        ctl.observe([engine(finished=10, generated=int(throughput(pool.inflight) * POLL_SECONDS))])


def test_quantile_interpolates():
    assert quantile([0.5], 0.9) == 0.5
    assert quantile([0.0, 0.0, 0.2, 1.0], 0.9) == pytest.approx(0.76)


def test_throughput_is_per_second_of_poll_interval():
    load = RoleLoad(running=1, queued=0, preemptions=0, finished=1, expired=0, kv_usage=0.0, generated=1000)
    # A poll loop slowed to 10 s under load reads the same throughput, not twice as much
    assert per_second([Poll({"colocated": load}, inflight=10, seconds=5.0)] * 2) == 200
    assert per_second([Poll({"colocated": load}, inflight=10, seconds=10.0)] * 2) == 100


def test_queued_share_adds_roles_and_counts_idle_units():
    ctl, _ = controller(inflight=60)
    # Prefill queue plus decode's pending KV pulls, over all units in flight (most of them idle on tools)
    ctl.observe([engine("p0", "prefill", running=2, waiting=4), engine("d0", "decode", running=40, waiting=2)])
    assert ctl.queued_share == pytest.approx(0.1)
    assert ctl.queued_share_by_role == {"prefill": pytest.approx(4 / 60), "decode": pytest.approx(2 / 60)}
    assert ctl.growth_multiplier == 1.0 and ctl.signal == "soft"
    assert ctl.capacity == 1_000_000  # the running set's capacity: decode engines only


def test_ramp_grows_per_turnover_tapered_by_the_queue():
    ctl, pool = controller()
    turnover(ctl, pool)
    ctl.observe([engine()])
    assert ctl.cap == pytest.approx(300, rel=0.02)  # empty queues: x3 per turnover

    ctl, pool = controller()
    turnover(ctl, pool)
    ctl.observe([engine(waiting=5)])  # halfway to the target
    assert ctl.cap == pytest.approx(200, rel=0.02)

    ctl, pool = controller(inflight=50)
    turnover(ctl, pool)
    ctl.observe([engine()])
    assert pool.limits == []  # the cap does not bind admission


def test_ramp_ends_where_growth_stops_paying():
    ctl, pool = controller()
    # Throughput stops growing at 150 units in flight (e.g. a saturated API server: no engine queue)
    run(ctl, pool, lambda inflight: 10 * min(inflight, 150), polls=400)
    assert not ctl.ramping
    assert 130 <= ctl.max_inflight < 400


def test_probe_keeps_a_step_up_that_pays():
    ctl, pool = controller()
    ctl.ramping = False
    run(ctl, pool, lambda inflight: 10 * inflight, polls=400)  # throughput in proportion to the pool
    assert ctl.max_inflight > 200 and all(limit >= 100 for limit in pool.limits)


def test_probe_returns_from_a_step_up_that_buys_nothing():
    ctl, pool = controller()
    ctl.ramping = False
    run(ctl, pool, lambda inflight: 10 * min(inflight, 100), polls=300)  # the knee is at 100
    assert max(pool.limits) == 125  # one step up, rejected, then one step down, rejected
    assert 86 <= ctl.max_inflight <= 100 and ctl.elasticity is not None


def test_queue_trims_the_surplus_without_cancelling():
    ctl, pool = controller()
    for _ in range(LOAD_WINDOW_POLLS):
        ctl.observe([engine(waiting=30)])
    assert pool.limits == [int(100 * (1 - 0.3 + 0.1))] and pool.cancelled == []  # the units queued past the target

    ctl.observe([engine(waiting=30)])
    assert len(pool.limits) == 1  # no repeat before the pool drained to the cap


def test_queue_still_overloaded_after_a_drained_trim_cancels():
    ctl, pool = controller()
    for _ in range(LOAD_WINDOW_POLLS):
        ctl.observe([engine(waiting=30)])
    pool.inflight = pool.limits[-1]
    turnover(ctl, pool)
    for _ in range(LOAD_WINDOW_POLLS):
        ctl.observe([engine(waiting=int(0.6 * pool.inflight))])
    target = int(pool.inflight * (1 - 0.6 + 0.1))
    assert pool.limits[1:] == [target] and pool.cancelled == [pool.inflight - target]


def test_kv_gauge_applies_until_recompute_is_measured():
    ctl, pool = controller(measured=False)
    ctl.observe([engine("e0", kv_usage=0.95), *(engine(f"e{i}", kv_usage=0.1) for i in range(1, 4))])
    assert pool.limits == []  # one hot engine is a placement problem
    ctl.observe([engine(f"e{i}", kv_usage=0.85) for i in range(4)])
    assert pool.limits == [82] and pool.cancelled == []  # 100 * 0.7 / 0.85, soft

    ctl, pool = controller()
    ctl.observe([engine(f"e{i}", kv_usage=0.85) for i in range(4)])
    assert pool.limits == []  # measured recompute: the gauge misses idle sessions' evicted caches


def test_hisparse_host_pool_replaces_the_gpu_gauge():
    ctl, _ = controller()
    ctl.observe([engine(kv_usage=0.95, host_kv_usage=0.2)])
    assert ctl.kv_usage == 0.2


def test_preemption_rate_trims_without_cancelling():
    ctl, pool = controller()
    ctl.observe([engine(f"e{i}", finished=50, preemptions=1 if i == 0 else 0) for i in range(4)])
    assert ctl.preemption_rate == 1 / 200 and pool.limits == []

    ctl.observe([engine(f"e{i}", finished=50, preemptions=10) for i in range(4)])
    assert pool.limits == [80] and pool.cancelled == [] and ctl.ceiling == 100


def test_weight_update_pause_freezes_and_skips_the_backlog():
    ctl, pool = controller()
    ctl.pause()
    ctl.observe([engine(running=0, waiting=50, preemptions=5)])
    assert not ctl.polls

    ctl.resume()
    for _ in range(RESUME_SKIP_POLLS):
        ctl.observe([engine(waiting=50)])
    assert not ctl.polls and pool.limits == []
    for _ in range(LOAD_WINDOW_POLLS):
        ctl.observe([engine(waiting=50)])
    assert pool.limits == [int(100 * (1 - 0.5 + 0.1))]  # past the backlog a standing queue trims again


def test_throughput_cliff_cuts_and_cancels():
    ctl, pool = controller()
    turnover(ctl, pool)  # a pool that turned over carries its steady load
    healthy = [engine(waiting=10, finished=10, generated=1000, latency=20.0)]
    for _ in range(THROUGHPUT_WINDOW_POLLS):
        ctl.observe(healthy)
    assert ctl.best_throughput == pytest.approx(1000 / POLL_SECONDS, rel=0.01)  # tokens per second

    thrashing = [engine(waiting=10, finished=4, generated=400, latency=4 * 10.0)]  # 10 s per call, was 2 s
    polls = 0
    while not pool.cancelled:
        ctl.observe(thrashing)
        polls += 1
    assert polls > 1.5 * THROUGHPUT_WINDOW_POLLS  # the window filled with the thrash, then it held for a window
    assert pool.limits == [50] and pool.cancelled == [50] and ctl.ceiling == 50  # cut to half at most


def test_slower_environment_is_not_a_cliff():
    ctl, pool = controller()
    turnover(ctl, pool)
    for _ in range(THROUGHPUT_WINDOW_POLLS):
        ctl.observe([engine(waiting=10, finished=10, generated=1000, latency=20.0)])
    for _ in range(2 * THROUGHPUT_WINDOW_POLLS):
        ctl.observe([engine(waiting=10, finished=4, generated=400, latency=8.0)])  # fewer calls, as fast
    assert pool.cancelled == []


def test_throughput_cliff_without_the_latency_metric_cuts_on_a_standing_queue():
    ctl, pool = controller()
    turnover(ctl, pool)
    for _ in range(THROUGHPUT_WINDOW_POLLS):
        ctl.observe([engine(waiting=10, finished=10, generated=1000)])  # no e2e latency histogram exported
    collapsed = [engine(waiting=80, finished=1, generated=100)]  # most units stuck in the engine queue
    for _ in range(3 * THROUGHPUT_WINDOW_POLLS):
        ctl.observe(collapsed)
        if pool.cancelled:
            break
    assert pool.cancelled  # the turnover never comes past a cliff; the cut does not wait for it


def test_projection_counts_running_units_to_the_max_context():
    ctl, pool = controller(inflight=10, measured=False)
    # Ten units made calls 2..10, their context growing 1000 tokens per call
    for call in range(2, 11):
        for _ in range(10):
            ctl.record_calls(1000 * call, 0, call)
    ctl.observe([engine()])
    # Still running: projected to grow on to the 32k max context, so the pool carries a third of its load
    assert ctl.steady_context == pytest.approx(17_026, rel=0.01)
    assert ctl.maturity == pytest.approx(6000 / 17_026, rel=0.01)

    # All ended after their 10th call: the pool already carries its steady load
    for _ in range(10):
        ctl.record_episode(10_000, 10)
    ctl.observe([engine()])
    assert ctl.steady_context == pytest.approx(6000)
    assert ctl.maturity == pytest.approx(1.0)
