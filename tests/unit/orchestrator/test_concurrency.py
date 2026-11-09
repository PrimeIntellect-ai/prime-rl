import pytest

from prime_rl.configs.orchestrator import ConcurrencyConfig
from prime_rl.orchestrator import concurrency
from prime_rl.orchestrator.concurrency import (
    DWELL_SECONDS,
    GROWTH_MAX,
    LOAD_WINDOW_POLLS,
    QUEUE_TARGET,
    RESUME_SKIP_POLLS,
    ConcurrencyController,
    EngineLoadSample,
    Poll,
    RoleLoad,
    Window,
)
from prime_rl.orchestrator.live import context
from prime_rl.orchestrator.types import LiveTrace

POLL_SECONDS = 5.0
DROP_WINDOW_POLLS = int(DWELL_SECONDS / POLL_SECONDS)


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
    waiting: int = 0,
    finished: int = 0,
    generated: int = 0,
    latency: float = 0.0,
) -> EngineLoadSample:
    return EngineLoadSample(
        engine_id=engine_id,
        role=role,
        kv_capacity_tokens=1_000_000,
        max_model_len=32_768,
        waiting=waiting,
        finished_delta=finished,
        generation_tokens_delta=generated,
        latency_delta=latency,
    )


class Pool:
    """The dispatcher side: units in flight, and what the controller asked for."""

    def __init__(self, inflight: int):
        self.inflight = inflight
        self.limits: list[int] = []
        self.cancelled: list[int] = []


def controller(inflight: int = 100) -> tuple[ConcurrencyController, Pool]:
    ctl = ConcurrencyController(ConcurrencyConfig(initial_inflight=100), fallback_cost=1)
    pool = Pool(inflight)
    ctl.bind(set_limit=pool.limits.append, get_inflight=lambda: pool.inflight, on_overload=pool.cancelled.append)
    return ctl, pool


def turnover(ctl: ConcurrencyController, pool: Pool) -> None:
    """Every unit in flight completes (and is replaced) a little over once."""
    for _ in range(pool.inflight + 1):
        ctl.record_episode(10_000, 16)


def run(ctl: ConcurrencyController, pool: Pool, throughput, polls: int, queued=lambda inflight: 0) -> None:
    """The pool fills the cap, turns over every 50 polls, and the engines generate ``throughput(inflight)`` tokens
    per second with ``queued(inflight)`` requests waiting."""
    for _ in range(polls):
        pool.inflight = ctl.max_inflight
        for _ in range(max(1, pool.inflight // 50)):
            ctl.record_episode(10_000, 16)
        sample = engine(
            finished=10, generated=int(throughput(pool.inflight) * POLL_SECONDS), waiting=queued(pool.inflight)
        )
        ctl.observe([sample])


def trimmed(inflight: int, queued_share: float) -> int:
    """The cap a queue trim leaves: the units queued beyond the target go."""
    return int(inflight * (1 - queued_share + QUEUE_TARGET))


def test_throughput_is_per_second_of_poll_interval():
    load = RoleLoad(queued=0, finished=1, generated=1000)
    # A poll loop slowed to 10 s under load reads the same throughput, not twice as much
    for seconds, expected in ((5.0, 200), (10.0, 100)):
        window = Window()
        for _ in range(2):
            window.add(Poll({"colocated": load}, inflight=10, seconds=seconds))
        assert window.throughput == expected


def test_queued_share_adds_roles_and_counts_idle_units():
    ctl, _ = controller(inflight=60)
    # Prefill queue plus decode's pending KV pulls, over all units in flight (most of them idle on tools)
    ctl.observe([engine("p0", "prefill", waiting=4), engine("d0", "decode", waiting=2)])
    assert ctl.queued_share == pytest.approx(0.1)
    assert ctl.queued_share_by_role == {"prefill": pytest.approx(4 / 60), "decode": pytest.approx(2 / 60)}
    assert ctl.headroom == 0.0
    assert ctl.capacity == 1_000_000  # the running set's capacity: decode engines only


@pytest.mark.parametrize(
    ("inflight", "waiting", "cap"),
    [(100, 0, 100 * GROWTH_MAX), (100, 5, 100 * (1 + GROWTH_MAX) / 2), (50, 0, 100)],
    ids=["empty-queues", "queue-halfway-to-target", "cap-not-binding"],
)
def test_climb_grows_per_turnover_tapered_by_the_queue(inflight, waiting, cap):
    ctl, pool = controller(inflight)
    turnover(ctl, pool)
    ctl.observe([engine(waiting=waiting)])
    assert ctl.cap == pytest.approx(cap, rel=0.01)


def test_turnovers_without_a_measurement_do_not_grow_the_cap():
    ctl, pool = controller()
    for _ in range(5):  # /metrics unreachable while units keep completing
        turnover(ctl, pool)
        ctl.observe([])
    turnover(ctl, pool)
    ctl.observe([engine()])
    assert ctl.cap == pytest.approx(100 * GROWTH_MAX, rel=0.01)  # one measured turnover's growth, not 3**6


def test_fast_climb_ends_where_growth_stops_paying():
    ctl, pool = controller()
    # Throughput stops growing at 150 units in flight (e.g. a saturated API server: no engine queue)
    run(ctl, pool, lambda inflight: 10 * min(inflight, 150), polls=400)
    assert not ctl.climbing
    assert 130 <= ctl.max_inflight < 400


def test_step_up_that_pays_is_kept():
    ctl, pool = controller()
    ctl.climbing = False
    run(ctl, pool, lambda inflight: 10 * inflight, polls=400)  # throughput in proportion to the pool
    assert ctl.max_inflight > 200 and all(limit >= 100 for limit in pool.limits)


def test_step_up_that_buys_nothing_returns():
    ctl, pool = controller()
    ctl.climbing = False
    run(ctl, pool, lambda inflight: 10 * min(inflight, 100), polls=400)  # the knee is at 100
    rejected = pool.limits.index(100)
    assert max(pool.limits) == 150  # one step up, rejected
    assert all(75 <= limit <= 115 for limit in pool.limits[rejected:])  # then small steps around the knee


def test_queue_does_not_hold_growth_that_pays():
    ctl, pool = controller()
    ctl.climbing = False
    # Units wait on KV loads (a queued share past the target) while throughput still grows in proportion
    run(ctl, pool, lambda inflight: 10 * inflight, polls=600, queued=lambda inflight: inflight * 12 // 100)
    assert ctl.paying and ctl.max_inflight > 200


def test_queue_trims_then_cancels_when_a_drained_trim_did_not_clear_it():
    ctl, pool = controller()
    for _ in range(LOAD_WINDOW_POLLS):
        ctl.observe([engine(waiting=30)])
    assert pool.limits == [trimmed(100, 0.3)] and pool.cancelled == []

    ctl.observe([engine(waiting=30)])
    assert len(pool.limits) == 1  # no repeat before the pool drained to the cap

    pool.inflight = pool.limits[-1]
    turnover(ctl, pool)
    for _ in range(LOAD_WINDOW_POLLS):
        ctl.observe([engine(waiting=int(0.6 * pool.inflight))])
    target = trimmed(pool.inflight, 0.6)
    assert pool.limits[1:] == [target] and pool.cancelled == [pool.inflight - target]


def test_weight_update_pause_freezes_and_skips_the_backlog():
    ctl, pool = controller()
    ctl.pause()
    ctl.observe([engine(waiting=50)])
    assert not ctl.polls

    ctl.drops = 1
    ctl.resume()
    assert ctl.drops == 0 and ctl.window.seconds == 0  # no drop window straddles the update
    for _ in range(RESUME_SKIP_POLLS):
        ctl.observe([engine(waiting=50)])
    assert not ctl.polls and pool.limits == []
    for _ in range(LOAD_WINDOW_POLLS):
        ctl.observe([engine(waiting=50)])
    assert pool.limits == [trimmed(100, 0.5)]  # past the backlog a standing queue trims again


HEALTHY = engine(waiting=5, finished=10, generated=1000, latency=20.0)  # 2 s per call


def settle(ctl: ConcurrencyController, pool: Pool, healthy: EngineLoadSample = HEALTHY) -> None:
    """A pool that turned over (so it carries its steady load) runs ``healthy`` for a drop window: the best one."""
    turnover(ctl, pool)
    ctl.climbing = False
    for _ in range(DROP_WINDOW_POLLS + 1):  # the first poll has no interval
        ctl.observe([healthy])


def degrade(ctl: ConcurrencyController, pool: Pool, sample: EngineLoadSample, windows: int) -> None:
    """The engines report ``sample`` for up to ``windows`` drop windows, until the controller cancels."""
    for _ in range(windows * DROP_WINDOW_POLLS):
        ctl.observe([sample])
        if pool.cancelled:
            break


def test_throughput_drop_trims_then_cancels():
    ctl, pool = controller()
    settle(ctl, pool)
    thrashing = engine(waiting=5, finished=4, generated=400, latency=4 * 10.0)  # 10 s per call
    degrade(ctl, pool, thrashing, windows=2)
    assert pool.limits == [50] and pool.cancelled == []  # trimmed in proportion, to half at most

    degrade(ctl, pool, thrashing, windows=1)  # still thrashing: units finish too slowly to drain
    assert pool.cancelled == [50]


def test_slower_environment_is_not_a_cliff():
    ctl, pool = controller()
    settle(ctl, pool)
    degrade(ctl, pool, engine(waiting=5, finished=4, generated=400, latency=8.0), windows=4)  # fewer calls, as fast
    assert pool.cancelled == [] and pool.limits  # trimmed, never cancelled


def test_throughput_drop_without_the_latency_metric_cancels_on_a_standing_queue():
    ctl, pool = controller()
    settle(ctl, pool, engine(waiting=5, finished=10, generated=1000))  # no e2e latency histogram exported
    degrade(ctl, pool, engine(waiting=80, finished=1, generated=100), windows=4)  # most units stuck in the queue
    assert pool.cancelled


def test_projection_counts_running_units_to_the_max_context():
    ctl, pool = controller(inflight=10)
    # Ten units made calls 2..10, their context growing 1000 tokens per call
    for call in range(2, 11):
        for _ in range(10):
            ctl.record_call(1000 * call, call)
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

    # Every unit's 3rd call carried no context (e.g. a fork): the walk steps over the gap to the tail instead of
    # stopping after the first call and reading a young pool as already at its steady load
    ctl, pool = controller(inflight=10)
    for call in (2, 4, 5, 6, 7, 8, 9, 10):
        for _ in range(10):
            ctl.record_call(1000 * call, call)
    ctl.observe([engine()])
    assert ctl.steady_context > 15_000 and ctl.maturity < 0.5


def test_call_context_is_the_history_it_extends():
    def usage(prompt: int, cached: int | None) -> dict:
        return {"prompt_tokens": prompt, "completion_tokens": 50, "cached_input_tokens": cached}

    trace = LiveTrace()
    assert context(trace, usage(1000, 0)) == 0  # first call: no history yet
    assert context(trace, usage(600, 1008)) == 1000
    assert context(trace, usage(2200, 400)) == 1608  # evicted or not, the history is the same
    assert context(trace, usage(100, 0)) == 0  # history shrank: rewritten
    assert context(trace, usage(300, None)) == 0  # no cache usage reported: input unknown
    assert trace.prev_input is None
    assert context(trace, usage(500, 0)) == 0  # chain restarts
