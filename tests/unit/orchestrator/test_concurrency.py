import asyncio
import uuid
from types import SimpleNamespace

import pytest

from prime_rl.configs.orchestrator import ConcurrencyConfig
from prime_rl.orchestrator import concurrency
from prime_rl.orchestrator.concurrency import (
    DWELL_SECONDS,
    GROWTH_MAX,
    LOAD_WINDOW_POLLS,
    QUEUE_TARGET,
    RESUME_SKIP_POLLS,
    STEP_MIN,
    ConcurrencyController,
    EngineLoadSample,
    Poll,
    RoleLoad,
    Window,
)
from prime_rl.orchestrator.dispatcher import Dispatcher
from prime_rl.orchestrator.live import context
from prime_rl.orchestrator.trainer_capacity import TrainerCapacity
from prime_rl.orchestrator.types import InflightEpisode, LiveTrace, Policy

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
        self.finished = 0.0
        self.limits: list[int] = []
        self.shed: list[int] = []


def controller(inflight: int = 100, trainer: TrainerCapacity | None = None) -> tuple[ConcurrencyController, Pool]:
    ctl = ConcurrencyController(ConcurrencyConfig(initial_inflight=100), fallback_cost=1)
    pool = Pool(inflight)
    ctl.bind(
        set_limit=pool.limits.append,
        get_inflight=lambda: pool.inflight,
        on_overload=pool.shed.append,
        get_train_finished=lambda: pool.finished,
        trainer=trainer,
    )
    return ctl, pool


def trainer(rollouts_per_busy_s: float, busy: float = 1.0) -> TrainerCapacity:
    """A trainer that consumes ``rollouts_per_busy_s`` while busy, busy ``busy`` of each 100 s step."""
    capacity = TrainerCapacity()
    for step in range(4):
        capacity.shipped(
            int(rollouts_per_busy_s * 100 * busy), requested_at=100.0 * step - 100 * (1 - busy), now=100.0 * step
        )
    capacity.rollouts_shipped = 0  # the simulated pools ship every rollout they finish: no shipped share to apply
    return capacity


def turnover(ctl: ConcurrencyController, pool: Pool) -> None:
    """Every unit in flight completes (and is replaced) a little over once."""
    for _ in range(pool.inflight + 1):
        ctl.record_episode(10_000, 16)


def run(ctl: ConcurrencyController, pool: Pool, throughput, polls: int, queued=lambda inflight: 0, each=None) -> None:
    """The pool fills the cap, turns over every 50 polls, and the engines generate ``throughput(inflight)`` tokens
    per second with ``queued(inflight)`` requests waiting."""
    for _ in range(polls):
        pool.inflight = ctl.max_inflight
        for _ in range(max(1, pool.inflight // 50)):
            ctl.record_episode(10_000, 16)
        pool.finished += pool.inflight / 50
        sample = engine(
            finished=10, generated=int(throughput(pool.inflight) * POLL_SECONDS), waiting=queued(pool.inflight)
        )
        ctl.observe([sample])
        if each is not None:
            each()


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


@pytest.mark.parametrize("hungry", [False, True], ids=["no-trainer", "trainer-hungrier-than-the-engines"])
def test_step_up_that_pays_is_kept(hungry):
    ctl, pool = controller(trainer=trainer(100.0) if hungry else None)
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


def test_fed_trainer_cuts_the_cap_and_a_starving_one_takes_the_cut_back():
    # The pool finishes 0.4 train rollouts/s at 100 units, in proportion to the pool, and engine throughput grows
    # with it; the trainer consumes 0.24/s, so beyond ~66 units rollouts only go stale
    capacity = trainer(0.24)
    ctl, pool = controller(trainer=capacity)
    ctl.climbing = False
    run(ctl, pool, lambda inflight: 10 * inflight, polls=600)
    assert max(pool.limits) < 100  # never restored, although every cut lost throughput
    assert 66 <= ctl.max_inflight <= 80  # just above the 0.264 rollouts/s the trainer needs with its margin
    cap = ctl.max_inflight
    capacity.shipped(24, requested_at=430.0, now=500.0)  # the trainer waited 70% of its step
    ctl.trainer_shipped()
    assert ctl.max_inflight > cap


def test_queue_does_not_hold_growth_that_pays():
    ctl, pool = controller()
    ctl.climbing = False
    # Units wait on KV loads (a queued share past the target) while throughput still grows in proportion
    run(ctl, pool, lambda inflight: 10 * inflight, polls=600, queued=lambda inflight: inflight * 12 // 100)
    assert ctl.paying and ctl.max_inflight > 200


def test_queue_trims_then_sheds_when_a_drained_trim_did_not_clear_it():
    ctl, pool = controller()
    for _ in range(LOAD_WINDOW_POLLS):
        ctl.observe([engine(waiting=30)])
    assert pool.limits == [trimmed(100, 0.3)] and pool.shed == []

    ctl.observe([engine(waiting=30)])
    assert len(pool.limits) == 1  # no repeat before the pool drained to the cap

    pool.inflight = pool.limits[-1]
    turnover(ctl, pool)
    for _ in range(LOAD_WINDOW_POLLS):
        ctl.observe([engine(waiting=int(0.6 * pool.inflight))])
    target = trimmed(pool.inflight, 0.6)
    assert pool.limits[1:] == [target] and pool.shed == [pool.inflight - target]


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
    """The engines report ``sample`` for up to ``windows`` drop windows, until the controller sheds."""
    for _ in range(windows * DROP_WINDOW_POLLS):
        ctl.observe([sample])
        if pool.shed:
            break


def test_throughput_drop_trims_then_sheds():
    ctl, pool = controller()
    settle(ctl, pool)
    thrashing = engine(waiting=5, finished=4, generated=400, latency=4 * 10.0)  # 10 s per call
    degrade(ctl, pool, thrashing, windows=2)
    assert pool.limits == [50] and pool.shed == []  # trimmed in proportion, to half at most

    degrade(ctl, pool, thrashing, windows=1)  # still thrashing: units finish too slowly to drain
    assert pool.shed == [50]


def test_slower_environment_is_not_a_cliff():
    ctl, pool = controller()
    settle(ctl, pool)
    degrade(ctl, pool, engine(waiting=5, finished=4, generated=400, latency=8.0), windows=4)  # fewer calls, as fast
    assert pool.shed == [] and pool.limits  # trimmed, never shed


def test_throughput_drop_without_the_latency_metric_sheds_on_a_standing_queue():
    ctl, pool = controller()
    settle(ctl, pool, engine(waiting=5, finished=10, generated=1000))  # no e2e latency histogram exported
    degrade(ctl, pool, engine(waiting=80, finished=1, generated=100), windows=4)  # most units stuck in the queue
    assert pool.shed


def test_projection_counts_running_units_to_the_max_context():
    ctl, pool = controller(inflight=10)
    # Ten units made calls 2..10, their context growing 1000 tokens per call
    for call in range(2, 11):
        for _ in range(10):
            ctl.record_call(1000 * call, call)
    ctl.observe([engine()])
    # Still running: projected to grow on to the 32k max context, so the pool carries a third of its load
    assert ctl.projection.steady_context == pytest.approx(17_026, rel=0.01)
    assert ctl.projection.maturity == pytest.approx(6000 / 17_026, rel=0.01)

    # All ended after their 10th call: the pool already carries its steady load
    for _ in range(10):
        ctl.record_episode(10_000, 10)
    ctl.observe([engine()])
    assert ctl.projection.steady_context == pytest.approx(6000)
    assert ctl.projection.maturity == pytest.approx(1.0)

    # Every unit's 3rd call carried no context (e.g. a fork): the walk steps over the gap to the tail instead of
    # stopping after the first call and reading a young pool as already at its steady load
    ctl, pool = controller(inflight=10)
    for call in (2, 4, 5, 6, 7, 8, 9, 10):
        for _ in range(10):
            ctl.record_call(1000 * call, call)
    ctl.observe([engine()])
    assert ctl.projection.steady_context > 15_000 and ctl.projection.maturity < 0.5


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


def test_train_admission_holds_permits_and_feeds_a_bottlenecked_trainer():
    unbatched, rate = [0], [None]
    dispatcher = Dispatcher(
        train_envs=None,
        eval_envs=None,
        train_source=None,
        eval_source=None,
        policy_clients=None,
        policy=Policy(version=0, model_name="policy"),
        progress=None,
        initial_max_inflight=16,
        max_inflight_ceiling=None,
        dispatch_per_minute=None,
        run_id="run",
        run_name=None,
        unconsumed_train_rollouts=lambda: unbatched[0],
        batch_size=lambda: 8,
        trainer_rate=lambda: rate[0],
    )
    dispatcher.current_inflight = 10
    dispatcher.inflight = {object(): SimpleNamespace(kind="train") for _ in range(10)}
    # Up to one batch waiting frees its permits: the trainer finds its next batch ready
    dispatcher.queued_train_rollouts = 3
    unbatched[0] = 5
    assert dispatcher.held_permits == 0
    assert dispatcher.train_admission_open()
    # Past it, every finished rollout holds its permit until the trainer takes it
    unbatched[0] = 11
    assert dispatcher.held_permits == 6
    assert not dispatcher.train_admission_open()

    # A bottlenecked trainer taking 0.5 rollouts/s over a 20 s generation lifetime needs 10 in flight, plus 10 %
    unbatched[0], rate[0] = 5, 0.5
    dispatcher.generation_s = 20.0
    assert dispatcher.trainer_target == pytest.approx(11.0)
    assert dispatcher.train_admission_open()
    unbatched[0] = 6
    assert not dispatcher.train_admission_open()
    rate[0] = None
    assert dispatcher.train_admission_open()

    episode = SimpleNamespace(run=SimpleNamespace(work=SimpleNamespace(type="train")), traces=[object(), object()])
    dispatcher.out_q.put_nowait(episode)
    assert asyncio.run(dispatcher.get()) is episode
    assert dispatcher.queued_train_rollouts == 1


def test_starved_trainer_steps_up_until_fed_and_cuts_stay_above_the_starving_cap():
    # As above, but the trainer starves below 70 units: a bump still short keeps stepping up, never back down
    capacity = trainer(0.24)
    ctl, pool = controller(trainer=capacity)
    ctl.climbing = False
    ctl.cap = ctl.max_inflight = 60
    ctl.fed_cut = True
    capacity.shipped(24, requested_at=430.0, now=500.0)  # waited 70% of its step at 60 units
    ctl.trainer_shipped()
    bumped = ctl.max_inflight
    run(ctl, pool, lambda inflight: 10 * inflight, polls=600)
    assert bumped > 60 and min(pool.limits[pool.limits.index(bumped) :]) > 60


def test_starvation_after_a_throughput_drop_respects_its_ceiling():
    ctl, pool = controller(trainer=trainer(0.24))
    ctl.climbing = False
    ctl.cap = ctl.max_inflight = 80
    ctl.ceiling = 90.0  # a KV thrash above 90 trimmed the cap there earlier
    ctl.fed_cut = True
    ctl.trainer.shipped(24, requested_at=430.0, now=500.0)
    ctl.trainer_shipped()
    assert ctl.max_inflight == 90
    ctl.trim(70, 80, reason="queue", shed=False)
    assert not ctl.fed_cut  # a trim set the cap, so a later wait does not undo it


def test_starve_bump_past_a_ceiling_takes_the_smallest_step_and_never_lowers_the_cap():
    ctl, pool = controller(trainer=trainer(0.24))
    ctl.climbing = False
    ctl.cap = ctl.max_inflight = 120
    ctl.ceiling = 90.0  # an old throughput drop, which later steps climbed past
    ctl.fed_cut = True
    ctl.trainer.shipped(24, requested_at=430.0, now=500.0)
    ctl.trainer_shipped()
    assert ctl.max_inflight == int(120 * STEP_MIN)


def test_rollouts_that_never_ship_do_not_feed_the_trainer():
    # The pool finishes 0.4 train rollouts/s at 100 units, but only half of them ship (failed or filtered): the
    # trainer, eating 0.24/s, is fed only above ~132 units, so the cap is never cut below 100
    capacity = trainer(0.24)
    ctl, pool = controller(trainer=capacity)
    ctl.climbing = False
    shipped = lambda: setattr(capacity, "rollouts_shipped", (ctl.finished or 0) // 2)
    run(ctl, pool, lambda inflight: 10 * inflight, polls=600, each=shipped)
    assert min(pool.limits) >= 100


def test_overload_pauses_the_youngest_groups_and_resumes_them_first():
    sent: list[tuple[str, bool]] = []

    class Env:
        async def pause(self, request_id: str, paused: bool) -> bool:
            sent.append((request_id, paused))
            return True

    class Envs:
        def get(self, name: str) -> Env:
            return Env()

        def __iter__(self):
            return iter(())

    async def run() -> None:
        dispatcher = Dispatcher(
            train_envs=Envs(),
            eval_envs=None,
            train_source=None,
            eval_source=None,
            policy_clients=None,
            policy=Policy(version=0, model_name="policy"),
            progress=None,
            initial_max_inflight=6,
            max_inflight_ceiling=None,
            dispatch_per_minute=None,
            run_id="run",
            run_name=None,
        )
        groups = {name: uuid.uuid4() for name in "abc"}
        tasks = {}
        for i, (name, started) in enumerate([("a", 1.0), ("a", 5.0), ("b", 2.0), ("b", 3.0), ("c", 4.0), ("c", 4.5)]):
            meta = InflightEpisode(
                kind="train",
                env_name="env",
                group_id=groups[name],
                task=None,
                policy_version=0,
                step=0,
                started_at=started,
                dispatch_id=f"{name}{i}",
            )
            tasks[meta.dispatch_id] = asyncio.get_running_loop().create_future()
            dispatcher.inflight[tasks[meta.dispatch_id]] = meta
        dispatcher.current_inflight = 6

        # Group a started first; c is the youngest by its oldest member, then b
        dispatcher.set_limit(3)
        dispatcher.pause_inflight(3)
        assert dispatcher.current_inflight == 2 and len(dispatcher.paused) == 4
        await asyncio.gather(*dispatcher.pausing.values())
        assert sent == [("c4", True), ("c5", True), ("b2", True), ("b3", True)]

        # Free permits go to the paused episodes first, oldest pause first
        sent.clear()
        dispatcher.resume_paused()
        assert dispatcher.current_inflight == 3 and len(dispatcher.paused) == 3
        await asyncio.gather(*dispatcher.pausing.values())
        assert sent == [("c4", False)]

        # A paused episode that ends gave its permit back already
        dispatcher.leave(tasks["b3"])
        assert dispatcher.current_inflight == 3 and len(dispatcher.paused) == 2
        dispatcher.leave(tasks["c4"])
        assert dispatcher.current_inflight == 2

    asyncio.run(run())


def test_a_pause_the_worker_missed_gives_the_permit_back_and_a_resume_is_repeated():
    sent: list[tuple[str, bool]] = []
    # A run still on the wire (missed), a pause whose answer was lost, a resume that fails once
    answers = {("a0", True): [False], ("a1", True): [ConnectionError("lost")], ("a1", False): [True]}
    answers[("a2", True)] = [True]
    answers[("a2", False)] = [ConnectionError("down"), True]

    class Env:
        async def pause(self, request_id: str, paused: bool) -> bool:
            sent.append((request_id, paused))
            answer = answers[(request_id, paused)].pop(0)
            if isinstance(answer, Exception):
                raise answer
            return answer

    class Envs:
        def get(self, name: str) -> Env:
            return Env()

        def __iter__(self):
            return iter(())

    async def run() -> None:
        dispatcher = Dispatcher(
            train_envs=Envs(),
            eval_envs=None,
            train_source=None,
            eval_source=None,
            policy_clients=None,
            policy=Policy(version=0, model_name="policy"),
            progress=None,
            initial_max_inflight=3,
            max_inflight_ceiling=None,
            dispatch_per_minute=None,
            run_id="run",
            run_name=None,
        )
        group = uuid.uuid4()
        tasks = {}
        for i in range(3):
            meta = InflightEpisode(
                kind="train",
                env_name="env",
                group_id=group,
                task=None,
                policy_version=0,
                step=0,
                started_at=float(i),
                dispatch_id=f"a{i}",
            )
            tasks[meta.dispatch_id] = asyncio.get_running_loop().create_future()
            dispatcher.inflight[tasks[meta.dispatch_id]] = meta
        dispatcher.current_inflight = 3

        dispatcher.pause_inflight(3)
        assert dispatcher.current_inflight == 0
        while dispatcher.pausing:
            await asyncio.gather(*dispatcher.pausing.values())
        # a0 and a1 run on: a0 was never paused, and a1's pause was undone by a resume
        assert dispatcher.current_inflight == 2 and list(dispatcher.paused) == [tasks["a2"]]
        assert ("a1", False) in sent

        dispatcher.set_limit(3)
        dispatcher.resume_paused()
        while dispatcher.pausing:
            await asyncio.gather(*dispatcher.pausing.values())
        assert sent.count(("a2", False)) == 2 and dispatcher.current_inflight == 3

    asyncio.run(run())


def test_a_missed_pause_does_not_undo_a_later_pause():
    """A pause that misses after a resume and a new pause were already queued leaves
    them alone: the dispatcher still counts the episode paused, as the worker will."""
    sent: list[tuple[str, bool]] = []
    first_answer = asyncio.Event()

    class Env:
        async def pause(self, request_id: str, paused: bool) -> bool:
            sent.append((request_id, paused))
            if len(sent) == 1:
                await first_answer.wait()
                return False
            return True

    class Envs:
        def get(self, name: str) -> Env:
            return Env()

        def __iter__(self):
            return iter(())

    async def run() -> None:
        dispatcher = Dispatcher(
            train_envs=Envs(),
            eval_envs=None,
            train_source=None,
            eval_source=None,
            policy_clients=None,
            policy=Policy(version=0, model_name="policy"),
            progress=None,
            initial_max_inflight=1,
            max_inflight_ceiling=None,
            dispatch_per_minute=None,
            run_id="run",
            run_name=None,
        )
        meta = InflightEpisode(
            kind="train", env_name="env", group_id=uuid.uuid4(), task=None, policy_version=0, step=0, dispatch_id="a0"
        )
        task = asyncio.get_running_loop().create_future()
        dispatcher.inflight[task] = meta
        dispatcher.current_inflight = 1

        dispatcher.pause_inflight(1)
        await asyncio.sleep(0)
        dispatcher.resume_paused()
        dispatcher.pause_inflight(1)
        assert dispatcher.current_inflight == 0 and task in dispatcher.paused
        first_answer.set()
        while dispatcher.pausing:
            await asyncio.gather(*dispatcher.pausing.values())
        assert sent == [("a0", True), ("a0", False), ("a0", True)]
        assert dispatcher.current_inflight == 0 and task in dispatcher.paused

    asyncio.run(run())
