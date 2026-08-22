from collections.abc import Callable
from typing import cast

from httpx import AsyncClient

from prime_rl.configs.orchestrator import ConcurrencyConfig
from prime_rl.orchestrator.concurrency import ConcurrencyController, EngineLoadSample
from prime_rl.orchestrator.inference_metrics import (
    EngineSample,
    EngineSnapshot,
    MetricsEndpoint,
    TimedSnapshot,
    counter_rate,
)


def make_sample(
    *,
    throughput: float,
    kv_usage: float = 0.0,
    running: int = 1,
    waiting: int = 0,
    preemptions_delta: int = 0,
) -> EngineLoadSample:
    return EngineLoadSample(
        engine_id="decode0",
        role="decode",
        kv_capacity_tokens=None,
        max_model_len=None,
        kv_usage=kv_usage,
        running=running,
        waiting=waiting,
        waiting_capacity=waiting,
        preemptions_delta=preemptions_delta,
        generation_tokens_per_s=throughput,
    )


def make_controller(initial: int = 4, maximum: int = 32) -> tuple[ConcurrencyController, dict[str, int]]:
    controller = ConcurrencyController(
        ConcurrencyConfig(initial_inflight=initial, min_inflight=1, max_inflight=maximum),
        fallback_cost=1,
    )
    state = {"inflight": initial}

    def set_limit(limit: int) -> None:
        # The synthetic plant settles immediately at the requested WIP. Real
        # downward probes qualify only after the dispatcher drains to the cap.
        state["inflight"] = limit

    controller.bind(set_limit=set_limit, get_inflight=lambda: state["inflight"])
    return controller, state


def drive(
    controller: ConcurrencyController,
    state: dict[str, int],
    throughput_at: Callable[[int], float],
    *,
    kv_usage_at: Callable[[int], float] = lambda _: 0.0,
    polls: int = 200,
    complete_turnover: bool = True,
) -> None:
    for _ in range(polls):
        cap = int(state["inflight"])
        if complete_turnover:
            # Simulate one full pool of successful completions between polls.
            state["inflight"] = cap - 1
            for _ in range(cap):
                controller.record_episode("env", "train", tokens=1, duration=1.0)
            state["inflight"] = cap
        controller.observe(
            [
                make_sample(
                    throughput=throughput_at(cap),
                    kv_usage=kv_usage_at(cap),
                )
            ]
        )


def test_compute_bound_workload_uses_largest_cap_on_throughput_plateau() -> None:
    controller, state = make_controller()

    drive(controller, state, lambda cap: min(100.0 * cap, 800.0))

    assert controller.incumbent == 32


def test_kv_guardrail_stops_normal_attention_probe_before_memory_pressure() -> None:
    controller, state = make_controller()

    drive(
        controller,
        state,
        lambda cap: 100.0 * cap,
        kv_usage_at=lambda cap: cap / 12,
    )

    assert controller.incumbent == 7
    assert controller.max_inflight == 7


def test_low_agentic_inference_demand_never_ratchets_ceiling_down() -> None:
    controller, state = make_controller(initial=1024, maximum=1024)
    limits: list[int] = []
    controller.bind(set_limit=limits.append, get_inflight=lambda: state["inflight"])

    # Reproduce the run failure: a binding episode pool, low active inference,
    # no completed episodes, and throughput dominated by workload phase.
    noisy_throughput = iter([200.0, 20.0, 180.0, 15.0] * 25)
    for throughput in noisy_throughput:
        controller.observe([make_sample(throughput=throughput)])

    # Even after turnover, reaching the user ceiling is not a reason to probe
    # downward. Only explicit pressure may reduce it.
    controller.turnover = 10.0
    drive(controller, state, lambda _: 10.0, polls=50, complete_turnover=False)

    assert controller.max_inflight == 1024
    assert controller.incumbent == 1024
    assert controller.probe_phase == "baseline"
    assert limits == []


def test_transient_capacity_queue_does_not_reduce_concurrency() -> None:
    controller, state = make_controller(initial=32, maximum=32)

    controller.observe([make_sample(throughput=100.0, waiting=1)])
    controller.observe([make_sample(throughput=100.0, waiting=1)])
    controller.observe([make_sample(throughput=100.0, waiting=0)])

    assert controller.max_inflight == 32
    assert controller.incumbent == 32


def test_sustained_substantial_capacity_queue_still_cuts() -> None:
    controller, state = make_controller(initial=32, maximum=32)

    for _ in range(6):
        controller.observe([make_sample(throughput=100.0, running=1, waiting=1)])

    assert controller.max_inflight == 28


def test_pressure_cut_recovers_only_by_probing_up() -> None:
    controller, state = make_controller(initial=16, maximum=32)

    controller.observe([make_sample(throughput=100.0, preemptions_delta=1)])
    cut = controller.max_inflight
    assert cut == 12

    drive(controller, state, lambda cap: 100.0 * cap, polls=20)

    assert controller.incumbent > cut
    assert controller.max_inflight >= controller.incumbent


def test_engine_queue_aborts_probe_without_cancelling_work() -> None:
    controller, state = make_controller()
    cancelled: list[int] = []
    controller.bind(
        set_limit=lambda limit: state.__setitem__("inflight", limit),
        get_inflight=lambda: state["inflight"],
        on_overload=cancelled.append,
    )

    drive(controller, state, lambda cap: 100.0 * cap, polls=4)
    assert controller.probe_phase == "probe"
    assert controller.max_inflight > controller.incumbent

    controller.observe([make_sample(throughput=100.0, waiting=1)])

    assert controller.probe_phase == "baseline"
    assert controller.max_inflight == controller.incumbent
    assert cancelled == []


def test_zero_token_episode_does_not_unlock_throughput_probe() -> None:
    controller, state = make_controller()

    controller.record_episode("env", "train", tokens=0, duration=1.0)
    drive(controller, state, lambda cap: 100.0 * cap, polls=20, complete_turnover=False)

    assert controller.turnover == 0.0
    assert controller.max_inflight == 4


def test_generation_counter_rate_uses_interval_delta_and_rejects_reset() -> None:
    endpoint = MetricsEndpoint(client=cast(AsyncClient, None), role="decode", key="decode", name="decode0")
    previous = TimedSnapshot(
        timestamp=10.0,
        snapshot=EngineSnapshot(counters={"generation_tokens_total": 100.0}),
    )
    sample = EngineSample(
        endpoint=endpoint,
        engine_label="0",
        timestamp=15.0,
        snapshot=EngineSnapshot(counters={"generation_tokens_total": 200.0}),
    )

    assert counter_rate(sample, previous, ("generation_tokens", "generation_tokens_total")) == 20.0

    reset = EngineSample(
        endpoint=endpoint,
        engine_label="0",
        timestamp=20.0,
        snapshot=EngineSnapshot(counters={"generation_tokens_total": 10.0}),
    )
    assert counter_rate(reset, previous, ("generation_tokens", "generation_tokens_total")) is None


def test_kv_trim_drains_softly_then_cancels_only_above_hard_cap() -> None:
    controller, state = make_controller(initial=16)
    cancelled: list[int] = []
    controller.bind(
        set_limit=lambda limit: state.__setitem__("inflight", limit),
        get_inflight=lambda: 16,
        on_overload=cancelled.append,
    )

    controller.observe([make_sample(throughput=100.0, kv_usage=0.81)])
    assert controller.max_inflight == 13
    assert cancelled == []

    controller.trim_cooldown = 0
    controller.draining = False
    controller.observe([make_sample(throughput=100.0, kv_usage=0.91)])
    assert controller.max_inflight == 12
    assert cancelled == [4]
