from collections.abc import Callable
from typing import cast

from httpx import AsyncClient

from prime_rl.configs.orchestrator import ConcurrencyConfig
from prime_rl.orchestrator.concurrency import (
    PREEMPTION_CUT_FRACTION,
    PROBE_FACTOR,
    QUEUE_PERSISTENCE_POLLS,
    ConcurrencyController,
    EngineLoadSample,
)
from prime_rl.orchestrator.inference_metrics import (
    EngineSample,
    EngineSnapshot,
    MetricsEndpoint,
    TimedSnapshot,
    counter_rate,
)


def make_sample(
    *, throughput: float, kv_usage: float = 0.0, waiting: int = 0, preemptions: int = 0
) -> EngineLoadSample:
    return EngineLoadSample(
        engine_id="decode0",
        role="decode",
        kv_capacity_tokens=None,
        max_model_len=None,
        kv_usage=kv_usage,
        running=1,
        waiting=waiting,
        waiting_capacity=waiting,
        preemptions_delta=preemptions,
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
) -> None:
    for _ in range(polls):
        cap = state["inflight"]
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


def test_controller_reprobes_down_when_workload_optimum_decreases() -> None:
    controller, state = make_controller()
    drive(controller, state, lambda cap: min(100.0 * cap, 800.0))
    assert controller.incumbent == 32

    def shifted_throughput(cap: int) -> float:
        if cap <= 5:
            return 200.0 * cap
        return 1_000.0 / (1 + 0.1 * (cap - 5))

    drive(controller, state, shifted_throughput, polls=500)

    assert controller.incumbent == 5


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
    assert controller.probe_direction == "down"
    assert controller.max_inflight == controller.incumbent
    assert cancelled == []


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


def test_persistent_queue_soft_cuts_without_cancelling_active_work() -> None:
    inflight = 1_000
    limits: list[int] = []
    cancellations: list[int] = []
    controller = ConcurrencyController(
        ConcurrencyConfig(initial_inflight=inflight, max_inflight=2_000),
        fallback_cost=131_072,
    )
    controller.bind(
        set_limit=limits.append,
        get_inflight=lambda: inflight,
        on_overload=cancellations.append,
    )

    for _ in range(QUEUE_PERSISTENCE_POLLS):
        controller.observe([make_sample(throughput=100.0, waiting=60)])

    assert controller.signal == "soft"
    target = int(inflight / PROBE_FACTOR)
    assert controller.max_inflight == target
    assert limits == [target]
    assert cancellations == []
    assert controller.draining
    assert not controller.escalated


def test_preemption_hard_cuts_and_cancels_active_work() -> None:
    inflight = 1_000
    limits: list[int] = []
    cancellations: list[int] = []
    controller = ConcurrencyController(
        ConcurrencyConfig(initial_inflight=inflight, max_inflight=2_000),
        fallback_cost=131_072,
    )
    controller.bind(
        set_limit=limits.append,
        get_inflight=lambda: inflight,
        on_overload=cancellations.append,
    )

    controller.observe([make_sample(throughput=100.0, preemptions=1)])

    target = int(inflight * PREEMPTION_CUT_FRACTION)
    assert controller.signal == "hard"
    assert controller.max_inflight == target
    assert limits == [target]
    assert cancellations == [inflight - target]
    assert controller.draining
    assert controller.escalated
