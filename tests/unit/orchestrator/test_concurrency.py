from prime_rl.configs.orchestrator import ConcurrencyConfig
from prime_rl.orchestrator.concurrency import (
    PREEMPTION_CUT_FRACTION,
    QUEUE_CUT_FRACTION,
    QUEUE_PERSISTENCE_POLLS,
    ConcurrencyController,
    EngineLoadSample,
)


def _sample(*, waiting: int = 0, preemptions: int = 0) -> EngineLoadSample:
    return EngineLoadSample(
        engine_id="decode-0",
        role="decode",
        kv_capacity_tokens=1_000_000,
        max_model_len=131_072,
        kv_usage=0.3,
        running=100,
        waiting=waiting,
        waiting_capacity=waiting,
        preemptions_delta=preemptions,
    )


def _controller(inflight: int = 1_000) -> tuple[ConcurrencyController, list[int], list[int]]:
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
    return controller, limits, cancellations


def test_persistent_queue_soft_cuts_without_cancelling_active_work() -> None:
    controller, limits, cancellations = _controller()

    for _ in range(QUEUE_PERSISTENCE_POLLS):
        controller.observe([_sample(waiting=60)])

    assert controller.signal == "soft"
    assert controller.max_inflight == int(1_000 * QUEUE_CUT_FRACTION)
    assert limits == [int(1_000 * QUEUE_CUT_FRACTION)]
    assert cancellations == []
    assert controller.draining
    assert not controller.escalated


def test_preemption_hard_cuts_and_cancels_active_work() -> None:
    controller, limits, cancellations = _controller()

    controller.observe([_sample(preemptions=1)])

    target = int(1_000 * PREEMPTION_CUT_FRACTION)
    assert controller.signal == "hard"
    assert controller.max_inflight == target
    assert limits == [target]
    assert cancellations == [1_000 - target]
    assert controller.draining
    assert controller.escalated
