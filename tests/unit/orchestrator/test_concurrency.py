import time

from prime_rl.configs.orchestrator import ConcurrencyConfig
from prime_rl.orchestrator import concurrency as module
from prime_rl.orchestrator.concurrency import ConcurrencyController, EngineLoadSample
from tests.unit.orchestrator.fakes import RecordingHooks


def sample(*, usage=0.1, running=10, waiting=0, capacity=100_000, preemptions=0, engine="e0", role=None):
    return EngineLoadSample(
        engine_id=engine,
        role=role,
        kv_capacity_tokens=capacity,
        max_model_len=1000,
        kv_usage=usage,
        running=running,
        waiting=waiting,
        waiting_capacity=None,
        preemptions_delta=preemptions,
    )


def make_controller(inflight: int, **config):
    hooks = RecordingHooks()
    controller = ConcurrencyController(ConcurrencyConfig(**config), fallback_cost=1000)
    state = {"inflight": inflight}
    controller.bind(
        set_limit=hooks.record("set_limit"),
        get_inflight=lambda: state["inflight"],
        on_overload=hooks.record("on_overload"),
    )
    return controller, hooks, state


def test_first_poll_derives_the_cap_from_kv_capacity():
    controller, hooks, _ = make_controller(inflight=0, max_inflight=1024)
    controller.observe([sample(capacity=50_000)])
    # 50k KV tokens / 1000 max_model_len
    assert controller.max_inflight == 50
    assert hooks["set_limit"] == [(50,)]


def test_completions_grow_the_cap_while_engines_are_clear_and_the_cap_binds():
    controller, hooks, state = make_controller(inflight=10, initial_inflight=10, max_inflight=1024)
    controller.observe([sample(usage=0.0)])
    for _ in range(10):
        controller.record_episode("env", "train", tokens=100, duration=1.0)
    assert controller.max_inflight > 10
    assert hooks["set_limit"][-1][0] == controller.max_inflight


def test_error_completions_never_grow():
    controller, hooks, _ = make_controller(inflight=10, initial_inflight=10, max_inflight=1024)
    controller.observe([sample(usage=0.0)])
    for _ in range(10):
        controller.record_episode("env", "train", tokens=0, duration=1.0)
    assert controller.max_inflight == 10
    assert hooks["set_limit"] == []


def test_growth_gate_expires_when_metrics_stall(monkeypatch):
    controller, _, _ = make_controller(inflight=10, initial_inflight=10, max_inflight=1024)
    controller.observe([sample(usage=0.0)])
    real_monotonic = time.monotonic
    monkeypatch.setattr(module.time, "monotonic", lambda: real_monotonic() + module.GROWTH_GATE_TTL_S + 1)
    controller.record_episode("env", "train", tokens=100, duration=1.0)
    assert controller.max_inflight == 10


def test_soft_trim_lowers_the_cap_without_cancelling():
    controller, hooks, _ = make_controller(inflight=100, initial_inflight=100, max_inflight=1024)
    controller.observe([sample(usage=0.85)])
    # 100 * 0.7 / 0.85
    assert controller.max_inflight == 82
    assert hooks["on_overload"] == []
    assert controller.signal == "soft"


def test_hard_trim_also_sheds_the_excess():
    controller, hooks, _ = make_controller(inflight=100, initial_inflight=100, max_inflight=1024)
    controller.observe([sample(usage=0.95)])
    assert controller.max_inflight == 73
    assert hooks["on_overload"] == [(27,)]


def test_trim_respects_the_cooldown():
    controller, hooks, _ = make_controller(inflight=100, initial_inflight=100, max_inflight=1024)
    controller.observe([sample(usage=0.85)])
    controller.observe([sample(usage=0.85)])
    assert len(hooks["set_limit"]) == 1


def test_preemptions_cut_and_latch_the_drain():
    controller, hooks, state = make_controller(inflight=100, initial_inflight=100, max_inflight=1024)
    controller.observe([sample(preemptions=3)])
    assert controller.max_inflight == 80
    assert hooks["on_overload"] == [(20,)]
    assert controller.draining
    # while draining, a second overload changes nothing
    controller.observe([sample(preemptions=3)])
    assert controller.max_inflight == 80
    # inflight drops below the cap on a quiet poll: the drain releases
    state["inflight"] = 70
    controller.observe([sample()])
    assert not controller.draining


def test_repeat_overload_inside_the_grace_window_cuts_harder():
    controller, hooks, state = make_controller(inflight=100, initial_inflight=100, max_inflight=1024)
    controller.observe([sample(preemptions=1)])
    state["inflight"] = 60
    controller.observe([sample()])  # drain releases, escalation armed
    controller.observe([sample(preemptions=1)])
    # escalated fraction 0.5 of 60
    assert controller.max_inflight == 30


def test_persistent_queue_overload_cuts_after_the_persistence_window():
    controller, hooks, _ = make_controller(inflight=100, initial_inflight=100, max_inflight=1024)
    for _ in range(module.QUEUE_PERSISTENCE_POLLS - 1):
        controller.observe([sample(running=10, waiting=20)])
    assert controller.max_inflight == 100
    controller.observe([sample(running=10, waiting=20)])
    assert controller.max_inflight == 90
    assert controller.signal == "hard"


def test_prefill_engines_do_not_count_as_pressure():
    controller, hooks, _ = make_controller(inflight=100, initial_inflight=100, max_inflight=1024)
    controller.observe([sample(usage=0.99, preemptions=5, role="prefill")])
    assert controller.max_inflight == 100
    assert controller.signal == "clear"


def test_cap_stays_inside_the_configured_band():
    controller, _, _ = make_controller(inflight=4, min_inflight=4, max_inflight=8)
    controller.observe([sample(capacity=1_000_000)])
    assert controller.max_inflight == 8
    controller.observe([sample(usage=0.99, preemptions=1)])
    assert controller.max_inflight == 4
