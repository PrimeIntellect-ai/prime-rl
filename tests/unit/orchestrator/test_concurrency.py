import pytest

from prime_rl.configs.orchestrator import ConcurrencyConfig, RecomputeConfig
from prime_rl.orchestrator.concurrency import (
    QUEUE_PERSISTENCE_POLLS,
    ConcurrencyController,
    EngineLoadSample,
    quantile,
)


def engine(
    engine_id: str = "e0",
    role: str | None = None,
    *,
    running: int = 10,
    waiting: int = 0,
    kv_usage: float = 0.3,
    preemptions: int = 0,
    finished: int = 0,
    expired: int = 0,
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
        expired_kv_delta=expired,
        host_kv_usage=host_kv_usage,
    )


def controller(inflight: int = 20, recompute: bool = False) -> tuple[ConcurrencyController, list[int], list[int]]:
    config = ConcurrencyConfig(
        initial_inflight=20, recompute=RecomputeConfig(grow_below=0.4, trim_above=0.7) if recompute else None
    )
    ctl = ConcurrencyController(config, fallback_cost=1)
    limits, cancelled = [], []
    ctl.bind(set_limit=limits.append, get_inflight=lambda: inflight, on_overload=cancelled.append)
    return ctl, limits, cancelled


def test_quantile_interpolates():
    assert quantile([0.5], 0.9) == 0.5
    assert quantile([0.0, 0.0, 0.2, 1.0], 0.9) == pytest.approx(0.76)


def test_kv_trim_ignores_one_hot_engine():
    ctl, limits, _ = controller()
    ctl.observe([engine("e0", kv_usage=0.95), *(engine(f"e{i}", kv_usage=0.1) for i in range(1, 4))])
    assert ctl.kv_usage < 0.8 and limits == []

    ctl.observe([engine(f"e{i}", kv_usage=0.85) for i in range(4)])
    assert limits == [16]  # pool-wide pressure still soft-trims (20 * 0.7 / 0.85)


def test_hisparse_host_pool_replaces_the_gpu_gauge():
    ctl, _, _ = controller()
    ctl.observe([engine(kv_usage=0.95, host_kv_usage=0.2)])
    assert ctl.kv_usage == 0.2


def test_preemptions_cut_by_pooled_rate():
    ctl, limits, cancelled = controller()
    ctl.observe([engine(f"e{i}", finished=50) for i in range(4)])
    ctl.observe([engine("e0", preemptions=1, finished=50), *(engine(f"e{i}", finished=50) for i in range(1, 4))])
    assert ctl.preemption_rate == 1 / 400 and ctl.signal == "clear" and limits == []

    ctl.observe([engine(f"e{i}", preemptions=2, finished=50) for i in range(4)])
    assert ctl.signal == "hard" and limits == [16] and cancelled == [4]


def test_prefill_saturation_binds_in_pd():
    ctl, limits, cancelled = controller()
    samples = [engine("p0", "prefill", running=2, waiting=4), engine("d0", "decode", running=40)]
    ctl.observe(samples)
    assert ctl.saturation_by_role == {"prefill": 2.0, "decode": 0.0}
    assert not ctl.can_grow
    for _ in range(QUEUE_PERSISTENCE_POLLS - 1):
        ctl.observe(samples)
    assert limits == [18] and cancelled == [2]  # persistent prefill queue: the cliff
    assert ctl.capacity == 1_000_000  # bootstrap capacity counts decode engines only


def test_weight_update_pause_freezes_and_skips_the_backlog():
    ctl, limits, _ = controller()
    ctl.pause()
    ctl.observe([engine(running=0, waiting=50, preemptions=5)])
    assert ctl.signal == "clear" and not ctl.load_polls

    ctl.resume()
    backlog = [engine(running=10, waiting=20)]
    for _ in range(QUEUE_PERSISTENCE_POLLS):
        ctl.observe(backlog)
    assert limits == [] and ctl.queue_overload_polls == 0 and not ctl.can_grow
    for _ in range(QUEUE_PERSISTENCE_POLLS):
        ctl.observe(backlog)
    assert limits == [18]  # past the window a persistent queue cuts again


def test_expired_leases_trim_busy_engines():
    ctl, limits, cancelled = controller(recompute=True)
    ctl.record_calls(1_000_000, 0)
    ctl.observe(
        [
            engine("p0", "prefill", running=4, waiting=1, finished=100, expired=5),
            engine("d0", "decode", running=40, finished=100),
        ]
    )
    assert ctl.expired_lease_rate == 0.05 and ctl.waste == 0.05 and ctl.engines_busy
    assert limits == [18] and cancelled == []
