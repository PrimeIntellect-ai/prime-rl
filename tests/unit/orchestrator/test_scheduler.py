from types import SimpleNamespace

from prime_rl.orchestrator.scheduler import InflightRolloutInfo, Scheduler


def _build_scheduler_stub() -> Scheduler:
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.inflight_requests = {
        object(): InflightRolloutInfo(
            off_policy_steps=0,
            client_config=SimpleNamespace(api_base_url="http://client-a"),
            group_id=1,
        ),
        object(): InflightRolloutInfo(
            off_policy_steps=2,
            client_config=SimpleNamespace(api_base_url="http://client-b"),
            group_id=2,
        ),
    }
    scheduler.group_rollouts_to_schedule = {
        1: 2,
        2: 0,
        3: 1,
    }
    scheduler.wait_for_ckpt_time = 0.0
    scheduler.update_weights_time = 0.0
    scheduler.step = 7
    scheduler.ckpt_step = 5
    scheduler.cancelled_rollouts_count = 4
    scheduler.inference_pool = SimpleNamespace(get_metrics=lambda: {"pool/servers": 1.0})
    return scheduler


def test_inflight_sample_count_includes_pending_group_rollouts():
    scheduler = _build_scheduler_stub()

    assert scheduler._inflight_rollout_count() == 2
    assert scheduler._inflight_sample_count() == 5


def test_get_metrics_reports_distinct_inflight_counts():
    scheduler = _build_scheduler_stub()

    metrics = scheduler.get_metrics()

    assert metrics["batch/inflight_rollouts"] == 2
    assert metrics["batch/inflight_samples"] == 5
    assert metrics["batch/off_policy_level/max"] == 2
    assert metrics["batch/off_policy_level/mean"] == 1.0
    assert metrics["batch/off_policy_level/min"] == 0
    assert metrics["batch/cancelled_rollouts"] == 4
    assert metrics["pool/servers"] == 1.0
    assert scheduler.cancelled_rollouts_count == 0
