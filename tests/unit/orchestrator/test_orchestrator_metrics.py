import time

import pytest
from prometheus_client import CollectorRegistry

from prime_rl.orchestrator.metrics import OrchestratorPrometheusMetrics
from tests.utils import prom_collect_samples, prom_sample_key, prom_sample_value


def test_orchestrator_metrics_update_and_cleanup():
    registry = CollectorRegistry()
    metrics = OrchestratorPrometheusMetrics(registry)
    env_names = {"env_a", "env_b"}

    update_start = time.time()
    metrics.update(
        {
            "step": 7,
            "progress/total_tokens": 120,
            "progress/total_samples": 12,
            "perf/throughput": 55.0,
            "reward/mean": 0.4,
            "reward/std": 0.2,
            "reward/min": 0.1,
            "reward/max": 0.8,
            "reward/median": 0.35,
            "batch/effective_batch_size": 0.6,
            "batch/solve_none": 0.2,
            "batch/solve_all": 0.1,
            "progress/ckpt_step": 6,
            "time/step": 1.5,
            "time/generate_completions": 0.7,
            "batch/async_level": 2,
            "batch/inflight_rollouts": 4,
            "batch/inflight_samples": 8,
            "batch/off_policy_level/max": 3,
            "batch/off_policy_level/mean": 1.5,
            "batch/cancelled_rollouts": 1,
            "error/mean": 0.05,
            "seq_len/mean": 128,
            "decode_len/mean": 64,
            "event_loop_lag/mean": 0.02,
            "reward/env_a": 0.5,
            "reward/env_b": 0.3,
            "batch/env_a": 0.55,
            "batch/env_b": 0.45,
            "worker/worker_a/pending": 3,
            "worker/worker_b/pending": 1,
            "worker_lag/worker_a/mean": 0.1,
            "worker_lag/worker_b/max": 0.2,
            "pool/easy": 0.2,
            "pool/normal": 0.6,
            "pool/hard": 0.2,
        },
        env_names=env_names,
    )
    update_end = time.time()

    samples = prom_collect_samples(registry)
    assert prom_sample_value(samples, "orchestrator_step") == pytest.approx(7.0)
    assert prom_sample_value(samples, "orchestrator_total_tokens") == pytest.approx(120.0)
    assert prom_sample_value(samples, "orchestrator_total_samples") == pytest.approx(12.0)
    assert prom_sample_value(samples, "orchestrator_throughput_tokens_per_sec") == pytest.approx(55.0)
    assert prom_sample_value(samples, "orchestrator_reward_mean") == pytest.approx(0.4)
    assert prom_sample_value(samples, "orchestrator_reward_std") == pytest.approx(0.2)
    assert prom_sample_value(samples, "orchestrator_reward_min") == pytest.approx(0.1)
    assert prom_sample_value(samples, "orchestrator_reward_max") == pytest.approx(0.8)
    assert prom_sample_value(samples, "orchestrator_reward_median") == pytest.approx(0.35)
    assert prom_sample_value(samples, "orchestrator_effective_batch_size") == pytest.approx(0.6)
    assert prom_sample_value(samples, "orchestrator_solve_none") == pytest.approx(0.2)
    assert prom_sample_value(samples, "orchestrator_solve_all") == pytest.approx(0.1)
    assert prom_sample_value(samples, "orchestrator_ckpt_step") == pytest.approx(6.0)
    assert prom_sample_value(samples, "orchestrator_step_duration_seconds") == pytest.approx(1.5)
    assert prom_sample_value(samples, "orchestrator_generate_completions_duration_seconds") == pytest.approx(0.7)
    assert prom_sample_value(samples, "orchestrator_async_level") == pytest.approx(2.0)
    assert prom_sample_value(samples, "orchestrator_off_policy_level_max") == pytest.approx(3.0)
    assert prom_sample_value(samples, "orchestrator_off_policy_level_mean") == pytest.approx(1.5)
    assert prom_sample_value(samples, "orchestrator_cancelled_rollouts") == pytest.approx(1.0)
    assert prom_sample_value(samples, "orchestrator_error_rate") == pytest.approx(0.05)
    assert prom_sample_value(samples, "orchestrator_seq_len_mean") == pytest.approx(128.0)
    assert prom_sample_value(samples, "orchestrator_decode_len_mean") == pytest.approx(64.0)
    assert prom_sample_value(samples, "orchestrator_event_loop_lag_seconds", stat="mean") == pytest.approx(0.02)
    assert prom_sample_value(samples, "orchestrator_env_reward_mean", env="env_a") == pytest.approx(0.5)
    assert prom_sample_value(samples, "orchestrator_env_reward_mean", env="env_b") == pytest.approx(0.3)
    assert prom_sample_key("orchestrator_env_reward_mean", env="std") not in samples
    assert prom_sample_key("orchestrator_env_reward_mean", env="min") not in samples
    assert prom_sample_key("orchestrator_env_reward_mean", env="max") not in samples
    assert prom_sample_key("orchestrator_env_reward_mean", env="median") not in samples
    assert prom_sample_value(samples, "orchestrator_env_batch_ratio", env="env_a") == pytest.approx(0.55)
    assert prom_sample_value(samples, "orchestrator_env_batch_ratio", env="env_b") == pytest.approx(0.45)
    assert prom_sample_value(samples, "orchestrator_worker_pending_count", worker="worker_a") == pytest.approx(3.0)
    assert prom_sample_value(samples, "orchestrator_worker_pending_count", worker="worker_b") == pytest.approx(1.0)
    assert prom_sample_value(
        samples, "orchestrator_worker_lag_seconds", worker="worker_a", stat="mean"
    ) == pytest.approx(0.1)
    assert prom_sample_value(
        samples, "orchestrator_worker_lag_seconds", worker="worker_b", stat="max"
    ) == pytest.approx(0.2)
    assert prom_sample_value(samples, "orchestrator_pool_ratio", pool="easy") == pytest.approx(0.2)
    assert prom_sample_value(samples, "orchestrator_pool_ratio", pool="normal") == pytest.approx(0.6)
    assert prom_sample_value(samples, "orchestrator_pool_ratio", pool="hard") == pytest.approx(0.2)
    last_step_ts = prom_sample_value(samples, "orchestrator_last_step_timestamp_seconds")
    assert update_start <= last_step_ts <= update_end

    assert prom_sample_key("orchestrator_env_batch_ratio", env="inflight_rollouts") not in samples
    assert prom_sample_key("orchestrator_env_batch_ratio", env="inflight_samples") not in samples

    metrics.update(
        {
            "step": 8,
            "reward/env_a": 0.6,
            "batch/env_a": 0.7,
            "worker/worker_a/pending": 2,
            "worker_lag/worker_a/mean": 0.05,
        },
        env_names=env_names,
    )

    samples = prom_collect_samples(registry)
    assert prom_sample_value(samples, "orchestrator_step") == pytest.approx(8.0)
    assert prom_sample_value(samples, "orchestrator_env_reward_mean", env="env_a") == pytest.approx(0.6)
    assert prom_sample_value(samples, "orchestrator_env_batch_ratio", env="env_a") == pytest.approx(0.7)
    assert prom_sample_value(samples, "orchestrator_worker_pending_count", worker="worker_a") == pytest.approx(2.0)
    assert prom_sample_value(
        samples, "orchestrator_worker_lag_seconds", worker="worker_a", stat="mean"
    ) == pytest.approx(0.05)

    assert prom_sample_key("orchestrator_env_reward_mean", env="env_b") not in samples
    assert prom_sample_key("orchestrator_env_batch_ratio", env="env_b") not in samples
    assert prom_sample_key("orchestrator_worker_pending_count", worker="worker_b") not in samples
    assert prom_sample_key("orchestrator_worker_lag_seconds", worker="worker_b", stat="max") not in samples
