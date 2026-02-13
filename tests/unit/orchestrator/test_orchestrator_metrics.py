from prometheus_client import CollectorRegistry, generate_latest

from prime_rl.orchestrator.metrics import OrchestratorPrometheusMetrics


def test_orchestrator_metrics_update_and_cleanup():
    registry = CollectorRegistry()
    metrics = OrchestratorPrometheusMetrics(registry)

    metrics.update(
        {
            "step": 7,
            "progress/total_tokens": 120,
            "progress/total_samples": 12,
            "perf/throughput": 55.0,
            "reward/mean": 0.4,
            "batch/effective_batch_size": 0.6,
            "batch/solve_none": 0.2,
            "batch/solve_all": 0.1,
            "progress/ckpt_step": 6,
            "time/step": 1.5,
            "time/generate_completions": 0.7,
            "batch/async_level": 2,
            "batch/off_policy_level/max": 3,
            "batch/off_policy_level/mean": 1.5,
            "batch/cancelled_rollouts": 1,
            "error/mean": 0.05,
            "seq_len/mean": 128,
            "completion_len/mean": 64,
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
        }
    )

    content = generate_latest(registry).decode()
    assert "orchestrator_step 7.0" in content
    assert "orchestrator_throughput_tokens_per_sec 55.0" in content
    assert 'orchestrator_env_reward_mean{env="env_a"} 0.5' in content
    assert 'orchestrator_env_batch_ratio{env="env_b"} 0.45' in content
    assert 'orchestrator_worker_pending_count{worker="worker_a"} 3.0' in content
    assert 'orchestrator_worker_lag_seconds{stat="max",worker="worker_b"} 0.2' in content
    assert 'orchestrator_event_loop_lag_seconds{stat="mean"} 0.02' in content
    assert "orchestrator_last_step_timestamp_seconds" in content

    metrics.update(
        {
            "step": 8,
            "reward/env_a": 0.6,
            "batch/env_a": 0.7,
            "worker/worker_a/pending": 2,
            "worker_lag/worker_a/mean": 0.05,
        }
    )

    content = generate_latest(registry).decode()
    assert 'env="env_b"' not in content
    assert 'worker="worker_b"' not in content
