import time

import pytest
from prometheus_client import CollectorRegistry

from prime_rl.trainer.metrics import RunStats, TrainerPrometheusMetrics
from tests.utils import prom_collect_samples, prom_sample_key, prom_sample_value


def test_trainer_metrics_update():
    registry = CollectorRegistry()
    metrics = TrainerPrometheusMetrics(registry)

    update_start = time.time()
    metrics.update(
        step=5,
        loss=0.25,
        throughput=100.0,
        grad_norm=1.5,
        peak_memory_gib=2.0,
        learning_rate=1e-4,
        mfu=10.0,
        entropy=0.1,
        mismatch_kl=0.05,
    )
    update_end = time.time()

    samples = prom_collect_samples(registry)

    assert prom_sample_value(samples, "trainer_step") == pytest.approx(5.0)
    assert prom_sample_value(samples, "trainer_loss") == pytest.approx(0.25)
    assert prom_sample_value(samples, "trainer_throughput_tokens_per_sec") == pytest.approx(100.0)
    assert prom_sample_value(samples, "trainer_grad_norm") == pytest.approx(1.5)
    assert prom_sample_value(samples, "trainer_peak_memory_gib") == pytest.approx(2.0)
    assert prom_sample_value(samples, "trainer_learning_rate") == pytest.approx(1e-4)
    assert prom_sample_value(samples, "trainer_mfu_percent") == pytest.approx(10.0)
    assert prom_sample_value(samples, "trainer_entropy") == pytest.approx(0.1)
    assert prom_sample_value(samples, "trainer_mismatch_kl") == pytest.approx(0.05)
    assert prom_sample_value(samples, "trainer_kl_ent_ratio") == pytest.approx(0.5)
    last_step_ts = prom_sample_value(samples, "trainer_last_step_timestamp_seconds")
    assert update_start <= last_step_ts <= update_end


def test_trainer_metrics_run_cleanup():
    registry = CollectorRegistry()
    metrics = TrainerPrometheusMetrics(registry)

    metrics.update_runs(
        runs_discovered=2,
        runs_max=4,
        run_stats=[
            RunStats(run_id="run_a", step=10, total_tokens=1000, learning_rate=1e-4, ready=True),
            RunStats(run_id="run_b", step=20, total_tokens=2000, learning_rate=2e-4, ready=False),
        ],
    )
    samples = prom_collect_samples(registry)
    assert prom_sample_value(samples, "trainer_runs_discovered") == pytest.approx(2.0)
    assert prom_sample_value(samples, "trainer_runs_active") == pytest.approx(2.0)
    assert prom_sample_value(samples, "trainer_runs_ready") == pytest.approx(1.0)
    assert prom_sample_value(samples, "trainer_runs_max") == pytest.approx(4.0)
    assert prom_sample_value(samples, "trainer_run_step", run="run_a") == pytest.approx(10.0)
    assert prom_sample_value(samples, "trainer_run_step", run="run_b") == pytest.approx(20.0)
    assert prom_sample_value(samples, "trainer_run_tokens", run="run_a") == pytest.approx(1000.0)
    assert prom_sample_value(samples, "trainer_run_tokens", run="run_b") == pytest.approx(2000.0)
    assert prom_sample_value(samples, "trainer_run_learning_rate", run="run_a") == pytest.approx(1e-4)
    assert prom_sample_value(samples, "trainer_run_learning_rate", run="run_b") == pytest.approx(2e-4)
    assert prom_sample_value(samples, "trainer_run_ready", run="run_a") == pytest.approx(1.0)
    assert prom_sample_value(samples, "trainer_run_ready", run="run_b") == pytest.approx(0.0)

    metrics.update_runs(
        runs_discovered=1,
        runs_max=4,
        run_stats=[RunStats(run_id="run_a", step=11, total_tokens=1100, learning_rate=1e-4, ready=True)],
    )
    samples = prom_collect_samples(registry)
    assert prom_sample_value(samples, "trainer_runs_discovered") == pytest.approx(1.0)
    assert prom_sample_value(samples, "trainer_runs_active") == pytest.approx(1.0)
    assert prom_sample_value(samples, "trainer_runs_ready") == pytest.approx(1.0)
    assert prom_sample_value(samples, "trainer_runs_max") == pytest.approx(4.0)
    assert prom_sample_value(samples, "trainer_run_step", run="run_a") == pytest.approx(11.0)
    assert prom_sample_value(samples, "trainer_run_tokens", run="run_a") == pytest.approx(1100.0)
    assert prom_sample_value(samples, "trainer_run_learning_rate", run="run_a") == pytest.approx(1e-4)
    assert prom_sample_value(samples, "trainer_run_ready", run="run_a") == pytest.approx(1.0)
    assert prom_sample_key("trainer_run_step", run="run_b") not in samples
    assert prom_sample_key("trainer_run_tokens", run="run_b") not in samples
    assert prom_sample_key("trainer_run_learning_rate", run="run_b") not in samples
    assert prom_sample_key("trainer_run_ready", run="run_b") not in samples
