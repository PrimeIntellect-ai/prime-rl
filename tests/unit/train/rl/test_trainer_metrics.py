from prometheus_client import CollectorRegistry, generate_latest

from prime_rl.trainer.rl.metrics import RunStats, TrainerPrometheusMetrics


def test_trainer_metrics_update():
    registry = CollectorRegistry()
    metrics = TrainerPrometheusMetrics(registry)

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

    content = generate_latest(registry).decode()
    assert "trainer_step 5.0" in content
    assert "trainer_loss 0.25" in content
    assert "trainer_throughput_tokens_per_sec 100.0" in content
    assert "trainer_mismatch_kl 0.05" in content
    assert "trainer_last_step_timestamp_seconds" in content


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
    content = generate_latest(registry).decode()
    assert "trainer_runs_discovered 2.0" in content
    assert 'trainer_run_step{run="run_a"} 10.0' in content
    assert 'trainer_run_step{run="run_b"} 20.0' in content

    metrics.update_runs(
        runs_discovered=1,
        runs_max=4,
        run_stats=[RunStats(run_id="run_a", step=11, total_tokens=1100, learning_rate=1e-4, ready=True)],
    )
    content = generate_latest(registry).decode()
    assert 'trainer_run_step{run="run_a"} 11.0' in content
    assert "run_b" not in content
