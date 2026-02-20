"""Prometheus metric definitions for trainer processes."""

import time
from dataclasses import dataclass

from prometheus_client import CollectorRegistry, Gauge


@dataclass
class RunStats:
    """Statistics for a single run/LoRA adapter."""

    run_id: str
    step: int
    total_tokens: int
    learning_rate: float
    ready: bool


class TrainerPrometheusMetrics:
    """Container for trainer Prometheus metrics and update helpers."""

    def __init__(self, registry: CollectorRegistry):
        """Register all trainer metrics in the provided Prometheus registry."""
        self.step = Gauge("trainer_step", "Current training step", registry=registry)
        self.loss = Gauge("trainer_loss", "Current training loss", registry=registry)
        self.throughput = Gauge(
            "trainer_throughput_tokens_per_sec", "Training throughput in tokens/sec", registry=registry
        )
        self.last_step_ts = Gauge(
            "trainer_last_step_timestamp_seconds", "Unix timestamp of last step", registry=registry
        )
        self.grad_norm = Gauge("trainer_grad_norm", "Gradient norm", registry=registry)
        self.peak_mem = Gauge("trainer_peak_memory_gib", "Peak GPU memory in GiB", registry=registry)
        self.lr = Gauge("trainer_learning_rate", "Current learning rate", registry=registry)
        self.mfu = Gauge("trainer_mfu_percent", "Model FLOPS utilization %", registry=registry)
        self.entropy = Gauge("trainer_entropy", "Mean entropy", registry=registry)
        self.mismatch_kl = Gauge(
            "trainer_mismatch_kl", "KL divergence between trainer and inference model", registry=registry
        )
        self.kl_ent_ratio = Gauge("trainer_kl_ent_ratio", "Ratio of mismatch KL to entropy", registry=registry)

        # Aggregate run metrics
        self.runs_discovered = Gauge("trainer_runs_discovered", "Number of run folders discovered", registry=registry)
        self.runs_active = Gauge("trainer_runs_active", "Number of runs with assigned slots", registry=registry)
        self.runs_ready = Gauge("trainer_runs_ready", "Number of runs ready for gradient updates", registry=registry)
        self.runs_max = Gauge("trainer_runs_max", "Maximum run capacity", registry=registry)

        # Per-run metrics with labels
        self.run_step = Gauge("trainer_run_step", "Training step for run", ["run"], registry=registry)
        self.run_tokens = Gauge("trainer_run_tokens", "Total tokens processed by run", ["run"], registry=registry)
        self.run_learning_rate = Gauge(
            "trainer_run_learning_rate", "Current learning rate for run", ["run"], registry=registry
        )
        self.run_ready = Gauge(
            "trainer_run_ready", "Whether run is ready for updates (1=ready, 0=not ready)", ["run"], registry=registry
        )

        # Track known run labels for cleanup
        self._known_runs: set[str] = set()

    def update(
        self,
        step: int,
        loss: float,
        throughput: float,
        grad_norm: float,
        peak_memory_gib: float,
        learning_rate: float,
        mfu: float = 0.0,
        entropy: float = 0.0,
        mismatch_kl: float = 0.0,
    ) -> None:
        """Update step-level trainer metrics after a training step."""
        self.step.set(step)
        self.loss.set(loss)
        self.throughput.set(throughput)
        self.grad_norm.set(grad_norm)
        self.peak_mem.set(peak_memory_gib)
        self.lr.set(learning_rate)
        self.mfu.set(mfu)
        self.entropy.set(entropy)
        self.mismatch_kl.set(mismatch_kl)
        if entropy > 0:
            self.kl_ent_ratio.set(mismatch_kl / entropy)
        self.last_step_ts.set(time.time())

    def update_runs(self, runs_discovered: int, runs_max: int, run_stats: list[RunStats]) -> None:
        """Update run/LoRA metrics.

        Args:
            runs_discovered: Number of run_* folders found in output directory
            runs_max: Maximum run capacity
            run_stats: List of per-run statistics
        """
        self.runs_discovered.set(runs_discovered)
        self.runs_active.set(len(run_stats))
        self.runs_ready.set(sum(1 for r in run_stats if r.ready))
        self.runs_max.set(runs_max)

        current_runs = {r.run_id for r in run_stats}
        removed_runs = self._known_runs - current_runs
        for run_id in removed_runs:
            self.run_step.remove(run_id)
            self.run_tokens.remove(run_id)
            self.run_learning_rate.remove(run_id)
            self.run_ready.remove(run_id)

        for run in run_stats:
            self.run_step.labels(run=run.run_id).set(run.step)
            self.run_tokens.labels(run=run.run_id).set(run.total_tokens)
            self.run_learning_rate.labels(run=run.run_id).set(run.learning_rate)
            self.run_ready.labels(run=run.run_id).set(1 if run.ready else 0)

        self._known_runs = current_runs
