"""Prometheus metric definitions for the orchestrator."""

import time

from prometheus_client import CollectorRegistry, Gauge


class OrchestratorPrometheusMetrics:
    """Container for orchestrator Prometheus metrics and update helpers."""

    def __init__(self, registry: CollectorRegistry):
        """Register all orchestrator metrics in the provided Prometheus registry."""
        self.step = Gauge("orchestrator_step", "Current orchestrator step", registry=registry)
        self.total_tokens = Gauge("orchestrator_total_tokens", "Total tokens processed", registry=registry)
        self.total_samples = Gauge("orchestrator_total_samples", "Total samples processed", registry=registry)
        self.throughput = Gauge("orchestrator_throughput_tokens_per_sec", "Tokens per second", registry=registry)
        self.reward_mean = Gauge("orchestrator_reward_mean", "Mean reward", registry=registry)
        self.effective_batch_size = Gauge(
            "orchestrator_effective_batch_size", "Effective batch size ratio", registry=registry
        )
        self.solve_none = Gauge("orchestrator_solve_none", "Fraction solving none", registry=registry)
        self.solve_all = Gauge("orchestrator_solve_all", "Fraction solving all", registry=registry)
        self.ckpt_step = Gauge("orchestrator_ckpt_step", "Current checkpoint step from trainer", registry=registry)
        self.last_step_ts = Gauge(
            "orchestrator_last_step_timestamp_seconds", "Unix timestamp of last step", registry=registry
        )
        self.step_duration = Gauge("orchestrator_step_duration_seconds", "Step duration", registry=registry)
        self.generate_completions_duration = Gauge(
            "orchestrator_generate_completions_duration_seconds", "Generation duration", registry=registry
        )
        self.async_level = Gauge("orchestrator_async_level", "Steps ahead of trainer", registry=registry)
        self.off_policy_max = Gauge("orchestrator_off_policy_level_max", "Max off-policy steps", registry=registry)
        self.off_policy_mean = Gauge("orchestrator_off_policy_level_mean", "Mean off-policy steps", registry=registry)
        self.cancelled_rollouts = Gauge(
            "orchestrator_cancelled_rollouts", "Cancelled rollouts this step", registry=registry
        )
        self.error_rate = Gauge("orchestrator_error_rate", "Error rate across rollouts", registry=registry)
        self.seq_len_mean = Gauge("orchestrator_seq_len_mean", "Mean sequence length", registry=registry)
        self.completion_len_mean = Gauge(
            "orchestrator_completion_len_mean", "Mean completion length", registry=registry
        )
        self.event_loop_lag = Gauge(
            "orchestrator_event_loop_lag_seconds", "Event loop lag", ["stat"], registry=registry
        )
        self.env_reward = Gauge(
            "orchestrator_env_reward_mean", "Mean reward per environment", ["env"], registry=registry
        )
        self.env_batch_ratio = Gauge(
            "orchestrator_env_batch_ratio", "Batch fraction per environment", ["env"], registry=registry
        )
        self.worker_pending = Gauge(
            "orchestrator_worker_pending_count", "Pending requests per worker", ["worker"], registry=registry
        )
        self.worker_lag = Gauge(
            "orchestrator_worker_lag_seconds", "Worker event loop lag", ["worker", "stat"], registry=registry
        )
        self.pool_ratio = Gauge("orchestrator_pool_ratio", "Pool distribution", ["pool"], registry=registry)

        self._known_reward_envs: set[str] = set()
        self._known_batch_envs: set[str] = set()
        self._known_workers: set[str] = set()
        self._known_worker_lag: set[tuple[str, str]] = set()

    def update(self, to_log: dict[str, float]) -> None:
        """
        Update orchestrator metrics from a single monitor log payload.

        Note: only using a subset of metrics from the monitor log that are
        stable, low cardinality metrics that are suitable for Prometheus.
        """
        self.step.set(to_log.get("step", 0))
        self.total_tokens.set(to_log.get("progress/total_tokens", 0))
        self.total_samples.set(to_log.get("progress/total_samples", 0))
        self.throughput.set(to_log.get("perf/throughput", 0))
        self.reward_mean.set(to_log.get("reward/mean", 0))
        self.effective_batch_size.set(to_log.get("batch/effective_batch_size", 0))
        self.solve_none.set(to_log.get("batch/solve_none", 0))
        self.solve_all.set(to_log.get("batch/solve_all", 0))
        self.ckpt_step.set(to_log.get("progress/ckpt_step", 0))
        self.last_step_ts.set(time.time())

        self.step_duration.set(to_log.get("time/step", 0))
        self.generate_completions_duration.set(to_log.get("time/generate_completions", 0))
        self.async_level.set(to_log.get("batch/async_level", 0))
        self.off_policy_max.set(to_log.get("batch/off_policy_level/max", 0))
        self.off_policy_mean.set(to_log.get("batch/off_policy_level/mean", 0))
        self.cancelled_rollouts.set(to_log.get("batch/cancelled_rollouts", 0))
        self.error_rate.set(to_log.get("error/mean", 0))
        self.seq_len_mean.set(to_log.get("seq_len/mean", 0))
        self.completion_len_mean.set(to_log.get("completion_len/mean", 0))

        for stat in ["min", "mean", "med", "p90", "max"]:
            key = f"event_loop_lag/{stat}"
            if key in to_log:
                self.event_loop_lag.labels(stat=stat).set(to_log[key])

        current_reward_envs = set()
        current_batch_envs = set()
        for key, value in to_log.items():
            if key.startswith("reward/") and key != "reward/mean":
                env = key.replace("reward/", "", 1)
                current_reward_envs.add(env)
                self.env_reward.labels(env=env).set(value)

            if key.startswith("batch/"):
                if key in {
                    "batch/solve_none",
                    "batch/solve_all",
                    "batch/effective_batch_size",
                    "batch/async_level",
                    "batch/inflight_rollouts",
                    "batch/inflight_samples",
                    "batch/cancelled_rollouts",
                }:
                    continue
                if key.startswith("batch/off_policy_level/"):
                    continue
                env = key.replace("batch/", "", 1)
                current_batch_envs.add(env)
                self.env_batch_ratio.labels(env=env).set(value)

        removed_reward_envs = self._known_reward_envs - current_reward_envs
        for env in removed_reward_envs:
            self.env_reward.remove(env)

        removed_batch_envs = self._known_batch_envs - current_batch_envs
        for env in removed_batch_envs:
            self.env_batch_ratio.remove(env)

        self._known_reward_envs = current_reward_envs
        self._known_batch_envs = current_batch_envs

        current_workers = set()
        current_worker_lag = set()
        for key, value in to_log.items():
            if key.startswith("worker/") and key.endswith("/pending"):
                worker = key.replace("worker/", "", 1).replace("/pending", "", 1)
                current_workers.add(worker)
                self.worker_pending.labels(worker=worker).set(value)
            if key.startswith("worker_lag/"):
                _, worker, stat = key.split("/", 2)
                current_worker_lag.add((worker, stat))
                self.worker_lag.labels(worker=worker, stat=stat).set(value)

        removed_workers = self._known_workers - current_workers
        for worker in removed_workers:
            self.worker_pending.remove(worker)
        self._known_workers = current_workers

        removed_worker_lag = self._known_worker_lag - current_worker_lag
        for worker, stat in removed_worker_lag:
            self.worker_lag.remove(worker, stat)
        self._known_worker_lag = current_worker_lag

        for pool in ["easy", "normal", "hard"]:
            key = f"pool/{pool}"
            if key in to_log:
                self.pool_ratio.labels(pool=pool).set(to_log[key])
