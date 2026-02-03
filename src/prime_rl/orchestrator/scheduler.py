"""
Scheduler that runs environments in subprocesses.

Isolates event loop lag from environment execution.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import NamedTuple

from prime_rl.orchestrator.buffer import Buffer
from prime_rl.orchestrator.config import EnvConfig, OrchestratorConfig
from prime_rl.orchestrator.env_worker import EnvWorker, WorkerDiedError
from prime_rl.orchestrator.utils import get_sampling_args
from prime_rl.utils.client import InferencePool
from prime_rl.utils.config import ClientConfig
from prime_rl.utils.logger import ProgressTracker, get_logger
from prime_rl.utils.pathing import get_env_worker_log_file
from prime_rl.utils.temp_scheduling import compute_temperature
from prime_rl.utils.utils import (
    get_broadcast_dir,
    get_latest_ckpt_step,
    get_step_path,
    wait_for_path,
)


class InflightRolloutInfo(NamedTuple):
    """Metadata for an in-flight group rollout request."""

    off_policy_steps: int
    worker: EnvWorker
    request_id: str


class Scheduler:
    """Asynchronously schedules group rollout requests using subprocess workers.

    Runs environment execution in separate processes to isolate event loop lag
    from the main orchestrator process.

    References:
    - AReal: https://arxiv.org/abs/2505.24298v1
    - PipelineRL: https://arxiv.org/abs/2509.19128v1
    """

    def __init__(
        self,
        client_config: ClientConfig,
        env_configs: list[EnvConfig],
        buffer: Buffer,
        config: OrchestratorConfig,
        oversampling_factor: float,
        max_async_level: int,
        max_off_policy_steps: int,
        strict_async_level: bool,
        inference_pool: InferencePool,
        lora_name: str | None = None,
        output_dir: Path | None = None,
    ):
        self.logger = get_logger()
        self.client_config = client_config
        self.buffer = buffer
        self.config = config
        self.batch_size = config.batch_size
        self.rollouts_per_example = config.rollouts_per_example
        self.seq_len = config.seq_len
        self.problems_per_batch = int(oversampling_factor * self.batch_size // self.rollouts_per_example)
        self.max_async_level = max_async_level
        self.max_off_policy_steps = max_off_policy_steps
        self.strict_async_level = strict_async_level
        self.lora_name = lora_name
        initial_temp = compute_temperature(step=0, sampling_config=config.sampling, max_steps=config.max_steps)
        self.sampling_args = get_sampling_args(config.sampling, temperature=initial_temp)
        self.model_name = self.config.model.name
        self.json_logging = config.log.json_logging

        # Inference pool - used for admin operations (adapter sync) and metrics
        self.inference_pool = inference_pool

        # Build example lookup dicts per env (example_id -> example)
        self.example_lookups: dict[str, dict[int, dict]] = {}
        for env_config in env_configs:
            env_name = env_config.name or env_config.id
            self.example_lookups[env_name] = buffer.example_buffer[env_name].copy()

        # Create workers - multiple per env
        self.workers_per_env = config.workers_per_env or 1
        self.workers: dict[str, list[EnvWorker]] = {}
        self.env_names: list[str] = []
        for env_config in env_configs:
            env_name = env_config.name or env_config.id
            self.env_names.append(env_name)
            self.workers[env_name] = []

            # Setup log file if env worker file logging is enabled (all workers share one file)
            env_log_file = None
            if config.log.env_worker_logs and output_dir is not None:
                env_log_file = get_env_worker_log_file(output_dir, env_name)
                env_log_file.parent.mkdir(parents=True, exist_ok=True)

            for worker_idx in range(self.workers_per_env):
                # Start with base model name - workers will be updated after LoRA is loaded
                worker = EnvWorker(
                    env_id=env_config.id,
                    env_args=env_config.args,
                    client_config=client_config,
                    model_name=self.model_name,
                    seq_len=config.seq_len,
                    interleaved_rollouts=config.trajectory_strategy == "interleaved",
                    max_concurrent_groups=(config.max_concurrent // (self.rollouts_per_example * self.workers_per_env))
                    if config.max_concurrent is not None
                    else -1,
                    tasks_per_minute=config.tasks_per_minute or -1,
                    example_lookup=self.example_lookups[env_name],
                    worker_name=f"{env_name}_{worker_idx}",
                    log_level=config.log.level,
                    vf_log_level=config.log.vf_level,
                    log_file=str(env_log_file) if env_log_file else None,
                    max_restarts=config.max_env_worker_restarts,
                    json_logging=config.log.json_logging,
                )
                self.workers[env_name].append(worker)

        # Track in-flight requests: future -> info
        self.inflight_group_rollouts: dict[asyncio.Future, InflightRolloutInfo] = {}

        self.step, self.ckpt_step = 0, 0
        self.checkpoint_ready = asyncio.Event()
        self.checkpoint_ready.set()
        self.update_weights_time, self.wait_for_ckpt_time = 0, 0
        self.cancelled_rollouts_count = 0
        self.last_batch_generation_time = 0.0

        # Background tasks
        self._response_collectors: list[asyncio.Task] = []

    async def start(self):
        """Start all workers and response collectors."""
        total_workers = sum(len(workers) for workers in self.workers.values())
        self.logger.info(f"Starting {total_workers} env worker(s) ({self.workers_per_env} per env)")
        for workers in self.workers.values():
            for worker in workers:
                worker.start()
                # Start response collector for each worker
                task = asyncio.create_task(worker.collect_responses())
                self._response_collectors.append(task)

    async def stop(self):
        """Stop all workers and collectors."""
        for task in self._response_collectors:
            task.cancel()
        for workers in self.workers.values():
            for worker in workers:
                worker.stop()

    def set_sampling_args(self, sampling_args: dict) -> None:
        """Update sampling args for future rollout requests."""
        self.sampling_args = sampling_args

    def cancel_all_inflight_rollouts(self):
        """Cancel all in-flight rollout requests.

        Used when weights are updated to discard stale rollouts generated with old weights.
        """
        count = len(self.inflight_group_rollouts)
        for future in list(self.inflight_group_rollouts.keys()):
            if not future.done():
                future.cancel()
        self.inflight_group_rollouts.clear()
        self.cancelled_rollouts_count += count

    async def schedule_group_rollout(self):
        """Asynchronously schedules a group rollout request."""
        example = self.buffer.sample_examples(n=1)[0]

        # Route to worker for this example's environment
        task = example["task"]
        workers = self.workers[task]
        worker = min(workers, key=lambda w: w.pending_count)

        future, request_id = await worker.submit_request(
            example_id=example["example_id"],
            rollouts_per_example=self.config.rollouts_per_example,
            sampling_args=self.sampling_args,
        )

        self.inflight_group_rollouts[future] = InflightRolloutInfo(
            off_policy_steps=0,
            worker=worker,
            request_id=request_id,
        )

    async def update_policy_loop(self):
        """Continuously checks for new policy checkpoints."""
        while True:
            await self.update_policy()
            await asyncio.sleep(1)

    async def update_policy(self):
        """Updates the policy to the latest available checkpoint. Aborts rollout requests that are older than the max retention steps."""
        latest_ckpt_step = get_latest_ckpt_step(get_broadcast_dir(self.config.output_dir)) or 0
        async_away_ckpt_step = max(self.step - self.max_async_level, 0)
        next_ckpt_step = (
            async_away_ckpt_step if self.strict_async_level else max(async_away_ckpt_step, latest_ckpt_step)
        )

        if next_ckpt_step > self.ckpt_step:
            if next_ckpt_step == async_away_ckpt_step:
                self.logger.info(
                    f"Orchestrator paused: waiting for trainer process to complete checkpoint {next_ckpt_step} "
                    f"(>{self.max_async_level} step(s) ahead). Training is progressing normally."
                )
                self.checkpoint_ready.clear()
                wait_for_ckpt_start_time = time.perf_counter()
                await wait_for_path(get_step_path(get_broadcast_dir(self.config.output_dir), next_ckpt_step) / "STABLE")
                self.wait_for_ckpt_time = time.perf_counter() - wait_for_ckpt_start_time
                self.logger.info(
                    f"Orchestrator resumed: checkpoint {next_ckpt_step} ready (after {self.wait_for_ckpt_time:.2f}s)"
                )

            self.logger.debug(
                f"Got new policy with step {next_ckpt_step}. Updating weights and cancelling old rollout requests."
            )

            # Update weights on inference servers
            update_weights_start_time = time.perf_counter()
            weights_path = get_step_path(get_broadcast_dir(self.config.output_dir), next_ckpt_step)
            await self.inference_pool.update_weights(weights_path, lora_name=self.lora_name, step=next_ckpt_step)
            self.update_weights_time = time.perf_counter() - update_weights_start_time
            self.logger.debug(f"Updated weights to step {next_ckpt_step} in {self.update_weights_time:.2f}s")

            if self.lora_name is not None:
                self.model_name = self.lora_name
                # Update workers to use LoRA name for future requests
                for workers in self.workers.values():
                    for worker in workers:
                        worker.update_model_name(self.lora_name)

            self.checkpoint_ready.set()

            # Handle off-policy tracking - cancel old requests
            futures_to_remove = []
            futures_to_update = []

            for future, info in self.inflight_group_rollouts.items():
                if info.off_policy_steps > self.max_off_policy_steps:
                    if not future.done():
                        future.cancel()
                    futures_to_remove.append((future, info.worker))
                else:
                    futures_to_update.append((future, info.off_policy_steps + 1, info.worker, info.request_id))

            # Remove cancelled
            for future, worker in futures_to_remove:
                self.inflight_group_rollouts.pop(future, None)
            self.cancelled_rollouts_count += len(futures_to_remove)

            # Update off-policy steps for remaining
            for future, off_policy_steps, worker, request_id in futures_to_update:
                if future in self.inflight_group_rollouts:
                    self.inflight_group_rollouts[future] = InflightRolloutInfo(
                        off_policy_steps=off_policy_steps,
                        worker=worker,
                        request_id=request_id,
                    )

            if len(futures_to_remove) > 0:
                self.logger.warning(
                    f"Cancelled {len(futures_to_remove)} old rollout requests (will refill naturally). Consider increasing max_off_policy_steps to avoid this."
                )

            self.ckpt_step = next_ckpt_step

    async def generate_batch(self, step: int) -> list[dict]:
        """Generate a batch of rollouts using workers.

        Returns list of result dicts (not vf.State, since those stay in workers).
        """
        self.step = step
        batch_start_time = time.perf_counter()

        # Schedule initial tasks
        self.logger.debug("Starting to generate batch rollouts")
        while len(self.inflight_group_rollouts) < self.problems_per_batch:
            await self.schedule_group_rollout()

        batch_rollouts: list[dict] = []
        pbar = ProgressTracker(
            total=self.config.batch_size, desc="Generating rollouts (train)", json_logging=self.json_logging, step=step
        )

        while len(batch_rollouts) < self.config.batch_size:
            # Wait for at least one future to complete
            done, _ = await asyncio.wait(
                self.inflight_group_rollouts.keys(),
                return_when=asyncio.FIRST_COMPLETED,
            )

            await self.checkpoint_ready.wait()

            for finished_future in done:
                if len(batch_rollouts) >= self.config.batch_size:
                    batch_rollouts = batch_rollouts[: self.config.batch_size]
                    break

                # Safely pop the future from tracking
                if self.inflight_group_rollouts.pop(finished_future, None) is None:
                    continue

                try:
                    group_results: list[dict] = finished_future.result()

                    # Update buffer with results
                    self.buffer.update(group_results)
                    accepted_rollouts = self.buffer.sample_rollouts(n=self.config.rollouts_per_example)

                    batch_rollouts.extend(accepted_rollouts)
                    pbar.update(len(accepted_rollouts))

                except asyncio.CancelledError:
                    pass  # Request was cancelled, will be rescheduled
                except WorkerDiedError as e:
                    # Worker died - check if it will auto-restart or if orchestrator should crash
                    self.logger.warning(f"Rollout lost due to worker death: {e}")
                    # Check if any worker has permanently failed (exceeded max restarts)
                    self._check_for_fatal_worker_errors()
                except Exception as e:
                    self.logger.warning(f"Rollout failed: {e}")

                await self.schedule_group_rollout()

        pbar.close()
        self.last_batch_generation_time = time.perf_counter() - batch_start_time
        return batch_rollouts

    def _check_for_fatal_worker_errors(self):
        """Check if any worker has permanently failed and raise the error to crash orchestrator."""
        for workers in self.workers.values():
            for worker in workers:
                if worker._fatal_error is not None:
                    self.logger.error(f"Worker '{worker.worker_name}' permanently failed, crashing orchestrator")
                    raise worker._fatal_error

    @property
    def max_off_policy_level(self) -> int:
        if not self.inflight_group_rollouts:
            return 0
        return max(info.off_policy_steps for info in self.inflight_group_rollouts.values())

    @property
    def min_off_policy_level(self) -> int:
        if not self.inflight_group_rollouts:
            return 0
        return min(info.off_policy_steps for info in self.inflight_group_rollouts.values())

    @property
    def mean_off_policy_level(self) -> float:
        if not self.inflight_group_rollouts:
            return 0
        steps = [info.off_policy_steps for info in self.inflight_group_rollouts.values()]
        return sum(steps) / len(steps)

    @property
    def async_level(self) -> int:
        return self.step - self.ckpt_step

    def get_metrics(self) -> dict[str, float]:
        metrics = {
            "time/wait_for_ckpt": self.wait_for_ckpt_time,
            "time/update_weights": self.update_weights_time,
            "batch/async_level": self.async_level,
            "batch/inflight_rollouts": len(self.inflight_group_rollouts),
            "batch/inflight_samples": len(self.inflight_group_rollouts) * self.rollouts_per_example,
            "batch/off_policy_level/max": self.max_off_policy_level,
            "batch/off_policy_level/mean": self.mean_off_policy_level,
            "batch/off_policy_level/min": self.min_off_policy_level,
            "batch/cancelled_rollouts": self.cancelled_rollouts_count,
        }
        self.cancelled_rollouts_count = 0

        # Add per-worker lag metrics and pending counts
        for workers in self.workers.values():
            for worker in workers:
                worker_key = worker.worker_name
                # Track pending count per worker (useful for debugging load balancing)
                metrics[f"worker/{worker_key}/pending"] = worker.pending_count
                if worker.latest_lag_metrics:
                    for metric_name, value in worker.latest_lag_metrics.items():
                        # e.g. "worker_lag/env_0/max"
                        metrics[f"worker_lag/{worker_key}/{metric_name.split('/')[-1]}"] = value

        # Add inference pool metrics (e.g. elastic pool server counts)
        metrics.update(self.inference_pool.get_metrics())

        return metrics
