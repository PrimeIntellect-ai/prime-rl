"""Shipper: the trainer-facing end of the pipeline.

Takes each cut ``TrainBatch``, holds it until inference serves a recent enough
policy, packs and sends it, advances the step, checkpoints, and reports the step's
metrics. It owns ``Progress`` (the step every other component reads) and the lag
gate: dispatch pauses while the batch being collected runs more than
``target_lag`` versions ahead of the policy inference serves. After the final
batch it asks the pipeline to drain."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any

from prime_rl import monitors as default_monitors
from prime_rl.configs.orchestrator import CheckpointConfig, ShipperConfig
from prime_rl.orchestrator.algo.routing import is_trainable
from prime_rl.orchestrator.annotations import stamp_batch
from prime_rl.orchestrator.ckpt import CheckpointManager
from prime_rl.orchestrator.metrics import TrainEpisodes, dispatch_failure_metrics
from prime_rl.orchestrator.packing import BatchPacker
from prime_rl.orchestrator.train_source import TrainSource
from prime_rl.orchestrator.types import DispatchFailure, Progress, TrainBatch
from prime_rl.orchestrator.utils import episode_group_id, episode_staleness, trim_process_memory
from prime_rl.transports.batch.base import BatchSender
from prime_rl.utils.heartbeat import Heartbeat
from prime_rl.utils.logger import format_time, get_logger


class Shipper:
    def __init__(
        self,
        config: ShipperConfig,
        *,
        packer: BatchPacker,
        sender: BatchSender,
        ckpt_manager: CheckpointManager,
        ckpt_config: CheckpointConfig | None,
        train_source: TrainSource,
        heart: Heartbeat | None = None,
    ) -> None:
        self.config = config
        self.packer = packer
        self.sender = sender
        self.ckpt_manager = ckpt_manager
        self.ckpt_config = ckpt_config
        self.train_source = train_source
        self.heart = heart

        self.progress = Progress()
        self.draining = asyncio.Event()
        # Previous batch arrival, reset every ship so ``time/step`` is sink-to-sink cycle time.
        self.last_batch_at: float | None = None
        self.wait_for_policy_time = 0.0
        self.gate_closed_at: float | None = None
        self.gate_open = True

        self._version: Callable[[], int] = lambda: 0
        self._wait_for_version: Callable[..., Awaitable[bool]] | None = None
        self._gate: Callable[[bool], None] = lambda open: None
        self._on_drain: Callable[[str], Awaitable[None]] | None = None
        self.monitors: Any = default_monitors

    def bind(
        self,
        *,
        version: Callable[[], int] | None = None,
        wait_for_version: Callable[..., Awaitable[bool]] | None = None,
        gate: Callable[[bool], None] | None = None,
        on_drain: Callable[[str], Awaitable[None]] | None = None,
        monitors: Any = None,
    ) -> None:
        if version is not None:
            self._version = version
        if wait_for_version is not None:
            self._wait_for_version = wait_for_version
        if gate is not None:
            self._gate = gate
        if on_drain is not None:
            self._on_drain = on_drain
        if monitors is not None:
            self.monitors = monitors

    # ── state others read ──────────────────────────────────────────────────

    def step(self) -> int:
        """The batch being collected, 1-indexed and advanced right after a ship."""
        return self.progress.step

    def resume(self, step: int, progress: Progress | None) -> None:
        """Continue after checkpoint ``step``: restore the counters and collect ``step + 1``."""
        if progress is not None:
            self.progress = progress
        self.progress.step = step + 1

    def start_clock(self) -> None:
        """Anchor the step clock so the first step measures startup to first batch."""
        self.last_batch_at = time.perf_counter()

    # ── inbound ────────────────────────────────────────────────────────────

    async def on_version(self, _step: int) -> None:
        """Re-check the lag gate after inference applied a new policy."""
        self.update_gate()

    async def on_batch(self, batch: TrainBatch) -> None:
        """Ship one batch; a batch arriving while draining is dropped so nothing
        ships past ``max_steps``."""
        if self.draining.is_set():
            return
        config = self.config
        step = self.progress.step
        now = time.perf_counter()
        step_time = (now - self.last_batch_at) if self.last_batch_at is not None else 0.0
        self.last_batch_at = now

        # A resume can start past the end (checkpoint written at the final step, or a
        # lowered ``max_steps``): never ship beyond the budget.
        if config.max_steps is not None and step > config.max_steps:
            await self.start_draining(f"Step {step} exceeds max_steps={config.max_steps}")
            return
        if not batch.samples:
            get_logger().warning(
                f"Step {step}: skipping empty train batch after {len(batch.episodes)} finalized episodes"
            )
            return
        effective = batch.cohort.effective
        n_trainable = sum(is_trainable(record.trace) for record in effective.records)
        if effective.num_traces and n_trainable / effective.num_traces <= 0.1:
            get_logger().warning(
                f"Only {n_trainable}/{effective.num_traces} effective traces are trainable "
                f"({n_trainable / effective.num_traces:.1%}) — consider reviewing task difficulty"
            )

        # Ship batch ``step`` only once inference has applied v{step-1-target_lag}: fast
        # envs fill batches from buffered rollouts and would race arbitrarily far ahead
        # of the trainer otherwise. Always satisfiable: the trainer broadcasts every version.
        required_version = step - 1 - config.target_lag
        if self._version() < required_version:
            hold_start = time.perf_counter()
            await self.wait_for_version(required_version, f"to ship batch {step}")
            self.wait_for_policy_time += time.perf_counter() - hold_start

        # The effective (clean, trained-on) subset is logged at ship time as annotation
        # records against each trace's arrival record — never a second episode copy.
        await self.monitors.log(effective.vf_episodes, step, "train", "effective")
        await self.monitors.log_annotations(stamp_batch(effective.vf_episodes, step))

        pack_start_time = time.perf_counter()
        micro_batch_grid = await asyncio.to_thread(self.packer.pack, batch.samples)
        pack_time = time.perf_counter() - pack_start_time
        await self.sender.send(micro_batch_grid)
        self.progress.step += 1
        self.update_gate()
        save_ckpt_time = self.maybe_save_ckpt(step)
        trim_process_memory()

        await self.monitors.log(self.step_metrics(batch, step, step_time, pack_time, save_ckpt_time), step=step)
        self.warn_discards(batch, step_time)
        self.wait_for_policy_time = 0.0
        if self.heart is not None:
            self.heart.beat()
        self.progress.total_tokens += batch.episodes.num_total_tokens
        self.progress.total_samples += batch.episodes.num_traces
        self.progress.total_problems += self.num_tasks(batch)
        self.log_train_batch(batch, step=step, step_time=step_time)

        if config.max_steps is not None and step >= config.max_steps:
            await self.wait_for_version(step, "before shutdown", timeout=config.version_wait_timeout)
            # Drain right after the final batch: waiting for another to fill would burn
            # inference on data that can never train.
            await self.start_draining("Shipped the final batch")
        trim_process_memory()

    # ── internals ──────────────────────────────────────────────────────────

    async def wait_for_version(self, version: int, reason: str, *, timeout: float | None = None) -> None:
        if self._version() >= version:
            return
        if self._wait_for_version is None:
            raise RuntimeError("Shipper.wait_for_version is not bound")
        await self._wait_for_version(version, timeout=timeout, reason=reason)

    def update_gate(self) -> None:
        """Pause dispatch while the batch being collected runs more than ``target_lag``
        ahead of the policy inference serves. Steps are 1-indexed while versions are
        0-indexed, so the shipped-batch count is ``step - 1``."""
        version = self._version()
        lead = (self.progress.step - 1) - version
        if lead > self.config.target_lag:
            if self.gate_open:
                get_logger().info(
                    f"Pausing dispatcher until inference applies policy "
                    f"v{self.progress.step - 1 - self.config.target_lag} (currently v{version})"
                )
                self.gate_closed_at = time.perf_counter()
            self.gate_open = False
        else:
            if not self.gate_open:
                get_logger().info(f"Resuming dispatcher (policy v{version})")
                if self.gate_closed_at is not None:
                    self.wait_for_policy_time += time.perf_counter() - self.gate_closed_at
                    self.gate_closed_at = None
            self.gate_open = True
        self._gate(self.gate_open)

    async def start_draining(self, reason: str) -> None:
        """Stop scheduling train work and let the pipeline empty."""
        self.draining.set()
        if self._on_drain is not None:
            await self._on_drain(reason)

    def maybe_save_ckpt(self, step: int) -> float:
        """Checkpoint the step just shipped if it's an interval boundary. Returns the
        elapsed time (0.0 when no save happened)."""
        if self.ckpt_config is None or not self.ckpt_config.interval:
            return 0.0
        # The final step's checkpoint is written once at teardown (``save_final``).
        if self.config.max_steps is not None and step >= self.config.max_steps:
            return 0.0
        if step % self.ckpt_config.interval != 0:
            return 0.0
        get_logger().info(f"Saving checkpoint at step {step}")
        t = time.perf_counter()
        self.save_ckpt(step)
        return time.perf_counter() - t

    def save_ckpt(self, step: int) -> None:
        # Synchronous on purpose: the payload is tiny, and snapshotting on the event loop
        # keeps the dispatcher from mutating the train source mid-save
        self.ckpt_manager.save(step, self.progress, self.train_source.state_dict())

    def save_final(self) -> None:
        """``progress.step`` points at the next (unshipped) step; checkpoint the last
        finished one. No-op before the first ship or without a ckpt block."""
        if self.ckpt_config is None or self.progress.step <= 1:
            return
        self.progress.step -= 1
        get_logger().info(f"Saving final checkpoint at step {self.progress.step}")
        self.save_ckpt(self.progress.step)

    @staticmethod
    def num_tasks(batch: TrainBatch) -> int:
        group_ids = {episode_group_id(episode) for episode in batch.episodes}
        group_ids.update(failure.group_id for failure in batch.failures)
        return len(group_ids)

    def step_metrics(
        self, batch: TrainBatch, step: int, step_time: float, pack_time: float, save_ckpt_time: float
    ) -> dict[str, float]:
        """Episode metrics over the {agg,<env>} × {all,effective} matrix plus progress,
        timing, staleness and env-share accounting for one shipped step."""
        effective = batch.cohort.effective
        metrics: dict[str, float] = {}
        for subset, pool in (("all", batch.episodes), ("effective", effective)):
            metrics |= pool.metrics.to_wandb(prefix="train/agg", subset=subset)
            for env_name, env_pool in pool.by_env().items():
                metrics |= env_pool.metrics.to_wandb(prefix=f"train/{env_name}", subset=subset)
        total_attempts = len(batch.episodes) + len(batch.failures)
        metrics |= dispatch_failure_metrics(batch.failures, prefix="train/agg/all", total_attempts=total_attempts)
        failures_by_env: dict[str, list[DispatchFailure]] = {}
        for failure in batch.failures:
            failures_by_env.setdefault(failure.env_name, []).append(failure)
        episodes_by_env = batch.episodes.by_env()
        for env_name in set(episodes_by_env) | set(failures_by_env):
            env_failures = failures_by_env.get(env_name, [])
            env_attempts = len(episodes_by_env.get(env_name, TrainEpisodes())) + len(env_failures)
            metrics |= dispatch_failure_metrics(
                env_failures, prefix=f"train/{env_name}/all", total_attempts=env_attempts
            )

        num_tokens = batch.episodes.num_total_tokens
        metrics |= {
            "progress/tokens": num_tokens,
            "progress/input_tokens": sum(record.trace.num_input_tokens for record in effective.records),
            "progress/output_tokens": sum(record.trace.num_output_tokens for record in effective.records),
            "progress/rollouts": batch.episodes.num_traces,
            "progress/tasks": self.num_tasks(batch),
            "progress/total_tokens": self.progress.total_tokens,
            "progress/total_rollouts": self.progress.total_samples,
            "progress/total_tasks": self.progress.total_problems,
            "time/step": step_time,
            "time/pack": pack_time,
            "time/save_ckpt": save_ckpt_time,
            "time/wait_for_policy": self.wait_for_policy_time,
            "step": step,
        }
        # Staleness of the shipped cohort, decomposed into its in-flight and in-queue
        # shares; ``dropped`` counts queued traces the sweep voided since the last ship.
        staleness = [episode_staleness(episode, step) for episode in effective]
        if staleness:
            totals, in_flight, in_queue = (list(values) for values in zip(*staleness))
            metrics |= {
                "off_policy/mean": sum(totals) / len(totals),
                "off_policy/max": float(max(totals)),
                "off_policy/in_flight/mean": sum(in_flight) / len(in_flight),
                "off_policy/in_flight/max": float(max(in_flight)),
                "off_policy/in_queue/mean": sum(in_queue) / len(in_queue),
                "off_policy/in_queue/max": float(max(in_queue)),
            }
        metrics["off_policy/dropped"] = float(batch.stale_drops)
        for env_name, env_pool in episodes_by_env.items():
            metrics[f"batch/{env_name}"] = env_pool.num_traces / batch.episodes.num_traces
        metrics |= self.train_source.metrics()
        return metrics

    def warn_discards(self, batch: TrainBatch, step_time: float) -> None:
        active_step_time = max(step_time - self.wait_for_policy_time, 0.0)
        if step_time > 0 and self.wait_for_policy_time >= active_step_time:
            get_logger().warning(
                f"Orchestrator waited {format_time(self.wait_for_policy_time)} for policy updates, at least as long "
                f"as its {format_time(active_step_time)} active step time. Train-inference compute is imbalanced; "
                "add more trainer nodes."
            )
        shipped_episode_ids = {episode.id for episode in batch.cohort}
        discarded = [
            episode
            for episode in batch.episodes
            if episode.id not in shipped_episode_ids and episode.id not in batch.buffered_episode_ids
        ]
        stale_episodes = sum(episode.id in batch.episodes.cancelled for episode in discarded)
        errored_episodes = sum(
            episode.id not in batch.episodes.cancelled
            and (not episode.ok or any(trace.has_error for trace in episode.traces))
            for episode in discarded
        )
        num_attempts = len(batch.episodes) + len(batch.failures) + batch.cancelled_attempts
        num_discarded = len(discarded) + len(batch.failures) + batch.cancelled_attempts
        num_stale = stale_episodes + batch.stale_attempts
        num_errored = errored_episodes + len(batch.failures)
        num_no_signal = num_discarded - num_stale - num_errored
        if num_attempts and num_discarded / num_attempts > 0.5:
            get_logger().warning(
                f"Discarded {num_discarded}/{num_attempts} episodes ({num_discarded / num_attempts:.1%}): "
                f"stale={num_stale}, errored={num_errored}, no_signal={num_no_signal}. Review max_off_policy_steps, "
                "episode errors, and reward signal."
            )

    def log_train_batch(self, batch: TrainBatch, *, step: int, step_time: float) -> None:
        """Per-step ``Step …`` success line. Multi-env runs append an indented ``╰─`` line per env.
        Quality metrics (Reward, Trainable, Turns, Branches, Max Off-Policy, Truncation) are
        over exactly the traces shipped this step (``batch.cohort``); ``Error``, ``Cancelled``
        and ``Ratio`` describe the step's full arrival window."""
        episodes = batch.episodes
        effective = batch.cohort.effective
        eff = effective.metrics
        n_generated = episodes.num_traces
        n_effective = effective.num_traces
        n_trainable = sum(is_trainable(record.trace) for record in effective.records)
        trainable_rate = (n_trainable / n_effective) if n_effective else 0.0
        max_off_policy_steps = max((episode_staleness(episode, step)[0] for episode in effective), default=0)

        head = (
            f"Step {step} | {format_time(step_time):>7} | Reward {eff.reward.mean():.4f} | "
            f"Trainable {n_trainable}/{n_effective} ({trainable_rate:.1%}) | "
            f"Turns {eff.num_turns.mean():.1f} | Branches {eff.num_branches.mean():.1f} | "
            f"Max Off-Policy {max_off_policy_steps} | "
            f"Error {episodes.metrics.has_error.mean():.1%} | Cancelled {episodes.metrics.cancelled.mean():.1%} | "
            f"Truncation {eff.is_truncated.mean():.1%}"
        )
        if len(self.train_source.env_names) <= 1:
            get_logger().success(head)
            return

        window_by_env = episodes.by_env()
        shipped_by_env = effective.by_env()
        env_names = sorted(set(window_by_env) | set(shipped_by_env))
        name_width = max((len(name) for name in env_names), default=0)
        lines = [head]
        for env_name in env_names:
            pool = window_by_env.get(env_name, TrainEpisodes())
            env_eff_pool = shipped_by_env.get(env_name, TrainEpisodes())
            env_eff = env_eff_pool.metrics
            ratio = (pool.num_traces / n_generated) if n_generated else 0.0
            lines.append(
                f"╰─ {env_name:<{name_width}} | Ratio {ratio:.1%} | Reward {env_eff.reward.mean():.4f} | "
                f"Turns {env_eff.num_turns.mean():.1f} | Branches {env_eff.num_branches.mean():.1f} | "
                f"Max Off-Policy {max((episode_staleness(episode, step)[0] for episode in env_eff_pool), default=0)} | "
                f"Error {pool.metrics.has_error.mean():.1%} | Cancelled {pool.metrics.cancelled.mean():.1%} | "
                f"Truncation {env_eff.is_truncated.mean():.1%}"
            )
        get_logger().success("\n\t\t ".join(lines))
