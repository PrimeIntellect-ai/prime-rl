"""Shipper: the trainer-facing end of the pipeline.

Takes each cut ``Batch``, holds it until inference serves a recent enough policy,
packs and sends it, advances the step, checkpoints, and reports the step's metrics.
It owns ``Progress`` (the step every other component reads), the batch transport,
the checkpoint manager, and the lag gate: dispatch pauses while the batch being
collected runs more than ``max_off_policy_steps`` versions ahead of the policy
inference serves. After the final batch it asks the pipeline to drain."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any

from prime_rl import monitors as default_monitors
from prime_rl.configs.orchestrator import CheckpointConfig, OrchestratorConfig
from prime_rl.orchestrator.algo.routing import is_trainable
from prime_rl.orchestrator.annotations import stamp_batch
from prime_rl.orchestrator.ckpt import CheckpointManager
from prime_rl.orchestrator.metrics import Episodes
from prime_rl.orchestrator.packing import BatchPacker
from prime_rl.orchestrator.train_source import TrainSource
from prime_rl.orchestrator.types import Batch, Progress, cancel_reason, staleness, work_of
from prime_rl.orchestrator.utils import trim_process_memory
from prime_rl.transports.batch import setup_batch_sender
from prime_rl.utils.heartbeat import Heartbeat
from prime_rl.utils.logger import format_time, get_logger


class Shipper:
    def __init__(self, config: OrchestratorConfig, *, train_source: TrainSource, resume_step: int | None) -> None:
        self.max_steps = config.max_steps
        self.max_off_policy_steps = config.max_off_policy_steps
        self.train_source = train_source
        self.ckpt_config = config.ckpt
        self.ckpt = CheckpointManager(config.output_dir, config.ckpt or CheckpointConfig())

        self.progress = Progress()
        if resume_step is not None:
            resume = config.resume
            path = resume.dir / "orchestrator" if resume is not None and resume.dir is not None else None
            loaded = self.ckpt.load(resume_step, path=path)
            if loaded is not None:
                self.progress, state = loaded
                train_source.load_state_dict(state)
                get_logger().info(f"Resumed curriculum state for {', '.join(state['envs'])}")
            # The checkpoint finished ``resume_step``; the step derives from it (not the
            # loaded counter) so it stays coordinated with the trainer even when
            # ``ckpt.skip_progress`` leaves the counter unrestored.
            self.progress.step = resume_step + 1

        self.packer = BatchPacker(config)
        get_logger().info(f"Initializing micro batch sender ({config.rollout_transport})")
        self.sender = setup_batch_sender(
            config.output_dir, config.num_train_workers, self.progress.step, config.rollout_transport
        )
        self.heart = Heartbeat(config.heartbeat.url) if config.heartbeat is not None else None

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

    def step(self) -> int:
        """The batch being collected, 1-indexed and advanced right after a ship."""
        return self.progress.step

    def start_clock(self) -> None:
        """Anchor the step clock so the first step measures startup to first batch."""
        self.last_batch_at = time.perf_counter()

    def close(self) -> None:
        self.sender.close()

    # ── inbound ────────────────────────────────────────────────────────────

    async def on_version(self, _step: int) -> None:
        """Re-check the lag gate after inference applied a new policy."""
        self.update_gate()

    async def on_batch(self, batch: Batch) -> None:
        """Ship one batch; a batch arriving while draining is dropped so nothing
        ships past ``max_steps``."""
        if self.draining.is_set():
            return
        step = self.progress.step
        now = time.perf_counter()
        step_time = (now - self.last_batch_at) if self.last_batch_at is not None else 0.0
        self.last_batch_at = now

        # A resume can start past the end (checkpoint written at the final step, or a
        # lowered ``max_steps``): never ship beyond the budget.
        if self.max_steps is not None and step > self.max_steps:
            await self.start_draining(f"Step {step} exceeds max_steps={self.max_steps}")
            return
        episodes = Episodes(batch.groups)
        shipped = episodes.sampled
        n_trainable = sum(is_trainable(trace) for trace in shipped.traces)
        if n_trainable / shipped.num_traces <= 0.1:
            get_logger().warning(
                f"Only {n_trainable}/{shipped.num_traces} shipped traces are trainable "
                f"({n_trainable / shipped.num_traces:.1%}) — consider reviewing task difficulty"
            )

        # Ship batch ``step`` only once inference serves v{step-1-max_off_policy_steps}:
        # fast envs fill batches from buffered rollouts and would race arbitrarily far
        # ahead of the trainer otherwise. Always satisfiable: the trainer broadcasts every version.
        hold_start = time.perf_counter()
        await self.wait_for_version(step - 1 - self.max_off_policy_steps, f"to ship batch {step}")
        self.wait_for_policy_time += time.perf_counter() - hold_start

        # The shipped subset is logged at ship time as annotation records against each
        # trace's arrival record — never a second episode copy.
        await self.monitors.log(shipped.vf_episodes, step, "train", "effective")
        await self.monitors.log_annotations(stamp_batch(shipped.vf_episodes, step))

        pack_start = time.perf_counter()
        micro_batch_grid = await asyncio.to_thread(self.packer.pack, batch.samples)
        pack_time = time.perf_counter() - pack_start
        await self.sender.send(micro_batch_grid)
        self.progress.step += 1
        self.update_gate()
        save_ckpt_time = self.maybe_save_ckpt(step)
        trim_process_memory()

        num_tasks = len({group.id for group in batch.groups})
        await self.monitors.log(
            self.step_metrics(batch, step, step_time=step_time, pack_time=pack_time, save_ckpt_time=save_ckpt_time),
            step=step,
        )
        self.warn_discards(episodes, step_time)
        self.wait_for_policy_time = 0.0
        if self.heart is not None:
            self.heart.beat()
        self.progress.total_tokens += episodes.num_total_tokens
        self.progress.total_samples += episodes.num_traces
        self.progress.total_problems += num_tasks
        self.log_train_batch(episodes, step=step, step_time=step_time)

        if self.max_steps is not None and step >= self.max_steps:
            await self.wait_for_version(step, "before shutdown")
            # Drain right after the final batch: waiting for another to fill would burn
            # inference on data that can never train.
            await self.start_draining("Shipped the final batch")
        trim_process_memory()

    # ── internals ──────────────────────────────────────────────────────────

    async def wait_for_version(self, version: int, reason: str) -> None:
        if self._version() >= version:
            return
        if self._wait_for_version is None:
            raise RuntimeError("Shipper.wait_for_version is not bound")
        await self._wait_for_version(version, reason=reason)

    def update_gate(self) -> None:
        """Pause dispatch while the batch being collected runs more than
        ``max_off_policy_steps`` ahead of the policy inference serves. Steps are
        1-indexed while versions are 0-indexed, so the shipped-batch count is ``step - 1``."""
        version = self._version()
        lead = (self.progress.step - 1) - version
        if lead > self.max_off_policy_steps:
            if self.gate_open:
                get_logger().info(
                    f"Pausing dispatcher until inference applies policy "
                    f"v{self.progress.step - 1 - self.max_off_policy_steps} (currently v{version})"
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
        if self.max_steps is not None and step >= self.max_steps:
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
        self.ckpt.save(step, self.progress, self.train_source.state_dict())

    def save_final(self) -> None:
        """``progress.step`` points at the next (unshipped) step; checkpoint the last
        finished one. No-op before the first ship or without a ckpt block."""
        if self.ckpt_config is None or self.progress.step <= 1:
            return
        self.progress.step -= 1
        get_logger().info(f"Saving final checkpoint at step {self.progress.step}")
        self.save_ckpt(self.progress.step)

    # ── reporting ──────────────────────────────────────────────────────────

    def step_metrics(
        self, batch: Batch, step: int, *, step_time: float, pack_time: float, save_ckpt_time: float
    ) -> dict[str, float]:
        """Episode metrics over the {agg,<env>} × {all,effective} matrix plus progress,
        timing, staleness and env-share accounting for one shipped step."""
        episodes = Episodes(batch.groups)
        shipped = episodes.sampled
        metrics: dict[str, float] = {}
        for subset, pool in (("all", episodes), ("effective", shipped)):
            metrics |= pool.train_metrics("train/agg", subset=subset)
            for env_name, env_pool in pool.by_env().items():
                metrics |= env_pool.train_metrics(f"train/{env_name}", subset=subset)

        metrics |= {
            "progress/tokens": episodes.num_total_tokens,
            "progress/input_tokens": sum(trace.num_input_tokens for trace in shipped.traces),
            "progress/output_tokens": sum(trace.num_output_tokens for trace in shipped.traces),
            "progress/rollouts": episodes.num_traces,
            "progress/tasks": len({group.id for group in batch.groups}),
            "progress/total_tokens": self.progress.total_tokens,
            "progress/total_rollouts": self.progress.total_samples,
            "progress/total_tasks": self.progress.total_problems,
            "time/step": step_time,
            "time/pack": pack_time,
            "time/save_ckpt": save_ckpt_time,
            "time/wait_for_policy": self.wait_for_policy_time,
            "step": step,
        }
        # Staleness of the shipped cohort, decomposed into its in-flight share (weight
        # updates during generation) and the time spent queued between completion and ship.
        totals, in_flight = [], []
        for episode in shipped:
            total = staleness(episode, step)
            span = work_of(episode).policy
            totals.append(total)
            in_flight.append(min(total, span.drift) if span is not None else 0)
        if totals:
            in_queue = [total - flight for total, flight in zip(totals, in_flight, strict=True)]
            for name, values in (("", totals), ("/in_flight", in_flight), ("/in_queue", in_queue)):
                metrics[f"off_policy{name}/mean"] = sum(values) / len(values)
                metrics[f"off_policy{name}/max"] = float(max(values))
        metrics["off_policy/dropped"] = float(
            sum(len(episode.traces) for episode in episodes if cancel_reason(episode) == "stale")
        )
        for env_name, env_pool in episodes.by_env().items():
            metrics[f"batch/{env_name}"] = env_pool.num_traces / episodes.num_traces
        metrics |= self.train_source.metrics()
        return metrics

    def warn_discards(self, episodes: Episodes, step_time: float) -> None:
        active_step_time = max(step_time - self.wait_for_policy_time, 0.0)
        if step_time > 0 and self.wait_for_policy_time >= active_step_time:
            get_logger().warning(
                f"Orchestrator waited {format_time(self.wait_for_policy_time)} for policy updates, at least as long "
                f"as its {format_time(active_step_time)} active step time. Train-inference compute is imbalanced; "
                "add more trainer nodes."
            )
        attempts = len(episodes)
        discarded = attempts - len(episodes.sampled)
        stale = int(sum(episodes.cancelled.values))
        errored = int(sum(episodes.has_error.values))
        if attempts and discarded / attempts > 0.5:
            get_logger().warning(
                f"Discarded {discarded}/{attempts} episodes ({discarded / attempts:.1%}): "
                f"stale={stale}, errored={errored}, no_signal={discarded - stale - errored}. "
                "Review max_off_policy_steps, episode errors, and reward signal."
            )

    def log_train_batch(self, episodes: Episodes, *, step: int, step_time: float) -> None:
        """Per-step ``Step …`` success line. Multi-env runs append an indented ``╰─`` line per env.
        Quality metrics (Reward, Trainable, Turns, Branches, Max Off-Policy, Truncation) are
        over exactly the traces shipped this step; ``Error``, ``Cancelled`` and ``Ratio``
        describe the step's full arrival window."""

        def line(window: Episodes, head: str) -> str:
            shipped = window.sampled
            n_trainable = sum(is_trainable(trace) for trace in shipped.traces)
            n_shipped = shipped.num_traces
            max_staleness = max((staleness(episode, step) for episode in shipped), default=0)
            return (
                f"{head} | Reward {shipped.reward.mean():.4f} | "
                f"Trainable {n_trainable}/{n_shipped} ({(n_trainable / n_shipped) if n_shipped else 0.0:.1%}) | "
                f"Turns {shipped.num_turns.mean():.1f} | Branches {shipped.num_branches.mean():.1f} | "
                f"Max Off-Policy {max_staleness} | Error {window.has_error.mean():.1%} | "
                f"Cancelled {window.cancelled.mean():.1%} | Truncation {shipped.is_truncated.mean():.1%}"
            )

        lines = [line(episodes, f"Step {step} | {format_time(step_time):>7}")]
        by_env = episodes.by_env()
        if len(by_env) > 1:
            width = max(len(name) for name in by_env)
            for env_name, pool in by_env.items():
                ratio = pool.num_traces / episodes.num_traces if episodes.num_traces else 0.0
                lines.append(line(pool, f"╰─ {env_name:<{width}} | Ratio {ratio:.1%}"))
        get_logger().success("\n\t\t ".join(lines))
