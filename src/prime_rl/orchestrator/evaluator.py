"""Evaluator: eval epochs from trigger to report.

``trigger(step)`` fires the envs due at a step through the ``EvalSource`` and asks
the dispatcher to prefer eval. Every eval result the dispatcher delivers goes
through ``ingest`` into the ``EvalSink``; a completed epoch is logged through the
monitors and its metrics reported. Both the RL orchestrator and ``uv run eval`` use
it, so one epoch report exists."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import verifiers.v1 as vf

from prime_rl import monitors as default_monitors
from prime_rl.orchestrator.annotations import stamp_batch
from prime_rl.orchestrator.envs import EvalEnvs
from prime_rl.orchestrator.eval_sink import EvalSink
from prime_rl.orchestrator.eval_source import EvalSource
from prime_rl.orchestrator.metrics import dispatch_failure_metrics
from prime_rl.orchestrator.types import DispatchResult, EvalBatch
from prime_rl.orchestrator.utils import eval_work
from prime_rl.utils.logger import format_time, get_logger


@dataclass(frozen=True)
class EvaluatorConfig:
    """``max_steps`` is the final step, whose eval fires every env regardless of
    interval. A resumed run re-fires the evals due at ``resume_step`` only when
    ``retrigger_on_resume`` is set. ``upload_epochs`` hands each finished epoch to
    the monitors whole (``log_eval_epoch``), the way ``uv run eval`` publishes to
    the platform."""

    max_steps: int | None = None
    retrigger_on_resume: bool = True
    resume_step: int | None = None
    upload_epochs: bool = False


class Evaluator:
    def __init__(self, config: EvaluatorConfig, *, eval_source: EvalSource, eval_envs: EvalEnvs) -> None:
        self.config = config
        self.eval_source = eval_source
        self.sink = EvalSink(eval_envs=eval_envs)
        self.triggered_steps: set[int] = set()
        self.triggered_at: dict[tuple[str, int], float] = {}
        # Epochs with rollouts still owed, ``(env, step)``; ``changed`` pulses on every finalize.
        self.pending: set[tuple[str, int]] = set()
        self.changed = asyncio.Event()

        self._prefer_eval: Callable[[str], None] = lambda reason: None
        self.monitors: Any = default_monitors

    def bind(self, *, prefer_eval: Callable[[str], None] | None = None, monitors: Any = None) -> None:
        if prefer_eval is not None:
            self._prefer_eval = prefer_eval
        if monitors is not None:
            self.monitors = monitors

    # ── inbound ────────────────────────────────────────────────────────────

    async def trigger(self, step: int, *, force: bool = False) -> list[str]:
        """Fire the envs due at ``step`` and return their names; empty when none is due
        or the step already fired. The final step fires every env."""
        if step in self.triggered_steps:
            return []
        if self.config.resume_step == step and not self.config.retrigger_on_resume:
            return []
        is_final = self.config.max_steps is not None and step >= self.config.max_steps
        fired = self.eval_source.trigger(step, force=force or is_final)
        if not fired:
            return []
        self.triggered_steps.add(step)
        now = time.perf_counter()
        for env_name in fired:
            self.triggered_at[(env_name, step)] = now
            expected = self.sink.batch_size_for(env_name)
            await self.monitors.log_eval_plan(env_name, step, expected)
            if expected > 0:
                self.pending.add((env_name, step))
        queued = sum(
            request.rollouts or 0
            for request in self.eval_source.queue
            if request.step == step and request.env_name in fired
        )
        get_logger().info(f"Starting evals in {', '.join(fired)} at step {step} ({queued} total rollouts)")
        self._prefer_eval(f"eval was triggered at step {step}")
        return fired

    async def ingest(self, item: DispatchResult) -> None:
        """One dispatcher result of an eval epoch."""
        batch = self.sink.ingest(item)
        if batch is not None:
            await self.finalize(batch)

    async def restore(self, episode: vf.Episode) -> None:
        """An episode of the epoch that landed before a resume rejoins it."""
        await self.monitors.log([episode], eval_work(episode).step, "eval", "all")
        await self.ingest(episode)

    def is_pending(self, step: int, envs: list[str]) -> bool:
        return any((env_name, step) in self.pending for env_name in envs)

    # ── reporting ──────────────────────────────────────────────────────────

    async def finalize(self, batch: EvalBatch) -> None:
        """Persist and log one completed eval epoch through the monitors. The epoch
        counts as done only once everything is logged, so a caller waiting on
        ``changed`` can finalize the monitors right after."""
        try:
            await self.report(batch)
        finally:
            self.pending.discard((batch.env_name, batch.step))
            self.changed.set()

    async def report(self, batch: EvalBatch) -> None:
        if not batch.episodes and not batch.failures and not batch.cancelled:
            get_logger().warning(f"Eval @ step={batch.step} env={batch.env_name}: no attempts returned, skipping log")
            return

        # The non-errored subset is logged on epoch completion; the full returned
        # cohort already streamed into ``all`` on arrival.
        if batch.episodes.effective:
            await self.monitors.log(batch.episodes.effective.vf_episodes, batch.step, "eval", "effective")
            await self.monitors.log_annotations(stamp_batch(batch.episodes.effective.vf_episodes, batch.step))
        if self.config.upload_epochs:
            await self.monitors.log_eval_epoch(batch.env_name, batch.step, batch.episodes.vf_episodes)

        episodes = batch.episodes
        effective = episodes.effective
        metrics: dict[str, float] = {}
        for subset, pool in (("all", episodes), ("effective", effective)):
            metrics |= pool.metrics.to_wandb(prefix=f"eval/{batch.env_name}", subset=subset)
        total_attempts = len(episodes) + len(batch.failures) + batch.cancelled
        metrics |= dispatch_failure_metrics(
            batch.failures, prefix=f"eval/{batch.env_name}/all", total_attempts=total_attempts
        )
        if batch.cancelled:
            metrics[f"eval/{batch.env_name}/all/cancelled/count"] = float(batch.cancelled)
            metrics[f"eval/{batch.env_name}/all/cancelled/mean"] = batch.cancelled / total_attempts
        # The policy the epoch measured: the oldest version any of its rollouts started
        # on. Episodes dispatched before the step's weights applied carry an older span.
        versions = {span.start for episode in episodes if (span := eval_work(episode).policy) is not None}
        versions.update(failure.policy_version for failure in batch.failures)
        metrics[f"eval/{batch.env_name}/policy_version"] = float(min(versions, default=batch.step))
        metrics["step"] = float(batch.step)
        await self.monitors.log(metrics, step=batch.step)

        eff, full = effective.metrics, episodes.metrics
        triggered_at = self.triggered_at.pop((batch.env_name, batch.step), None)
        elapsed = (time.perf_counter() - triggered_at) if triggered_at is not None else 0.0
        if batch.cancelled:
            get_logger().warning(
                f"Partially evaluated {batch.env_name} (Step {batch.step}) | "
                f"{format_time(elapsed):>7} | Reward {eff.reward.mean():.4f} | "
                f"Error {full.has_error.mean():.1%} | "
                f"Completed {len(episodes)}/{total_attempts} | Cancelled {batch.cancelled}/{total_attempts}"
            )
            return
        get_logger().success(
            f"Evaluated {batch.env_name} (Step {batch.step}) | "
            f"{format_time(elapsed):>7} | Reward {eff.reward.mean():.4f} | "
            f"Turns {eff.num_turns.mean():.1f} | Branches {eff.num_branches.mean():.1f} | "
            f"Error {full.has_error.mean():.1%} | Truncation {eff.is_truncated.mean():.1%}"
        )

    def status(self) -> str | None:
        parts = []
        for env_name, _step, arrived, expected in sorted(self.sink.batch_progress()):
            parts.append(f"{env_name} {arrived}/{expected} ({arrived / expected:.1%})" if expected else env_name)
        return " | ".join(parts) if parts else None
