"""Evaluator: eval epochs from trigger to report.

``trigger(step)`` fires the envs due at a step through the ``EvalSource`` and asks
the dispatcher to prefer eval. Every eval episode the dispatcher delivers goes
through ``ingest``; once an env's epoch holds every attempt it owed, the epoch is
logged through the monitors and its metrics reported. Both the RL orchestrator and
``uv run eval`` use it, so one epoch report exists."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Any

import verifiers.v1 as vf

from prime_rl import monitors as default_monitors
from prime_rl.orchestrator.annotations import stamp_batch
from prime_rl.orchestrator.envs import EvalEnvs
from prime_rl.orchestrator.eval_source import EvalSource
from prime_rl.orchestrator.metrics import Episodes
from prime_rl.orchestrator.types import Epoch, Group, env_of, group_of, work_of
from prime_rl.utils.logger import format_time, get_logger


class Evaluator:
    def __init__(
        self,
        *,
        eval_source: EvalSource,
        eval_envs: EvalEnvs,
        max_steps: int | None = None,
        retrigger_on_resume: bool = True,
        resume_step: int | None = None,
        upload_epochs: bool = False,
    ) -> None:
        """``max_steps`` is the final step, whose eval fires every env regardless of
        interval. A resumed run re-fires the evals due at ``resume_step`` only when
        ``retrigger_on_resume`` is set. ``upload_epochs`` hands each finished epoch to
        the monitors whole (``log_eval_epoch``), the way ``uv run eval`` publishes to
        the platform."""
        self.eval_source = eval_source
        self.eval_envs = eval_envs
        self.max_steps = max_steps
        self.retrigger_on_resume = retrigger_on_resume
        self.resume_step = resume_step
        self.upload_epochs = upload_epochs

        self.triggered_steps: set[int] = set()
        self.triggered_at: dict[tuple[str, int], float] = {}
        self.epochs: dict[tuple[str, int], list[vf.Episode]] = {}
        """Episodes landed per open epoch ``(env, step)``; an epoch closes when every
        attempt it owed is in. ``changed`` pulses on every close."""
        self.changed = asyncio.Event()

        self._prefer_eval: Callable[[str], None] = lambda reason: None
        self.monitors: Any = default_monitors

    def bind(self, *, prefer_eval: Callable[[str], None] | None = None, monitors: Any = None) -> None:
        if prefer_eval is not None:
            self._prefer_eval = prefer_eval
        if monitors is not None:
            self.monitors = monitors

    def expected(self, env_name: str) -> int:
        """Every attempt of an env's epoch: its examples times its group size."""
        env = self.eval_envs.get(env_name)
        return len(env.examples) * env.config.group_size

    # ── inbound ────────────────────────────────────────────────────────────

    async def trigger(self, step: int, *, force: bool = False) -> list[str]:
        """Fire the envs due at ``step`` and return their names; empty when none is due
        or the step already fired. The final step fires every env."""
        if step in self.triggered_steps:
            return []
        if self.resume_step == step and not self.retrigger_on_resume:
            return []
        is_final = self.max_steps is not None and step >= self.max_steps
        fired = self.eval_source.trigger(step, force=force or is_final)
        if not fired:
            return []
        self.triggered_steps.add(step)
        now = time.perf_counter()
        for env_name in fired:
            self.triggered_at[(env_name, step)] = now
            expected = self.expected(env_name)
            await self.monitors.log_eval_plan(env_name, step, expected)
            if expected > 0:
                self.epochs.setdefault((env_name, step), [])
        queued = sum(
            request.rollouts or 0
            for request in self.eval_source.queue
            if request.step == step and request.env_name in fired
        )
        get_logger().info(f"Starting evals in {', '.join(fired)} at step {step} ({queued} total rollouts)")
        self._prefer_eval(f"eval was triggered at step {step}")
        return fired

    async def ingest(self, episode: vf.Episode) -> None:
        key = (env_of(episode), work_of(episode).step)
        episodes = self.epochs.setdefault(key, [])
        episodes.append(episode)
        if len(episodes) >= self.expected(key[0]):
            del self.epochs[key]
            groups: dict[str, list[vf.Episode]] = {}
            for episode in episodes:
                groups.setdefault(group_of(episode), []).append(episode)
            epoch = Epoch(key[0], key[1], [Group(key[0], gid, key[1], members) for gid, members in groups.items()])
            # The epoch counts as done only once everything is logged, so a caller
            # waiting on ``changed`` can finalize the monitors right after.
            try:
                await self.report(epoch)
            finally:
                self.changed.set()

    async def restore(self, episode: vf.Episode) -> None:
        """An episode of the epoch that landed before a resume rejoins it."""
        await self.monitors.log([episode], work_of(episode).step, "eval", "all")
        await self.ingest(episode)

    def is_pending(self, step: int, envs: list[str]) -> bool:
        return any((env_name, step) in self.epochs for env_name in envs)

    # ── reporting ──────────────────────────────────────────────────────────

    async def report(self, epoch: Epoch) -> None:
        episodes = Episodes(epoch.groups)
        if not episodes:
            get_logger().warning(f"Eval @ step={epoch.step} env={epoch.env}: no attempts returned, skipping log")
            return
        clean = episodes.clean

        # The clean subset is logged on epoch completion; the full returned cohort
        # already streamed into ``all`` on arrival.
        if clean:
            await self.monitors.log(clean.vf_episodes, epoch.step, "eval", "effective")
            await self.monitors.log_annotations(stamp_batch(clean.vf_episodes, epoch.step))
        if self.upload_epochs:
            await self.monitors.log_eval_epoch(epoch.env, epoch.step, episodes.vf_episodes)

        k = self.eval_envs.get(epoch.env).config.group_size
        metrics: dict[str, float] = {}
        for subset, pool in (("all", episodes), ("effective", clean)):
            metrics |= pool.eval_metrics(f"eval/{epoch.env}", subset=subset, k=k)
        # The policy the epoch measured: the oldest version any of its attempts started
        # on. Episodes dispatched before the step's weights applied carry an older span.
        versions = [span.start for episode in episodes if (span := work_of(episode).policy) is not None]
        metrics[f"eval/{epoch.env}/policy_version"] = float(min(versions, default=epoch.step))
        metrics["step"] = float(epoch.step)
        await self.monitors.log(metrics, step=epoch.step)

        triggered_at = self.triggered_at.pop((epoch.env, epoch.step), None)
        elapsed = (time.perf_counter() - triggered_at) if triggered_at is not None else 0.0
        cancelled = int(sum(episodes.cancelled.values))
        head = f"{epoch.env} (Step {epoch.step}) | {format_time(elapsed):>7} | Reward {clean.reward.mean():.4f}"
        if cancelled:
            get_logger().warning(
                f"Partially evaluated {head} | Error {episodes.has_error.mean():.1%} | "
                f"Completed {len(episodes) - cancelled}/{len(episodes)} | Cancelled {cancelled}/{len(episodes)}"
            )
            return
        get_logger().success(
            f"Evaluated {head} | Turns {clean.num_turns.mean():.1f} | Branches {clean.num_branches.mean():.1f} | "
            f"Error {episodes.has_error.mean():.1%} | Truncation {clean.is_truncated.mean():.1%}"
        )

    def status(self) -> str | None:
        parts = []
        for (env_name, _step), episodes in sorted(self.epochs.items()):
            expected = self.expected(env_name)
            parts.append(f"{env_name} {len(episodes)}/{expected} ({len(episodes) / expected:.1%})")
        return " | ".join(parts) if parts else None
