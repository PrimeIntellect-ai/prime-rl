"""TrainSink: three-level rollout sink for the training side.

1. ``process_rollout`` — eager per-rollout tokenization (overlaps with
   dispatcher producing more rollouts), then the env algorithm's
   ``finalize_rollout`` (rollout-local scoring + any reference I/O). Errored
   and untrainable rollouts skip this.
2. ``process_group`` — filters errored rollouts, hands the trainable
   survivors to the env algorithm's ``finalize_group`` (advantages +
   per-sample wire stamping), annotates degeneration detections, and drops
   zero-advantage rollouts before they consume batch budget.
3. ``process_batch`` — pops a ``batch_size`` cohort and assembles the
   trainer-bound ``TrainingSample`` list. Returns a ``TrainBatch``.

``add()`` takes one episode (``list[Rollout]``) and returns
``TrainBatch | None``; group accounting counts episodes, never loose traces.
I/O concerns (ship to trainer, save_rollouts, monitor.log) live on the
orchestrator.
"""

from __future__ import annotations

import asyncio
import uuid
from collections import defaultdict

from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.orchestrator.envs import TrainEnvs
from prime_rl.orchestrator.filters import (
    detect_gibberish,
    detect_repetition,
    gibberish_logprob_threshold,
    has_zero_advantage,
)
from prime_rl.orchestrator.metrics import TrainRollouts
from prime_rl.orchestrator.trajectories import trace_to_samples
from prime_rl.orchestrator.types import Rollout, TrainBatch
from prime_rl.transport import TrainingSample
from prime_rl.utils.logger import get_logger

# Warn every N consecutive finalized groups whose survivors were all dropped
# as zero-advantage — the batch isn't filling, usually a task-difficulty
# mismatch (rewards are homogeneous within every group)
ZERO_ADVANTAGE_STALL_WARN_GROUPS = 25


class TrainSink:
    """Three-level train sink. Constructed once, fed via ``add(rollout)``."""

    def __init__(
        self,
        config: OrchestratorConfig,
        *,
        tokenizer,
        train_envs: TrainEnvs,
        mm_token_type_ids_mapping: dict[int, int] | None,
    ) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self.train_envs = train_envs
        self.mm_token_type_ids_mapping = mm_token_type_ids_mapping
        self.batch_size = config.batch_size
        self.count_zero_advantage_in_batch = config.count_zero_advantage_in_batch
        self.gibberish_logprob_threshold = gibberish_logprob_threshold(tokenizer.vocab_size)

        # Observation window for the next shipped batch: rollouts of groups
        # finalized since the last ship (errored + filtered + survivors).
        # In-progress groups stay out until they finalize.
        self.pending_rollouts: TrainRollouts = TrainRollouts()
        # Keyed by the dispatcher's group UUID. ``(env_name, task_idx)``
        # isn't unique — the same task can be re-sampled while an
        # earlier group is still in flight
        self.pending_groups: dict[uuid.UUID, list[Rollout]] = defaultdict(list)
        # Episodes arrived per group — the finalization count (an episode may
        # add several traces to ``pending_groups`` but counts once here).
        self.pending_group_episodes: dict[uuid.UUID, int] = defaultdict(int)
        self.pending_batch: list[Rollout] = []
        # Consecutive finalized groups that contributed nothing to
        # ``pending_batch`` because every survivor was zero-advantage
        self.consecutive_zero_advantage_groups = 0

    def group_size_for(self, env_name: str) -> int:
        return self.train_envs.get(env_name).config.group_size

    def batch_progress(self) -> tuple[int, int]:
        """``(current, target)`` rollouts for the train batch — counts only
        ``pending_batch`` (survivors of finalized groups, queued for the
        trainer), so it's an honest 0→target fill. Partial-group arrivals are
        reported separately by ``buffered_count()``."""
        return len(self.pending_batch), self.batch_size

    def buffered_count(self) -> int:
        """Episodes that have arrived but sit in not-yet-complete groups —
        buffered in the sink ahead of the batch."""
        return sum(self.pending_group_episodes.values())

    def pending_batch_by_env(self) -> dict[str, int]:
        """Per-env breakdown of ``batch_progress()`` (``pending_batch`` only);
        values sum to the aggregate."""
        counts: dict[str, int] = defaultdict(int)
        for r in self.pending_batch:
            counts[r.env_name] += 1
        return dict(counts)

    async def add(self, episode: list[Rollout]) -> TrainBatch | None:
        """Process one episode arrival; finalize the group on the
        ``group_size``-th episode; return a ``TrainBatch`` if the finalization
        pushed (or left) the batch over its threshold. Arrivals into
        still-incomplete groups never ship a batch."""
        group_id = episode[0].group_id
        env_name = episode[0].env_name
        for rollout in episode:
            await self.process_rollout(rollout)
        self.pending_groups[group_id].extend(episode)
        self.pending_group_episodes[group_id] += 1
        if self.pending_group_episodes[group_id] < self.group_size_for(env_name):
            return None
        await self.process_group(group_id)
        # ``pending_batch`` only grows on group finalization, so readiness is
        # only re-checked here — the window of a shipped batch then always
        # contains at least the group that finalized it.
        if len(self.pending_batch) >= self.batch_size:
            return self.process_batch()
        return None

    async def process_rollout(self, rollout: Rollout) -> None:
        """Build training samples from the rollout's Trace (one per branch), walking the
        message graph. Training is renderer-only across all modes (RL/OPD student, SFT teacher),
        so every node already carries its tokens. Errored rollouts are dropped at the group
        level, so skip them here; untrainable traces never become training data."""
        if rollout.has_error or not rollout.agent.trainable:
            return
        samples = await asyncio.to_thread(
            trace_to_samples,
            rollout,
            env_name=rollout.env_name,
            mm_token_type_ids_mapping=self.mm_token_type_ids_mapping,
        )
        rollout.samples = samples or []
        # Arrival phase: rollout-local scoring (raw reward, echo observation
        # weighting, opd/opsd reference logprobs) runs as soon as the rollout is
        # tokenized — before its group is complete.
        await self.train_envs.get(rollout.env_name).algorithm.finalize_rollout(rollout)

    async def process_group(self, group_id: uuid.UUID) -> None:
        """Finalize one GRPO group: drop errored rollouts, assign advantages,
        annotate detections, append the informative survivors to
        ``pending_batch``."""
        group = self.pending_groups.pop(group_id, [])
        self.pending_group_episodes.pop(group_id, None)
        if not group:
            return
        # Window membership follows group finalization, not arrival: a rollout
        # only becomes observable (metrics / persistence) once its whole group
        # is finalized, so a batch's window never claims rollouts of a group
        # that ships later. Dropped groups still land here — they were observed.
        for r in group:
            self.pending_rollouts.append(r)
        env_name = group[0].env_name
        task_idx = group[0].task.data.idx
        survivors = [r for r in group if not r.has_error]
        num_errored = len(group) - len(survivors)

        env = self.train_envs.get(env_name)
        # Untrainable traces carry no samples and must not skew the group baseline.
        survivors = [r for r in survivors if r.agent.trainable]
        if not survivors:
            get_logger().debug(
                f"Finished group | env={env_name} task_idx={task_idx} | "
                f"rollouts={len(group)} (errored={num_errored}) | dropped: no trainable survivors"
            )
            return

        # Advantages + per-sample wire stamping (advantage stream, loss
        # routing) are the algorithm's job (finalize_group); the sink only
        # owns the grouping mechanics.
        await env.algorithm.finalize_group(survivors)

        # The env has a single sampling temperature; fan it out per token
        # (context tokens are masked out, so their temperature is don't-care).
        temperature = env.sampling_args["temperature"]
        for r in survivors:
            for sample in r.samples:
                sample.temperatures = [temperature] * len(sample.token_ids)

        # Degeneration detection is monitor-only (metrics); the zero-advantage
        # check drops — a rollout whose advantage stream is all zero carries no
        # learning signal, unless the env's algorithm trains without one (echo).
        num_zero_advantage = 0
        appended = 0
        for r in survivors:
            r.filter_results = {
                "gibberish": detect_gibberish(r, self.gibberish_logprob_threshold),
                "repetition": detect_repetition(r),
                "zero_advantage": has_zero_advantage(r),
            }
            r.is_filtered = r.filter_results["zero_advantage"] and not env.algorithm.trains_on_zero_advantage
            if r.is_filtered:
                num_zero_advantage += 1
                # Opt-in: a dropped rollout still occupies a batch slot, so the
                # per-step sampling effort stays fixed while the trained-on
                # sample count varies with the zero-advantage rate.
                if not self.count_zero_advantage_in_batch:
                    continue
            self.pending_batch.append(r)
            appended += 1

        if appended:
            self.consecutive_zero_advantage_groups = 0
        else:
            self.consecutive_zero_advantage_groups += 1
            if self.consecutive_zero_advantage_groups % ZERO_ADVANTAGE_STALL_WARN_GROUPS == 0:
                get_logger().warning(
                    f"{self.consecutive_zero_advantage_groups} consecutive groups dropped as zero-advantage — "
                    "the batch isn't filling; check task difficulty (rewards are homogeneous within every group)"
                )

        # Per-group summary. One line per finalized group.
        rewards = [r.reward for r in survivors]
        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
        get_logger().debug(
            f"Finished group | env={env_name} task_idx={task_idx} | "
            f"rollouts={len(group)} (errored={num_errored}, zero_advantage={num_zero_advantage}) | "
            f"reward={avg_reward:.4f}"
        )

    def process_batch(self) -> TrainBatch:
        """Pop a ``batch_size`` cohort off ``pending_batch`` and assemble the
        trainer-bound ``TrainingSample`` list. Overflow stays for the next
        batch."""
        cohort = self.pending_batch[: self.batch_size]
        self.pending_batch = self.pending_batch[self.batch_size :]

        # Samples are pre-built by ``process_rollout``; ``process_group`` already stamped the
        # advantage stream and loss routing on each sample. Zero-advantage rollouts kept in the
        # budget by ``count_zero_advantage_in_batch`` don't ship.
        samples: list[TrainingSample] = [sample for r in cohort if not r.is_filtered for sample in r.samples]

        # ``rollouts`` is the observation window — every rollout of every group finalized since the
        # last ship (errored + filtered + survivors) — while ``samples`` is the shipped cohort's
        # trainable payload. ``rollouts.effective`` / ``rollouts.metrics`` derive the clean subset +
        # metric views on demand. Reset the window only when the batch actually ships (non-empty
        # samples) — an empty batch is dropped unlogged by the orchestrator, so keep accumulating its
        # finalized groups (and any overflow) into the next shipped batch's window.
        rollouts = self.pending_rollouts
        if samples:
            self.pending_rollouts = TrainRollouts()
        return TrainBatch(rollouts=rollouts, samples=samples)
