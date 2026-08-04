"""TrainSink: three-level rollout sink for the training side.

1. ``process_rollout`` — eager per-rollout tokenization (overlaps with
   dispatcher producing more rollouts), the degeneracy detectors, then the env
   algorithm's ``finalize_rollout`` (rollout-local scoring + any reference
   I/O). Errored and untrainable rollouts skip this.
2. ``process_group`` — filters errored rollouts, hands the episodes narrowed
   to their trainable survivors to the env algorithm's ``finalize_group``
   (advantages + per-sample wire stamping), then applies the drop policy —
   the one place a rollout is kept out of training.
3. ``process_batch`` — pops a cohort and flattens it into the trainer-bound
   ``TrainingSample`` list. Returns a ``TrainBatch``.

``add()`` takes one ``Episode`` and returns
``TrainBatch | None``; group accounting counts episodes, never loose traces.
I/O concerns (ship to trainer, save_episodes, monitor.log) live on the
orchestrator.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict

from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.orchestrator.detectors import Detector, detect, drop_reasons
from prime_rl.orchestrator.envs import TrainEnvs
from prime_rl.orchestrator.metrics import TrainRollouts
from prime_rl.orchestrator.trajectories import trace_to_samples
from prime_rl.orchestrator.types import (
    Episode,
    TrainBatch,
    TrainRollout,
    env_name_of,
    group_id_of,
    group_rollouts,
    narrow,
    rollouts_of,
)
from prime_rl.transport import TrainingSample
from prime_rl.utils.logger import get_logger


def payload_tokens(rollout: TrainRollout) -> int:
    """Token cost of the rollout's trainer-bound payload — the samples built by
    ``process_rollout``. This is what actually ships: forked traces can drop
    branches with no trainable tokens, so ``Trace.num_total_tokens`` (which sums
    over all branches) may overcount. For linear traces the two agree.

    Zero-payload rollouts (no trainable samples at all) fall back to the trace
    total so they still advance token batching — a degenerate all-zero-payload
    stream then ships empty batches and trips the orchestrator's
    consecutive-empty-batch abort instead of stalling the readiness check."""
    return sum(len(sample.token_ids) for sample in rollout.samples) or rollout.num_total_tokens


class TrainSink:
    """Three-level train sink. Constructed once, fed via ``add(rollout)``."""

    def __init__(
        self,
        config: OrchestratorConfig,
        *,
        tokenizer,
        train_envs: TrainEnvs,
        mm_token_type_ids_mapping: dict[int, int] | None,
        batch_size: int | None,
        token_batch_size: int | None,
        detectors: list[Detector],
        drop_detections: list[str],
        drop_zero_advantage: bool,
    ) -> None:
        assert (batch_size is None) != (token_batch_size is None), (
            "Exactly one of batch_size / token_batch_size must be set"
        )
        self.config = config
        self.tokenizer = tokenizer
        self.train_envs = train_envs
        self.mm_token_type_ids_mapping = mm_token_type_ids_mapping
        self.batch_size = batch_size
        self.token_batch_size = token_batch_size
        self.detectors = detectors
        self.drop_detections = drop_detections
        self.drop_zero_advantage = drop_zero_advantage

        # Observation window for the next shipped batch: rollouts of groups
        # finalized since the last ship (errored + filtered + survivors).
        # In-progress groups stay out until they finalize.
        self.pending_rollouts: TrainRollouts = TrainRollouts()
        # Keyed by the dispatcher's group UUID. ``(env_name, task_idx)``
        # isn't unique — the same task can be re-sampled while an
        # earlier group is still in flight
        self.pending_groups: dict[str, list[Episode]] = defaultdict(list)
        # Episodes arrived per group — the finalization count (an episode may
        # add several traces to ``pending_groups`` but counts once here).
        self.pending_group_episodes: dict[str, int] = defaultdict(int)
        self.pending_batch: list[TrainRollout] = []
        # Running payload-token total of ``pending_batch`` (token-batched
        # runs), kept in sync on append/pop so the readiness check never
        # re-sums per arrival.
        self.pending_tokens: int = 0

        # Reset by the orchestrator after each ship via ``reset_pre_filter_stats``
        self.pre_filter_seen = 0
        self.pre_filter_dropped = 0
        self.pre_filter_dropped_by_name: dict[str, int] = {}

    def group_size_for(self, env_name: str) -> int:
        return self.train_envs.get(env_name).config.group_size

    def batch_progress(self) -> tuple[int, int, str]:
        """``(current, target, unit)`` for the train batch — counts only
        ``pending_batch`` (survivors of finalized groups, queued for the
        trainer), so it's an honest 0→target fill. Partial-group arrivals are
        reported separately by ``buffered_count()``."""
        if self.batch_size is not None:
            return len(self.pending_batch), self.batch_size, "rollouts"
        assert self.token_batch_size is not None
        return self.pending_tokens, self.token_batch_size, "tokens"

    def buffered_count(self) -> int:
        """Episodes that have arrived but sit in not-yet-complete groups
        (non-group-scoring envs) — buffered in the sink ahead of the batch."""
        return sum(
            self.pending_group_episodes.get(group_id, 0)
            for group_id, episodes in self.pending_groups.items()
            if episodes and not self.train_envs.get(episodes[0].env_name).requires_group_scoring
        )

    def pending_batch_by_env(self) -> dict[str, int]:
        """Per-env breakdown of ``batch_progress()`` (``pending_batch`` only);
        values sum to the aggregate."""
        counts: dict[str, int] = defaultdict(int)
        for r in self.pending_batch:
            counts[r.env_name] += 1
        return dict(counts)

    async def add(self, episode: Episode) -> TrainBatch | None:
        """Process one episode arrival; finalize the group on the
        ``group_size``-th episode; return a ``TrainBatch`` if the finalization
        pushed (or left) the batch over its threshold. Arrivals into
        still-incomplete groups never ship a batch. A failed episode brings no
        rollouts, but still counts toward the group so finalization triggers."""
        group_id = group_id_of(episode)
        env_name = env_name_of(episode)
        for rollout in rollouts_of(episode):
            await self.process_rollout(rollout)
        self.pending_groups[group_id].append(episode)
        self.pending_group_episodes[group_id] += 1
        if self.pending_group_episodes[group_id] < self.group_size_for(env_name):
            return None
        await self.process_group(group_id)
        # ``pending_batch`` only grows on group finalization, so readiness is
        # only re-checked here — the window of a shipped batch then always
        # contains at least the group that finalized it.
        ready = (
            len(self.pending_batch) >= self.batch_size
            if self.batch_size is not None
            else self.pending_tokens >= (self.token_batch_size or 0)
        )
        if ready:
            return self.process_batch()
        return None

    async def process_rollout(self, rollout: TrainRollout) -> None:
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
        detect(self.detectors, rollout)
        # Arrival phase: rollout-local scoring (raw reward, echo observation
        # weighting, opd/opsd reference logprobs) runs as soon as the rollout is
        # tokenized — before its group is complete.
        await self.train_envs.get(rollout.env_name).algorithm.finalize_rollout(rollout)

    async def process_group(self, group_id: str) -> None:
        """Finalize one GRPO group: drop errored rollouts (the whole group
        when ``requires_group_scoring`` and any failed), assign advantages,
        apply the drop policy, append what survives to ``pending_batch``."""
        episodes = self.pending_groups.pop(group_id, [])
        self.pending_group_episodes.pop(group_id, None)
        if not episodes:
            return
        # Read the group's facts off an episode, not a trace: every episode in it may have
        # produced none (a whole group cancelled off-policy).
        env_name = episodes[0].env_name
        group = [t for e in episodes for t in rollouts_of(e)]
        # Window membership follows group finalization, not arrival: a rollout
        # only becomes observable (metrics / persistence) once its whole group
        # is finalized, so a batch's window never claims rollouts of a group
        # that ships later. Dropped groups still land here — they were observed.
        for episode in episodes:
            self.pending_rollouts.append(episode)
        task_idx = group[0].task.data.idx if group else -1
        num_errored = sum(r.has_error for r in group)

        # Group-scoring envs: any failure makes survivors' rewards unsafe
        # (computed relative to the missing ones)
        env = self.train_envs.get(env_name)
        if num_errored > 0 and env.requires_group_scoring:
            get_logger().debug(
                f"Finished group | env={env_name} task_idx={task_idx} | "
                f"rollouts={len(group)} (errored={num_errored}) | dropped: group-scored partial"
            )
            return
        # Untrainable traces carry no samples and must not skew the group baseline. The cohort
        # stays episodes so the algorithm can still see which attempts shared one.
        cohort = [n for e in episodes if (n := narrow(e, lambda r: not r.has_error and r.agent.trainable))]
        survivors = group_rollouts(cohort)
        if not survivors:
            get_logger().debug(
                f"Finished group | env={env_name} task_idx={task_idx} | "
                f"rollouts={len(group)} (errored={num_errored}) | dropped: no trainable survivors"
            )
            return

        # Advantages + per-sample wire stamping (advantage stream, loss
        # routing) are the algorithm's job (finalize_group); the sink only
        # owns the grouping mechanics.
        await env.algorithm.finalize_group(cohort)

        # The env has a single sampling temperature; fan it out per token
        # (context tokens are masked out, so their temperature is don't-care).
        temperature = env.sampling_args["temperature"]
        for r in survivors:
            for sample in r.samples:
                sample.temperatures = [temperature] * len(sample.token_ids)

        # Credit is assigned, so the drop decision can be made now: every detection was already
        # measured at tokenization, and zero credit is only knowable after the group scored.
        dropped_by_reason: dict[str, int] = {}
        num_dropped = 0
        for r in survivors:
            self.pre_filter_seen += 1
            reasons = drop_reasons(
                r, drop_detections=self.drop_detections, drop_zero_advantage=self.drop_zero_advantage
            )
            r.is_filtered = bool(reasons)
            if reasons:
                self.pre_filter_dropped += 1
                num_dropped += 1
                for reason in reasons:
                    self.pre_filter_dropped_by_name[reason] = self.pre_filter_dropped_by_name.get(reason, 0) + 1
                    dropped_by_reason[reason] = dropped_by_reason.get(reason, 0) + 1
                continue
            self.pending_batch.append(r)
            if self.token_batch_size is not None:
                self.pending_tokens += payload_tokens(r)

        rewards = [r.reward for r in survivors]
        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
        drop_str = ", ".join(f"{n}={c}" for n, c in dropped_by_reason.items()) if dropped_by_reason else "—"
        get_logger().debug(
            f"Finished group | env={env_name} task_idx={task_idx} | "
            f"rollouts={len(group)} (errored={num_errored}, dropped={num_dropped}) | "
            f"reward={avg_reward:.4f} | dropped: {drop_str}"
        )

    def process_batch(self) -> TrainBatch:
        """Pop a cohort off ``pending_batch`` (by rollout count when ``batch_size`` is set, by
        token count when ``token_batch_size`` is set) and flatten it into the trainer-bound
        ``TrainingSample`` list. Overflow stays for the next batch."""
        if self.batch_size is not None:
            cohort = self.pending_batch[: self.batch_size]
            self.pending_batch = self.pending_batch[self.batch_size :]
        else:
            assert self.token_batch_size is not None
            cut = 0
            running = 0
            for i, r in enumerate(self.pending_batch):
                running += payload_tokens(r)
                cut = i + 1
                if running >= self.token_batch_size:
                    break
            cohort = self.pending_batch[:cut]
            self.pending_batch = self.pending_batch[cut:]
            self.pending_tokens -= running

        # Samples are pre-built by ``process_rollout``; ``process_group`` already stamped the
        # advantage stream and loss routing on each sample, and decided what ships. Past this line
        # the batch is samples — the episode it came from has served its purpose.
        samples: list[TrainingSample] = [sample for r in cohort for sample in r.samples]

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

    def reset_pre_filter_stats(self) -> None:
        self.pre_filter_seen = 0
        self.pre_filter_dropped = 0
        self.pre_filter_dropped_by_name.clear()
