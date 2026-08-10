"""TrainSink: three-level rollout sink for the training side.

1. ``process_rollout`` — eager per-rollout tokenization (overlaps with
   dispatcher producing more rollouts), then the env algorithm's
   ``finalize_rollout`` (rollout-local scoring + any reference I/O). Errored
   and untrainable rollouts skip this.
2. ``process_group`` — filters errored rollouts, hands the trainable
   survivors to the env algorithm's ``finalize_group`` (advantages +
   per-sample wire stamping), runs the pre-batch filter pass.
3. ``process_batch`` — applies post-batch filter annotations and assembles
   the trainer-bound ``TrainingSample`` list. Returns a ``TrainBatch``.

``add()`` takes one v1 ``Episode`` and returns
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
from prime_rl.orchestrator.filters import RolloutFilter, apply_filters
from prime_rl.orchestrator.metrics import TrainEpisodes
from prime_rl.orchestrator.trajectories import trace_to_samples
from prime_rl.orchestrator.types import Episode, Rollout, TrainBatch
from prime_rl.transport import TrainingSample
from prime_rl.utils.logger import get_logger


def payload_tokens(rollout: Rollout) -> int:
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
    """Three-level train sink. Constructed once, fed via ``add(episode)``."""

    def __init__(
        self,
        config: OrchestratorConfig,
        *,
        tokenizer,
        train_envs: TrainEnvs,
        mm_token_type_ids_mapping: dict[int, int] | None,
        batch_size: int | None,
        token_batch_size: int | None,
        pre_filters: list[RolloutFilter],
        post_filters: list[RolloutFilter],
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
        self.pre_filters = pre_filters
        self.post_filters = post_filters

        # Keyed by the dispatcher's group UUID. ``(env_name, task_idx)``
        # isn't unique — the same task can be re-sampled while an
        # earlier group is still in flight
        self.pending_groups: dict[uuid.UUID, list[Episode]] = defaultdict(list)
        self.ready_groups: list[list[Rollout]] = []
        # Observation window for the next successful ship. In-progress groups
        # stay out until they finalize.
        self.pending_episodes: list[Episode] = []
        self.pending_selected: int = 0
        # Running payload-token total of ready groups for token-batched runs.
        self.pending_tokens: int = 0

        # Reset by the orchestrator after each ship via ``reset_pre_filter_stats``
        self.pre_filter_seen = 0
        self.pre_filter_dropped = 0
        self.pre_filter_dropped_by_name: dict[str, int] = {}

    def group_size_for(self, env_name: str) -> int:
        return self.train_envs.get(env_name).config.group_size

    def batch_progress(self) -> tuple[int, int, str]:
        """``(current, target, unit)`` for the train batch — counts only
        survivors of finalized groups queued for the trainer, so it's an
        honest 0→target fill. Partial-group arrivals are
        reported separately by ``buffered_count()``."""
        if self.batch_size is not None:
            return self.pending_selected, self.batch_size, "rollouts"
        assert self.token_batch_size is not None
        return self.pending_tokens, self.token_batch_size, "tokens"

    def buffered_count(self) -> int:
        """Episodes buffered in not-yet-complete groups ahead of the batch."""
        return sum(len(episodes) for episodes in self.pending_groups.values())

    def pending_batch_by_env(self) -> dict[str, int]:
        """Per-env breakdown of ``batch_progress()`` (selected rollouts only);
        values sum to the aggregate."""
        counts: dict[str, int] = defaultdict(int)
        for group in self.ready_groups:
            for rollout in group:
                counts[rollout.env_name] += 1
        return dict(counts)

    async def add(self, episode: Episode) -> TrainBatch | None:
        """Process one episode arrival; finalize the group on the
        ``group_size``-th episode; return a ``TrainBatch`` if the finalization
        pushed (or left) the batch over its threshold. Arrivals into
        still-incomplete groups never ship a batch."""
        group_id = episode.group_id
        env_name = episode.env_name
        if episode.ok:
            for trace in episode.traces:
                await self.process_rollout(trace)
        self.pending_groups[group_id].append(episode)
        if len(self.pending_groups[group_id]) < self.group_size_for(env_name):
            return None
        await self.process_group(group_id)
        # The ready cohort only grows on group finalization, so readiness is
        # only re-checked here — the window of a shipped batch then always
        # contains at least the group that finalized it.
        ready = (
            self.pending_selected >= self.batch_size
            if self.batch_size is not None
            else self.pending_tokens >= (self.token_batch_size or 0)
        )
        if ready:
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
        """Finalize one GRPO group: drop errored rollouts, assign advantages, run
        pre-batch filters, and append the completed group to the ready cohort."""
        episodes = self.pending_groups.pop(group_id, [])
        if not episodes:
            return
        group = [trace for episode in episodes for trace in episode.traces]
        # Window membership follows group finalization, not arrival: a rollout
        # only becomes observable (metrics / persistence) once its whole group
        # is finalized, so a batch's window never claims rollouts of a group
        # that ships later. Dropped groups still land here — they were observed.
        self.pending_episodes.extend(episodes)
        env_name = episodes[0].env_name
        task_idx = episodes[0].task_data.idx
        survivors = [trace for episode in episodes if episode.ok for trace in episode.traces if not trace.has_error]
        num_failed_episodes = sum(not episode.ok for episode in episodes)
        num_errored_traces = sum(r.has_error for r in group)

        env = self.train_envs.get(env_name)
        # Untrainable traces carry no samples and must not skew the group baseline.
        survivors = [r for r in survivors if r.agent.trainable]
        if not survivors:
            get_logger().debug(
                f"Finished group | env={env_name} task_idx={task_idx} | "
                f"episodes={len(episodes)} (failed={num_failed_episodes}) | "
                f"traces={len(group)} (errored={num_errored_traces}) | dropped: no trainable survivors"
            )
            self.ready_groups.append([])
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

        if self.pre_filters:
            apply_filters(self.pre_filters, survivors)
        filtered_by_name: dict[str, int] = {}
        num_filtered = 0
        for r in survivors:
            self.pre_filter_seen += 1
            if r.is_filtered:
                self.pre_filter_dropped += 1
                num_filtered += 1
                for name, hit in r.filter_results.items():
                    if hit:
                        self.pre_filter_dropped_by_name[name] = self.pre_filter_dropped_by_name.get(name, 0) + 1
                        filtered_by_name[name] = filtered_by_name.get(name, 0) + 1
                continue
            # Reset annotations so the post-batch filter pass starts clean
            r.filter_results = {}
            r.is_filtered = False
            self.pending_selected += 1
            if self.token_batch_size is not None:
                self.pending_tokens += payload_tokens(r)

        selected = [r for r in survivors if not r.is_filtered]
        self.ready_groups.append(selected)

        # Per-group summary. One line per finalized group; per-filter
        # detection breakdown lives at debug level in ``apply_filters``
        rewards = [r.reward for r in survivors]
        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
        filter_str = ", ".join(f"{n}={c}" for n, c in filtered_by_name.items()) if filtered_by_name else "—"
        get_logger().debug(
            f"Finished group | env={env_name} task_idx={task_idx} | "
            f"episodes={len(episodes)} (failed={num_failed_episodes}) | "
            f"traces={len(group)} (errored={num_errored_traces}, filtered={num_filtered}) | "
            f"reward={avg_reward:.4f} | filters: {filter_str}"
        )

    def process_batch(self) -> TrainBatch:
        """Pop all ready groups as one cohort, preserving group atomicity."""
        groups, self.ready_groups = self.ready_groups, []
        cohort = [rollout for group in groups for rollout in group]
        self.pending_selected = 0
        self.pending_tokens = 0

        if self.post_filters:
            apply_filters(self.post_filters, cohort)

        # Samples are pre-built by ``process_rollout``; ``process_group`` already stamped the
        # advantage stream and loss routing on each sample. Filtered rollouts don't ship.
        samples: list[TrainingSample] = [sample for r in cohort if not r.is_filtered for sample in r.samples]

        # Episodes are the observation window; trace and metric views are derived from them.
        episodes = TrainEpisodes(self.pending_episodes)
        if samples:
            self.pending_episodes = []
        return TrainBatch(episodes=episodes, samples=samples)

    def reset_pre_filter_stats(self) -> None:
        self.pre_filter_seen = 0
        self.pre_filter_dropped = 0
        self.pre_filter_dropped_by_name.clear()
