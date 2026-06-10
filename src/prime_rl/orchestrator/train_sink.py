"""TrainSink: three-level rollout sink for the training side.

1. ``process_rollout`` — eager per-rollout tokenization (overlaps with
   dispatcher producing more rollouts). Errored rollouts skip this.
2. ``process_group`` — filters errored rollouts, computes advantages over
   survivors, runs the pre-batch filter pass.
3. ``process_batch`` — applies post-batch filter annotations and assembles
   the trainer-bound ``TrainingSample`` list. Returns a ``TrainBatch``.

``add()`` returns ``TrainBatch | None``. I/O concerns (ship to trainer,
save_rollouts, monitor.log, teacher logprobs) live on the orchestrator.
"""

from __future__ import annotations

import asyncio
import uuid
from collections import defaultdict
from typing import cast

import verifiers as vf

from prime_rl.configs.orchestrator import OrchestratorConfig, RAEAdvantageConfig
from prime_rl.orchestrator.advantage import assign_advantages
from prime_rl.orchestrator.envs import TrainEnvs
from prime_rl.orchestrator.filters import RolloutFilter, apply_filters
from prime_rl.orchestrator.multi_agent_advantage import (
    RAEState,
    compute_rae_advantages,
    extract_episode_pairs_for_multi_agent,
    fan_out_trainable_for_multi_agent,
)
from prime_rl.orchestrator.trajectories import (
    backfill_rollout_tokens,
    interleave_rollout,
    offload_images_to_disk,
)
from prime_rl.orchestrator.types import RAEStats, TrainBatch, TrainBatchMetrics, TrainRollout
from prime_rl.transport import TrainingSample
from prime_rl.utils.logger import get_logger


class TrainSink:
    """Three-level train sink. Constructed once, fed via ``add(rollout)``."""

    def __init__(
        self,
        config: OrchestratorConfig,
        *,
        tokenizer,
        renderer,
        train_envs: TrainEnvs,
        mm_token_type_ids_mapping: dict[int, int] | None,
        batch_size: int | None,
        token_batch_size: int | None,
        rae_state: RAEState | None,
        pre_filters: list[RolloutFilter],
        post_filters: list[RolloutFilter],
    ) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self.renderer = renderer
        self.train_envs = train_envs
        self.mm_token_type_ids_mapping = mm_token_type_ids_mapping
        self.batch_size = batch_size
        self.token_batch_size = token_batch_size
        # Per-env advantage fns live on each ``TrainEnv``; the shared RAE state
        # (multi-agent ``rae`` envs) is owned by the orchestrator so
        # checkpoints can round-trip it.
        if rae_state is None and any(isinstance(env_cfg.advantage, RAEAdvantageConfig) for env_cfg in config.train.env):
            raise ValueError(
                "advantage.type='rae' requires the orchestrator-owned RAEState "
                "(it is checkpointed across resumes); constructing a fresh state "
                "inside the sink would silently diverge from the checkpoint"
            )
        self.rae_state = rae_state
        self.pre_filters = pre_filters
        self.post_filters = post_filters

        # RAE observability accumulated across groups since the last ship;
        # popped into ``TrainBatchMetrics`` by ``process_batch``
        self.rae_step_stats: RAEStats | None = None

        # Keyed by the dispatcher's group UUID. ``(env_name, example_id)``
        # isn't unique — the same example can be re-sampled while an
        # earlier group is still in flight
        self.pending_groups: dict[uuid.UUID, list[TrainRollout]] = defaultdict(list)
        self.pending_batch: list[TrainRollout] = []
        self.pending_batch_tokens: int = 0
        self.pending_episode_rollouts: dict[uuid.UUID, TrainRollout] = {}

        # Reset by the orchestrator after each ship via ``reset_pre_filter_stats``
        self.pre_filter_seen = 0
        self.pre_filter_dropped = 0
        self.pre_filter_dropped_by_name: dict[str, int] = {}

        # Per-env arrival / error counters since the last ship; reset in
        # ``process_batch``. Fuel for the per-env success log breakdown
        self.arrivals_by_env: dict[str, int] = defaultdict(int)
        self.errors_by_env: dict[str, int] = defaultdict(int)

    def group_size_for(self, env_name: str) -> int:
        return self.train_envs.get(env_name).config.group_size

    def in_progress_groups(self) -> list[list[TrainRollout]]:
        """Per-rollout groups currently accumulating in ``pending_groups`` —
        i.e. groups that haven't hit ``group_size`` yet, so the pipeline log
        can reflect partial-group progress. Skips group-scoring envs (whose
        rollouts only make sense as a unit — the user expects per-group
        fill, not per-rollout, for those)."""
        out: list[list[TrainRollout]] = []
        for rollouts in self.pending_groups.values():
            if not rollouts:
                continue
            env_name = rollouts[0].env_name
            if self.train_envs.get(env_name).requires_group_scoring:
                continue
            out.append(rollouts)
        return out

    def batch_progress(self) -> tuple[int, int, str]:
        """``(current, target, unit)`` for the train batch — counts only
        ``pending_batch`` (survivors of finalized groups, queued for the
        trainer), so it's an honest 0→target fill. Partial-group arrivals are
        reported separately by ``buffered_count()``."""
        if self.batch_size is not None:
            return len(self.pending_batch), self.batch_size, "rollouts"
        return self.pending_batch_tokens, cast(int, self.token_batch_size), "tokens"

    def buffered_count(self) -> int:
        """Rollouts that have arrived but sit in not-yet-complete groups
        (non-group-scoring envs) — buffered in the sink ahead of the batch."""
        return sum(len(group) for group in self.in_progress_groups())

    def pending_batch_by_env(self) -> dict[str, int]:
        """Per-env breakdown of ``batch_progress()`` (``pending_batch`` only);
        values sum to the aggregate."""
        counts: dict[str, int] = defaultdict(int)
        for r in self.pending_batch:
            counts[r.env_name] += 1
        return dict(counts)

    async def add(self, rollout: TrainRollout) -> TrainBatch | None:
        """Process one arrival; finalize the group on the ``group_size``-th
        arrival; return a ``TrainBatch`` if the batch threshold is met."""
        await self.process_rollout(rollout)
        env_name = rollout.env_name
        self.arrivals_by_env[env_name] += 1
        if rollout.error is not None:
            self.errors_by_env[env_name] += 1
        self.pending_groups[rollout.group_id].append(rollout)
        if len(self.pending_groups[rollout.group_id]) >= self.group_size_for(env_name):
            await self.process_group(rollout.group_id)
        ready = (
            len(self.pending_batch) >= self.batch_size
            if self.batch_size is not None
            else self.pending_batch_tokens >= cast(int, self.token_batch_size)
        )
        if ready:
            return self.process_batch()
        return None

    async def process_rollout(self, rollout: TrainRollout) -> None:
        """Tokenize the rollout eagerly. Backfills tokens if the env didn't
        return them (SFT against external teacher APIs); errored rollouts
        skip tokenization and get dropped at the group level."""
        if rollout.error is not None:
            return
        if self._is_multi_agent_episode_rollout(rollout):
            return
        raw = rollout.raw
        needs_backfill = any(s["tokens"] is None for s in raw.get("trajectory") or [])
        if needs_backfill:
            await asyncio.to_thread(backfill_rollout_tokens, raw, self.tokenizer, renderer=self.renderer)
        samples = await asyncio.to_thread(
            interleave_rollout,
            raw,
            mm_token_type_ids_mapping=self.mm_token_type_ids_mapping,
            env_name=rollout.env_name,
        )
        rollout.samples = samples or []
        # Offload base64 image bytes to disk as soon as the rollout is
        # tokenized, so memory stays flat instead of holding every buffered
        # rollout's images until the batch ships (no-op for text-only).
        await asyncio.to_thread(offload_images_to_disk, [raw], self.config.output_dir)
        self._fill_token_usage_from_samples(rollout)

    async def process_group(self, group_id: uuid.UUID) -> None:
        """Finalize one GRPO group: drop errored rollouts (the whole group
        when ``requires_group_scoring`` and any failed), assign advantages,
        run pre-batch filters, append survivors to ``pending_batch``."""
        group = self.pending_groups.pop(group_id, [])
        if not group:
            return
        env_name = group[0].env_name
        example_id = group[0].example_id
        survivors = [r for r in group if r.error is None]
        num_errored = len(group) - len(survivors)

        # Group-scoring envs: any failure makes survivors' rewards unsafe
        # (computed relative to the missing ones)
        env = self.train_envs.get(env_name)
        if num_errored > 0 and env.requires_group_scoring:
            get_logger().debug(
                f"Finished group | env={env_name} example_id={example_id} | "
                f"rollouts={len(group)} (errored={num_errored}) | dropped: group-scored partial"
            )
            return
        if not survivors:
            get_logger().debug(
                f"Finished group | env={env_name} example_id={example_id} | "
                f"rollouts={len(group)} (errored={num_errored}) | dropped: all failed"
            )
            return

        episode_rollouts: list[TrainRollout] | None = None
        if env.is_multi_agent:
            episode_rollouts = survivors
            survivors = await self.project_multi_agent_group(episode_rollouts)
            if not survivors:
                get_logger().debug(
                    f"Finished group | env={env_name} example_id={example_id} | "
                    f"rollouts={len(group)} (errored={num_errored}) | dropped: no trainable member rollouts"
                )
                return
        else:
            assign_advantages(survivors, self.train_envs.get(env_name).advantage_fn)

        # Propagate to the pre-tokenized samples so the orchestrator can
        # collect samples at ship time without re-walking rollouts. The env
        # has a single sampling temperature; fan it out across each sample's
        # completion tokens here (interleave leaves it empty).
        for r in survivors:
            temperature = r.raw.get("sampling_args", {}).get("temperature", env.sampling_args.get("temperature", 1.0))
            for sample in r.samples:
                sample.advantage = r.advantage
                sample.reward = r.reward
                sample.env_name = r.env_name
                sample.training_mode = self.config.training_mode
                sample.completion_temperatures = [temperature] * len(sample.completion_ids)

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
            if r.source_rollout_id is not None and episode_rollouts is not None:
                for episode in episode_rollouts:
                    if episode.rollout_id == r.source_rollout_id:
                        self.pending_episode_rollouts[episode.rollout_id] = episode
                        break
            self.pending_batch.append(r)
            self.pending_batch_tokens += self.rollout_token_count(r)

        # Per-group summary. One line per finalized group; per-filter
        # detection breakdown lives at debug level in ``apply_filters``
        rewards = [r.reward for r in survivors]
        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
        filter_str = ", ".join(f"{n}={c}" for n, c in filtered_by_name.items()) if filtered_by_name else "—"
        get_logger().debug(
            f"Finished group | env={env_name} example_id={example_id} | "
            f"rollouts={len(group)} (errored={num_errored}, filtered={num_filtered}) | "
            f"reward={avg_reward:.4f} | filters: {filter_str}"
        )

    async def project_multi_agent_group(self, episodes: list[TrainRollout]) -> list[TrainRollout]:
        if self.rae_state is None:
            raise RuntimeError("Multi-agent training requires advantage.type='rae'.")

        raw_episodes: list[vf.RolloutOutput] = []
        for episode in episodes:
            episode.raw["env_name"] = episode.env_name
            raw_episodes.append(episode.raw)
        member_raws, episode_to_member_idxs = fan_out_trainable_for_multi_agent(raw_episodes, self.config.multi_agent)
        # Pairs come from the UNFILTERED mar_score: frame derivation and
        # zero-sum validation must not depend on which seats train_one kept
        episode_pairs = extract_episode_pairs_for_multi_agent(raw_episodes, self.config.multi_agent)
        advantages, rae_stats = compute_rae_advantages(member_raws, self.rae_state, episode_pairs=episode_pairs)
        if self.rae_step_stats is None:
            self.rae_step_stats = rae_stats
        else:
            self.rae_step_stats.merge(rae_stats)

        out: list[TrainRollout] = []
        for episode, member_idxs in zip(episodes, episode_to_member_idxs):
            for member_idx in member_idxs:
                raw = vf.RolloutOutput(dict(member_raws[member_idx]))
                self._inherit_episode_accounting(raw, episode)
                member = TrainRollout(
                    raw=raw,
                    env_name=episode.env_name,
                    example_id=episode.example_id,
                    group_id=episode.group_id,
                    policy_version=episode.policy_version,
                    off_policy_steps=episode.off_policy_steps,
                    source_rollout_id=episode.rollout_id,
                    advantage=advantages[member_idx],
                )
                out.append(member)
        # Tokenization is independent per member: ``process_rollout`` mutates
        # only the member's own raw/samples and uses the shared
        # tokenizer/renderer read-only, so members can overlap. ``out`` keeps
        # the deterministic episode x member order regardless.
        await asyncio.gather(*(self.process_rollout(member) for member in out))
        return out

    def _is_multi_agent_episode_rollout(self, rollout: TrainRollout) -> bool:
        return self.train_envs.get(rollout.env_name).is_multi_agent and "mar_score" in rollout.raw

    @staticmethod
    def _inherit_episode_accounting(raw: vf.RolloutOutput, episode: TrainRollout) -> None:
        raw.setdefault("env_name", episode.env_name)
        raw.setdefault("trajectory_id", f"{episode.rollout_id}:{raw.get('member_id', 'member')}")
        raw.setdefault("completion", None)
        # Member-level truncation: a member row is truncated iff one of its
        # own trajectory steps hit a generation-length stop. Prompt-overflow
        # truncation produces no trajectory step and stays episode-level,
        # visible via the inherited stop_condition ("prompt_too_long").
        raw["is_truncated"] = any(step["is_truncated"] for step in raw["trajectory"])
        raw.setdefault("stop_condition", episode.raw.get("stop_condition"))
        raw.setdefault("metrics", episode.raw.get("metrics", {}))
        raw.setdefault("timing", episode.raw.get("timing", {}))
        raw.setdefault("token_usage", episode.raw.get("token_usage", {}))

    @staticmethod
    def _fill_token_usage_from_samples(rollout: TrainRollout) -> None:
        prefill = 0
        decode = 0
        for sample in rollout.samples:
            sample_decode = sum(sample.completion_mask)
            decode += sample_decode
            prefill += len(sample.prompt_ids) + len(sample.completion_mask) - sample_decode
        rollout.raw["token_usage"] = {
            "input_tokens": float(prefill),
            "output_tokens": float(decode),
            "final_input_tokens": float(prefill),
            "final_output_tokens": float(decode),
        }

    def process_batch(self) -> TrainBatch:
        """Pop a cohort off ``pending_batch`` (by rollout count when
        ``batch_size`` is set, by token count when ``token_batch_size`` is
        set), apply post-batch filter annotations, and assemble the
        trainer-bound ``TrainingSample`` list. Overflow stays for the next
        batch."""
        if self.batch_size is not None:
            cohort = self.pending_batch[: self.batch_size]
            self.pending_batch = self.pending_batch[self.batch_size :]
            self.pending_batch_tokens -= sum(self.rollout_token_count(r) for r in cohort)
        else:
            token_batch_size = cast(int, self.token_batch_size)
            cut = 0
            running = 0
            for i, r in enumerate(self.pending_batch):
                running += self.rollout_token_count(r)
                cut = i + 1
                if running >= token_batch_size:
                    break
            cohort = self.pending_batch[:cut]
            self.pending_batch = self.pending_batch[cut:]
            self.pending_batch_tokens -= running

        if self.post_filters:
            apply_filters(self.post_filters, cohort)
        episode_rollouts = self.pop_episode_rollouts_for(cohort)

        # Samples are pre-built by ``process_rollout``; ``process_group``
        # already set advantage/reward on each sample
        samples: list[TrainingSample] = []
        prefill_lens: list[int] = []
        decode_lens: list[int] = []
        samples_per_rollout: list[int] = []
        num_prefill = 0
        num_decode = 0
        for r in cohort:
            samples_per_rollout.append(len(r.samples))
            prefill = 0
            decode = 0
            for sample in r.samples:
                sample_decode = sum(sample.completion_mask)
                sample_prefill = len(sample.prompt_ids) + len(sample.completion_mask) - sample_decode
                decode += sample_decode
                prefill += sample_prefill
                if not r.is_filtered:
                    samples.append(sample)
            prefill_lens.append(prefill)
            decode_lens.append(decode)
            num_prefill += prefill
            num_decode += decode

        n_trainable = sum(1 for r in cohort if not r.is_filtered)

        metrics = TrainBatchMetrics(
            n_trainable=n_trainable,
            num_prefill_tokens=num_prefill,
            num_decode_tokens=num_decode,
            rollout_prefill_lens=prefill_lens,
            rollout_decode_lens=decode_lens,
            samples_per_rollout=samples_per_rollout,
            samples_shipped=len(samples),
            arrivals_by_env=dict(self.arrivals_by_env),
            errors_by_env=dict(self.errors_by_env),
            rae_stats=self.rae_step_stats,
        )
        self.rae_step_stats = None
        self.arrivals_by_env.clear()
        self.errors_by_env.clear()
        return TrainBatch(rollouts=cohort, samples=samples, metrics=metrics, episode_rollouts=episode_rollouts)

    @staticmethod
    def rollout_token_count(rollout: TrainRollout) -> int:
        usage = rollout.raw["token_usage"]
        return int(usage["final_input_tokens"] + usage["final_output_tokens"])

    def pop_episode_rollouts_for(self, cohort: list[TrainRollout]) -> list[TrainRollout] | None:
        seen: set[uuid.UUID] = set()
        episodes: list[TrainRollout] = []
        for rollout in cohort:
            source_id = rollout.source_rollout_id
            if source_id is None or source_id in seen:
                continue
            seen.add(source_id)
            episode = self.pending_episode_rollouts.pop(source_id, None)
            if episode is not None:
                episodes.append(episode)
        return episodes or None

    def reset_pre_filter_stats(self) -> None:
        self.pre_filter_seen = 0
        self.pre_filter_dropped = 0
        self.pre_filter_dropped_by_name.clear()
