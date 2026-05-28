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

from prime_rl.configs.orchestrator import AdvantageConfig, OrchestratorConfig
from prime_rl.orchestrator.advantage import assign_advantages, setup_advantage_fn
from prime_rl.orchestrator.envs import TrainEnvs
from prime_rl.orchestrator.filters import RolloutFilter, apply_filters
from prime_rl.orchestrator.trajectories import (
    backfill_rollout_tokens,
    interleave_rollout,
)
from prime_rl.orchestrator.types import TrainBatch, TrainBatchMetrics, TrainRollout
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
        advantage_config: AdvantageConfig | None,
        pre_filters: list[RolloutFilter],
        post_filters: list[RolloutFilter],
    ) -> None:
        assert (batch_size is None) != (token_batch_size is None), (
            "Exactly one of batch_size / token_batch_size must be set"
        )
        self.config = config
        self.tokenizer = tokenizer
        self.renderer = renderer
        self.train_envs = train_envs
        self.mm_token_type_ids_mapping = mm_token_type_ids_mapping
        self.batch_size = batch_size
        self.token_batch_size = token_batch_size
        # Built once — custom advantage funcs do an ``import_object`` and
        # we don't want to pay that per group. ``None`` = reward-only path.
        self.advantage_fn = setup_advantage_fn(advantage_config) if advantage_config is not None else None
        self.pre_filters = pre_filters
        self.post_filters = post_filters

        # Keyed by the dispatcher's group UUID. ``(env_name, example_id)``
        # isn't unique — the same example can be re-sampled while an
        # earlier group is still in flight.
        self.pending_groups: dict[uuid.UUID, list[TrainRollout]] = defaultdict(list)
        self.pending_batch: list[TrainRollout] = []

        # Reset by the orchestrator after each ship via ``reset_pre_filter_stats``.
        self.pre_filter_seen = 0
        self.pre_filter_dropped = 0
        self.pre_filter_dropped_by_name: dict[str, int] = {}

        # Per-env arrival / error counters since the last ship; reset in
        # ``process_batch``. Fuel for the per-env success log breakdown.
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
        """``(current, target, unit)`` for the in-progress train batch.
        Counts ``pending_batch`` plus per-rollout partial groups from
        non-group-scoring envs so the counter ticks per-rollout (group-
        scoring envs jump in ``group_size`` increments)."""
        in_progress_rollouts = [r for group in self.in_progress_groups() for r in group]
        if self.batch_size is not None:
            return len(self.pending_batch) + len(in_progress_rollouts), self.batch_size, "rollouts"
        assert self.token_batch_size is not None
        tokens = sum(
            r.raw["token_usage"]["final_input_tokens"] + r.raw["token_usage"]["final_output_tokens"]
            for r in self.pending_batch + in_progress_rollouts
        )
        return tokens, self.token_batch_size, "tokens"

    def pending_batch_by_env(self) -> dict[str, int]:
        """Per-env breakdown of ``batch_progress()``; values sum to the
        aggregate."""
        counts: dict[str, int] = defaultdict(int)
        for r in self.pending_batch:
            counts[r.env_name] += 1
        for group in self.in_progress_groups():
            for r in group:
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
            self.process_group(rollout.group_id)
        ready = (
            len(self.pending_batch) >= self.batch_size
            if self.batch_size is not None
            else sum(
                r.raw["token_usage"]["final_input_tokens"] + r.raw["token_usage"]["final_output_tokens"]
                for r in self.pending_batch
            )
            >= (self.token_batch_size or 0)
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

    def process_group(self, group_id: uuid.UUID) -> None:
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
        # (computed relative to the missing ones).
        env = self.train_envs.get(env_name)
        if num_errored > 0 and env.requires_group_scoring:
            get_logger().debug(
                f"Group | env={env_name} example_id={example_id} | "
                f"rollouts={len(group)} (errored={num_errored}) | dropped: group-scored partial"
            )
            return
        if not survivors:
            get_logger().debug(
                f"Group | env={env_name} example_id={example_id} | "
                f"rollouts={len(group)} (errored={num_errored}) | dropped: all failed"
            )
            return

        assign_advantages(survivors, self.advantage_fn)

        # Propagate to the pre-tokenized samples so the orchestrator can
        # collect samples at ship time without re-walking rollouts.
        for r in survivors:
            for sample in r.samples:
                sample.advantage = r.advantage
                sample.reward = r.reward
                sample.env_name = r.env_name
                sample.training_mode = self.config.training_mode

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
            # Reset annotations so the post-batch filter pass starts clean.
            r.filter_results = {}
            r.is_filtered = False
            self.pending_batch.append(r)

        # Per-group summary. One line per finalized group; per-filter
        # detection breakdown lives at debug level in ``apply_filters``.
        rewards = [r.reward for r in survivors]
        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
        filter_str = ", ".join(f"{n}={c}" for n, c in filtered_by_name.items()) if filtered_by_name else "—"
        get_logger().debug(
            f"Group | env={env_name} example_id={example_id} | "
            f"rollouts={len(group)} (errored={num_errored}, filtered={num_filtered}) | "
            f"reward={avg_reward:.4f} | filters: {filter_str}"
        )

    def process_batch(self) -> TrainBatch:
        """Pop a cohort off ``pending_batch`` (by rollout count when
        ``batch_size`` is set, by token count when ``token_batch_size`` is
        set), apply post-batch filter annotations, and assemble the
        trainer-bound ``TrainingSample`` list. Overflow stays for the next
        batch."""
        if self.batch_size is not None:
            cohort = self.pending_batch[: self.batch_size]
            self.pending_batch = self.pending_batch[self.batch_size :]
        else:
            assert self.token_batch_size is not None
            cut = 0
            running = 0
            for i, r in enumerate(self.pending_batch):
                running += r.raw["token_usage"]["final_input_tokens"] + r.raw["token_usage"]["final_output_tokens"]
                cut = i + 1
                if running >= self.token_batch_size:
                    break
            cohort = self.pending_batch[:cut]
            self.pending_batch = self.pending_batch[cut:]

        if self.post_filters:
            apply_filters(self.post_filters, cohort)

        # Samples are pre-built by ``process_rollout``; ``process_group``
        # already set advantage/reward on each sample.
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
        )
        self.arrivals_by_env.clear()
        self.errors_by_env.clear()
        return TrainBatch(rollouts=cohort, samples=samples, metrics=metrics)

    def reset_pre_filter_stats(self) -> None:
        self.pre_filter_seen = 0
        self.pre_filter_dropped = 0
        self.pre_filter_dropped_by_name.clear()
