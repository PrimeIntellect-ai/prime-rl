"""TrainSink: three-level rollout sink for the training side.

Each rollout passes through three processing levels, each defined by a
``process_*`` method on the sink (same shape as ``EvalSink``):

1. ``process_rollout(rollout)`` — runs on every ``add()``. Eager
   tokenization (``backfill_rollout_tokens`` + ``interleave_rollout``), so
   the heavy per-rollout CPU work overlaps with the dispatcher producing
   more rollouts instead of stacking up at ship time. Errored rollouts
   (``raw["error"]`` set) skip tokenization — they'll be dropped at the
   group level.
2. ``process_group(key)`` — runs when ``group_size`` rollouts have arrived
   for ``(env, example_id)``. Filters out errored rollouts (and drops the
   whole group when the env ``requires_group_scoring`` and any rollout in
   the group failed); computes advantages over survivors; runs the
   pre-batch filter pass; appends survivors to ``pending_batch``.
3. ``process_batch()`` — runs when ``pending_batch`` has enough rollouts
   (``batch_size`` rollouts or ``token_batch_size`` tokens). Pops a cohort,
   applies post-batch filter annotations, and assembles the trainer-bound
   ``TrainingSample`` list. Returns a ``TrainBatch``.

``add()`` returns ``TrainBatch | None`` directly (mirrors ``EvalSink.add``);
no separate ``pop_batch`` / ``is_batch_done`` API needed. The dispatcher
emits every dispatched rollout (success or error) exactly once so the
count-based finalization always fires.

I/O concerns (ship to trainer, save_rollouts to disk, offload images,
teacher_logprobs for opd, metrics build, monitor.log) live on the
orchestrator.
"""

from __future__ import annotations

import asyncio
import uuid
from collections import defaultdict

import verifiers as vf

from prime_rl.configs.orchestrator import AdvantageConfig, OrchestratorConfig
from prime_rl.orchestrator.advantage import compute_advantages
from prime_rl.orchestrator.envs import TrainEnvs
from prime_rl.orchestrator.filters import RolloutFilter, apply_filters
from prime_rl.orchestrator.trajectories import (
    backfill_rollout_tokens,
    interleave_rollout,
)
from prime_rl.orchestrator.types import Rollout, TrainBatch, TrainBatchMetrics
from prime_rl.orchestrator.vf_utils import get_seq_len
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
        self.advantage_config = advantage_config
        self.pre_filters = pre_filters
        self.post_filters = post_filters

        # In-progress GRPO groups keyed by the dispatcher's group UUID.
        # The sink finalizes a group once arrivals reach the per-env
        # ``group_size`` (looked up via ``group_size_for(env_name)``) —
        # works because the dispatcher emits every dispatched rollout
        # (success or error) exactly once. We can't key on
        # ``(env_name, example_id)`` because the same example can be
        # re-sampled while an earlier group is still in flight.
        self.pending_groups: dict[uuid.UUID, list[Rollout]] = defaultdict(list)
        # Survivors of the pre-filter pass — waiting to ship. Singular
        # because train has one batch in flight at a time (unlike eval,
        # which can have multiple ``(env, eval_step)`` epochs in parallel).
        self.pending_batch: list[vf.RolloutOutput] = []

        # Per-batch pre-filter detection counters; reset by the orchestrator
        # after each ship via ``reset_pre_filter_stats``.
        self.pre_filter_seen = 0
        self.pre_filter_dropped = 0
        self.pre_filter_dropped_by_name: dict[str, int] = {}

        # Per-env arrival/error counters since the last ship — fuel for the
        # per-env breakdown in the success log. Reset in ``process_batch``.
        self.arrivals_by_env: dict[str, int] = defaultdict(int)
        self.errors_by_env: dict[str, int] = defaultdict(int)

    def group_size_for(self, env_name: str) -> int:
        return self.train_envs.get(env_name).config.group_size

    def batch_progress(self) -> tuple[int, int, str]:
        """``(current, target, unit)`` for the in-progress train batch — fuel
        for the orchestrator's pipeline log. Returns rollout count vs
        ``batch_size`` when rollout-batching, or token count vs
        ``token_batch_size`` when token-batching."""
        if self.batch_size is not None:
            return len(self.pending_batch), self.batch_size, "rollouts"
        assert self.token_batch_size is not None
        return sum(get_seq_len(r) for r in self.pending_batch), self.token_batch_size, "tokens"

    # ── ingest ────────────────────────────────────────────────────────────

    async def add(self, rollout: Rollout) -> TrainBatch | None:
        """Process one arrival. Runs the per-rollout step always; finalizes
        the group on the ``group_size``-th arrival; returns a ``TrainBatch``
        if that arrival brought ``pending_batch`` to the batch threshold."""
        assert rollout.kind == "train", "TrainSink only handles train rollouts"
        await self.process_rollout(rollout)
        env_name = rollout.raw["env_name"]
        self.arrivals_by_env[env_name] += 1
        if rollout.raw.get("error") is not None:
            self.errors_by_env[env_name] += 1
        self.pending_groups[rollout.group_id].append(rollout)
        if len(self.pending_groups[rollout.group_id]) >= self.group_size_for(env_name):
            self.process_group(rollout.group_id)
        # Mirror ``EvalSink.add``: trigger ``process_batch`` once the buffer
        # has reached the configured rollout/token threshold, otherwise
        # return None and wait for more arrivals.
        ready = (
            len(self.pending_batch) >= self.batch_size
            if self.batch_size is not None
            else sum(get_seq_len(r) for r in self.pending_batch) >= (self.token_batch_size or 0)
        )
        if ready:
            return self.process_batch()
        return None

    # ── level 1: per-rollout (tokenization) ───────────────────────────────

    async def process_rollout(self, rollout: Rollout) -> None:
        """Tokenize this rollout eagerly. Backfills tokens if the env didn't
        return them (sft against external teacher APIs), then runs
        ``interleave_rollout`` to produce one or more ``TrainingSample``\\ s
        attached as ``raw["_samples"]``. Errored rollouts skip tokenization;
        they'll be dropped at the group level.
        """
        raw = rollout.raw
        raw["_policy_version"] = rollout.policy_version
        if raw.get("error") is not None:
            raw["_samples"] = []
            return

        needs_backfill = any(s["tokens"] is None for s in raw.get("trajectory") or [])
        if needs_backfill:
            await asyncio.to_thread(backfill_rollout_tokens, raw, self.tokenizer, renderer=self.renderer)
        samples = await asyncio.to_thread(
            interleave_rollout, raw, mm_token_type_ids_mapping=self.mm_token_type_ids_mapping
        )
        raw["_samples"] = samples or []

    # ── level 2: per-group (filter errors + advantages + pre-filter) ──────

    def process_group(self, group_id: uuid.UUID) -> None:
        """Finalize one GRPO group: filter errored rollouts (and drop the
        whole group when ``requires_group_scoring`` is set and any failed),
        compute advantages over survivors, propagate them onto the per-
        rollout ``TrainingSample``\\ s, then run the pre-batch filter pass.
        Survivors (not filtered) land in ``pending_batch``.
        """
        group = self.pending_groups.pop(group_id, [])
        if not group:
            return
        all_raws = [r.raw for r in group]
        env_name = all_raws[0]["env_name"]
        example_id = all_raws[0]["example_id"]
        survivors = [raw for raw in all_raws if raw.get("error") is None]
        num_errored = len(all_raws) - len(survivors)

        # Group-scoring envs: any failure makes the surviving rollouts'
        # rewards unsafe (computed relative to the missing ones). Drop.
        env = self.train_envs.get(env_name)
        if num_errored > 0 and env.requires_group_scoring:
            get_logger().debug(
                f"Group | env={env_name} example_id={example_id} | "
                f"rollouts={len(all_raws)} (errored={num_errored}) | dropped: group-scored partial"
            )
            return
        if not survivors:
            get_logger().debug(
                f"Group | env={env_name} example_id={example_id} | "
                f"rollouts={len(all_raws)} (errored={num_errored}) | dropped: all failed"
            )
            return

        # Advantages over surviving rollouts only.
        if self.advantage_config is not None:
            compute_advantages(survivors, self.advantage_config)
        else:
            for raw in survivors:
                raw["advantage"] = raw.get("reward", 0.0)

        # Propagate advantages + reward + env to the pre-tokenized samples,
        # so the orchestrator can just collect samples at ship time without
        # re-walking rollouts.
        for raw in survivors:
            for sample in raw.get("_samples", []):
                sample.advantage = raw.get("advantage")
                sample.reward = raw.get("reward")
                sample.env_name = raw.get("env_name")
                sample.training_mode = self.config.training_mode

        # Pre-batch filter pass.
        if self.pre_filters:
            apply_filters(self.pre_filters, survivors)
        filtered_by_name: dict[str, int] = {}
        num_filtered = 0
        for raw in survivors:
            self.pre_filter_seen += 1
            if raw.get("is_filtered"):
                self.pre_filter_dropped += 1
                num_filtered += 1
                for name, hit in (raw.get("filters") or {}).items():
                    if hit:
                        self.pre_filter_dropped_by_name[name] = self.pre_filter_dropped_by_name.get(name, 0) + 1
                        filtered_by_name[name] = filtered_by_name.get(name, 0) + 1
                continue
            # Reset annotations so the post-batch filter pass starts clean.
            raw["filters"] = {}
            raw["is_filtered"] = False
            self.pending_batch.append(raw)

        # Per-group summary. One line per finalized group; per-filter
        # detection breakdown lives at debug level in ``apply_filters``.
        rewards = [raw.get("reward", 0.0) for raw in survivors]
        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
        filter_str = ", ".join(f"{n}={c}" for n, c in filtered_by_name.items()) if filtered_by_name else "—"
        get_logger().debug(
            f"Group | env={env_name} example_id={example_id} | "
            f"rollouts={len(all_raws)} (errored={num_errored}, filtered={num_filtered}) | "
            f"reward={avg_reward:.4f} | filters: {filter_str}"
        )

    # ── level 3: per-batch (post-filter + samples assembly) ───────────────

    def process_batch(self) -> TrainBatch:
        """Pop a cohort off ``pending_batch`` (by rollout count when
        ``batch_size`` is set, by token count when ``token_batch_size`` is
        set), apply post-batch filter annotations, and assemble the trainer-
        bound ``TrainingSample`` list from already-tokenized rollouts. Any
        overflow stays in ``pending_batch`` for the next batch."""
        if self.batch_size is not None:
            cohort = self.pending_batch[: self.batch_size]
            self.pending_batch = self.pending_batch[self.batch_size :]
        else:
            assert self.token_batch_size is not None
            cut = 0
            running = 0
            for i, r in enumerate(self.pending_batch):
                running += get_seq_len(r)
                cut = i + 1
                if running >= self.token_batch_size:
                    break
            cohort = self.pending_batch[:cut]
            self.pending_batch = self.pending_batch[cut:]

        if self.post_filters:
            apply_filters(self.post_filters, cohort)
        else:
            for r in cohort:
                r.setdefault("filters", {})
                r.setdefault("is_filtered", False)

        # Collect tokenized samples + per-rollout stats. Samples were pre-built
        # by ``process_rollout``; the per-group hook already set advantage/reward
        # on each sample.
        samples: list[TrainingSample] = []
        prefill_lens: list[int] = []
        decode_lens: list[int] = []
        samples_per_rollout: list[int] = []
        num_prefill = 0
        num_decode = 0
        for raw in cohort:
            rollout_samples = raw.get("_samples", []) or []
            samples_per_rollout.append(len(rollout_samples))
            prefill = 0
            decode = 0
            for sample in rollout_samples:
                sample_decode = sum(sample.completion_mask)
                sample_prefill = len(sample.prompt_ids) + len(sample.completion_mask) - sample_decode
                decode += sample_decode
                prefill += sample_prefill
                if not raw.get("is_filtered"):
                    samples.append(sample)
            prefill_lens.append(prefill)
            decode_lens.append(decode)
            num_prefill += prefill
            num_decode += decode

        n_trainable = sum(1 for r in cohort if not r.get("is_filtered"))

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
