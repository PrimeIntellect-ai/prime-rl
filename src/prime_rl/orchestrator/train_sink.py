"""TrainSink: three-level rollout sink for the training side.

Each rollout passes through three processing levels, each defined by a
``process_*`` method on the sink:

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
   pre-batch filter pass.
3. ``process_batch(cohort)`` — runs when the survivor buffer reaches
   ``batch_size`` (or ``token_batch_size`` tokens). Applies post-batch
   filter annotations and assembles the trainer-bound ``TrainingSample``
   list. Returns a ``TrainBatch`` (raw cohort + samples + counters).

The sink owns the boundary signals: ``add()`` returns True iff a full
batch is ready, and ``is_batch_done()`` re-checks after each pop. The
dispatcher emits every dispatched rollout (success or error) exactly once
so the count-based finalization always fires.

I/O concerns (ship to trainer, save_rollouts to disk, offload images,
teacher_logprobs for opd, metrics build, monitor.log) live on the
orchestrator.
"""

from __future__ import annotations

import asyncio
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
from prime_rl.orchestrator.types import ProcessResult, Rollout, TrainBatch
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
        self.group_size = config.group_size
        self.logger = get_logger()

        # In-progress GRPO groups keyed by (env_name, example_id). The sink
        # finalizes a group once ``len(pending_groups[key]) == group_size``
        # — works because the dispatcher emits every dispatched rollout
        # (success or error) exactly once.
        self.pending_groups: dict[tuple[str, int], list[Rollout]] = defaultdict(list)
        # Survivors of the pre-filter pass — waiting to ship.
        self.batch_buf: list[vf.RolloutOutput] = []

        # Per-batch pre-filter detection counters; reset by the orchestrator
        # after each ship via ``reset_pre_filter_stats``.
        self.pre_filter_seen = 0
        self.pre_filter_dropped = 0
        self.pre_filter_dropped_by_name: dict[str, int] = {}

    # ── ingest ────────────────────────────────────────────────────────────

    async def add(self, rollout: Rollout) -> bool:
        """Process one arrival. Runs the per-rollout step always; finalizes
        the group on the ``group_size``-th arrival. Returns True iff the
        survivor buffer has reached the batch threshold (so the orchestrator
        should call ``pop_batch``)."""
        assert rollout.kind == "train", "TrainSink only handles train rollouts"
        await self.process_rollout(rollout)
        key = (rollout.env_name, rollout.example_id)
        self.pending_groups[key].append(rollout)
        if len(self.pending_groups[key]) >= self.group_size:
            self.process_group(key)
        return self.is_batch_done()

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

    def process_group(self, key: tuple[str, int]) -> None:
        """Finalize one GRPO group: filter errored rollouts (and drop the
        whole group when ``requires_group_scoring`` is set and any failed),
        compute advantages over survivors, propagate them onto the per-
        rollout ``TrainingSample``\\ s, then run the pre-batch filter pass.
        Survivors (not filtered) land in ``batch_buf``.
        """
        env_name, _example_id = key
        group = self.pending_groups.pop(key, [])
        if not group:
            return
        all_raws = [r.raw for r in group]
        survivors = [raw for raw in all_raws if raw.get("error") is None]
        num_errored = len(all_raws) - len(survivors)

        # Group-scoring envs: any failure makes the surviving rollouts'
        # rewards unsafe (computed relative to the missing ones). Drop.
        env = self.train_envs.get(env_name)
        if num_errored > 0 and env.requires_group_scoring:
            self.logger.warning(
                f"Dropping group-scored train group ({env_name}) — {num_errored}/{len(all_raws)} rollouts failed"
            )
            return
        if not survivors:
            self.logger.warning(f"Dropping train group ({env_name}) — all {len(all_raws)} rollouts failed")
            return
        if num_errored > 0:
            self.logger.warning(
                f"Partial train group ({env_name}) — {len(survivors)}/{len(all_raws)} survived ({num_errored} failed)"
            )

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
        for raw in survivors:
            self.pre_filter_seen += 1
            if raw.get("is_filtered"):
                self.pre_filter_dropped += 1
                for name, hit in (raw.get("filters") or {}).items():
                    if hit:
                        self.pre_filter_dropped_by_name[name] = self.pre_filter_dropped_by_name.get(name, 0) + 1
                continue
            # Reset annotations so the post-batch filter pass starts clean.
            raw["filters"] = {}
            raw["is_filtered"] = False
            self.batch_buf.append(raw)

    # ── level 3: per-batch (post-filter + samples assembly) ───────────────

    def is_batch_done(self) -> bool:
        """True iff the survivor buffer has reached ``batch_size`` rollouts
        (or ``token_batch_size`` tokens)."""
        if self.batch_size is not None:
            return len(self.batch_buf) >= self.batch_size
        assert self.token_batch_size is not None
        return sum(get_seq_len(r) for r in self.batch_buf) >= self.token_batch_size

    def pop_batch(self) -> TrainBatch:
        """Pop a batch off the survivors buffer and run ``process_batch`` on it."""
        cohort = self.pop_cohort()
        return self.process_batch(cohort)

    def pop_cohort(self) -> list[vf.RolloutOutput]:
        if self.batch_size is not None:
            cohort = self.batch_buf[: self.batch_size]
            self.batch_buf = self.batch_buf[self.batch_size :]
            return cohort
        assert self.token_batch_size is not None
        cut = 0
        running = 0
        for i, r in enumerate(self.batch_buf):
            running += get_seq_len(r)
            cut = i + 1
            if running >= self.token_batch_size:
                break
        cohort = self.batch_buf[:cut]
        self.batch_buf = self.batch_buf[cut:]
        return cohort

    def process_batch(self, cohort: list[vf.RolloutOutput]) -> TrainBatch:
        """Apply post-batch filters (annotation only) and assemble the trainer-
        bound ``TrainingSample`` list from already-tokenized rollouts.
        """
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

        result = ProcessResult(
            n_trainable=n_trainable,
            num_prefill_tokens=num_prefill,
            num_decode_tokens=num_decode,
            rollout_prefill_lens=prefill_lens,
            rollout_decode_lens=decode_lens,
            samples_per_rollout=samples_per_rollout,
            samples_shipped=len(samples),
        )
        return TrainBatch(rollouts=cohort, samples=samples, result=result)

    def reset_pre_filter_stats(self) -> None:
        self.pre_filter_seen = 0
        self.pre_filter_dropped = 0
        self.pre_filter_dropped_by_name.clear()
