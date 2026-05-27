"""TrainSink: three-level rollout sink for the training side.

Each rollout passes through three processing levels, each defined by a
``process_*`` method on the sink:

1. ``process_rollout(rollout)`` — runs on every ``add()``. Eager
   tokenization (``backfill_rollout_tokens`` + ``interleave_rollout``), so
   the heavy per-rollout CPU work overlaps with the dispatcher producing
   more rollouts instead of stacking up at ship time.
2. ``process_group(key)`` — runs when ``rollout.is_group_done=True`` for
   the ``(env, example_id)`` GRPO group. Computes advantages, propagates
   them onto the per-rollout ``TrainingSample``\\ s produced in step 1,
   then applies the pre-batch filters and drops filtered rollouts.
3. ``process_batch(batch)`` — runs when ``batch_ready()``. Applies post-
   batch filter annotations (no drops; filter rates only); the result is
   a ``TrainBatchResult`` carrying tokenized ``TrainingSample``\\ s, the
   per-rollout cohort (for metrics), and a per-batch metrics dict.

Batch readiness is sink-internal: ``batch_size`` rollouts (or
``token_batch_size`` tokens) of survivors accumulate, then the next
``add()`` returns ``True`` and the orchestrator calls ``pop_batch``.

I/O concerns (ship to trainer, save_rollouts to disk, offload images,
teacher_logprobs for opd, monitor.log) live on the orchestrator.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import verifiers as vf

from prime_rl.configs.orchestrator import AdvantageConfig, OrchestratorConfig
from prime_rl.orchestrator.advantage import compute_advantages
from prime_rl.orchestrator.filters import RolloutFilter, apply_filters
from prime_rl.orchestrator.trajectories import (
    backfill_rollout_tokens,
    interleave_rollout,
)
from prime_rl.orchestrator.vf_utils import get_seq_len
from prime_rl.orchestrator_v2.ckpt import Progress
from prime_rl.orchestrator_v2.dispatcher import Rollout
from prime_rl.orchestrator_v2.metrics import MetricsBuilder, ProcessResult
from prime_rl.transport import TrainingSample
from prime_rl.utils.logger import get_logger


@dataclass
class TrainBatchResult:
    """Everything the orchestrator needs after one batch is processed.

    The ``samples`` list is the trainer-bound payload (filtered survivors
    only — ``post_batch_filters`` with ``enforce=True`` drop here). The
    ``rollouts`` list is the full cohort including filtered ones, kept for
    metric aggregation. ``metrics`` is the W&B dict, fully built; the
    orchestrator just hands it to ``monitor.log``.
    """

    rollouts: list[vf.RolloutOutput]
    samples: list[TrainingSample]
    metrics: dict[str, Any]
    result: ProcessResult


class TrainSink:
    """Three-level train sink. Constructed once, fed via ``add(rollout)``."""

    def __init__(
        self,
        config: OrchestratorConfig,
        *,
        tokenizer,
        renderer,
        mm_token_type_ids_mapping: dict[int, int] | None,
        batch_size: int | None,
        token_batch_size: int | None,
        advantage_config: AdvantageConfig | None,
        pre_filters: list[RolloutFilter],
        post_filters: list[RolloutFilter],
        metrics_builder: MetricsBuilder,
    ) -> None:
        assert (batch_size is None) != (token_batch_size is None), (
            "Exactly one of batch_size / token_batch_size must be set"
        )
        self.config = config
        self.tokenizer = tokenizer
        self.renderer = renderer
        self.mm_token_type_ids_mapping = mm_token_type_ids_mapping
        self.batch_size = batch_size
        self.token_batch_size = token_batch_size
        self.advantage_config = advantage_config
        self.pre_filters = pre_filters
        self.post_filters = post_filters
        self.metrics_builder = metrics_builder
        self.logger = get_logger()

        # In-progress GRPO groups keyed by (env_name, example_id). Each group
        # holds the surviving ``Rollout``\\ s as they arrive; finalized on
        # ``is_group_done=True``.
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
        """Run the per-rollout level (always) and per-group level (on
        ``is_group_done``). Returns True if the batch is now ready to pop."""
        assert rollout.kind == "train", "TrainSink only handles train rollouts"
        await self.process_rollout(rollout)
        key = (rollout.env_name, rollout.example_id)
        self.pending_groups[key].append(rollout)
        if rollout.is_group_done:
            self.process_group(key)
        return self.batch_ready()

    # ── level 1: per-rollout (tokenization) ───────────────────────────────

    async def process_rollout(self, rollout: Rollout) -> None:
        """Tokenize this rollout eagerly. Backfills tokens if the env didn't
        return them (sft against external teacher APIs), then runs
        ``interleave_rollout`` to produce one or more ``TrainingSample``\\ s
        attached as ``raw["_samples"]``.
        """
        raw = rollout.raw
        raw["_policy_version"] = rollout.policy_version

        needs_backfill = any(s["tokens"] is None for s in raw.get("trajectory") or [])
        if needs_backfill:
            await asyncio.to_thread(backfill_rollout_tokens, raw, self.tokenizer, renderer=self.renderer)
        samples = await asyncio.to_thread(
            interleave_rollout, raw, mm_token_type_ids_mapping=self.mm_token_type_ids_mapping
        )
        raw["_samples"] = samples or []

    # ── level 2: per-group (advantages + pre-filter) ──────────────────────

    def process_group(self, key: tuple[str, int]) -> None:
        """Compute advantages over the surviving rollouts in this GRPO group,
        propagate them onto the per-rollout ``TrainingSample``\\ s, then run
        the pre-batch filter pass. Survivors (not filtered) land in
        ``batch_buf``.
        """
        group = self.pending_groups.pop(key, [])
        if not group:
            return
        raws = [r.raw for r in group]

        # Advantages over the (possibly partial) GRPO group.
        if self.advantage_config is not None:
            compute_advantages(raws, self.advantage_config)
        else:
            for raw in raws:
                raw["advantage"] = raw.get("reward", 0.0)

        # Propagate advantages + reward + env to the pre-tokenized samples,
        # so the orchestrator can just collect samples at ship time without
        # re-walking rollouts.
        for raw in raws:
            for sample in raw.get("_samples", []):
                sample.advantage = raw.get("advantage")
                sample.reward = raw.get("reward")
                sample.env_name = raw.get("env_name")
                sample.training_mode = self.config.training_mode

        # Pre-batch filter pass.
        if self.pre_filters:
            apply_filters(self.pre_filters, raws)
        for raw in raws:
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

    # ── level 3: per-batch (post-filter + metrics) ────────────────────────

    def batch_ready(self) -> bool:
        if self.batch_size is not None:
            return len(self.batch_buf) >= self.batch_size
        assert self.token_batch_size is not None
        return sum(get_seq_len(r) for r in self.batch_buf) >= self.token_batch_size

    def pop_batch(
        self,
        *,
        step: int,
        progress: Progress,
        dispatcher_gauges: dict[str, float],
        dispatcher_drain: dict[str, float],
        step_time: float,
        save_ckpt_time: float,
    ) -> TrainBatchResult:
        """Pop a batch off the survivors buffer and run ``process_batch`` on it."""
        cohort = self._pop_cohort()
        return self.process_batch(
            cohort,
            step=step,
            progress=progress,
            dispatcher_gauges=dispatcher_gauges,
            dispatcher_drain=dispatcher_drain,
            step_time=step_time,
            save_ckpt_time=save_ckpt_time,
        )

    def _pop_cohort(self) -> list[vf.RolloutOutput]:
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

    def process_batch(
        self,
        cohort: list[vf.RolloutOutput],
        *,
        step: int,
        progress: Progress,
        dispatcher_gauges: dict[str, float],
        dispatcher_drain: dict[str, float],
        step_time: float,
        save_ckpt_time: float,
    ) -> TrainBatchResult:
        """Apply post-batch filters (annotation only), assemble the trainer-
        bound ``TrainingSample`` list from already-tokenized rollouts, and
        build the per-step metrics dict.
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
            parallel_preprocess_time=0.0,  # tokenization is per-rollout now
            teacher_logprobs_time=0.0,  # set by the orchestrator after teacher logprobs
            samples_shipped=len(samples),
        )

        metrics = self.metrics_builder.build(
            step=step,
            rollouts=cohort,
            result=result,
            progress=progress,
            dispatcher_gauges=dispatcher_gauges,
            dispatcher_drain=dispatcher_drain,
            step_time=step_time,
            save_ckpt_time=save_ckpt_time,
            pre_filter_seen=self.pre_filter_seen,
            pre_filter_dropped=self.pre_filter_dropped,
            pre_filter_dropped_by_name=dict(self.pre_filter_dropped_by_name),
        )

        return TrainBatchResult(rollouts=cohort, samples=samples, metrics=metrics, result=result)

    def reset_pre_filter_stats(self) -> None:
        self.pre_filter_seen = 0
        self.pre_filter_dropped = 0
        self.pre_filter_dropped_by_name.clear()
