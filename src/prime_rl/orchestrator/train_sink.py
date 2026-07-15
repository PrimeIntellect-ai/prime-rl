"""TrainSink: three-level rollout sink for the training side.

1. ``process_rollout`` — eager per-rollout tokenization (overlaps with
   dispatcher producing more rollouts), then the env algorithm's
   ``finalize_rollout`` (rollout-local scoring + any reference I/O). Errored
   rollouts skip this.
2. ``process_group`` — filters errored rollouts, hands survivors to the env
   algorithm's ``finalize_group`` (advantages + per-sample wire stamping),
   runs the pre-batch filter pass.
3. ``process_batch`` — applies post-batch filter annotations and assembles
   the trainer-bound ``TrainingSample`` list. Returns a ``TrainBatch``.

``add()`` returns ``TrainBatch | None``. I/O concerns (ship to trainer,
save_rollouts, monitor.log) live on the orchestrator.
"""

from __future__ import annotations

import asyncio
import uuid
from collections import defaultdict

from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.orchestrator.envs import TrainEnvs
from prime_rl.orchestrator.filters import RolloutFilter, apply_filters
from prime_rl.orchestrator.metrics import TrainRollouts
from prime_rl.orchestrator.trajectories import qa_recycle_samples, trace_to_samples
from prime_rl.orchestrator.types import Rollout, TrainBatch
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

        # Observation window for the next shipped batch: rollouts of groups
        # finalized since the last ship (errored + filtered + survivors).
        # In-progress groups stay out until they finalize.
        self.pending_rollouts: TrainRollouts = TrainRollouts()
        # Keyed by the dispatcher's group UUID. ``(env_name, task_idx)``
        # isn't unique — the same task can be re-sampled while an
        # earlier group is still in flight
        self.pending_groups: dict[uuid.UUID, list[Rollout]] = defaultdict(list)
        self.pending_batch: list[Rollout] = []
        # A5 lessons are group-owned until a post-filter survivor from that group ships,
        # so filtering/splitting cannot silently lose them.
        self._group_meta_samples: dict[uuid.UUID, list[TrainingSample]] = {}
        # Running trainer-payload total, including TTT recycle/meta samples.
        self.pending_tokens: int = 0

        # Reset by the orchestrator after each ship via ``reset_pre_filter_stats``
        self.pre_filter_seen = 0
        self.pre_filter_dropped = 0
        self.pre_filter_dropped_by_name: dict[str, int] = {}

        # Lazy per-env chat clients for TTT meta-extraction (a shared client would pin
        # every env to whichever pool triggered meta-lessons first).
        self._meta_clients: dict[str, object] = {}

        # A5 meta-extraction outcome counters — a rising dropped rate means the arm is
        # quietly running without lessons.
        self.meta_groups_ok = 0
        self.meta_groups_dropped = 0

    @staticmethod
    def _sample_tokens(samples: list[TrainingSample]) -> int:
        return sum(len(sample.token_ids) for sample in samples)

    def _refresh_pending_tokens(self) -> None:
        """Recompute token-batch progress after a cohort is removed."""
        if self.token_batch_size is None:
            return
        self.pending_tokens = sum(payload_tokens(r) for r in self.pending_batch)
        pending_group_ids = {r.group_id for r in self.pending_batch}
        self.pending_tokens += sum(
            self._sample_tokens(samples)
            for group_id, samples in self._group_meta_samples.items()
            if group_id in pending_group_ids
        )

    def group_size_for(self, env_name: str) -> int:
        return self.train_envs.get(env_name).config.group_size

    def in_progress_groups(self) -> list[list[Rollout]]:
        """Per-rollout groups currently accumulating in ``pending_groups`` —
        i.e. groups that haven't hit ``group_size`` yet, so the pipeline log
        can reflect partial-group progress. Skips group-scoring envs (whose
        rollouts only make sense as a unit — the user expects per-group
        fill, not per-rollout, for those)."""
        out: list[list[Rollout]] = []
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
        assert self.token_batch_size is not None
        return self.pending_tokens, self.token_batch_size, "tokens"

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

    async def add(self, rollout: Rollout) -> TrainBatch | None:
        """Process one arrival; finalize the group on the ``group_size``-th
        arrival; return a ``TrainBatch`` if the finalization pushed (or left)
        the batch over its threshold. Arrivals into still-incomplete groups
        never ship a batch."""
        await self.process_rollout(rollout)
        env_name = rollout.env_name
        self.pending_groups[rollout.group_id].append(rollout)
        if len(self.pending_groups[rollout.group_id]) < self.group_size_for(env_name):
            return None
        await self.process_group(rollout.group_id)
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

    async def process_rollout(self, rollout: Rollout) -> None:
        """Build training samples from the rollout's Trace (one per branch), walking the
        message graph. Training is renderer-only across all modes (RL/OPD student, SFT teacher),
        so every node already carries its tokens. Errored rollouts are dropped at the group
        level, so skip them here."""
        if rollout.has_error:
            return
        # ttt_qa branches may sample at their own temperature (QAConfig.temperature;
        # None = same as the rollout, so nothing to stamp).
        env = self.train_envs.get(rollout.env_name)
        ttt_config = getattr(env.config, "ttt", None)
        active_ttt = ttt_config is not None and ttt_config.enabled
        qa_config = getattr(ttt_config, "qa", None) if active_ttt else None
        qa_temperature = getattr(qa_config, "temperature", None)
        try:
            samples = await asyncio.to_thread(
                trace_to_samples,
                rollout,
                env_name=rollout.env_name,
                mm_token_type_ids_mapping=self.mm_token_type_ids_mapping,
                qa_temperature=qa_temperature,
                rollout_temperature=env.sampling_args["temperature"],
            )
        except Exception as exc:
            if not active_ttt:
                # Preserve Prime-RL's existing fail-fast behavior outside the feature.
                raise
            # TTT checkpoint/version mapping can reject a malformed recorded branch.
            # Contain that rollout so the group logic drops it instead of killing the run.
            import verifiers.v1 as vf

            get_logger().warning(f"Tokenization failed for rollout in group {rollout.group_id}", exc_info=True)
            rollout.errors.append(vf.Error(type="TokenizationError", message=str(exc)))
            rollout.samples = []
            return
        rollout.samples = samples or []
        # Arrival phase: rollout-local scoring (raw reward, echo observation
        # weighting, opd/opsd reference logprobs) runs as soon as the rollout is
        # tokenized — before its group is complete.
        await self.train_envs.get(rollout.env_name).algorithm.finalize_rollout(rollout)

    async def process_group(self, group_id: uuid.UUID) -> None:
        """Finalize one GRPO group: drop errored rollouts (the whole group
        when ``requires_group_scoring`` and any failed), assign advantages,
        run pre-batch filters, append survivors to ``pending_batch``."""
        group = self.pending_groups.pop(group_id, [])
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

        # Group-scoring envs: any failure makes survivors' rewards unsafe
        # (computed relative to the missing ones)
        env = self.train_envs.get(env_name)
        if num_errored > 0 and env.requires_group_scoring:
            get_logger().debug(
                f"Finished group | env={env_name} task_idx={task_idx} | "
                f"rollouts={len(group)} (errored={num_errored}) | dropped: group-scored partial"
            )
            return
        if not survivors:
            get_logger().debug(
                f"Finished group | env={env_name} task_idx={task_idx} | "
                f"rollouts={len(group)} (errored={num_errored}) | dropped: all failed"
            )
            return

        # Advantages + per-sample wire stamping (advantage stream, loss
        # routing) are the algorithm's job (finalize_group); the sink only
        # owns the grouping mechanics.
        await env.algorithm.finalize_group(survivors)

        # TTT Q&A recycling: append ce-routed Q&A samples AFTER group finalization so
        # the advantage broadcast/stamping never sees them.
        ttt_config = getattr(env.config, "ttt", None)
        # Gate on `enabled`: a disabled TTT block must run neither QA-to-policy path.
        qa_config = ttt_config.qa if ttt_config is not None and ttt_config.enabled else None
        if qa_config is not None and qa_config.recycle_to_policy:
            for r in survivors:
                try:
                    r.samples.extend(await asyncio.to_thread(qa_recycle_samples, r, self.tokenizer, env_name))
                except Exception:
                    # QA recycling is enrichment (chat templates can choke on odd tool
                    # schemas): skip this rollout's recycling, never error the rollout.
                    get_logger().warning(f"QA recycling failed | env={env_name} task_idx={task_idx}", exc_info=True)

        # TTT group-level meta-extraction (see qa_meta.py); samples stay group-owned
        # until a member survives both filter passes.
        meta_samples: list[TrainingSample] = []
        if qa_config is not None and qa_config.meta_lessons and len(survivors) >= 2:
            from prime_rl.orchestrator.qa_meta import extract_meta_lessons, meta_lesson_samples
            from prime_rl.utils.client import openai_client_from_config

            try:
                if env_name not in self._meta_clients:
                    self._meta_clients[env_name] = openai_client_from_config(await env.sampler.pool.get_eval_client())
                items = await extract_meta_lessons(
                    survivors, qa_config, self._meta_clients[env_name], env.sampler.pool.model_name
                )
                if items:
                    meta_samples = await asyncio.to_thread(
                        meta_lesson_samples, items, survivors, self.tokenizer, env_name
                    )
                    get_logger().debug(
                        f"TTT meta-lessons | env={env_name} task_idx={task_idx} | {len(items)} lesson(s) extracted"
                    )
                self.meta_groups_ok += 1
            except Exception:
                # Enrichment, never correctness: a bad group must not kill the run.
                self.meta_groups_dropped += 1
                get_logger().warning(
                    f"Meta-lesson extraction failed | env={env_name} task_idx={task_idx}", exc_info=True
                )

        # The env has a single sampling temperature; fan it out per token
        # (context tokens are masked out, so their temperature is don't-care).
        # Only fill unstamped samples: ce recycle/meta samples (T=1) and ttt_qa
        # branches (qa.temperature) already carry theirs.
        temperature = env.sampling_args["temperature"]
        queued_group_member = False
        for r in survivors:
            for sample in r.samples:
                if not sample.temperatures:
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
            self.pending_batch.append(r)
            queued_group_member = True
            if self.token_batch_size is not None:
                self.pending_tokens += payload_tokens(r)

        if meta_samples and queued_group_member:
            self._group_meta_samples[group_id] = meta_samples
            if self.token_batch_size is not None:
                self.pending_tokens += self._sample_tokens(meta_samples)

        # Per-group summary. One line per finalized group; per-filter
        # detection breakdown lives at debug level in ``apply_filters``
        rewards = [r.reward for r in survivors]
        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
        filter_str = ", ".join(f"{n}={c}" for n, c in filtered_by_name.items()) if filtered_by_name else "—"
        get_logger().debug(
            f"Finished group | env={env_name} task_idx={task_idx} | "
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
            counted_meta_groups: set[uuid.UUID] = set()
            for i, r in enumerate(self.pending_batch):
                running += payload_tokens(r)
                if r.group_id not in counted_meta_groups:
                    running += self._sample_tokens(self._group_meta_samples.get(r.group_id, []))
                    counted_meta_groups.add(r.group_id)
                cut = i + 1
                if running >= self.token_batch_size:
                    break
            cohort = self.pending_batch[:cut]
            self.pending_batch = self.pending_batch[cut:]
            self._refresh_pending_tokens()

        if self.post_filters:
            apply_filters(self.post_filters, cohort)

        # Samples are pre-built by ``process_rollout``; ``process_group`` already stamped the
        # advantage stream and loss routing on each sample. Filtered rollouts don't ship.
        samples: list[TrainingSample] = [sample for r in cohort if not r.is_filtered for sample in r.samples]

        # Deliver a group's meta lessons exactly once with its first post-filter
        # survivor. If this cohort contains only filtered members but later members
        # remain queued, retain the lessons for the later cohort.
        remaining_group_ids = {r.group_id for r in self.pending_batch}
        for group_id in dict.fromkeys(r.group_id for r in cohort):
            meta_samples = self._group_meta_samples.get(group_id)
            if not meta_samples:
                continue
            if any(r.group_id == group_id and not r.is_filtered for r in cohort):
                samples.extend(meta_samples)
                del self._group_meta_samples[group_id]
            elif group_id not in remaining_group_ids:
                del self._group_meta_samples[group_id]
        self._refresh_pending_tokens()

        # The observation window contains every rollout of every group finalized since
        # the last ship; samples are the current cohort's trainer-bound payload.

        rollouts = self.pending_rollouts
        if samples:
            self.pending_rollouts = TrainRollouts()
        return TrainBatch(rollouts=rollouts, samples=samples)

    def reset_pre_filter_stats(self) -> None:
        self.pre_filter_seen = 0
        self.pre_filter_dropped = 0
        self.pre_filter_dropped_by_name.clear()
        self.meta_groups_ok = 0
        self.meta_groups_dropped = 0
