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
import functools
import uuid
from collections import defaultdict

from prime_rl.configs.losses import apply_echo_override
from prime_rl.configs.orchestrator import AdvantageConfig, OrchestratorConfig
from prime_rl.orchestrator.advantage import assign_advantages, setup_advantage_fn
from prime_rl.orchestrator.echo import build_echo_annotations
from prime_rl.orchestrator.envs import TrainEnvs
from prime_rl.orchestrator.filters import RolloutFilter, apply_filters
from prime_rl.orchestrator.trajectories import (
    backfill_rollout_tokens,
    interleave_rollout,
    offload_images_to_disk,
)
from prime_rl.orchestrator.types import TrainBatch, TrainBatchMetrics, TrainRollout
from prime_rl.transport import TrainingSample
from prime_rl.utils.logger import get_logger
from prime_rl.utils.utils import import_object


def _sample_has_trainable_tokens(sample: TrainingSample) -> bool:
    """Whether a sample contributes any gradient: a primary (completion) token after
    any per-env zeroing, or an echo token. Echo position 0 is excluded to match the
    trainer (the first token has no shifted current-token logprob)."""
    if any(sample.completion_mask):
        return True
    echo_alpha = sample.echo_alpha
    return echo_alpha is not None and any(a is not None for a in echo_alpha[1:])


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
        self._echo_cache: dict[str, tuple] = {}
        self._primary_cache: dict[str, bool] = {}
        self.mm_token_type_ids_mapping = mm_token_type_ids_mapping
        self.batch_size = batch_size
        self.token_batch_size = token_batch_size
        # Built once — custom advantage funcs do an ``import_object`` and
        # we don't want to pay that per group. ``None`` = reward-only path
        self.advantage_fn = setup_advantage_fn(advantage_config) if advantage_config is not None else None
        self.pre_filters = pre_filters
        self.post_filters = post_filters

        # Keyed by the dispatcher's group UUID. ``(env_name, example_id)``
        # isn't unique — the same example can be re-sampled while an
        # earlier group is still in flight
        self.pending_groups: dict[uuid.UUID, list[TrainRollout]] = defaultdict(list)
        self.pending_batch: list[TrainRollout] = []

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
        assert self.token_batch_size is not None
        tokens = sum(
            r.raw["token_usage"]["final_input_tokens"] + r.raw["token_usage"]["final_output_tokens"]
            for r in self.pending_batch
        )
        return tokens, self.token_batch_size, "tokens"

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

    def _resolve_echo(self, env_name: str):
        """Resolve the env's enabled echo term (with per-env overrides applied) and
        its bound filter fn, cached per env. At most one echo term may be enabled."""
        if env_name not in self._echo_cache:
            env_config = self.train_envs.get(env_name).config
            enabled = env_config.enabled_losses
            echo_terms = [
                term for term in self.config.losses if term.type == "echo" and (enabled is None or term.name in enabled)
            ]
            if len(echo_terms) > 1:
                raise ValueError(
                    f"At most one echo term may be enabled per env (env {env_name!r}): "
                    f"{[term.name for term in echo_terms]}"
                )
            term = echo_terms[0] if echo_terms else None
            if term is not None:
                override = env_config.loss_overrides.get(term.name)
                if override:
                    term = apply_echo_override(term, override)
            filter_fn = None
            if term is not None and term.filter is not None:
                fn = import_object(term.filter.import_path)
                filter_fn = functools.partial(fn, **term.filter.kwargs)
            self._echo_cache[env_name] = (term, filter_fn)
        return self._echo_cache[env_name]

    def _primary_enabled(self, env_name: str) -> bool:
        """Whether the rl-mode primary (the rl/custom term) is enabled for this env. sft/opd
        dispatch to fixed cores and are always on; only rl-mode can be disabled per env via
        enabled_losses, in which case that env's samples ship with a zeroed completion mask
        (echo, which has its own mask, still applies)."""
        if env_name not in self._primary_cache:
            if self.config.training_mode != "rl":
                self._primary_cache[env_name] = True
            else:
                primary = next((term for term in self.config.losses if term.type in ("rl", "custom")), None)
                enabled = self.train_envs.get(env_name).config.enabled_losses
                self._primary_cache[env_name] = primary is not None and (enabled is None or primary.name in enabled)
        return self._primary_cache[env_name]

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

        echo_term, echo_filter_fn = self._resolve_echo(rollout.env_name)
        echo_annotations = await asyncio.to_thread(build_echo_annotations, raw, echo_term, echo_filter_fn)

        samples = await asyncio.to_thread(
            interleave_rollout,
            raw,
            mm_token_type_ids_mapping=self.mm_token_type_ids_mapping,
            env_name=rollout.env_name,
            echo_annotations=echo_annotations,
        )
        rollout.samples = samples or []
        # Offload base64 image bytes to disk as soon as the rollout is
        # tokenized, so memory stays flat instead of holding every buffered
        # rollout's images until the batch ships (no-op for text-only).
        await asyncio.to_thread(offload_images_to_disk, [raw], self.config.output_dir)

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

        assign_advantages(survivors, self.advantage_fn)

        # Propagate to the pre-tokenized samples so the orchestrator can
        # collect samples at ship time without re-walking rollouts. The env
        # has a single sampling temperature; fan it out across each sample's
        # completion tokens here (interleave leaves it empty).
        temperature = env.sampling_args["temperature"]
        for r in survivors:
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
            self.pending_batch.append(r)

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
        # already set advantage/reward on each sample
        samples: list[TrainingSample] = []
        prefill_lens: list[int] = []
        decode_lens: list[int] = []
        samples_per_rollout: list[int] = []
        num_prefill = 0
        num_decode = 0
        n_trainable = 0
        for r in cohort:
            samples_per_rollout.append(len(r.samples))
            prefill = 0
            decode = 0
            rollout_trainable = False
            for sample in r.samples:
                sample_decode = sum(sample.completion_mask)
                sample_prefill = len(sample.prompt_ids) + len(sample.completion_mask) - sample_decode
                decode += sample_decode
                prefill += sample_prefill
                if not r.is_filtered:
                    # Disable the primary loss for this env by zeroing the completion mask
                    # (done after the decode/prefill metric above so throughput stays accurate).
                    if not self._primary_enabled(r.env_name):
                        sample.completion_mask = [False] * len(sample.completion_mask)
                    samples.append(sample)
                    rollout_trainable = rollout_trainable or _sample_has_trainable_tokens(sample)
            prefill_lens.append(prefill)
            decode_lens.append(decode)
            num_prefill += prefill
            num_decode += decode
            # Count a rollout as trainable only if it ships at least one loss-bearing token
            # (primary or echo) — a primary-disabled env with no echo tokens contributes nothing.
            if rollout_trainable:
                n_trainable += 1

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
