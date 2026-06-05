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
from dataclasses import dataclass

from prime_rl.configs.losses import apply_term_override, is_primary, overlay_terms, to_echo_config
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


@dataclass
class WeightInputs:
    """Inputs to a custom overlay weight resolver: returns a per-token weight list for ``sample``
    (length = prompt + completion). ``rollouts`` is the full GRPO group (each with its
    advantage/reward/raw trajectory), so the resolver can compute group-relative weights."""

    sample: TrainingSample
    rollouts: list[TrainRollout]


def _sample_has_trainable_tokens(sample: TrainingSample) -> bool:
    """Whether a sample contributes any gradient: a primary (completion) token after any per-env
    zeroing, or an overlay token with a nonzero weight. Position 0 is excluded to match the trainer
    (no shifted current-token logprob); a zero weight contributes no gradient, so it doesn't count."""
    if any(sample.completion_mask):
        return True
    overlays = sample.overlay_alphas
    if overlays is None:
        return False
    return any(any(a is not None and a != 0.0 for a in alphas[1:]) for alphas in overlays.values())


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
        self._overlay_cache: dict[str, list] = {}
        self._primary_cache: dict[str, bool] = {}
        # Advantage-weighted overlay terms: name -> tau. Their per-token alpha ships as a 1.0
        # eligibility marker (resolved at tokenization, before advantages) and is scaled by the
        # rollout's advantage x tau once advantages are assigned (see process_group).
        self._advantage_overlay_taus: dict[str, float] = {
            term.name: term.weight.tau for term in overlay_terms(config.losses) if term.weight.type == "advantage"
        }
        # Custom-weighted overlay terms: name -> (resolver fn, kwargs). The resolver runs per-rollout
        # in process_group (post-advantage) and fills the eligible tokens' per-token weights.
        self._custom_weight_fns: dict[str, tuple] = {
            term.name: (import_object(term.weight.import_path), term.weight.kwargs)
            for term in overlay_terms(config.losses)
            if term.weight.type == "custom"
        }
        # The primary's advantage weight tau is applied orchestrator-side (so any primary core, not
        # just dppo_kl, gets the resolved advantage × tau). Default 1.0 = no-op = bit-identical RL.
        _primary = next((term for term in config.losses if is_primary(term)), None)
        self._primary_tau: float = (
            _primary.weight.tau if _primary is not None and _primary.weight.type == "advantage" else 1.0
        )
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

    def _resolve_overlays(self, env_name: str):
        """Resolve the env's enabled overlay terms (per-env overrides applied) into a list of
        ``(term_name, resolved EchoLossConfig, bound filter fn)``, cached per env."""
        if env_name not in self._overlay_cache:
            env_config = self.train_envs.get(env_name).config
            enabled = env_config.enabled_losses
            resolved = []
            for term in overlay_terms(self.config.losses):
                if enabled is not None and term.name not in enabled:
                    continue
                if term.name in env_config.loss_overrides:
                    term = apply_term_override(term, env_config.loss_overrides[term.name])
                echo_config = to_echo_config(term)
                filter_fn = None
                if echo_config.filter is not None:
                    fn = import_object(echo_config.filter.import_path)
                    filter_fn = functools.partial(fn, **echo_config.filter.kwargs)
                resolved.append((term.name, echo_config, filter_fn))
            self._overlay_cache[env_name] = resolved
        return self._overlay_cache[env_name]

    def _primary_enabled(self, env_name: str) -> bool:
        """Whether the rl-mode primary (the rl/custom term) is enabled for this env. sft/opd
        dispatch to fixed cores and are always on; only rl-mode can be disabled per env via
        enabled_losses, in which case that env's samples ship with a zeroed completion mask
        (echo, which has its own mask, still applies)."""
        if env_name not in self._primary_cache:
            if self.config.training_mode != "rl":
                self._primary_cache[env_name] = True
            else:
                primary = next((term for term in self.config.losses if is_primary(term)), None)
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

        overlay_annotations = {}
        for name, echo_config, filter_fn in self._resolve_overlays(rollout.env_name):
            ann = await asyncio.to_thread(build_echo_annotations, raw, echo_config, filter_fn)
            if ann is not None:
                overlay_annotations[name] = ann

        samples = await asyncio.to_thread(
            interleave_rollout,
            raw,
            mm_token_type_ids_mapping=self.mm_token_type_ids_mapping,
            env_name=rollout.env_name,
            overlay_annotations=overlay_annotations,
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
                # The primary's per-token advantage is the GRPO advantage × the advantage-weight tau.
                sample.advantage = r.advantage * self._primary_tau if r.advantage is not None else r.advantage
                sample.reward = r.reward
                sample.env_name = r.env_name
                sample.training_mode = self.config.training_mode
                sample.completion_temperatures = [temperature] * len(sample.completion_ids)
                # Scale advantage-weighted overlays by this rollout's advantage (x tau); their alpha
                # arrives as a 1.0 eligibility marker since overlay tokens are resolved before advantages.
                if self._advantage_overlay_taus and sample.overlay_alphas is not None and r.advantage is not None:
                    for name, tau in self._advantage_overlay_taus.items():
                        alphas = sample.overlay_alphas.get(name)
                        if alphas is not None:
                            scale = r.advantage * tau
                            sample.overlay_alphas[name] = [a * scale if a is not None else None for a in alphas]
                # Custom-weighted overlays: a per-rollout resolver fills the eligible tokens' weights.
                if self._custom_weight_fns and sample.overlay_alphas is not None:
                    for name, (weight_fn, kwargs) in self._custom_weight_fns.items():
                        alphas = sample.overlay_alphas.get(name)
                        if alphas is None:
                            continue
                        weights = weight_fn(WeightInputs(sample=sample, rollouts=survivors), **kwargs)
                        if not isinstance(weights, list) or len(weights) != len(alphas):
                            raise ValueError(
                                f"custom weight {name!r} must return list[float] of length {len(alphas)} "
                                f"(prompt + completion), got {type(weights).__name__} of length "
                                f"{len(weights) if isinstance(weights, list) else 'n/a'}"
                            )
                        sample.overlay_alphas[name] = [
                            float(w) if m is not None else None for w, m in zip(weights, alphas)
                        ]

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
