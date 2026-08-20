"""TrainSink: three-level rollout sink for the training side.

1. ``process_rollout`` — eager per-rollout tokenization (overlaps with
   dispatcher producing more rollouts), then the env algorithm's
   ``finalize_rollout`` (rollout-local scoring + any reference I/O). Errored
   and untrainable rollouts skip this.
2. ``process_group`` — removes errored rollouts, hands the trainable
   survivors to the env algorithm's ``finalize_group`` (advantages +
   per-sample wire stamping), then asks the source curriculum whether the
   result should train.
3. ``process_batch`` — assembles the trainer-bound ``TrainingSample`` list
   and optionally removes zero-advantage RL payload.

``add()`` takes one episode (``list[Rollout]``) and returns
``TrainBatch | None``; group accounting counts episodes, never loose traces.
I/O concerns (ship to trainer, monitors.log) live on the
orchestrator.
"""

from __future__ import annotations

import asyncio
import uuid
from collections import defaultdict
from collections.abc import Callable

from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.orchestrator.envs import TrainEnvs
from prime_rl.orchestrator.metrics import TrainRollouts
from prime_rl.orchestrator.trajectories import trace_to_samples
from prime_rl.orchestrator.types import Rollout, TrainBatch
from prime_rl.transports.rollouts import TrainingSample
from prime_rl.utils.logger import get_logger

MAX_CONSECUTIVE_ZERO_OUTPUT_BATCH_EQUIVALENTS = 10


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


def _prune_zero_advantages(sample: TrainingSample) -> bool:
    """Remove zero-advantage tokens from the RL component.

    Return whether the sample still carries any RL, CE, or reference-KL
    component and therefore needs to be shipped.
    """
    if sample.advantages is None:
        return True

    if sample.rl_weights is None:
        rl_weights = [1.0 if trainable else 0.0 for trainable in sample.mask]
    else:
        rl_weights = list(sample.rl_weights)

    changed = False
    for index, (trainable, advantage, weight) in enumerate(
        zip(sample.mask, sample.advantages, rl_weights, strict=True)
    ):
        if trainable and advantage == 0.0 and weight != 0.0:
            rl_weights[index] = 0.0
            changed = True

    if not changed:
        return True

    sample.rl_weights = rl_weights
    has_rl = any(trainable and weight != 0.0 for trainable, weight in zip(sample.mask, rl_weights, strict=True))
    has_ce = sample.ce_weights is not None and any(weight != 0.0 for weight in sample.ce_weights)
    has_ref_kl = sample.ref_kl_weights is not None and any(weight != 0.0 for weight in sample.ref_kl_weights)
    return has_rl or has_ce or has_ref_kl


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
        on_result: Callable[[list[Rollout]], bool] | None = None,
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
        self.on_result = on_result

        # Observation window for the next shipped batch: rollouts of groups
        # finalized since the last ship (errored + rejected + admitted).
        # In-progress groups stay out until they finalize.
        self.pending_rollouts: TrainRollouts = TrainRollouts()
        # Keyed by the dispatcher's group UUID. ``(env_name, task_idx)``
        # isn't unique — the same task can be re-sampled while an
        # earlier group is still in flight
        self.pending_groups: dict[uuid.UUID, list[Rollout]] = defaultdict(list)
        # Episodes arrived per group — the finalization count (an episode may
        # add several traces to ``pending_groups`` but counts once here).
        self.pending_group_episodes: dict[uuid.UUID, int] = defaultdict(int)
        self.pending_batch: list[Rollout] = []
        # Running payload-token total of ``pending_batch`` (token-batched
        # runs), kept in sync on append/pop so the readiness check never
        # re-sums per arrival.
        self.pending_tokens: int = 0
        # Finalized work since the most recent positive contribution, measured
        # in the active batch unit.
        self.zero_output_units: int = 0
        self.reported_zero_output_windows: int = 0

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
        """Episodes that have arrived but sit in not-yet-complete groups —
        buffered in the sink ahead of the batch."""
        return sum(self.pending_group_episodes.values())

    def pending_batch_by_env(self) -> dict[str, int]:
        """Per-env breakdown of ``batch_progress()`` (``pending_batch`` only);
        values sum to the aggregate."""
        counts: dict[str, int] = defaultdict(int)
        for r in self.pending_batch:
            counts[r.env_name] += 1
        return dict(counts)

    async def add(self, episode: list[Rollout]) -> TrainBatch | None:
        """Process one episode arrival; finalize the group on the
        ``group_size``-th episode; return a ``TrainBatch`` if the finalization
        pushed (or left) the batch over its threshold. Arrivals into
        still-incomplete groups never ship a batch."""
        group_id = episode[0].group_id
        env_name = episode[0].env_name
        for rollout in episode:
            await self.process_rollout(rollout)
        self.pending_groups[group_id].extend(episode)
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
        """Finalize one group, ask its curriculum for admission, and queue it."""
        group = self.pending_groups.pop(group_id, [])
        self.pending_group_episodes.pop(group_id, None)
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

        env = self.train_envs.get(env_name)
        # Untrainable traces carry no samples and must not skew the group baseline.
        survivors = [r for r in survivors if r.agent.trainable]
        if not survivors:
            self._admit(group)
            self._record_zero_output(group)
            get_logger().debug(
                f"Finished group | env={env_name} task_idx={task_idx} | "
                f"rollouts={len(group)} (errored={num_errored}) | dropped: no trainable survivors"
            )
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

        if not self._admit(group):
            self._record_zero_output(group)
            get_logger().debug(
                f"Finished group | env={env_name} task_idx={task_idx} | "
                f"rollouts={len(group)} (errored={num_errored}) | rejected by curriculum"
            )
            return

        for r in survivors:
            self.pending_batch.append(r)
            if self.token_batch_size is not None:
                self.pending_tokens += payload_tokens(r)
        self.zero_output_units = 0
        self.reported_zero_output_windows = 0

        rewards = [r.reward for r in survivors]
        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
        get_logger().debug(
            f"Finished group | env={env_name} task_idx={task_idx} | "
            f"rollouts={len(group)} (errored={num_errored}) | reward={avg_reward:.4f}"
        )

    def _admit(self, group: list[Rollout]) -> bool:
        admitted = self.on_result(group) if self.on_result is not None else True
        for rollout in group:
            rollout.is_admitted = admitted
        return admitted

    def _record_zero_output(self, group: list[Rollout]) -> None:
        if self.batch_size is not None:
            self.zero_output_units += len(group)
        else:
            payload = sum(payload_tokens(rollout) for rollout in group)
            self.zero_output_units += payload or self.config.seq_len * len(group)
        self._check_zero_output_budget()

    def _check_zero_output_budget(self) -> None:
        target = self.batch_size if self.batch_size is not None else self.token_batch_size
        assert target is not None
        windows = self.zero_output_units // target
        if windows <= self.reported_zero_output_windows:
            return
        self.reported_zero_output_windows = windows
        get_logger().warning(
            f"No admitted train payload after {self.zero_output_units} finalized units "
            f"(consecutive zero-output batch equivalents: "
            f"{windows}/{MAX_CONSECUTIVE_ZERO_OUTPUT_BATCH_EQUIVALENTS})"
        )
        if windows >= MAX_CONSECUTIVE_ZERO_OUTPUT_BATCH_EQUIVALENTS:
            raise RuntimeError(
                f"{windows} consecutive zero-output batch equivalents — "
                "check the curriculum admission policy and task difficulty."
            )

    def process_batch(self) -> TrainBatch:
        """Pop a cohort off ``pending_batch`` (by rollout count when
        ``batch_size`` is set, by token count when ``token_batch_size`` is
        set), and assemble the trainer-bound ``TrainingSample`` list. Overflow
        stays for the next batch."""
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

        if self.config.train.filter_zero_advantages:
            for rollout in cohort:
                rollout.samples = [sample for sample in rollout.samples if _prune_zero_advantages(sample)]
        samples: list[TrainingSample] = [sample for rollout in cohort for sample in rollout.samples]

        # ``rollouts`` is the observation window — every rollout of every group finalized since the
        # last ship (errored + rejected + admitted) — while ``samples`` is the shipped cohort's
        # trainable payload. ``rollouts.effective`` / ``rollouts.metrics`` derive the clean subset +
        # metric views on demand. Reset the window only when the batch actually ships (non-empty
        # samples) — an empty batch is dropped unlogged by the orchestrator, so keep accumulating its
        # finalized groups (and any overflow) into the next shipped batch's window.
        rollouts = self.pending_rollouts
        if samples:
            self.pending_rollouts = TrainRollouts()
        return TrainBatch(rollouts=rollouts, samples=samples)
