"""Training-side episode, group, and batch assembly.

``add()`` takes one completed episode, ``fail()`` a request that produced no
episode, and ``cancel()`` a ``GroupCancellation`` for some of a group's
episodes. Before every readiness check the sink sweeps ``pending_batch`` for
traces past ``max_off_policy_steps`` — this sweep, not the dispatcher's
in-flight cancel, is what guarantees nothing stale ships. Staleness is per
episode: a stale episode is dropped alone, and its group is scored from the
members that remain.

A group is scored once all its members are accounted for. For ``grpo``-family
algorithms, the staleness failsafe scores earlier: once a finished fresh
member reaches ``max_off_policy_steps`` (its last admissible batch) and at
least ``failsafe_min_members`` members have finished, the group is scored over
every finished member (earlier waves and stale members included), and the
members at the bound are queued at the front of ``pending_batch`` so the next
cut takes them. Younger finished members wait for the group to complete or
for their own bound."""

from __future__ import annotations

import asyncio
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field

import verifiers.v1 as vf

from prime_rl.configs.algorithm import GRPOAlgoConfig
from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.orchestrator.algo.base import iter_trainable_traces
from prime_rl.orchestrator.algo.routing import stamp_loss_routing
from prime_rl.orchestrator.envs import TrainEnvs
from prime_rl.orchestrator.metrics import TrainEpisodes
from prime_rl.orchestrator.train_source import TrainSource
from prime_rl.orchestrator.trajectories import trace_to_samples
from prime_rl.orchestrator.types import DispatchFailure, GroupCancellation, Progress, TrainBatch
from prime_rl.orchestrator.utils import episode_env_name, episode_group_id, rollout_age, train_work
from prime_rl.transports.batch import TrainingSample
from prime_rl.utils.logger import get_logger


@dataclass
class ScoredGroup:
    """A group the failsafe split into waves, while members are still owed."""

    members: list[vf.Episode] = field(default_factory=list)
    """Members scored in earlier waves, or stale; later waves are scored with them."""
    accounted: int = 0
    """Members delivered, failed, or cancelled in earlier waves."""


def _prune_zero_advantages(sample: TrainingSample) -> bool:
    """Remove zero-advantage tokens from the RL component."""
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


def _trainable(episode: vf.Episode) -> bool:
    return any(True for _ in iter_trainable_traces([episode]))


class TrainSink:
    """Score native episodes, admit groups, then compile trainer payloads."""

    def __init__(
        self,
        config: OrchestratorConfig,
        *,
        tokenizer,
        train_envs: TrainEnvs,
        progress: Progress,
        batch_size: int,
        train_source: TrainSource,
    ) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self.train_envs = train_envs
        self.progress = progress
        self.batch_size = batch_size
        self.train_source = train_source

        self.pending_episodes = TrainEpisodes()
        self.pending_failures: list[DispatchFailure] = []
        self.pending_cancelled_attempts = 0
        self.pending_stale_attempts = 0
        self.pending_groups: dict[str, list[vf.Episode]] = defaultdict(list)
        self.pending_group_failures: dict[str, list[DispatchFailure]] = defaultdict(list)
        # Their ``count``s fill in for the episodes the group will never deliver.
        self.pending_group_cancellations: dict[str, list[GroupCancellation]] = defaultdict(list)
        self.scored_groups: dict[str, ScoredGroup] = {}
        self.pending_batch: dict[str, list[TrainingSample]] = {}
        self.episode_by_trace: dict[str, vf.Episode] = {}
        # Queued traces voided by the staleness sweep since the last ship;
        # read and reset by the orchestrator's per-step metrics.
        self.stale_drops = 0
        # Step of the last full staleness sweep — queued traces only age when
        # ``progress.step`` advances, so one full sweep per step suffices.
        self._swept_step = 0
        self._failsafe_swept_step = 0
        # Read and reset by the orchestrator's per-step metrics.
        self.event_counts: Counter[str] = Counter()
        self.zero_output_units = 0
        self.reported_zero_output_windows = 0

    def group_size_for(self, env_name: str) -> int:
        return self.train_envs.get(env_name).config.group_size

    def batch_progress(self) -> tuple[int, int]:
        return len(self.pending_batch), self.batch_size

    def buffered_count(self) -> int:
        episodes = sum(len(group) for group in self.pending_groups.values())
        failures = sum(len(group) for group in self.pending_group_failures.values())
        return episodes + failures

    def pending_batch_by_env(self) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for trace_id in self.pending_batch:
            episode = self.episode_by_trace[trace_id]
            counts[episode_env_name(episode)] += 1
        return dict(counts)

    async def add(self, episode: vf.Episode) -> TrainBatch | None:
        """Process one completed episode and return a batch when ready."""
        await self.process_episode(episode)
        group_id = episode_group_id(episode)
        self.pending_groups[group_id].append(episode)
        return await self._settle(group_id, episode_env_name(episode))

    async def cancel(self, cancellation: GroupCancellation) -> TrainBatch | None:
        """Process a cancellation marker: its ``count`` completes the group's
        episode accounting so finalization still fires."""
        self.pending_group_cancellations[cancellation.group_id].append(cancellation)
        return await self._settle(cancellation.group_id, cancellation.env_name)

    async def fail(self, failure: DispatchFailure) -> TrainBatch | None:
        """Count a request failure toward its group without presenting it as
        an episode to the algorithm or curriculum."""
        if failure.kind != "train":
            raise ValueError(f"TrainSink cannot process a {failure.kind} dispatch failure")
        self.pending_group_failures[failure.group_id].append(failure)
        return await self._settle(failure.group_id, failure.env_name)

    async def _settle(self, group_id: str, env_name: str) -> TrainBatch | None:
        """Score the group if it is complete or the failsafe is due, sweep the
        other groups for the failsafe, and return a batch when ready."""
        processed = True
        if self._group_complete(group_id, env_name):
            await self.process_group(group_id)
        elif self._failsafe_due(group_id, env_name):
            await self.process_group(group_id, failsafe=True)
        else:
            processed = False
        if not await self._sweep_failsafe() and not processed:
            return None
        return self._maybe_batch()

    def _group_complete(self, group_id: str, env_name: str) -> bool:
        cancelled = sum(cancellation.count for cancellation in self.pending_group_cancellations[group_id])
        failed = len(self.pending_group_failures[group_id])
        scored = self.scored_groups.get(group_id)
        accounted = scored.accounted if scored is not None else 0
        return accounted + len(self.pending_groups[group_id]) + failed + cancelled >= self.group_size_for(env_name)

    def _failsafe_due(self, group_id: str, env_name: str) -> bool:
        """Whether a finished member of the group is at ``max_off_policy_steps``
        and at least ``failsafe_min_members`` members have finished (fresh,
        stale, or scored in earlier waves)."""
        algo = self.train_envs.get(env_name).config.algo
        if not isinstance(algo, GRPOAlgoConfig):
            return False
        state = self.scored_groups.get(group_id)
        pending = [episode for episode in self.pending_groups[group_id] if _trainable(episode)]
        earlier = sum(_trainable(episode) for episode in state.members) if state is not None else 0
        return len(pending) + earlier >= algo.failsafe_min_members and any(
            self._age(episode) == self.config.max_off_policy_steps for episode in pending
        )

    async def _sweep_failsafe(self) -> bool:
        """Fire the failsafe for groups that aged into it and return whether
        any did. Ages only move with ``progress.step``, so this runs once per
        step, before the step's first cut; arrivals are checked in
        ``_settle``."""
        if self._failsafe_swept_step == self.progress.step:
            return False
        self._failsafe_swept_step = self.progress.step
        due = [
            group_id
            for group_id, group in self.pending_groups.items()
            if group and self._failsafe_due(group_id, episode_env_name(group[0]))
        ]
        for group_id in due:
            await self.process_group(group_id, failsafe=True)
        return bool(due)

    def _maybe_batch(self) -> TrainBatch | None:
        """Sweep stale queued traces, then cut a batch if the survivors still
        meet the threshold."""
        self._drop_stale()
        return self.process_batch() if len(self.pending_batch) >= self.batch_size else None

    def _drop_stale(self, trace_ids: Iterable[str] | None = None) -> None:
        """Void queued traces past ``max_off_policy_steps``. This sweep is the
        hard guarantee on trained staleness; the dispatcher's in-flight cancel
        only saves compute.
        Queued traces only age when ``progress.step`` advances, so the full
        sweep runs once per step; ``trace_ids`` scopes the check to a freshly
        inserted group, whose traces may already be stale on arrival.
        Frozen-sourced episodes (no policy span) never go stale.

        A swept trace whose window already shipped is visible only in
        ``stale_drops`` — its episode was reported with that window, and
        re-observing it would double-count its stats."""
        if trace_ids is None:
            if self._swept_step == self.progress.step:
                return
            self._swept_step = self.progress.step
            trace_ids = list(self.pending_batch)
        dropped = 0
        for trace_id in trace_ids:
            episode = self.episode_by_trace[trace_id]
            if not self._is_stale(episode):
                continue
            del self.pending_batch[trace_id]
            del self.episode_by_trace[trace_id]
            self.pending_episodes.cancelled.add(episode.id)
            dropped += 1
        if dropped:
            self.stale_drops += dropped
            get_logger().warning(
                f"Dropped {dropped} queued traces past max_off_policy_steps={self.config.max_off_policy_steps}. "
                "Consider increasing it to avoid this."
            )

    def _age(self, episode: vf.Episode) -> int:
        policy = train_work(episode).policy
        return 0 if policy is None else rollout_age(policy.start, self.progress.step)

    def _is_stale(self, episode: vf.Episode) -> bool:
        return self._age(episode) > self.config.max_off_policy_steps

    async def process_episode(self, episode: vf.Episode) -> None:
        """Run rollout-local algorithm work on one native episode."""
        env_name = episode_env_name(episode)
        await self.train_envs.get(env_name).algorithm.finalize_episode(episode)

    async def process_group(self, group_id: str, *, failsafe: bool = False) -> None:
        group = self.pending_groups.pop(group_id, [])
        failures = self.pending_group_failures.pop(group_id, [])
        cancellations = self.pending_group_cancellations.pop(group_id, [])
        if not group and not failures and not cancellations:
            return

        env_name = (
            episode_env_name(group[0]) if group else (failures[0].env_name if failures else cancellations[0].env_name)
        )
        env = self.train_envs.get(env_name)
        traces = [trace for episode in group for trace in episode.traces]
        task_idx = next((trace.task.data.idx for trace in traces), None)
        num_errored = (
            sum(trace.has_error for trace in traces)
            + sum(not episode.ok for episode in group if not episode.traces)
            + len(failures)
        )
        cancelled = sum(cancellation.count for cancellation in cancellations)
        n_owed = len(group) + len(failures) + cancelled
        self.pending_failures.extend(failures)
        self.pending_cancelled_attempts += cancelled
        self.pending_stale_attempts += sum(c.count for c in cancellations if c.reason == "stale")

        # Stale members are left out like errored ones, before the algorithm
        # and the curriculum see the group.
        stale = [episode for episode in group if self._is_stale(episode)]
        group = [episode for episode in group if not self._is_stale(episode)]
        self.pending_episodes.extend(stale, admitted=False, cancelled=True)
        survivors = [trace for _, trace in iter_trainable_traces(group)]
        state = self.scored_groups.pop(group_id, None)
        first = state is None
        if first and not failsafe:
            if survivors:
                await env.algorithm.finalize_group(group)
        else:
            state = state or ScoredGroup()
            if survivors:
                await env.algorithm.finalize_group(state.members + stale + group)
            if failsafe:
                # Only members at the bound ship now. Younger ones wait for the
                # group to complete, or for the failsafe at their own bound.
                waiting = [e for e in group if self._age(e) < self.config.max_off_policy_steps]
                if waiting:
                    self.pending_groups[group_id] = waiting
                group = [e for e in group if self._age(e) == self.config.max_off_policy_steps]
                survivors = [trace for _, trace in iter_trainable_traces(group)]
                n_owed -= len(waiting)
            state.members += stale + group
            state.accounted += n_owed
            if state.accounted < self.group_size_for(env_name):
                self.scored_groups[group_id] = state
            if first:
                self.event_counts["failsafe/groups_triggered"] += 1
        if first and not failsafe:
            admitted = bool(group) and self.train_source.on_result(group)
        else:
            # Early waves only pass the curriculum's gates; its sampler sees
            # the group once, with every finished member, when it completes.
            admitted = bool(group) and self.train_source.on_result(group, observe=False)
            if not failsafe:
                self.train_source.observe(state.members)
        if not survivors or not admitted:
            self.pending_episodes.extend(group, admitted=admitted)
            self._record_zero_output(group, survivors, n_owed)
            reason = "no trainable survivors" if not survivors else "rejected by curriculum"
            get_logger().debug(
                f"Dropped group | env={env_name} task_idx={task_idx} | "
                f"episodes={len(group)} traces={len(traces)} (errored={num_errored}) | reason={reason}"
            )
            return

        samples_by_trace: dict[str, list[TrainingSample]] = {}
        temperature = env.sampling_args["temperature"]
        for trace in survivors:
            samples = await asyncio.to_thread(trace_to_samples, trace, env_name=env_name)
            for sample in samples:
                sample.temperatures = [temperature] * len(sample.token_ids)
                if env.requires_sampling_masks and sample.sampling_mask is None:
                    # Rollout logprobs are mask-renormalized; training without the masks
                    # silently biases every importance ratio.
                    raise RuntimeError(
                        f"env '{env_name}' samples with truncation (top_p/top_k) but its rollouts "
                        "carry no sampling masks. Set `enable_return_sampling_mask = true` on "
                        "the inference server config (the rl entrypoint does this automatically) - "
                        "it requires vLLM's native sampling-mask capture (>= 0.28)."
                    )
                stamp_loss_routing(sample, env.algorithm.action_loss_type)
            if self.config.constant_trainer_batch_size:
                samples = [sample for sample in samples if _prune_zero_advantages(sample)]
            if samples:
                samples_by_trace[trace.id] = samples

        self.pending_episodes.extend(group, sampled_trace_ids=set(samples_by_trace), admitted=True)
        if not samples_by_trace:
            self._record_zero_output(group, survivors, n_owed)
            return

        if failsafe:
            # These members are at the bound: only the batch being collected can take them.
            self.pending_batch = samples_by_trace | self.pending_batch
        else:
            self.pending_batch.update(samples_by_trace)
        for episode in group:
            for trace in episode.traces:
                if trace.id in samples_by_trace:
                    self.episode_by_trace[trace.id] = episode
        self._drop_stale(samples_by_trace)
        # A fully-voided group shipped nothing — advance the zero-output tally
        # instead of resetting it, or a stalled trainer plus a tight bound
        # could void groups forever without ever surfacing the warning.
        if not any(trace_id in self.pending_batch for trace_id in samples_by_trace):
            self._record_zero_output(group, [], n_owed)
            return
        self.zero_output_units = 0
        self.reported_zero_output_windows = 0

    def _record_zero_output(self, group: list[vf.Episode], survivors: list[vf.Trace], n_owed: int) -> None:
        """``n_owed`` counts the group's full episode budget (arrived +
        cancelled), so dropped groups advance the zero-output tally at the
        same rate as fully-delivered ones."""
        returned_traces = sum(len(episode.traces) for episode in group)
        self.zero_output_units += len(survivors) or returned_traces or n_owed
        self._warn_zero_output()

    def _warn_zero_output(self) -> None:
        """Warn once per batch-equivalent of finalized units that shipped no
        payload, so a run that produces no training signal stays visible in the
        logs without aborting."""
        windows = self.zero_output_units // self.batch_size
        if windows <= self.reported_zero_output_windows:
            return
        self.reported_zero_output_windows = windows
        get_logger().warning(
            f"No admitted train payload after {self.zero_output_units} finalized units "
            f"({windows} zero-output batch equivalents)"
        )

    def process_batch(self) -> TrainBatch:
        selected = list(self.pending_batch.items())[: self.batch_size]

        selected_by_trace = dict(selected)
        selected_ids = set(selected_by_trace)
        for trace_id in selected_ids:
            del self.pending_batch[trace_id]

        if not self.config.constant_trainer_batch_size:
            selected_by_trace = {
                trace_id: [sample for sample in samples if _prune_zero_advantages(sample)]
                for trace_id, samples in selected
            }
            selected_by_trace = {trace_id: samples for trace_id, samples in selected_by_trace.items() if samples}
        samples = [sample for trace_samples in selected_by_trace.values() for sample in trace_samples]

        shipped_ids = set(selected_by_trace)
        buffered_episode_ids = {self.episode_by_trace[trace_id].id for trace_id in self.pending_batch}
        traces_by_episode: dict[int, list[vf.Trace]] = defaultdict(list)
        selected_episodes: dict[int, vf.Episode] = {}
        for trace_id in selected_ids:
            episode = self.episode_by_trace.pop(trace_id)
            if trace_id in shipped_ids:
                selected_episodes[id(episode)] = episode
                traces_by_episode[id(episode)].extend(trace for trace in episode.traces if trace.id == trace_id)
        cohort_episodes = [
            episode.model_copy(update={"traces": traces_by_episode[id(episode)]})
            for episode in selected_episodes.values()
        ]
        cohort = TrainEpisodes(cohort_episodes, sampled_trace_ids=shipped_ids)

        episodes = self.pending_episodes
        failures = self.pending_failures
        cancelled_attempts = self.pending_cancelled_attempts
        stale_attempts = self.pending_stale_attempts
        if samples:
            self.pending_episodes = TrainEpisodes()
            self.pending_failures = []
            self.pending_cancelled_attempts = 0
            self.pending_stale_attempts = 0
        return TrainBatch(
            episodes=episodes,
            cohort=cohort,
            samples=samples,
            failures=failures,
            buffered_episode_ids=buffered_episode_ids,
            cancelled_attempts=cancelled_attempts,
            stale_attempts=stale_attempts,
        )
