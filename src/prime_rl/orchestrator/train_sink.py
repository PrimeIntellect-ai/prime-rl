"""Training-side episode, group, and batch assembly.

``add()`` takes one completed episode and ``cancel()`` a dropped group's
``GroupCancellation``; both return ``TrainBatch | None``. Before every readiness
check the sink sweeps ``pending_batch`` for traces past ``max_off_policy_steps`` —
this sweep, not the dispatcher's in-flight cancel, is what guarantees nothing
stale ships."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Callable, Iterable

import verifiers.v1 as vf

from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.orchestrator.algo.base import iter_trainable_traces
from prime_rl.orchestrator.algo.routing import stamp_loss_routing
from prime_rl.orchestrator.envs import TrainEnvs
from prime_rl.orchestrator.metrics import TrainEpisodes
from prime_rl.orchestrator.trajectories import trace_to_samples
from prime_rl.orchestrator.types import GroupCancellation, Progress, TrainBatch
from prime_rl.orchestrator.utils import episode_env_name, episode_group_id, min_fresh_version, train_work
from prime_rl.transports.rollouts import TrainingSample
from prime_rl.utils.logger import get_logger

MAX_CONSECUTIVE_ZERO_OUTPUT_BATCH_EQUIVALENTS = 10


def payload_tokens(samples: list[TrainingSample], trace: vf.Trace | None = None) -> int:
    """Token cost of one trainer-bound trace."""
    return sum(len(sample.token_ids) for sample in samples) or (trace.num_total_tokens if trace is not None else 0)


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


class TrainSink:
    """Score native episodes, admit groups, then compile trainer payloads."""

    def __init__(
        self,
        config: OrchestratorConfig,
        *,
        tokenizer,
        train_envs: TrainEnvs,
        progress: Progress,
        batch_size: int | None,
        token_batch_size: int | None,
        on_result: Callable[[list[vf.Episode]], bool] | None = None,
    ) -> None:
        assert (batch_size is None) != (token_batch_size is None), (
            "Exactly one of batch_size / token_batch_size must be set"
        )
        self.config = config
        self.tokenizer = tokenizer
        self.train_envs = train_envs
        self.progress = progress
        self.batch_size = batch_size
        self.token_batch_size = token_batch_size
        self.on_result = on_result

        self.pending_episodes = TrainEpisodes()
        self.pending_groups: dict[str, list[vf.Episode]] = defaultdict(list)
        # A dropped group's terminal marker; its ``count`` fills in for the
        # episodes the group will never deliver.
        self.pending_group_cancellations: dict[str, GroupCancellation] = {}
        self.pending_batch: dict[str, list[TrainingSample]] = {}
        self.episode_by_trace: dict[str, vf.Episode] = {}
        self.pending_tokens = 0
        # Queued traces voided by the staleness sweep since the last ship;
        # read and reset by the orchestrator's per-step metrics.
        self.stale_drops = 0
        # Step of the last full staleness sweep — queued traces only age when
        # ``progress.step`` advances, so one full sweep per step suffices.
        self._swept_step = 0
        self.zero_output_units = 0
        self.reported_zero_output_windows = 0

    def group_size_for(self, env_name: str) -> int:
        return self.train_envs.get(env_name).config.group_size

    def batch_progress(self) -> tuple[int, int, str]:
        if self.batch_size is not None:
            return len(self.pending_batch), self.batch_size, "traces"
        assert self.token_batch_size is not None
        return self.pending_tokens, self.token_batch_size, "tokens"

    def buffered_count(self) -> int:
        return sum(len(group) for group in self.pending_groups.values())

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
        env_name = episode_env_name(episode)
        self.pending_groups[group_id].append(episode)
        if not self._group_complete(group_id, env_name):
            return None
        await self.process_group(group_id)
        return self._maybe_batch()

    async def cancel(self, cancellation: GroupCancellation) -> TrainBatch | None:
        """Process a dropped group's terminal marker: its ``count`` completes
        the group's episode accounting so finalization still fires. A
        ``stale`` drop also voids the group's already-arrived episodes in
        ``process_group`` — they share the group's dispatch version, so they
        are equally stale."""
        self.pending_group_cancellations[cancellation.group_id] = cancellation
        if not self._group_complete(cancellation.group_id, cancellation.env_name):
            return None
        await self.process_group(cancellation.group_id)
        return self._maybe_batch()

    def _group_complete(self, group_id: str, env_name: str) -> bool:
        cancellation = self.pending_group_cancellations.get(group_id)
        cancelled = cancellation.count if cancellation is not None else 0
        return len(self.pending_groups[group_id]) + cancelled >= self.group_size_for(env_name)

    def _maybe_batch(self) -> TrainBatch | None:
        """Sweep stale queued traces, then cut a batch if the survivors still
        meet the threshold."""
        self._drop_stale()
        ready = (
            len(self.pending_batch) >= self.batch_size
            if self.batch_size is not None
            else self.pending_tokens >= (self.token_batch_size or 0)
        )
        return self.process_batch() if ready else None

    def _drop_stale(self, trace_ids: Iterable[str] | None = None) -> None:
        """Void queued traces past ``max_off_policy_steps``. The batch being
        collected is ``progress.step`` and trains policy v{step-1}, so a
        queued trace generated from v{k} would ship at staleness
        ``(step-1) - k``. This sweep is the hard guarantee on trained
        staleness; the dispatcher's in-flight cancel only saves compute.
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
        min_version = min_fresh_version(self.progress.step, self.config.max_off_policy_steps)
        if min_version <= 0:
            return
        dropped = 0
        for trace_id in trace_ids:
            episode = self.episode_by_trace[trace_id]
            policy = train_work(episode).policy
            if policy is None or policy.start >= min_version:
                continue
            samples = self.pending_batch.pop(trace_id)
            if self.token_batch_size is not None:
                self.pending_tokens -= payload_tokens(samples, self._trace(trace_id))
            del self.episode_by_trace[trace_id]
            self.pending_episodes.cancelled.add(episode.id)
            dropped += 1
        if dropped:
            self.stale_drops += dropped
            get_logger().warning(
                f"Dropped {dropped} queued traces past max_off_policy_steps={self.config.max_off_policy_steps}. "
                "Consider increasing it to avoid this."
            )

    async def process_episode(self, episode: vf.Episode) -> None:
        """Run rollout-local algorithm work on one native episode."""
        env_name = episode_env_name(episode)
        await self.train_envs.get(env_name).algorithm.finalize_episode(episode)

    async def process_group(self, group_id: str) -> None:
        group = self.pending_groups.pop(group_id, [])
        cancellation = self.pending_group_cancellations.pop(group_id, None)
        if not group and cancellation is None:
            return

        env_name = episode_env_name(group[0]) if group else cancellation.env_name
        env = self.train_envs.get(env_name)
        traces = [trace for episode in group for trace in episode.traces]
        task_idx = next((trace.task.data.idx for trace in traces), None)
        num_errored = sum(trace.has_error for trace in traces) + sum(
            not episode.ok for episode in group if not episode.traces
        )
        n_owed = len(group) + (cancellation.count if cancellation is not None else 0)

        # A stale drop voids the whole group: every member shares the dispatch
        # version, so the arrived episodes are exactly as stale as the
        # cancelled tail. Stale groups bypass the curriculum — a pipeline
        # decision is not a task result.
        if cancellation is not None and cancellation.reason == "stale":
            self.pending_episodes.extend(group, admitted=False, cancelled=True)
            self._record_zero_output(group, [], n_owed)
            get_logger().debug(
                f"Finished group | env={env_name} task_idx={task_idx} | "
                f"episodes={len(group)} traces={len(traces)} (errored={num_errored}) | dropped: cancelled (stale)"
            )
            return

        survivors = [trace for _, trace in iter_trainable_traces(group)]
        if survivors:
            await env.algorithm.finalize_group(group)
        admitted = self._admit(group) if group else False
        if not survivors or not admitted:
            self.pending_episodes.extend(group, admitted=admitted)
            self._record_zero_output(group, survivors, n_owed)
            reason = "no trainable survivors" if not survivors else "rejected by curriculum"
            get_logger().debug(
                f"Finished group | env={env_name} task_idx={task_idx} | "
                f"episodes={len(group)} traces={len(traces)} (errored={num_errored}) | dropped: {reason}"
            )
            return

        samples_by_trace: dict[str, list[TrainingSample]] = {}
        temperature = env.sampling_args["temperature"]
        for trace in survivors:
            samples = await asyncio.to_thread(trace_to_samples, trace, env_name=env_name)
            for sample in samples:
                sample.temperatures = [temperature] * len(sample.token_ids)
                stamp_loss_routing(sample, env.algorithm.action_loss_type)
            if samples:
                samples_by_trace[trace.id] = samples

        self.pending_episodes.extend(group, sampled_trace_ids=set(samples_by_trace), admitted=True)
        if not samples_by_trace:
            self._record_zero_output(group, survivors, n_owed)
            return

        self.pending_batch.update(samples_by_trace)
        for episode in group:
            for trace in episode.traces:
                if trace.id in samples_by_trace:
                    self.episode_by_trace[trace.id] = episode
        if self.token_batch_size is not None:
            self.pending_tokens += sum(
                payload_tokens(samples, self._trace(trace_id)) for trace_id, samples in samples_by_trace.items()
            )
        self._drop_stale(samples_by_trace)
        # A group's traces share one dispatch version, so the insertion sweep
        # voids all or none of them. A fully-voided group shipped nothing —
        # advance the zero-output budget instead of resetting it, or a stalled
        # trainer plus a tight bound could void groups forever without ever
        # tripping the abort.
        if not any(trace_id in self.pending_batch for trace_id in samples_by_trace):
            self._record_zero_output(group, [], n_owed)
            return
        self.zero_output_units = 0
        self.reported_zero_output_windows = 0

        rewards = [trace.reward for trace in survivors if trace.id in samples_by_trace]
        avg_reward = sum(rewards) / len(rewards)
        get_logger().debug(
            f"Finished group | env={env_name} task_idx={task_idx} | "
            f"episodes={len(group)} traces={len(traces)} (errored={num_errored}) | reward={avg_reward:.4f}"
        )

    def _trace(self, trace_id: str) -> vf.Trace:
        episode = self.episode_by_trace[trace_id]
        return next(trace for trace in episode.traces if trace.id == trace_id)

    def _admit(self, group: list[vf.Episode]) -> bool:
        return self.on_result(group) if self.on_result is not None else True

    def _record_zero_output(self, group: list[vf.Episode], survivors: list[vf.Trace], n_owed: int) -> None:
        """``n_owed`` counts the group's full episode budget (arrived +
        cancelled), so dropped groups advance the zero-output budget at the
        same rate as fully-delivered ones."""
        if self.batch_size is not None:
            returned_traces = sum(len(episode.traces) for episode in group)
            self.zero_output_units += len(survivors) or returned_traces or n_owed
        else:
            survivor_tokens = sum(trace.num_total_tokens for trace in survivors)
            episode_tokens = sum(episode.num_total_tokens for episode in group)
            self.zero_output_units += survivor_tokens or episode_tokens or self.config.seq_len * n_owed
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
                "check the curriculum admission policy, task difficulty, and staleness drops."
            )

    def process_batch(self) -> TrainBatch:
        items = list(self.pending_batch.items())
        if self.batch_size is not None:
            selected = items[: self.batch_size]
        else:
            assert self.token_batch_size is not None
            cut = 0
            running = 0
            for index, (trace_id, samples) in enumerate(items):
                running += payload_tokens(samples, self._trace(trace_id))
                cut = index + 1
                if running >= self.token_batch_size:
                    break
            selected = items[:cut]
            self.pending_tokens -= running

        selected_by_trace = dict(selected)
        selected_ids = set(selected_by_trace)
        for trace_id in selected_ids:
            del self.pending_batch[trace_id]

        if self.config.train.filter_zero_advantages:
            selected_by_trace = {
                trace_id: [sample for sample in samples if _prune_zero_advantages(sample)]
                for trace_id, samples in selected
            }
            selected_by_trace = {trace_id: samples for trace_id, samples in selected_by_trace.items() if samples}
        samples = [sample for trace_samples in selected_by_trace.values() for sample in trace_samples]

        shipped_ids = set(selected_by_trace)
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
        if samples:
            self.pending_episodes = TrainEpisodes()
        return TrainBatch(episodes=episodes, cohort=cohort, samples=samples)
