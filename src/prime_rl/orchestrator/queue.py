"""Queue: compiled train groups waiting for the trainer.

The sink hands over one ``FinalizedGroup`` at a time. The queue keeps the traces
that may train, sweeps the ones that went stale, and cuts a ``TrainBatch`` when the
target is met. It also keeps the batch window's accounting — every episode, failure
and cancellation since the last cut — so the batch it emits describes the whole
window, not just the shipped cohort. Its metrics say which side of the pipeline
limits progress: a queue pinned at zero means rollouts cannot keep up, a queue
pinned at its target means the trainer is the constraint and staleness climbs.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Awaitable, Callable, Iterable

import verifiers.v1 as vf

from prime_rl.configs.orchestrator import QueueConfig
from prime_rl.orchestrator.metrics import TrainEpisodes
from prime_rl.orchestrator.types import DispatchFailure, FinalizedGroup, TrainBatch
from prime_rl.orchestrator.utils import episode_env_name, min_fresh_version, train_work
from prime_rl.transports.batch import TrainingSample
from prime_rl.utils.logger import get_logger


def payload_tokens(samples: list[TrainingSample], trace: vf.Trace | None = None) -> int:
    """Token cost of one trainer-bound trace."""
    return sum(len(sample.token_ids) for sample in samples) or (trace.num_total_tokens if trace is not None else 0)


def prune_zero_advantages(sample: TrainingSample) -> bool:
    """Remove zero-advantage tokens from the RL component; False when nothing trains."""
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


class Queue:
    def __init__(self, config: QueueConfig) -> None:
        self.config = config
        self._step: Callable[[], int] = lambda: 1
        self._on_batch: Callable[[TrainBatch], Awaitable[None]] | None = None

        self.pending_episodes = TrainEpisodes()
        self.pending_failures: list[DispatchFailure] = []
        self.pending_cancelled_attempts = 0
        self.pending_stale_attempts = 0
        self.pending_batch: dict[str, list[TrainingSample]] = {}
        self.episode_by_trace: dict[str, vf.Episode] = {}
        self.pending_tokens = 0
        self.stale_drops = 0
        self.total_stale_drops = 0
        # Queued traces only age when the step advances, so one full sweep per step.
        self._swept_step = 0
        self.zero_output_units = 0
        self.reported_zero_output_windows = 0

    def bind(self, *, step: Callable[[], int], on_batch: Callable[[TrainBatch], Awaitable[None]]) -> None:
        self._step = step
        self._on_batch = on_batch

    # ── inbound ────────────────────────────────────────────────────────────

    async def put(self, group: FinalizedGroup) -> None:
        """Account one finished group, queue its payload, and cut a batch when ready."""
        self.pending_failures.extend(group.failures)
        if group.cancellation is not None:
            self.pending_cancelled_attempts += group.cancellation.count
            if group.stale:
                self.pending_stale_attempts += group.cancellation.count

        if group.stale:
            # Every member shares the dispatch version, so the arrived episodes are
            # exactly as stale as the cancelled tail.
            self.pending_episodes.extend(group.episodes, admitted=False, cancelled=True)
            self._record_zero_output(group, [])
        elif not group.samples:
            self.pending_episodes.extend(group.episodes, admitted=group.admitted)
            self._record_zero_output(group, group.survivors)
        else:
            self.pending_episodes.extend(group.episodes, sampled_trace_ids=set(group.samples), admitted=True)
            self.pending_batch.update(group.samples)
            for episode in group.episodes:
                for trace in episode.traces:
                    if trace.id in group.samples:
                        self.episode_by_trace[trace.id] = episode
            if self.config.token_batch_size is not None:
                self.pending_tokens += sum(
                    payload_tokens(samples, self._trace(trace_id)) for trace_id, samples in group.samples.items()
                )
            # A group's traces share one dispatch version, so the insertion sweep voids
            # all or none of them. A fully-voided group shipped nothing: advance the
            # zero-output tally instead of resetting it.
            self._drop_stale(group.samples)
            if any(trace_id in self.pending_batch for trace_id in group.samples):
                self.zero_output_units = 0
                self.reported_zero_output_windows = 0
            else:
                self._record_zero_output(group, [])

        self._drop_stale()
        if self._ready():
            assert self._on_batch is not None, "Queue.on_batch is not bound"
            await self._on_batch(self.cut())

    # ── internals ──────────────────────────────────────────────────────────

    def _ready(self) -> bool:
        if self.config.batch_size is not None:
            return len(self.pending_batch) >= self.config.batch_size
        return self.pending_tokens >= (self.config.token_batch_size or 0)

    def _drop_stale(self, trace_ids: Iterable[str] | None = None) -> None:
        """Void queued traces past ``max_off_policy_steps``. The batch being collected
        is ``step`` and trains v{step-1}, so a trace generated from v{k} would ship at
        staleness ``(step-1) - k``. This sweep is the hard guarantee on trained
        staleness; the dispatcher's in-flight cancel only saves compute. ``trace_ids``
        scopes the check to a freshly inserted group. Frozen-sourced episodes (no
        policy span) never go stale."""
        step = self._step()
        if trace_ids is None:
            if self._swept_step == step:
                return
            self._swept_step = step
            trace_ids = list(self.pending_batch)
        min_version = min_fresh_version(step, self.config.max_off_policy_steps)
        if min_version <= 0:
            return
        dropped = 0
        for trace_id in trace_ids:
            episode = self.episode_by_trace[trace_id]
            policy = train_work(episode).policy
            if policy is None or policy.start >= min_version:
                continue
            samples = self.pending_batch.pop(trace_id)
            if self.config.token_batch_size is not None:
                self.pending_tokens -= payload_tokens(samples, self._trace(trace_id))
            del self.episode_by_trace[trace_id]
            self.pending_episodes.cancelled.add(episode.id)
            dropped += 1
        if dropped:
            self.stale_drops += dropped
            self.total_stale_drops += dropped
            get_logger().warning(
                f"Dropped {dropped} queued traces past max_off_policy_steps={self.config.max_off_policy_steps}. "
                "Consider increasing it to avoid this."
            )

    def _trace(self, trace_id: str) -> vf.Trace:
        episode = self.episode_by_trace[trace_id]
        return next(trace for trace in episode.traces if trace.id == trace_id)

    def _record_zero_output(self, group: FinalizedGroup, survivors: list[vf.Trace]) -> None:
        """``group.owed`` counts the full episode budget (arrived + cancelled), so
        dropped groups advance the tally at the same rate as delivered ones."""
        if self.config.batch_size is not None:
            returned_traces = sum(len(episode.traces) for episode in group.episodes)
            self.zero_output_units += len(survivors) or returned_traces or group.owed
        else:
            survivor_tokens = sum(trace.num_total_tokens for trace in survivors)
            episode_tokens = sum(episode.num_total_tokens for episode in group.episodes)
            self.zero_output_units += survivor_tokens or episode_tokens or self.config.seq_len * group.owed
        target = self.target
        windows = self.zero_output_units // target
        if windows <= self.reported_zero_output_windows:
            return
        self.reported_zero_output_windows = windows
        get_logger().warning(
            f"No admitted train payload after {self.zero_output_units} finalized units "
            f"({windows} zero-output batch equivalents)"
        )

    def cut(self) -> TrainBatch:
        items = list(self.pending_batch.items())
        if self.config.batch_size is not None:
            selected = items[: self.config.batch_size]
        else:
            assert self.config.token_batch_size is not None
            cut = 0
            running = 0
            for index, (trace_id, samples) in enumerate(items):
                running += payload_tokens(samples, self._trace(trace_id))
                cut = index + 1
                if running >= self.config.token_batch_size:
                    break
            selected = items[:cut]
            self.pending_tokens -= running

        selected_by_trace = dict(selected)
        selected_ids = set(selected_by_trace)
        for trace_id in selected_ids:
            del self.pending_batch[trace_id]

        if not self.config.constant_trainer_batch_size:
            selected_by_trace = {
                trace_id: [sample for sample in samples if prune_zero_advantages(sample)]
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

        batch = TrainBatch(
            episodes=self.pending_episodes,
            cohort=cohort,
            samples=samples,
            failures=self.pending_failures,
            buffered_episode_ids=buffered_episode_ids,
            cancelled_attempts=self.pending_cancelled_attempts,
            stale_attempts=self.pending_stale_attempts,
            stale_drops=self.stale_drops,
        )
        if samples:
            self.pending_episodes = TrainEpisodes()
            self.pending_failures = []
            self.pending_cancelled_attempts = 0
            self.pending_stale_attempts = 0
            self.stale_drops = 0
        return batch

    # ── observability ──────────────────────────────────────────────────────

    @property
    def target(self) -> int:
        target = self.config.batch_size if self.config.batch_size is not None else self.config.token_batch_size
        assert target is not None
        return target

    @property
    def size(self) -> int:
        """Queued units toward the target: traces, or tokens under token batching."""
        return len(self.pending_batch) if self.config.batch_size is not None else self.pending_tokens

    def by_env(self) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for trace_id in self.pending_batch:
            counts[episode_env_name(self.episode_by_trace[trace_id])] += 1
        return dict(counts)

    def staleness(self) -> list[int]:
        """Staleness each queued trace would ship at in the batch being collected."""
        step = self._step()
        out = []
        for episode in self.episode_by_trace.values():
            policy = train_work(episode).policy
            out.append(max(0, (step - 1) - policy.start) if policy is not None else 0)
        return out

    def status(self) -> str:
        size, target = self.size, self.target
        part = f"Train batch {size}/{target} ({size / target:.1%})"
        by_env = self.by_env()
        if len(by_env) > 1:
            part += " (" + ", ".join(f"{name}={count}" for name, count in sorted(by_env.items())) + ")"
        return part

    def gauges(self) -> dict[str, float]:
        staleness = self.staleness()
        return {
            "queue/size": float(self.size),
            "queue/traces": float(len(self.pending_batch)),
            "queue/tokens": float(self.pending_tokens),
            "queue/fill": self.size / self.target,
            "queue/staleness/max": float(max(staleness, default=0)),
            "queue/staleness/mean": sum(staleness) / len(staleness) if staleness else 0.0,
            "queue/dropped_stale": float(self.total_stale_drops),
        }
