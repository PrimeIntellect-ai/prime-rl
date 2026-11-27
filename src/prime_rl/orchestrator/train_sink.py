"""Training-side group finalization, scalar accounting, and batch assembly.

``add()`` takes one completed episode, ``fail()`` a request that produced no
episode, and ``cancel()`` a dropped group's ``GroupCancellation``. Before every
readiness check the sink sweeps ``pending_batch`` for traces past
``max_staleness`` — this sweep is what guarantees no stale token trains. Completed groups become numeric
observations; only queued traces retain verifier objects. The consumer drains
``take_rollout_window(force=True)`` before shipping a batch.

The sink never holds a sample's tokens: an env worker keeps each sample, and the trace's
info names it with a summary (``prime_rl.transports.payload``). The sink queues references
(``QueuedSample``) and decides from the summaries which samples still train; the trainer
applies the same decisions to the tokens."""

from __future__ import annotations

import asyncio
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.control.types import ConfigUpdate, Reconfigurable
from prime_rl.orchestrator.algo.base import iter_trainable_traces
from prime_rl.orchestrator.credit import branch_token_advantages, token_advantages, trace_advantage, trace_rl_weight
from prime_rl.orchestrator.envs import TrainEnvs
from prime_rl.orchestrator.metrics import RolloutWindow, TrainEpisodes
from prime_rl.orchestrator.types import DispatchFailure, GroupCancellation, Progress, TrainBatch
from prime_rl.orchestrator.utils import (
    episode_env_name,
    episode_group_id,
    min_fresh_version,
    train_work,
)
from prime_rl.transports.batch.materialize import mm_cut
from prime_rl.transports.batch.types import PAYLOAD_INFO_KEY, UNLABELED, UNVERSIONED, SampleRef
from prime_rl.transports.payload import delete_payloads, sample_payload_urls, trace_payload_urls, write_advantages
from prime_rl.utils.logger import get_logger

if TYPE_CHECKING:
    from prime_rl.env.heads import EpisodeHead, TraceHead

# Training samples reach the trainer only through env workers that keep them.
_KEEP_SAMPLES_HINT = (
    "Serve the env with prime-rl's `env-worker-pool` entrypoint, which keeps them for the trainer, "
    "and check its log for arrays it could not keep."
)


@dataclass
class QueuedSample:
    """A sample by reference, with how many of its tokens each policy version sampled: its
    trainable tokens, and the tokens of each loss component (``rl``, ``ce``, ``ref_kl``)."""

    ref: SampleRef
    trainable: dict[int, int]
    loss: dict[str, dict[int, int]]
    pruned: bool = False
    """Zero-advantage tokens are out of its rl component."""
    zero_advantage: bool = False
    """Its every token has advantage 0.0, so pruning takes it out of the rl component whole."""

    def is_fresh(self, version: int, min_version: int | None, max_version: int | None = None) -> bool:
        """Whether this sample's tokens of ``version`` train in a batch whose policies lie in
        ``[min_version, max_version]`` (None: unbounded on that side)."""
        if (min_version is None and max_version is None) or version == UNVERSIONED or self.ref.policy_start is None:
            return True
        effective = self.ref.policy_start if version == UNLABELED else version
        if min_version is not None and effective < min_version:
            return False
        if max_version is not None and effective > max_version:
            return False
        return True

    def has_loss(self, min_version: int | None, max_version: int | None = None) -> bool:
        """Whether any of its tokens still trains once tokens outside ``[min_version, max_version]`` are out."""
        for component, counts in self.loss.items():
            if component == "rl":
                if self.pruned and self.zero_advantage:
                    continue
                if self.ref.rl_weight is not None:
                    counts = self.trainable
            if any(count and self.is_fresh(version, min_version, max_version) for version, count in counts.items()):
                return True
        return False

    def stale_tokens(self, min_version: int, max_version: int | None = None) -> int:
        """Trainable tokens outside ``[min_version, max_version]``."""
        return sum(
            count for version, count in self.trainable.items() if not self.is_fresh(version, min_version, max_version)
        )

    @property
    def num_trainable(self) -> int:
        return sum(self.trainable.values())


def _counts(counts: dict[str, int]) -> dict[int, int]:
    return {int(version): count for version, count in counts.items()}


def queued_samples(
    trace: TraceHead, *, env_name: str, temperature: float, policy_start: int | None, seq_len: int
) -> list[QueuedSample]:
    """References to the samples an env worker kept for the trace, credited with its advantage
    and rl weight and cut to ``seq_len`` tokens. A trace credited per token has its advantages
    written into its samples (`write_token_advantages`) instead."""
    payload = trace.info.get(PAYLOAD_INFO_KEY)
    if payload is None:
        raise RuntimeError(f"env '{env_name}' kept no training samples for trace {trace.id}. {_KEEP_SAMPLES_HINT}")
    token_values = branch_token_advantages(trace)
    advantage = None if token_values is not None else trace_advantage(trace)
    rl_weight = trace_rl_weight(trace)
    samples = []
    for index, summary in payload["branches"].items():
        num_tokens = mm_cut(summary.get("images") or [], summary["num_tokens"], seq_len)
        images = [image for image in summary.get("images") or [] if image[0] + image[1] <= num_tokens]
        loss = {component: _counts(counts) for component, counts in summary["loss"].items()}
        if advantage is None and token_values is None and any(loss.get("rl", {}).values()):
            raise ValueError(
                f"sample from env '{env_name}' has rl member tokens but no advantages — "
                "the algorithm must credit its trace"
            )
        ref = SampleRef(
            url=payload["url"],
            id=summary["id"],
            num_tokens=num_tokens,
            env_name=env_name,
            trace_id=trace.id,
            branch_index=int(index),
            temperature=temperature,
            advantage=advantage,
            rl_weight=rl_weight,
            policy_start=policy_start,
            fields=summary["fields"],
            routed_shape=summary.get("routed_shape"),
            routed_dtype=summary.get("routed_dtype"),
            images=images or None,
        )
        zero = advantage == 0.0 if token_values is None else not token_values[int(index)].any()
        samples.append(QueuedSample(ref, _counts(summary["trainable"]), loss, zero_advantage=zero))
    return samples


def write_token_advantages(traces: Iterable[TraceHead]) -> list[TraceHead]:
    """Write the advantages of traces credited per token into the samples their env workers
    kept. The traces whose samples are no longer all kept."""
    lost = []
    for trace in traces:
        values = branch_token_advantages(trace)
        payload = trace.info.get(PAYLOAD_INFO_KEY)
        if values is None or payload is None:
            continue
        for index, summary in payload["branches"].items():
            if not write_advantages(f"{payload['url']}/{summary['id']}", values[int(index)]):
                lost.append(trace)
                break
    return lost


class TrainSink(Reconfigurable):
    """Score native episodes, admit groups, then compile trainer payloads."""

    def __init__(
        self,
        config: OrchestratorConfig,
        *,
        tokenizer,
        train_envs: TrainEnvs,
        progress: Progress,
        batch_size: int,
        on_result: Callable[[list[EpisodeHead]], bool] | None = None,
        on_groups_closed: Callable[[Iterable[str]], None] | None = None,
        group_size: Callable[[str], int | None] | None = None,
    ) -> None:
        """``on_groups_closed`` receives the ids of every group that closes: a group
        that finalized without anything to ship, or one whose batch shipped (`shipped`).
        ``group_size`` returns the size a group was drawn with, or None if unknown; an
        unknown group takes its env's current group size."""
        self.config = config
        self.tokenizer = tokenizer
        self.train_envs = train_envs
        self.progress = progress
        self.batch_size = batch_size
        self.on_result = on_result
        self.on_groups_closed = on_groups_closed
        self.group_size = group_size

        self.rollout_window = RolloutWindow()
        self.pending_groups: dict[str, list[EpisodeHead]] = defaultdict(list)
        self.pending_group_failures: dict[str, list[DispatchFailure]] = defaultdict(list)
        # A dropped group's terminal marker; its ``count`` fills in for the
        # episodes the group will never deliver.
        self.pending_group_cancellations: dict[str, GroupCancellation] = {}
        self.pending_batch: dict[str, list[QueuedSample]] = {}
        self.episode_by_trace: dict[str, EpisodeHead] = {}
        # Step of the last full staleness sweep — queued traces only age when
        # ``progress.step`` advances, so one full sweep per step suffices.
        self._swept_step = 0
        # Finalized units that shipped no payload since the last shipment, by drop reason.
        self.zero_output_units: Counter[str] = Counter()
        self.reported_zero_output_windows = 0

    def reconfigure(self, update: ConfigUpdate) -> None:
        """Queued traces stay queued: the next readiness check sweeps them against the new
        ``max_staleness`` and cuts batches of the new ``batch_size``."""
        self.config = update.config
        self.batch_size = update.config.batch_size
        self._swept_step = 0

    def _release(self, urls: Iterable[str]) -> None:
        """Env workers keep the arrays of delivered traces; the sink deletes those of every
        sample and trace it drops without shipping."""
        delete_payloads(urls)

    def release_samples(self, samples: Iterable[SampleRef]) -> None:
        """Delete the env workers' samples that will never ship."""
        self._release(sample_payload_urls(samples))

    def state_dict(self) -> dict:
        """What the sink holds between two results, for a rollout checkpoint: the groups
        still collecting episodes, and the traces queued for the next batches with
        their episodes. The containers are copies; the episodes and samples are not."""
        return {
            "pending_groups": {group_id: list(group) for group_id, group in self.pending_groups.items() if group},
            "pending_group_failures": {
                group_id: list(failures) for group_id, failures in self.pending_group_failures.items() if failures
            },
            "pending_group_cancellations": dict(self.pending_group_cancellations),
            "pending_batch": dict(self.pending_batch),
            "episode_by_trace": dict(self.episode_by_trace),
        }

    def load_state_dict(self, state: dict) -> None:
        """Take back what ``state_dict`` saved, except the entries of envs this run does
        not train on."""
        names = set(self.train_envs.names)
        for group_id, group in state["pending_groups"].items():
            if episode_env_name(group[0]) in names:
                self.pending_groups[group_id] = list(group)
        for group_id, failures in state["pending_group_failures"].items():
            if failures[0].env_name in names:
                self.pending_group_failures[group_id] = list(failures)
        for group_id, cancellation in state["pending_group_cancellations"].items():
            if cancellation.env_name in names:
                self.pending_group_cancellations[group_id] = cancellation
        for trace_id, samples in state["pending_batch"].items():
            episode = state["episode_by_trace"][trace_id]
            if episode_env_name(episode) in names:
                self.pending_batch[trace_id] = samples
                self.episode_by_trace[trace_id] = episode

    def group_size_for(self, group_id: str, env_name: str) -> int:
        size = self.group_size(group_id) if self.group_size is not None else None
        return size if size is not None else self.train_envs.get(env_name).config.group_size

    def batch_progress(self) -> tuple[int, int]:
        return len(self.pending_batch), self.batch_size

    def buffered_count(self) -> int:
        episodes = sum(len(group) for group in self.pending_groups.values())
        failures = sum(len(group) for group in self.pending_group_failures.values())
        return episodes + failures

    def batchable_rollouts(self) -> int:
        """Finished rollouts queued for a batch; those of groups still waiting for members can't be taken yet."""
        return len(self.pending_batch)

    def pending_batch_by_env(self) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for trace_id in self.pending_batch:
            episode = self.episode_by_trace[trace_id]
            counts[episode_env_name(episode)] += 1
        return dict(counts)

    async def add(self, episode: EpisodeHead) -> TrainBatch | None:
        """Process one completed episode and return a batch when ready."""
        group_id = episode_group_id(episode)
        env_name = episode_env_name(episode)
        self.pending_groups[group_id].append(episode)
        if not self._group_complete(group_id, env_name):
            return None
        await self.process_group(group_id)
        return await self.take_batch()

    async def cancel(self, cancellation: GroupCancellation) -> TrainBatch | None:
        """Process a dropped group's terminal marker: its ``count`` completes
        the group's episode accounting so finalization still fires. A
        ``shutdown`` drop also voids the group's already-arrived episodes in
        ``process_group``."""
        self.pending_group_cancellations[cancellation.group_id] = cancellation
        if not self._group_complete(cancellation.group_id, cancellation.env_name):
            return None
        await self.process_group(cancellation.group_id)
        return await self.take_batch()

    async def fail(self, failure: DispatchFailure) -> TrainBatch | None:
        """Count a request failure toward its group without presenting it as
        an episode to the algorithm or curriculum."""
        if failure.kind != "train":
            raise ValueError(f"TrainSink cannot process a {failure.kind} dispatch failure")
        self.pending_group_failures[failure.group_id].append(failure)
        if not self._group_complete(failure.group_id, failure.env_name):
            return None
        await self.process_group(failure.group_id)
        return await self.take_batch()

    def _group_complete(self, group_id: str, env_name: str) -> bool:
        cancellation = self.pending_group_cancellations.get(group_id)
        cancelled = cancellation.count if cancellation is not None else 0
        failed = len(self.pending_group_failures[group_id])
        return len(self.pending_groups[group_id]) + failed + cancelled >= self.group_size_for(group_id, env_name)

    async def take_batch(self) -> TrainBatch | None:
        """Sweep stale queued traces, then cut a batch if the survivors still
        meet the threshold."""
        self._drop_stale()
        return await self.process_batch() if len(self.pending_batch) >= self.batch_size else None

    def _drop_stale(self, trace_ids: Iterable[str] | None = None) -> None:
        """Mask queued tokens past ``max_staleness`` out of the loss and
        void the samples left with nothing to train. The batch being collected
        is ``progress.step`` and trains policy v{step-1}, so a token sampled by
        v{k} would ship at staleness ``(step-1) - k``. Every trace is checked by
        its token labels, not its dispatch version: an engine still installing
        the version an episode was dispatched under samples it with the
        previous one. Tokens without a policy label take the dispatch version.
        Queued tokens only age when ``progress.step`` advances, so the full
        sweep runs once per step; ``trace_ids`` scopes the check to a freshly
        inserted group. Frozen-sourced episodes (no policy span) never go stale.
        The trainer masks the same tokens when it builds the batch.

        Tokens of a version newer than v{step-1}, which the trainer cannot have
        published yet, are masked too: an engine that outlived a trainer restart
        from an earlier checkpoint samples them with the dead trainer's weights
        until it installs the new trainer's.

        Queue outcomes are counted separately from completed groups; observing
        the episode again here would double-count rollout statistics."""
        if trace_ids is None:
            if self._swept_step == self.progress.step:
                return
            self._swept_step = self.progress.step
            trace_ids = list(self.pending_batch)
        min_version = min_fresh_version(self.progress.step, self.config.max_staleness)
        max_version = self.progress.step - 1
        dropped = 0
        dropped_groups: set[str] = set()
        for trace_id in trace_ids:
            episode = self.episode_by_trace[trace_id]
            policy = train_work(episode).policy
            if policy is None:
                continue
            samples = self.pending_batch[trace_id]
            trainable = [
                not sample.stale_tokens(min_version, max_version) or sample.has_loss(min_version, max_version)
                for sample in samples
            ]
            self.release_samples(sample.ref for sample, keep in zip(samples, trainable) if not keep)
            if samples := [sample for sample, keep in zip(samples, trainable) if keep]:
                self.pending_batch[trace_id] = samples
                continue
            del self.pending_batch[trace_id]
            del self.episode_by_trace[trace_id]
            dropped_groups.add(episode_group_id(episode))
            dropped += 1
        self._close_unqueued(dropped_groups)
        if dropped:
            self.rollout_window.metrics.count("staleness/rollout/dropped", float(dropped))
            get_logger().warning(
                f"Dropped {dropped} queued traces with every token past max_staleness={self.config.max_staleness} "
                "or sampled by a policy newer than the trainer's. Consider increasing max_staleness if the former."
            )

    async def process_group(self, group_id: str) -> None:
        traces = [trace for episode in self.pending_groups.get(group_id, []) for trace in episode.traces]
        try:
            await self._process_group(group_id)
        finally:
            # Every delivered trace's arrays are either queued for a batch or deleted here.
            for trace in traces:
                queued = set(sample_payload_urls(sample.ref for sample in self.pending_batch.get(trace.id, [])))
                self._release(url for url in trace_payload_urls(trace) if url not in queued)

    async def _process_group(self, group_id: str) -> None:
        group = self.pending_groups.pop(group_id, [])
        failures = self.pending_group_failures.pop(group_id, [])
        cancellation = self.pending_group_cancellations.pop(group_id, None)
        if not group and not failures and cancellation is None:
            return

        env_name = (
            episode_env_name(group[0]) if group else (failures[0].env_name if failures else cancellation.env_name)
        )
        env = self.train_envs.get(env_name)
        traces = [trace for episode in group for trace in episode.traces]
        task_idx = next((trace.task.idx for trace in traces), None)
        num_errored = (
            sum(trace.has_error for trace in traces)
            + sum(not episode.ok for episode in group if not episode.traces)
            + len(failures)
        )
        n_owed = len(group) + len(failures) + (cancellation.count if cancellation is not None else 0)
        # A shutdown drop voids the whole group and bypasses the curriculum —
        # a pipeline decision is not a task result. The group stays open, so a
        # resume re-runs it.
        if cancellation is not None and cancellation.reason == "shutdown":
            self._record_group(group, failures, cancellation, env_name, admitted=False, cancelled=group)
            self._record_zero_output(group, [], n_owed, reason="cancelled")
            get_logger().debug(
                f"Dropped group | env={env_name} task_idx={task_idx} | "
                f"episodes={len(group)} traces={len(traces)} (errored={num_errored}) | "
                f"reason=cancelled ({cancellation.reason})"
            )
            return

        survivors = [trace for _, trace in iter_trainable_traces(group)]
        if survivors:
            await env.algorithm.score_group(group)
        admitted = self._admit(group) if group else False
        if not survivors or not admitted:
            self._close([group_id])
            self._record_group(group, failures, cancellation, env_name, admitted=admitted)
            reason = "no trainable survivors" if not survivors else "rejected by curriculum"
            self._record_zero_output(group, survivors, n_owed, reason=reason)
            get_logger().debug(
                f"Dropped group | env={env_name} task_idx={task_idx} | "
                f"episodes={len(group)} traces={len(traces)} (errored={num_errored}) | reason={reason}"
            )
            return

        per_token = [trace for trace in survivors if token_advantages(trace) is not None]
        lost = (
            {trace.id for trace in await asyncio.to_thread(write_token_advantages, per_token)} if per_token else set()
        )
        if lost:
            get_logger().warning(f"Dropped {len(lost)} traces whose env workers no longer keep their samples")
            self._release(url for trace in survivors if trace.id in lost for url in trace_payload_urls(trace))
        samples_by_trace: dict[str, list[QueuedSample]] = {}
        # The temperature the group sampled with, which a live update may have changed since.
        temperature = env.sampling_at(train_work(group[0]).step).temperature
        for episode, trace in iter_trainable_traces(group):
            if trace.id in lost:
                continue
            policy = train_work(episode).policy
            samples = queued_samples(
                trace,
                env_name=env_name,
                temperature=temperature,
                policy_start=policy.start if policy is not None else None,
                seq_len=self.config.seq_len,
            )
            for sample in samples:
                if env.requires_sampling_masks and "sampler_support_replayed" not in sample.ref.fields:
                    # Rollout logprobs are mask-renormalized; training without the masks
                    # silently biases every importance ratio.
                    raise RuntimeError(
                        f"env '{env_name}' samples with truncation (top_p/top_k) but its rollouts "
                        "carry no sampling masks. Set `enable_return_sampling_mask = true` on "
                        "the inference server config (the rl entrypoint does this automatically) - "
                        f"it requires vLLM's native sampling-mask capture (>= 0.28). {_KEEP_SAMPLES_HINT}"
                    )
                if env.requires_sampler_logprobs and "sampler_support_logprobs" not in sample.ref.fields:
                    raise RuntimeError(
                        f"env {env_name!r} requested sampler logprobs but returned none. {_KEEP_SAMPLES_HINT}"
                    )
            if self.config.constant_trainer_batch_size:
                samples = self._prune(samples)
            if samples:
                samples_by_trace[trace.id] = samples

        if not samples_by_trace:
            self._close([group_id])
            self._record_group(group, failures, cancellation, env_name)
            self._record_zero_output(group, survivors, n_owed, reason="zero advantage")
            return

        self.pending_batch.update(samples_by_trace)
        for episode in group:
            for trace in episode.traces:
                if trace.id in samples_by_trace:
                    self.episode_by_trace[trace.id] = episode.model_copy(update={"traces": [trace]})
        self._drop_stale(samples_by_trace)
        queued = set(samples_by_trace) & self.pending_batch.keys()
        stale = [
            episode
            for episode in group
            if any(trace.id in samples_by_trace for trace in episode.traces)
            and not any(trace.id in queued for trace in episode.traces)
        ]
        self._record_group(group, failures, cancellation, env_name, queued_trace_ids=queued, cancelled=stale)
        # A fully-voided group shipped nothing — advance the zero-output tally
        # instead of resetting it, or a stalled trainer plus a tight bound
        # could void groups forever without ever surfacing the warning.
        if not queued:
            self._record_zero_output(group, [], n_owed, reason="stale")
            return
        self.zero_output_units.clear()
        self.reported_zero_output_windows = 0

    def _record_group(
        self,
        group: list[EpisodeHead],
        failures: list[DispatchFailure],
        cancellation: GroupCancellation | None,
        env_name: str,
        *,
        admitted: bool = True,
        cancelled: Iterable[EpisodeHead] = (),
        queued_trace_ids: set[str] | None = None,
    ) -> None:
        episodes = TrainEpisodes(cancelled={episode.id for episode in cancelled})
        episodes.extend(group, admitted=admitted)
        self.rollout_window.observe(
            episodes, failures, cancellation, env_name=env_name, queued_trace_ids=queued_trace_ids or set()
        )

    def take_rollout_window(self, *, force: bool = False) -> RolloutWindow | None:
        """Transfer accumulated step statistics before shipment."""
        if not force:
            return None
        if not self.rollout_window.groups and not self.rollout_window.metrics.counts:
            return None
        window, self.rollout_window = self.rollout_window, RolloutWindow()
        return window

    def discard_queued(self) -> None:
        """Release accepted traces that cannot ship after training stops."""
        if self.pending_batch:
            self.rollout_window.metrics.count("rollout/queued/cancelled", float(len(self.pending_batch)))
        self.release_samples(sample.ref for samples in self.pending_batch.values() for sample in samples)
        self._release(
            url
            for group in self.pending_groups.values()
            for episode in group
            for trace in episode.traces
            for url in trace_payload_urls(trace)
        )
        self.pending_batch.clear()
        self.episode_by_trace.clear()

    def shipped(self, batch: TrainBatch) -> None:
        """Close the groups of a batch that the consumer shipped to the trainer."""
        self._close(episode_group_id(episode) for episode in batch.cohort.episodes)

    def _close(self, group_ids: Iterable[str]) -> None:
        if self.on_groups_closed is not None:
            self.on_groups_closed(group_ids)

    def _close_unqueued(self, group_ids: set[str]) -> None:
        """Close the groups left with no queued trace, which can no longer ship."""
        if group_ids:
            queued = {episode_group_id(episode) for episode in self.episode_by_trace.values()}
            self._close(group_ids - queued)

    def _admit(self, group: list[EpisodeHead]) -> bool:
        return self.on_result(group) if self.on_result is not None else True

    def _record_zero_output(
        self, group: list[EpisodeHead], survivors: list[TraceHead], n_owed: int, *, reason: str
    ) -> None:
        """``n_owed`` counts the group's full episode budget (arrived +
        cancelled), so dropped groups advance the zero-output tally at the
        same rate as fully-delivered ones."""
        returned_traces = sum(len(episode.traces) for episode in group)
        self.zero_output_units[reason] += len(survivors) or returned_traces or n_owed
        self._warn_zero_output()

    def _warn_zero_output(self) -> None:
        """Warn once per batch-equivalent of finalized units that shipped no
        payload, so a run that produces no training signal stays visible in the
        logs without aborting."""
        units = self.zero_output_units.total()
        windows = units // self.batch_size
        if windows <= self.reported_zero_output_windows:
            return
        self.reported_zero_output_windows = windows
        get_logger().warning(
            f"No admitted train payload after {units} finalized units "
            f"({windows} zero-output batch equivalents), by drop reason: "
            + ", ".join(f"{reason}={count}" for reason, count in self.zero_output_units.most_common())
        )

    @staticmethod
    def _prune(
        samples: list[QueuedSample], min_version: int | None = None, max_version: int | None = None
    ) -> list[QueuedSample]:
        """Take zero-advantage tokens out of the samples' rl component, and the samples left
        with nothing to train (tokens outside ``[min_version, max_version]`` already out) out of the queue."""
        for sample in samples:
            sample.pruned = True
        return [sample for sample in samples if sample.has_loss(min_version, max_version)]

    async def process_batch(self) -> TrainBatch:
        selected = list(self.pending_batch.items())[: self.batch_size]

        selected_by_trace = dict(selected)
        selected_ids = set(selected_by_trace)
        for trace_id in selected_ids:
            del self.pending_batch[trace_id]

        min_version = min_fresh_version(self.progress.step, self.config.max_staleness)
        max_version = self.progress.step - 1
        if not self.config.constant_trainer_batch_size:
            pruned = {trace_id: self._prune(samples, min_version, max_version) for trace_id, samples in selected}
            selected_by_trace = {trace_id: samples for trace_id, samples in pruned.items() if samples}
        samples = [sample for trace_samples in selected_by_trace.values() for sample in trace_samples]
        shipped = set(sample_payload_urls(sample.ref for sample in samples))
        self._release(
            url
            for _, trace_samples in selected
            for url in sample_payload_urls(sample.ref for sample in trace_samples)
            if url not in shipped
        )

        # Shared fork prefixes train only in their first branch, so each trainable token counts once.
        masked = sum(sample.stale_tokens(min_version, max_version) for sample in samples)
        kept = sum(sample.num_trainable for sample in samples) - masked
        self.rollout_window.metrics.rate("staleness/token/masked", {"": float(masked)}, kept + masked)

        shipped_ids = set(selected_by_trace)
        traces_by_episode: dict[str, list[TraceHead]] = defaultdict(list)
        selected_episodes: dict[str, EpisodeHead] = {}
        pruned_groups: set[str] = set()
        for trace_id in selected_ids:
            episode = self.episode_by_trace.pop(trace_id)
            if trace_id in shipped_ids:
                selected_episodes[episode.id] = episode
                traces_by_episode[episode.id].extend(episode.traces)
            else:
                pruned_groups.add(episode_group_id(episode))
        if pruned := len(selected_ids - shipped_ids):
            self.rollout_window.metrics.count("rollout/queued/pruned", float(pruned))
        # A group with a trace in this batch closes when the batch ships.
        self._close_unqueued(pruned_groups - {episode_group_id(episode) for episode in selected_episodes.values()})
        cohort_episodes = [
            episode.model_copy(update={"traces": traces_by_episode[episode.id]})
            for episode in selected_episodes.values()
        ]
        cohort = TrainEpisodes(cohort_episodes, sampled_trace_ids=shipped_ids)

        return TrainBatch(cohort=cohort, samples=[sample.ref for sample in samples])
