"""Training-side group finalization, scalar accounting, and batch assembly.

``add()`` takes one completed episode, ``fail()`` a request that produced no
episode, and ``cancel()`` a dropped group's ``GroupCancellation``. Before every
readiness check the sink sweeps ``pending_batch`` for traces past
``max_staleness`` — this sweep is what guarantees no stale token trains. Completed groups become numeric
observations; only queued traces retain verifier objects. The consumer drains
``take_rollout_window(force=True)`` before shipping a batch."""

from __future__ import annotations

import asyncio
import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from typing import Any

import numpy as np
import verifiers.v1 as vf

from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.control.types import ConfigUpdate, Reconfigurable
from prime_rl.orchestrator.algo.base import iter_trainable_traces
from prime_rl.orchestrator.algo.routing import stamp_loss_routing
from prime_rl.orchestrator.envs import TrainEnvs
from prime_rl.orchestrator.metrics import RolloutWindow, TrainEpisodes
from prime_rl.orchestrator.trajectories import iter_trainable_branches, trace_to_samples
from prime_rl.orchestrator.types import DispatchFailure, GroupCancellation, Progress, TaskRequest, TrainBatch
from prime_rl.orchestrator.utils import (
    episode_env_name,
    episode_group_id,
    min_fresh_version,
    node_policy_versions,
    train_work,
)
from prime_rl.transports.batch import TrainingSample
from prime_rl.transports.payload import delete_payloads, sample_payload_urls, trace_payload_urls
from prime_rl.utils.logger import get_logger


def _kept(sample: TrainingSample, field: str) -> bool:
    """Whether an env server keeps this array of the sample (see `prime_rl.transports.payload`)."""
    return sample.payload is not None and field in sample.payload.fields


# Training arrays reach the trainer only through env servers that keep them.
_KEEP_ARRAYS_HINT = (
    "Serve the env with prime-rl's `env-server` entrypoint, which keeps them for the trainer, "
    "and check its log for arrays it could not keep."
)


def _prune_zero_advantages(sample: TrainingSample) -> bool:
    """Remove zero-advantage tokens from the RL component."""
    if sample.advantages is None:
        return True

    mask = np.asarray(sample.mask, dtype=bool)
    advantages = np.asarray(sample.advantages)
    if sample.rl_weights is None:
        rl_weights = mask.astype(np.float32)
    else:
        rl_weights = np.array(sample.rl_weights, dtype=np.float32)
    if not len(mask) == len(advantages) == len(rl_weights):
        raise ValueError(
            f"mask, advantages and rl_weights must align: {len(mask)}, {len(advantages)}, {len(rl_weights)}"
        )

    pruned = mask & (advantages == 0.0) & (rl_weights != 0.0)
    if not pruned.any():
        return True

    rl_weights[pruned] = 0.0
    sample.rl_weights = rl_weights
    return _has_loss(sample)


def _has_loss(sample: TrainingSample) -> bool:
    mask = np.asarray(sample.mask, dtype=bool)
    if sample.rl_weights is not None:
        mask = mask & (np.asarray(sample.rl_weights) != 0.0)
    has_rl = bool(mask.any())
    has_ce = sample.ce_weights is not None and bool(np.any(np.asarray(sample.ce_weights) != 0.0))
    has_ref_kl = sample.ref_kl_weights is not None and bool(np.any(np.asarray(sample.ref_kl_weights) != 0.0))
    return has_rl or has_ce or has_ref_kl


def stale_token_runs(branch: vf.Branch, fallback: int, min_version: int) -> Iterator[tuple[int, int]]:
    """``[start, end)`` runs of the branch's sampled tokens whose policy
    version is older than ``min_version`` (see ``node_policy_versions``)."""
    offset = 0
    for node in branch.nodes:
        for start, end, version in node_policy_versions(node, fallback):
            if version < min_version:
                yield offset + start, offset + end
        offset += len(node.token_ids)


def token_staleness(branch: vf.Branch, fallback: int, trained_version: int) -> np.ndarray:
    """Per-token staleness of the branch when it trains on ``trained_version``
    (see ``node_policy_versions``); -1 on tokens the policy did not sample."""
    lengths = [len(node.token_ids) for node in branch.nodes]
    staleness = np.full(sum(lengths), -1, dtype=np.int16)
    offset = 0
    for node, length in zip(branch.nodes, lengths):
        if node.sampled:
            for start, end, version in node_policy_versions(node, fallback):
                staleness[offset + start : offset + end] = max(0, trained_version - version)
        offset += length
    return staleness


def mask_stale_tokens(sample: TrainingSample, branch: vf.Branch, fallback: int, min_version: int) -> int:
    """Remove the sample's stale tokens from every loss component and return
    how many trainable tokens this newly masked; fresher tokens keep training
    with their own inference logprobs and the sample's advantage."""
    masked = 0
    for start, end in stale_token_runs(branch, fallback, min_version):
        masked += int(sample.mask[start:end].sum())
        sample.mask[start:end] = False
        for weights in (sample.rl_weights, sample.ce_weights, sample.ref_kl_weights):
            if weights is not None:
                weights[start:end] = 0.0
    return masked


def _dump(episode: vf.Episode) -> dict[str, Any]:
    """Full-precision dump with raw-bytes token streams (a JSON dump rounds logprobs)."""
    return episode.model_dump(context={"token_arrays": True})


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
        on_result: Callable[[list[vf.Episode]], bool] | None = None,
    ) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self.train_envs = train_envs
        self.progress = progress
        self.batch_size = batch_size
        self.on_result = on_result

        self.rollout_window = RolloutWindow()
        self.pending_groups: dict[str, list[vf.Episode]] = defaultdict(list)
        self.pending_group_failures: dict[str, list[DispatchFailure]] = defaultdict(list)
        # A dropped group's terminal marker; its ``count`` fills in for the
        # episodes the group will never deliver.
        self.pending_group_cancellations: dict[str, GroupCancellation] = {}
        self.pending_batch: dict[str, list[TrainingSample]] = {}
        self.episode_by_trace: dict[str, vf.Episode] = {}
        # Step of the last full staleness sweep — queued traces only age when
        # ``progress.step`` advances, so one full sweep per step suffices.
        self._swept_step = 0
        self.zero_output_units = 0
        self.reported_zero_output_windows = 0

    def state_dict(self, open_groups: dict[str, tuple[str, vf.Task]]) -> dict[str, Any]:
        """Work a resume would otherwise regenerate: the accepted traces waiting
        for a batch (samples and episode), and every train group still open in the
        dispatcher (``open_groups``: group id -> env and task) with its finished
        episodes and failed requests. Copies are shallow: queued samples and finished episodes are not
        mutated before they ship. Parametrized pydantic episode classes do not
        pickle, so episodes are saved as dumps."""
        return {
            "pending_batch": dict(self.pending_batch),
            "queued_episodes": {trace_id: _dump(ep) for trace_id, ep in self.episode_by_trace.items()},
            "open_groups": {
                group_id: (
                    env_name,
                    task,
                    [_dump(ep) for ep in self.pending_groups.get(group_id, [])],
                    list(self.pending_group_failures.get(group_id, [])),
                )
                for group_id, (env_name, task) in open_groups.items()
            },
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> list[TaskRequest]:
        """Re-queue saved traces and finished group members. Like any queued
        trace, their tokens past ``max_staleness`` are masked by the staleness
        sweep. Saved failed requests still count toward their group. Returns one
        request per saved group for its missing members (same task, same group
        id), which the train source hands out before any new task.
        Env servers keep training arrays only while they run, so saved work whose arrays
        they kept is dropped: queued traces are lost and group members are rerun."""
        self.pending_batch = {
            trace_id: samples
            for trace_id, samples in state_dict["pending_batch"].items()
            if not sample_payload_urls(samples)
        }
        self.episode_by_trace = {
            trace_id: vf.WireEpisode.model_validate(ep)
            for trace_id, ep in state_dict["queued_episodes"].items()
            if trace_id in self.pending_batch
        }
        dropped = len(state_dict["pending_batch"]) - len(self.pending_batch)
        requests = []
        kept = 0
        for group_id, (env_name, task, dumps, failures) in state_dict["open_groups"].items():
            group_size = self.group_size_for(env_name)
            if len(dumps) + len(failures) >= group_size:
                raise ValueError(
                    f"Checkpoint has an open group for env {env_name!r} with {len(dumps)} finished episodes and "
                    f"{len(failures)} failed requests, but its group_size is now {group_size}. "
                    "Lowering group_size across a resume is not supported."
                )
            episodes = [vf.WireEpisode.model_validate(ep) for ep in dumps]
            episodes = [ep for ep in episodes if not any(trace_payload_urls(trace) for trace in ep.traces)]
            dropped += len(dumps) - len(episodes)
            if episodes:
                self.pending_groups[group_id] = episodes
                kept += len(episodes)
            if failures:
                self.pending_group_failures[group_id] = failures
            rollouts = group_size - len(episodes) - len(failures)
            requests.append(
                TaskRequest(env_name=env_name, task=task, step=self.progress.step, rollouts=rollouts, group_id=group_id)
            )
        get_logger().info(
            f"Resuming with {len(self.pending_batch)} queued traces and {len(requests)} open groups "
            f"({kept} finished episodes kept, {dropped} traces and episodes dropped with their env-server arrays)"
        )
        return requests

    def reconfigure(self, update: ConfigUpdate) -> None:
        """Queued traces stay queued: the next readiness check sweeps them against the new
        ``max_staleness`` and cuts batches of the new ``batch_size``."""
        self.config = update.config
        self.batch_size = update.config.batch_size
        self._swept_step = 0

    def _release(self, urls: Iterable[str]) -> None:
        """Env servers keep the arrays of delivered traces; the sink deletes those of every
        sample and trace it drops without shipping."""
        delete_payloads(urls)

    def release_samples(self, samples: Iterable[TrainingSample]) -> None:
        """Delete the env-server arrays of samples that will never ship."""
        self._release(sample_payload_urls(samples))

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
        return len(self.pending_groups[group_id]) + failed + cancelled >= self.group_size_for(env_name)

    async def take_batch(self) -> TrainBatch | None:
        """Sweep stale queued traces, then cut a batch if the survivors still
        meet the threshold."""
        self._drop_stale()
        return await self.process_batch() if len(self.pending_batch) >= self.batch_size else None

    def _drop_stale(self, trace_ids: Iterable[str] | None = None) -> None:
        """Mask queued tokens past ``max_staleness`` out of the loss and
        void the samples left with nothing to train. The batch being collected
        is ``progress.step`` and trains policy v{step-1}, so a token sampled by
        v{k} would ship at staleness ``(step-1) - k``. Tokens without a policy
        label take their episode's dispatch version, which no token of the
        episode predates. Queued tokens only age when ``progress.step``
        advances, so the full sweep runs once per step; ``trace_ids`` scopes
        the check to a freshly inserted group. Frozen-sourced episodes (no
        policy span) never go stale.

        Queue outcomes are counted separately from completed groups; observing
        the episode again here would double-count rollout statistics."""
        if trace_ids is None:
            if self._swept_step == self.progress.step:
                return
            self._swept_step = self.progress.step
            trace_ids = list(self.pending_batch)
        min_version = min_fresh_version(self.progress.step, self.config.max_staleness)
        if min_version <= 0:
            return
        dropped = 0
        for trace_id in trace_ids:
            episode = self.episode_by_trace[trace_id]
            policy = train_work(episode).policy
            if policy is None or policy.start >= min_version:
                continue
            branches = episode.traces[0].branches
            samples = self.pending_batch[trace_id]
            masked = [
                mask_stale_tokens(sample, branches[sample.branch_index], policy.start, min_version)
                for sample in samples
            ]
            trainable = [not n or _has_loss(sample) for sample, n in zip(samples, masked)]
            self.release_samples(sample for sample, keep in zip(samples, trainable) if not keep)
            if samples := [sample for sample, keep in zip(samples, trainable) if keep]:
                self.pending_batch[trace_id] = samples
                continue
            del self.pending_batch[trace_id]
            del self.episode_by_trace[trace_id]
            dropped += 1
        if dropped:
            self.rollout_window.metrics.count("staleness/rollout/dropped", float(dropped))
            get_logger().warning(
                f"Dropped {dropped} queued traces with every token past "
                f"max_staleness={self.config.max_staleness}. Consider increasing it to avoid this."
            )

    async def process_episode(self, episode: vf.Episode) -> None:
        """Run rollout-local algorithm work on one native episode."""
        env_name = episode_env_name(episode)
        await self.train_envs.get(env_name).algorithm.finalize_episode(episode)

    async def process_group(self, group_id: str) -> None:
        traces = [trace for episode in self.pending_groups.get(group_id, []) for trace in episode.traces]
        try:
            await self._process_group(group_id)
        finally:
            # Every delivered trace's arrays are either queued for a batch or deleted here.
            for trace in traces:
                queued = set(sample_payload_urls(self.pending_batch.get(trace.id, [])))
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
        task_idx = next((trace.task.data.idx for trace in traces), None)
        num_errored = (
            sum(trace.has_error for trace in traces)
            + sum(not episode.ok for episode in group if not episode.traces)
            + len(failures)
        )
        n_owed = len(group) + len(failures) + (cancellation.count if cancellation is not None else 0)
        # A shutdown drop voids the whole group and bypasses the curriculum —
        # a pipeline decision is not a task result.
        if cancellation is not None and cancellation.reason == "shutdown":
            self._record_group(group, failures, cancellation, env_name, admitted=False, cancelled=group)
            self._record_zero_output(group, [], n_owed)
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
            self._record_group(group, failures, cancellation, env_name, admitted=admitted)
            self._record_zero_output(group, survivors, n_owed)
            reason = "no trainable survivors" if not survivors else "rejected by curriculum"
            get_logger().debug(
                f"Dropped group | env={env_name} task_idx={task_idx} | "
                f"episodes={len(group)} traces={len(traces)} (errored={num_errored}) | reason={reason}"
            )
            return

        samples_by_trace: dict[str, list[TrainingSample]] = {}
        # The temperature the group sampled with, which a live update may have changed since.
        temperature = env.sampling_at(train_work(group[0]).step).temperature
        for trace in survivors:
            samples = await asyncio.to_thread(trace_to_samples, trace, env_name=env_name)
            for sample in samples:
                sample.temperatures = np.full(len(sample.token_ids), temperature, dtype=np.float32)
                if env.requires_sampling_masks and not _kept(sample, "sampling_mask"):
                    # Rollout logprobs are mask-renormalized; training without the masks
                    # silently biases every importance ratio.
                    raise RuntimeError(
                        f"env '{env_name}' samples with truncation (top_p/top_k) but its rollouts "
                        "carry no sampling masks. Set `enable_return_sampling_mask = true` on "
                        "the inference server config (the rl entrypoint does this automatically) - "
                        f"it requires vLLM's native sampling-mask capture (>= 0.28). {_KEEP_ARRAYS_HINT}"
                    )
                if env.requires_top_logprobs and not (
                    _kept(sample, "top_logprobs") or _kept(sample, "sampling_mask_logprobs")
                ):
                    raise RuntimeError(
                        f"env {env_name!r} requested sampler top logprobs but returned none. {_KEEP_ARRAYS_HINT}"
                    )
                stamp_loss_routing(sample, env.algorithm.action_loss_type)
            if self.config.constant_trainer_batch_size:
                samples = [sample for sample in samples if _prune_zero_advantages(sample)]
            if samples:
                samples_by_trace[trace.id] = samples

        if not samples_by_trace:
            self._record_group(group, failures, cancellation, env_name)
            self._record_zero_output(group, survivors, n_owed)
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
            self._record_zero_output(group, [], n_owed)
            return
        self.zero_output_units = 0
        self.reported_zero_output_windows = 0

    def _record_group(
        self,
        group: list[vf.Episode],
        failures: list[DispatchFailure],
        cancellation: GroupCancellation | None,
        env_name: str,
        *,
        admitted: bool = True,
        cancelled: Iterable[vf.Episode] = (),
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
        self.release_samples(sample for samples in self.pending_batch.values() for sample in samples)
        self._release(
            url
            for group in self.pending_groups.values()
            for episode in group
            for trace in episode.traces
            for url in trace_payload_urls(trace)
        )
        self.pending_batch.clear()
        self.episode_by_trace.clear()

    def _policy_start(self, trace_id: str) -> float:
        """Dispatch version of a queued trace; ``inf`` for frozen-sourced traces, which never go stale."""
        policy = train_work(self.episode_by_trace[trace_id]).policy
        return policy.start if policy is not None else math.inf

    def _admit(self, group: list[vf.Episode]) -> bool:
        return self.on_result(group) if self.on_result is not None else True

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

    async def process_batch(self) -> TrainBatch:
        # Oldest dispatch version first, so leftovers are the traces with the most staleness budget left.
        selected = sorted(self.pending_batch.items(), key=lambda item: self._policy_start(item[0]))[: self.batch_size]

        selected_by_trace = dict(selected)
        selected_ids = set(selected_by_trace)
        for trace_id in selected_ids:
            del self.pending_batch[trace_id]

        if not self.config.constant_trainer_batch_size:

            def prune_selected() -> dict[str, list[TrainingSample]]:
                pruned = {
                    trace_id: [sample for sample in samples if _prune_zero_advantages(sample)]
                    for trace_id, samples in selected
                }
                return {trace_id: samples for trace_id, samples in pruned.items() if samples}

            selected_by_trace = await asyncio.to_thread(prune_selected)
        samples = [sample for trace_samples in selected_by_trace.values() for sample in trace_samples]
        shipped = set(sample_payload_urls(samples))
        self._release(
            url for _, trace_samples in selected for url in sample_payload_urls(trace_samples) if url not in shipped
        )

        trained_version = self.progress.step - 1

        def stamp_staleness() -> None:
            for trace_id, trace_samples in selected_by_trace.items():
                episode = self.episode_by_trace[trace_id]
                if (policy := train_work(episode).policy) is None:
                    continue
                branches = episode.traces[0].branches
                for sample in trace_samples:
                    sample.staleness = token_staleness(branches[sample.branch_index], policy.start, trained_version)

        await asyncio.to_thread(stamp_staleness)

        min_version = min_fresh_version(self.progress.step, self.config.max_staleness)
        kept = masked = 0
        for trace_id, trace_samples in selected_by_trace.items():
            episode = self.episode_by_trace[trace_id]
            policy = train_work(episode).policy
            if policy is None or policy.start >= min_version:
                kept += sum(int(sample.mask.sum()) for sample in trace_samples)
                continue
            # Shared fork prefixes train only in their first branch; count each stale token once.
            trainable = {branch.index: (branch, mask) for branch, mask in iter_trainable_branches(episode.traces[0])}
            for sample in trace_samples:
                kept += int(sample.mask.sum())
                branch, mask = trainable[sample.branch_index]
                runs = stale_token_runs(branch, policy.start, min_version)
                masked += sum(int(mask[start:end].sum()) for start, end in runs)
        self.rollout_window.metrics.rate("staleness/token/masked", {"": float(masked)}, kept + masked)

        shipped_ids = set(selected_by_trace)
        traces_by_episode: dict[str, list[vf.Trace]] = defaultdict(list)
        selected_episodes: dict[str, vf.Episode] = {}
        for trace_id in selected_ids:
            episode = self.episode_by_trace.pop(trace_id)
            if trace_id in shipped_ids:
                selected_episodes[episode.id] = episode
                traces_by_episode[episode.id].extend(episode.traces)
        if pruned := len(selected_ids - shipped_ids):
            self.rollout_window.metrics.count("rollout/queued/pruned", float(pruned))
        cohort_episodes = [
            episode.model_copy(update={"traces": traces_by_episode[episode.id]})
            for episode in selected_episodes.values()
        ]
        cohort = TrainEpisodes(cohort_episodes, sampled_trace_ids=shipped_ids)

        return TrainBatch(cohort=cohort, samples=samples)
