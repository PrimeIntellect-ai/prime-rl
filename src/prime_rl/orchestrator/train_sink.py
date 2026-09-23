"""TrainSink: from dispatched results to finished train groups.

``ingest`` takes one dispatcher result — a completed episode, a request that
produced none, or a dropped group's cancellation — and holds it until its group
is complete. A complete group is scored by its env's algorithm, offered to the
curriculum, compiled into trainer samples, and handed on as one
``FinalizedGroup``; the queue decides when it ships."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Awaitable, Callable

import verifiers.v1 as vf

from prime_rl.orchestrator.algo.base import iter_trainable_traces
from prime_rl.orchestrator.algo.routing import stamp_loss_routing
from prime_rl.orchestrator.envs import TrainEnvs
from prime_rl.orchestrator.trajectories import trace_to_samples
from prime_rl.orchestrator.types import DispatchFailure, DispatchResult, FinalizedGroup, GroupCancellation
from prime_rl.orchestrator.utils import episode_env_name, episode_group_id
from prime_rl.transports.batch import TrainingSample
from prime_rl.utils.logger import get_logger


class TrainSink:
    def __init__(self, envs: TrainEnvs) -> None:
        self.envs = envs
        self._admit: Callable[[list[vf.Episode]], bool] = lambda group: True
        self._on_group: Callable[[FinalizedGroup], Awaitable[None]] | None = None

        self.pending_groups: dict[str, list[vf.Episode]] = defaultdict(list)
        self.pending_failures: dict[str, list[DispatchFailure]] = defaultdict(list)
        # A dropped group's terminal marker; its ``count`` fills in for the episodes
        # the group will never deliver.
        self.pending_cancellations: dict[str, GroupCancellation] = {}

    def bind(
        self,
        *,
        on_group: Callable[[FinalizedGroup], Awaitable[None]],
        admit: Callable[[list[vf.Episode]], bool] | None = None,
    ) -> None:
        self._on_group = on_group
        if admit is not None:
            self._admit = admit

    # ── inbound ────────────────────────────────────────────────────────────

    async def ingest(self, item: DispatchResult) -> None:
        if isinstance(item, GroupCancellation):
            group_id, env_name = item.group_id, item.env_name
            self.pending_cancellations[group_id] = item
        elif isinstance(item, DispatchFailure):
            group_id, env_name = item.group_id, item.env_name
            self.pending_failures[group_id].append(item)
        else:
            group_id, env_name = episode_group_id(item), episode_env_name(item)
            await self.envs.get(env_name).algorithm.finalize_episode(item)
            self.pending_groups[group_id].append(item)
        if self._group_complete(group_id, env_name):
            assert self._on_group is not None, "TrainSink.on_group is not bound"
            await self._on_group(await self.finalize_group(group_id))

    def _group_complete(self, group_id: str, env_name: str) -> bool:
        cancellation = self.pending_cancellations.get(group_id)
        cancelled = cancellation.count if cancellation is not None else 0
        failed = len(self.pending_failures[group_id])
        return len(self.pending_groups[group_id]) + failed + cancelled >= self.envs.get(env_name).config.group_size

    async def finalize_group(self, group_id: str) -> FinalizedGroup:
        group = self.pending_groups.pop(group_id, [])
        failures = self.pending_failures.pop(group_id, [])
        cancellation = self.pending_cancellations.pop(group_id, None)
        env_name = (
            episode_env_name(group[0]) if group else (failures[0].env_name if failures else cancellation.env_name)
        )
        env = self.envs.get(env_name)
        traces = [trace for episode in group for trace in episode.traces]
        task_idx = next((trace.task.data.idx for trace in traces), None)
        num_errored = (
            sum(trace.has_error for trace in traces)
            + sum(not episode.ok for episode in group if not episode.traces)
            + len(failures)
        )
        finalized = FinalizedGroup(
            env_name=env_name,
            episodes=group,
            samples={},
            survivors=[],
            failures=failures,
            cancellation=cancellation,
            admitted=False,
        )

        # A stale drop voids the whole group and bypasses the curriculum: a pipeline
        # decision is not a task result.
        if finalized.stale:
            get_logger().debug(
                f"Dropped group | env={env_name} task_idx={task_idx} | "
                f"episodes={len(group)} traces={len(traces)} (errored={num_errored}) | reason=cancelled (stale)"
            )
            return finalized

        survivors = [trace for _, trace in iter_trainable_traces(group)]
        finalized.survivors = survivors
        if survivors:
            await env.algorithm.finalize_group(group)
        finalized.admitted = self._admit(group) if group else False
        if not survivors or not finalized.admitted:
            reason = "no trainable survivors" if not survivors else "rejected by curriculum"
            get_logger().debug(
                f"Dropped group | env={env_name} task_idx={task_idx} | "
                f"episodes={len(group)} traces={len(traces)} (errored={num_errored}) | reason={reason}"
            )
            return finalized

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
            if samples:
                samples_by_trace[trace.id] = samples
        finalized.samples = samples_by_trace
        return finalized

    # ── observability ──────────────────────────────────────────────────────

    def buffered_count(self) -> int:
        """Episodes and failures of groups still waiting for their siblings."""
        episodes = sum(len(group) for group in self.pending_groups.values())
        failures = sum(len(group) for group in self.pending_failures.values())
        return episodes + failures

    def status(self) -> str | None:
        buffered = self.buffered_count()
        return f"+{buffered} buffered" if buffered else None
