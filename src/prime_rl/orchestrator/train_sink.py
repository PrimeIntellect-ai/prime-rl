"""TrainSink: from dispatched episodes to finished train groups.

``ingest`` takes one episode and holds it until its group is complete. A complete
group is scored by its env's algorithm, offered to the curriculum, compiled into
trainer samples, and handed on as one ``Group``; the queue decides when it ships."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Awaitable, Callable

import verifiers.v1 as vf

from prime_rl.orchestrator.algo.base import iter_trainable_traces
from prime_rl.orchestrator.algo.routing import stamp_loss_routing
from prime_rl.orchestrator.envs import TrainEnvs
from prime_rl.orchestrator.trajectories import trace_to_samples
from prime_rl.orchestrator.types import Group, cancel, cancel_reason, env_of, group_of, is_cancelled, work_of
from prime_rl.transports.batch import TrainingSample
from prime_rl.utils.logger import get_logger


def prune_zero_advantages(sample: TrainingSample) -> bool:
    """Zero the RL weight of zero-advantage tokens; False when nothing is left to train."""
    if sample.advantages is None:
        return True
    weights = list(sample.rl_weights) if sample.rl_weights is not None else [float(m) for m in sample.mask]
    changed = False
    for index, (trainable, advantage) in enumerate(zip(sample.mask, sample.advantages, strict=True)):
        if trainable and advantage == 0.0 and weights[index] != 0.0:
            weights[index] = 0.0
            changed = True
    if not changed:
        return True
    sample.rl_weights = weights
    has_rl = any(trainable and weight != 0.0 for trainable, weight in zip(sample.mask, weights, strict=True))
    has_ce = sample.ce_weights is not None and any(weight != 0.0 for weight in sample.ce_weights)
    has_ref_kl = sample.ref_kl_weights is not None and any(weight != 0.0 for weight in sample.ref_kl_weights)
    return has_rl or has_ce or has_ref_kl


class TrainSink:
    def __init__(self, envs: TrainEnvs) -> None:
        self.envs = envs
        self._admit: Callable[[list[vf.Episode]], bool] = lambda group: True
        self._on_group: Callable[[Group], Awaitable[None]] | None = None
        self.pending: dict[str, list[vf.Episode]] = defaultdict(list)

    def bind(
        self,
        *,
        on_group: Callable[[Group], Awaitable[None]],
        admit: Callable[[list[vf.Episode]], bool] | None = None,
    ) -> None:
        self._on_group = on_group
        if admit is not None:
            self._admit = admit

    async def ingest(self, episode: vf.Episode) -> None:
        env = self.envs.get(env_of(episode))
        if episode.traces:
            await env.algorithm.finalize_episode(episode)
        group_id = group_of(episode)
        group = self.pending[group_id]
        group.append(episode)
        if len(group) >= env.config.group_size:
            del self.pending[group_id]
            assert self._on_group is not None, "TrainSink.on_group is not bound"
            await self._on_group(await self.finalize(group))

    async def finalize(self, episodes: list[vf.Episode]) -> Group:
        env = self.envs.get(env_of(episodes[0]))
        group = Group(env.name, group_of(episodes[0]), work_of(episodes[0]).step, episodes, admitted=False)
        traces = group.traces
        task_idx = next((trace.task.data.idx for trace in traces), None)
        errored = sum(not episode.ok for episode in episodes)

        def dropped(reason: str) -> Group:
            get_logger().debug(
                f"Dropped group | env={env.name} task_idx={task_idx} | "
                f"episodes={len(episodes)} traces={len(traces)} (errored={errored}) | reason={reason}"
            )
            return group

        # Every member shares the dispatch version: one stale attempt voids the group,
        # and a pipeline decision is not a task result the curriculum should see.
        if any(cancel_reason(episode) == "stale" for episode in episodes):
            for episode in episodes:
                if not is_cancelled(episode):
                    cancel(episode, "stale")
            return dropped("stale")

        survivors = [trace for _, trace in iter_trainable_traces(episodes)]
        if not survivors:
            return dropped("no trainable survivors")
        await env.algorithm.finalize_group(episodes)
        group.admitted = self._admit([episode for episode in episodes if episode.traces])
        if not group.admitted:
            return dropped("rejected by curriculum")

        temperature = env.sampling_args["temperature"]
        for trace in survivors:
            samples = await asyncio.to_thread(trace_to_samples, trace, env_name=env.name)
            for sample in samples:
                sample.temperatures = [temperature] * len(sample.token_ids)
                if env.requires_sampling_masks and sample.sampling_mask is None:
                    # Rollout logprobs are mask-renormalized; training without the masks
                    # silently biases every importance ratio.
                    raise RuntimeError(
                        f"env '{env.name}' samples with truncation (top_p/top_k) but its rollouts "
                        "carry no sampling masks. Set `enable_return_sampling_mask = true` on "
                        "the inference server config (the rl entrypoint does this automatically) - "
                        "it requires vLLM's native sampling-mask capture (>= 0.28)."
                    )
                stamp_loss_routing(sample, env.algorithm.action_loss_type)
            samples = [sample for sample in samples if prune_zero_advantages(sample)]
            if samples:
                group.samples[trace.id] = samples
        return group

    def status(self) -> str | None:
        buffered = sum(len(group) for group in self.pending.values())
        return f"+{buffered} buffered" if buffered else None
