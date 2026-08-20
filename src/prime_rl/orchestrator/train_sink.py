"""Training-side episode, group, and batch assembly."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Callable

import verifiers.v1 as vf

from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.orchestrator.envs import TrainEnvs
from prime_rl.orchestrator.metrics import TrainEpisodes
from prime_rl.orchestrator.trajectories import trace_to_samples
from prime_rl.orchestrator.types import PreparedGroup, PreparedTrace, TrainBatch
from prime_rl.transports.rollouts import TrainingSample
from prime_rl.utils.logger import get_logger

MAX_CONSECUTIVE_ZERO_OUTPUT_BATCH_EQUIVALENTS = 10


def payload_tokens(prepared: PreparedTrace, trace: vf.Trace | None = None) -> int:
    """Token cost of one trainer-bound trace."""
    return sum(len(sample.token_ids) for sample in prepared) or (trace.num_total_tokens if trace is not None else 0)


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
    """Build training traces from episodes, finalize groups, and form batches."""

    def __init__(
        self,
        config: OrchestratorConfig,
        *,
        tokenizer,
        train_envs: TrainEnvs,
        mm_token_type_ids_mapping: dict[int, int] | None,
        batch_size: int | None,
        token_batch_size: int | None,
        on_result: Callable[[list[vf.Episode], PreparedGroup], bool] | None = None,
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

        self.pending_episodes = TrainEpisodes()
        self.pending_groups: dict[str, list[vf.Episode]] = defaultdict(list)
        self.prepared_by_episode: dict[str, PreparedGroup] = {}
        self.pending_batch: dict[str, PreparedTrace] = {}
        self.episode_by_trace: dict[str, vf.Episode] = {}
        self.pending_tokens = 0
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
            counts[episode.env.name or episode.env.id] += 1
        return dict(counts)

    async def add(self, episode: vf.Episode) -> TrainBatch | None:
        """Process one completed episode and return a batch when ready."""
        await self.process_episode(episode)
        group_id = episode.group_id
        if group_id is None:
            raise ValueError("Train episode is missing group_id")
        env_name = episode.env.name or episode.env.id
        group = self.pending_groups[group_id]
        group.append(episode)
        if len(group) < self.group_size_for(env_name):
            return None

        await self.process_group(group_id)
        ready = (
            len(self.pending_batch) >= self.batch_size
            if self.batch_size is not None
            else self.pending_tokens >= (self.token_batch_size or 0)
        )
        return self.process_batch() if ready else None

    async def process_episode(self, episode: vf.Episode) -> None:
        """Tokenize the clean trainable traces in one episode."""
        env_name = episode.env.name or episode.env.id
        env = self.train_envs.get(env_name)
        prepared: PreparedGroup = {}
        for trace in episode.traces:
            if trace.has_error or not trace.agent.trainable:
                continue
            samples = await asyncio.to_thread(
                trace_to_samples,
                trace,
                env_name=env_name,
                mm_token_type_ids_mapping=self.mm_token_type_ids_mapping,
            )
            prepared[trace.id] = samples or []
            await env.algorithm.finalize_trace(episode, trace, prepared[trace.id])
        self.prepared_by_episode[episode.id] = prepared

    async def process_group(self, group_id: str) -> None:
        group = self.pending_groups.pop(group_id, [])
        if not group:
            return

        env_name = group[0].env.name or group[0].env.id
        traces = [trace for episode in group for trace in episode.traces]
        task_idx = next((trace.task.data.idx for trace in traces), None)
        prepared = {
            trace_id: samples
            for episode in group
            for trace_id, samples in self.prepared_by_episode.pop(episode.id, {}).items()
        }
        num_errored = sum(trace.has_error for trace in traces) + sum(
            not episode.ok for episode in group if not episode.traces
        )

        if not prepared:
            admitted = self._admit(group, prepared)
            self.pending_episodes.extend(group, prepared, admitted=admitted)
            self._record_zero_output(group, prepared)
            get_logger().debug(
                f"Finished group | env={env_name} task_idx={task_idx} | "
                f"episodes={len(group)} traces={len(traces)} (errored={num_errored}) | "
                "dropped: no trainable survivors"
            )
            return

        env = self.train_envs.get(env_name)
        await env.algorithm.finalize_group(group, prepared)
        temperature = env.sampling_args["temperature"]
        for samples in prepared.values():
            for sample in samples:
                sample.temperatures = [temperature] * len(sample.token_ids)

        admitted = self._admit(group, prepared)
        self.pending_episodes.extend(group, prepared, admitted=admitted)
        if not admitted:
            self._record_zero_output(group, prepared)
            get_logger().debug(
                f"Finished group | env={env_name} task_idx={task_idx} | "
                f"episodes={len(group)} traces={len(traces)} (errored={num_errored}) | rejected by curriculum"
            )
            return

        self.pending_batch.update(prepared)
        for episode in group:
            for trace in episode.traces:
                if trace.id in prepared:
                    self.episode_by_trace[trace.id] = episode
        if self.token_batch_size is not None:
            self.pending_tokens += sum(
                payload_tokens(samples, self._trace(trace_id)) for trace_id, samples in prepared.items()
            )
        self.zero_output_units = 0
        self.reported_zero_output_windows = 0

        rewards = [trace.reward for trace in traces if trace.id in prepared]
        avg_reward = sum(rewards) / len(rewards)
        get_logger().debug(
            f"Finished group | env={env_name} task_idx={task_idx} | "
            f"episodes={len(group)} traces={len(traces)} (errored={num_errored}) | reward={avg_reward:.4f}"
        )

    def _trace(self, trace_id: str) -> vf.Trace:
        episode = self.episode_by_trace[trace_id]
        return next(trace for trace in episode.traces if trace.id == trace_id)

    def _admit(self, group: list[vf.Episode], prepared: PreparedGroup) -> bool:
        return self.on_result(group, prepared) if self.on_result is not None else True

    def _record_zero_output(self, group: list[vf.Episode], prepared: PreparedGroup) -> None:
        if self.batch_size is not None:
            returned_traces = sum(len(episode.traces) for episode in group)
            self.zero_output_units += len(prepared) or returned_traces or len(group)
        else:
            traces = {trace.id: trace for episode in group for trace in episode.traces}
            payload = sum(payload_tokens(samples, traces[trace_id]) for trace_id, samples in prepared.items())
            episode_tokens = sum(episode.num_total_tokens for episode in group)
            self.zero_output_units += payload or episode_tokens or self.config.seq_len * len(group)
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
        items = list(self.pending_batch.items())
        if self.batch_size is not None:
            selected = items[: self.batch_size]
        else:
            assert self.token_batch_size is not None
            cut = 0
            running = 0
            for index, (trace_id, prepared) in enumerate(items):
                running += payload_tokens(prepared, self._trace(trace_id))
                cut = index + 1
                if running >= self.token_batch_size:
                    break
            selected = items[:cut]
            self.pending_tokens -= running

        selected_prepared = dict(selected)
        selected_ids = set(selected_prepared)
        for trace_id in selected_ids:
            del self.pending_batch[trace_id]

        if self.config.train.filter_zero_advantages:
            for trace_id, prepared in selected:
                selected_prepared[trace_id] = [sample for sample in prepared if _prune_zero_advantages(sample)]
        samples = [sample for prepared in selected_prepared.values() for sample in prepared]

        traces_by_episode: dict[int, list[vf.Trace]] = defaultdict(list)
        selected_episodes: dict[int, vf.Episode] = {}
        for trace_id in selected_ids:
            episode = self.episode_by_trace.pop(trace_id)
            selected_episodes[id(episode)] = episode
            traces_by_episode[id(episode)].extend(trace for trace in episode.traces if trace.id == trace_id)
        cohort_episodes = [
            episode.model_copy(update={"traces": traces_by_episode[id(episode)]})
            for episode in selected_episodes.values()
        ]
        cohort = TrainEpisodes(cohort_episodes, selected_prepared)

        episodes = self.pending_episodes
        if samples:
            self.pending_episodes = TrainEpisodes()
        return TrainBatch(episodes=episodes, cohort=cohort, samples=samples)
