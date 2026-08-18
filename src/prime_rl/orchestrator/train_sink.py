"""Training-side episode, group, and batch assembly."""

from __future__ import annotations

import asyncio
import uuid
from collections import defaultdict
from collections.abc import Callable

from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.orchestrator.envs import TrainEnvs
from prime_rl.orchestrator.metrics import TrainEpisodes
from prime_rl.orchestrator.trajectories import trace_to_samples
from prime_rl.orchestrator.types import EpisodeRun, TrainBatch, TrainingTrace
from prime_rl.transport import TrainingSample
from prime_rl.utils.logger import get_logger

MAX_CONSECUTIVE_ZERO_OUTPUT_BATCH_EQUIVALENTS = 10


def payload_tokens(trace: TrainingTrace) -> int:
    """Token cost of one trainer-bound trace."""
    return sum(len(sample.token_ids) for sample in trace.samples) or trace.trace.num_total_tokens


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
        on_result: Callable[[list[EpisodeRun]], bool] | None = None,
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
        self.pending_groups: dict[uuid.UUID, list[EpisodeRun]] = defaultdict(list)
        self.pending_batch: list[TrainingTrace] = []
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
        for trace in self.pending_batch:
            counts[trace.context.env_name] += 1
        return dict(counts)

    async def add(self, run: EpisodeRun) -> TrainBatch | None:
        """Process one completed episode and return a batch when ready."""
        await self.process_episode(run)
        group_id = run.context.group_id
        env_name = run.context.env_name
        group = self.pending_groups[group_id]
        group.append(run)
        if len(group) < self.group_size_for(env_name):
            return None

        await self.process_group(group_id)
        ready = (
            len(self.pending_batch) >= self.batch_size
            if self.batch_size is not None
            else self.pending_tokens >= (self.token_batch_size or 0)
        )
        return self.process_batch() if ready else None

    async def process_episode(self, run: EpisodeRun) -> None:
        """Tokenize the clean trainable traces in one episode."""
        env = self.train_envs.get(run.context.env_name)
        for trace in run.traces:
            if trace.has_error or not trace.agent.trainable:
                continue
            samples = await asyncio.to_thread(
                trace_to_samples,
                trace,
                env_name=run.context.env_name,
                mm_token_type_ids_mapping=self.mm_token_type_ids_mapping,
            )
            training = TrainingTrace(
                context=run.context,
                episode=run.episode,
                trace=trace,
                samples=samples or [],
            )
            run.training.append(training)
            await env.algorithm.finalize_trace(training)

    async def process_group(self, group_id: uuid.UUID) -> None:
        group = self.pending_groups.pop(group_id, [])
        if not group:
            return

        self.pending_episodes.extend(group)
        env_name = group[0].context.env_name
        task_idx = group[0].context.task.data.idx
        traces = [trace for run in group for trace in run.traces]
        survivors = [training for run in group for training in run.training]
        num_errored = sum(trace.has_error for trace in traces) + sum(
            not run.episode.ok for run in group if not run.traces
        )

        if not survivors:
            self._admit(group)
            self._record_zero_output(group)
            get_logger().debug(
                f"Finished group | env={env_name} task_idx={task_idx} | "
                f"episodes={len(group)} traces={len(traces)} (errored={num_errored}) | "
                "dropped: no trainable survivors"
            )
            return

        env = self.train_envs.get(env_name)
        await env.algorithm.finalize_group(survivors)
        temperature = env.sampling_args["temperature"]
        for training in survivors:
            for sample in training.samples:
                sample.temperatures = [temperature] * len(sample.token_ids)

        if not self._admit(group):
            self._record_zero_output(group)
            get_logger().debug(
                f"Finished group | env={env_name} task_idx={task_idx} | "
                f"episodes={len(group)} traces={len(traces)} (errored={num_errored}) | rejected by curriculum"
            )
            return

        self.pending_batch.extend(survivors)
        if self.token_batch_size is not None:
            self.pending_tokens += sum(payload_tokens(training) for training in survivors)
        self.zero_output_units = 0
        self.reported_zero_output_windows = 0

        rewards = [training.trace.reward for training in survivors]
        avg_reward = sum(rewards) / len(rewards)
        get_logger().debug(
            f"Finished group | env={env_name} task_idx={task_idx} | "
            f"episodes={len(group)} traces={len(traces)} (errored={num_errored}) | reward={avg_reward:.4f}"
        )

    def _admit(self, group: list[EpisodeRun]) -> bool:
        admitted = self.on_result(group) if self.on_result is not None else True
        for run in group:
            run.is_admitted = admitted
        return admitted

    def _record_zero_output(self, group: list[EpisodeRun]) -> None:
        if self.batch_size is not None:
            self.zero_output_units += len(group)
        else:
            payload = sum(payload_tokens(trace) for run in group for trace in run.training)
            episode_tokens = sum(run.episode.num_total_tokens for run in group)
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
        if self.batch_size is not None:
            cohort = self.pending_batch[: self.batch_size]
            self.pending_batch = self.pending_batch[self.batch_size :]
        else:
            assert self.token_batch_size is not None
            cut = 0
            running = 0
            for index, trace in enumerate(self.pending_batch):
                running += payload_tokens(trace)
                cut = index + 1
                if running >= self.token_batch_size:
                    break
            cohort = self.pending_batch[:cut]
            self.pending_batch = self.pending_batch[cut:]
            self.pending_tokens -= running

        if self.config.train.filter_zero_advantages:
            for trace in cohort:
                trace.samples = [sample for sample in trace.samples if _prune_zero_advantages(sample)]
        samples = [sample for trace in cohort for sample in trace.samples]

        episodes = self.pending_episodes
        if samples:
            self.pending_episodes = TrainEpisodes()
        return TrainBatch(episodes=episodes, samples=samples)
