"""Evaluation-side episode, group, and epoch assembly."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from prime_rl.orchestrator.envs import EvalEnvs
from prime_rl.orchestrator.metrics import EvalEpisodes
from prime_rl.orchestrator.types import DispatchFailure, EvalBatch, GroupCancellation
from prime_rl.orchestrator.utils import episode_env_name, episode_group_id, eval_work
from prime_rl.utils.logger import get_logger

if TYPE_CHECKING:
    import verifiers.v1 as vf


class EvalSink:
    """Collect completed evaluation episodes into per-environment epochs."""

    def __init__(self, *, eval_envs: EvalEnvs) -> None:
        self.eval_envs = eval_envs
        self.pending_groups: dict[str, list[vf.Episode]] = defaultdict(list)
        self.pending_group_failures: dict[str, list[DispatchFailure]] = defaultdict(list)
        self.pending_group_cancellations: dict[str, GroupCancellation] = {}
        self.pending_batches: dict[tuple[str, int], list[vf.Episode]] = defaultdict(list)
        self.pending_batch_failures: dict[tuple[str, int], list[DispatchFailure]] = defaultdict(list)
        self.pending_batch_cancellations: dict[tuple[str, int], int] = defaultdict(int)
        self.expected_batch_sizes: dict[tuple[str, int], int] = {}

    def add(self, episode: vf.Episode) -> EvalBatch | None:
        env_name = episode_env_name(episode)
        group_id = episode_group_id(episode)
        eval_step = eval_work(episode).step
        bkey = (env_name, eval_step)
        group = self.pending_groups[group_id]
        group.append(episode)
        if self._group_size(group_id) >= self.group_size_for(env_name):
            self.process_group(group_id)
        if self._batch_size(bkey) >= self.batch_size_for(env_name, eval_step):
            return self.process_batch(bkey)
        return None

    def fail(self, failure: DispatchFailure) -> EvalBatch | None:
        """Count a request failure toward its group and eval epoch without
        manufacturing a verifier episode."""
        if failure.kind != "eval":
            raise ValueError(f"EvalSink cannot process a {failure.kind} dispatch failure")
        group_id = failure.group_id
        bkey = (failure.env_name, failure.step)
        self.pending_group_failures[group_id].append(failure)
        if self._group_size(group_id) >= self.group_size_for(failure.env_name):
            self.process_group(group_id)
        if self._batch_size(bkey) >= self.batch_size_for(failure.env_name, failure.step):
            return self.process_batch(bkey)
        return None

    def cancel(self, cancellation: GroupCancellation) -> EvalBatch | None:
        """Count a dispatcher-cancelled group toward its superseded eval epoch."""
        if cancellation.kind != "eval":
            raise ValueError(f"EvalSink cannot process a {cancellation.kind} group cancellation")
        group_id = cancellation.group_id
        bkey = (cancellation.env_name, cancellation.step)
        self.pending_group_cancellations[group_id] = cancellation
        if self._group_size(group_id) >= self.group_size_for(cancellation.env_name):
            self.process_group(group_id)
        if self._batch_size(bkey) >= self.batch_size_for(cancellation.env_name, cancellation.step):
            return self.process_batch(bkey)
        return None

    def _group_size(self, group_id: str) -> int:
        cancellation = self.pending_group_cancellations.get(group_id)
        return (
            len(self.pending_groups[group_id])
            + len(self.pending_group_failures[group_id])
            + (cancellation.count if cancellation is not None else 0)
        )

    def _batch_size(self, key: tuple[str, int]) -> int:
        return (
            len(self.pending_batches[key])
            + len(self.pending_batch_failures[key])
            + self.pending_batch_cancellations[key]
        )

    def group_size_for(self, env_name: str) -> int:
        return self.eval_envs.get(env_name).config.group_size

    def set_batch_size(self, env_name: str, step: int, size: int) -> None:
        self.expected_batch_sizes[(env_name, step)] = size

    def batch_size_for(self, env_name: str, step: int | None = None) -> int:
        if step is not None:
            key = (env_name, step)
            if key in self.expected_batch_sizes:
                return self.expected_batch_sizes[key]
        env = self.eval_envs.get(env_name)
        return len(env.examples) * env.config.group_size

    def batch_progress(self) -> list[tuple[str, int, int, int, int]]:
        keys = (
            set(self.pending_batches)
            | set(self.pending_batch_failures)
            | set(self.pending_batch_cancellations)
            | set(self.expected_batch_sizes)
        )
        batch_counts = {key: self._batch_size(key) for key in keys}
        buffered: dict[tuple[str, int], int] = {}
        group_ids = set(self.pending_groups) | set(self.pending_group_failures) | set(self.pending_group_cancellations)
        for group_id in group_ids:
            group = self.pending_groups[group_id]
            failures = self.pending_group_failures[group_id]
            cancellation = self.pending_group_cancellations.get(group_id)
            if not group and not failures and cancellation is None:
                continue
            if group:
                env_name = episode_env_name(group[0])
                eval_step = eval_work(group[0]).step
            elif failures:
                env_name = failures[0].env_name
                eval_step = failures[0].step
            else:
                env_name = cancellation.env_name
                eval_step = cancellation.step
            key = (env_name, eval_step)
            buffered[key] = (
                buffered.get(key, 0)
                + len(group)
                + len(failures)
                + (cancellation.count if cancellation is not None else 0)
            )
        return [
            (
                env_name,
                eval_step,
                batch_counts.get((env_name, eval_step), 0),
                self.batch_size_for(env_name, eval_step),
                buffered.get((env_name, eval_step), 0),
            )
            for env_name, eval_step in set(batch_counts) | set(buffered)
        ]

    def has_pending_group(self, group_id: str) -> bool:
        return bool(
            self.pending_groups.get(group_id)
            or self.pending_group_failures.get(group_id)
            or self.pending_group_cancellations.get(group_id)
        )

    def process_group(self, group_id: str) -> None:
        group = self.pending_groups.pop(group_id, [])
        failures = self.pending_group_failures.pop(group_id, [])
        cancellation = self.pending_group_cancellations.pop(group_id, None)
        if not group and not failures and cancellation is None:
            return
        if group:
            env_name = episode_env_name(group[0])
            eval_step = eval_work(group[0]).step
        elif failures:
            env_name = failures[0].env_name
            eval_step = failures[0].step
        else:
            env_name = cancellation.env_name
            eval_step = cancellation.step
        key = (env_name, eval_step)
        self.pending_batches[key].extend(group)
        self.pending_batch_failures[key].extend(failures)
        if cancellation is not None:
            self.pending_batch_cancellations[key] += cancellation.count

        traces = [trace for episode in group for trace in episode.traces]
        survivors = [trace for trace in traces if not trace.has_error]
        num_errored = len(traces) - len(survivors) + sum(not episode.ok for episode in group if not episode.traces)
        rewards = [trace.reward for trace in survivors]
        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
        task_idx = next((trace.task.data.idx for trace in traces), None)
        get_logger().debug(
            f"Finished group | env={env_name} task_idx={task_idx} "
            f"eval_step={eval_step} | episodes={len(group)} dispatch_failures={len(failures)} "
            f"cancelled={cancellation.count if cancellation is not None else 0} "
            f"traces={len(traces)} (errored={num_errored + len(failures)}) | reward={avg_reward:.4f}"
        )

    def process_batch(self, key: tuple[str, int]) -> EvalBatch:
        env_name, step = key
        episodes = self.pending_batches.pop(key, [])
        failures = self.pending_batch_failures.pop(key, [])
        cancelled = self.pending_batch_cancellations.pop(key, 0)
        self.expected_batch_sizes.pop(key, None)
        return EvalBatch(
            env_name=env_name,
            step=step,
            episodes=EvalEpisodes(episodes, group_size=self.group_size_for(env_name)),
            failures=failures,
            cancelled=cancelled,
        )
