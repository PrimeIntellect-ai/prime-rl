"""Evaluation-side episode, group, and epoch assembly."""

from __future__ import annotations

import uuid
from collections import defaultdict

from prime_rl.orchestrator.envs import EvalEnvs
from prime_rl.orchestrator.metrics import EvalEpisodes
from prime_rl.orchestrator.types import EpisodeRun, EvalBatch
from prime_rl.utils.logger import get_logger


class EvalSink:
    """Collect completed evaluation episodes into per-environment epochs."""

    def __init__(self, *, eval_envs: EvalEnvs) -> None:
        self.eval_envs = eval_envs
        self.pending_groups: dict[uuid.UUID, list[EpisodeRun]] = defaultdict(list)
        self.pending_batches: dict[tuple[str, int], list[EpisodeRun]] = defaultdict(list)

    def add(self, run: EpisodeRun) -> EvalBatch | None:
        env_name = run.context.env_name
        group_id = run.context.group_id
        eval_step = run.context.eval_step
        assert eval_step is not None
        bkey = (env_name, eval_step)
        group = self.pending_groups[group_id]
        group.append(run)
        if len(group) >= self.group_size_for(env_name):
            self.process_group(group_id)
        if len(self.pending_batches[bkey]) >= self.batch_size_for(env_name):
            return self.process_batch(bkey)
        return None

    def group_size_for(self, env_name: str) -> int:
        return self.eval_envs.get(env_name).config.group_size

    def batch_size_for(self, env_name: str) -> int:
        env = self.eval_envs.get(env_name)
        return len(env.examples) * env.config.group_size

    def batch_progress(self) -> list[tuple[str, int, int, int, int]]:
        batch_counts = {key: len(runs) for key, runs in self.pending_batches.items()}
        buffered: dict[tuple[str, int], int] = {}
        for group in self.pending_groups.values():
            if not group:
                continue
            context = group[0].context
            assert context.eval_step is not None
            key = (context.env_name, context.eval_step)
            buffered[key] = buffered.get(key, 0) + len(group)
        return [
            (
                env_name,
                eval_step,
                batch_counts.get((env_name, eval_step), 0),
                self.batch_size_for(env_name),
                buffered.get((env_name, eval_step), 0),
            )
            for env_name, eval_step in set(batch_counts) | set(buffered)
        ]

    def process_group(self, group_id: uuid.UUID) -> None:
        group = self.pending_groups.pop(group_id, [])
        if not group:
            return
        context = group[0].context
        assert context.eval_step is not None
        self.pending_batches[(context.env_name, context.eval_step)].extend(group)

        traces = [trace for run in group for trace in run.traces]
        survivors = [trace for trace in traces if not trace.has_error]
        num_errored = len(traces) - len(survivors) + sum(not run.episode.ok for run in group if not run.traces)
        rewards = [trace.reward for trace in survivors]
        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
        get_logger().debug(
            f"Finished group | env={context.env_name} task_idx={context.task.data.idx} "
            f"eval_step={context.eval_step} | episodes={len(group)} traces={len(traces)} "
            f"(errored={num_errored}) | reward={avg_reward:.4f}"
        )

    def process_batch(self, key: tuple[str, int]) -> EvalBatch:
        env_name, step = key
        episodes = self.pending_batches.pop(key, [])
        return EvalBatch(env_name=env_name, step=step, episodes=EvalEpisodes(episodes))
