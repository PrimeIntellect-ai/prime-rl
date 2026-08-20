"""Evaluation-side episode, group, and epoch assembly."""

from __future__ import annotations

from collections import defaultdict

import verifiers.v1 as vf

from prime_rl.orchestrator.envs import EvalEnvs
from prime_rl.orchestrator.metrics import EvalEpisodes
from prime_rl.orchestrator.provenance import episode_env_name, episode_group_id, eval_work
from prime_rl.orchestrator.types import EvalBatch
from prime_rl.utils.logger import get_logger


class EvalSink:
    """Collect completed evaluation episodes into per-environment epochs."""

    def __init__(self, *, eval_envs: EvalEnvs) -> None:
        self.eval_envs = eval_envs
        self.pending_groups: dict[str, list[vf.Episode]] = defaultdict(list)
        self.pending_batches: dict[tuple[str, int], list[vf.Episode]] = defaultdict(list)

    def add(self, episode: vf.Episode) -> EvalBatch | None:
        env_name = episode_env_name(episode)
        group_id = episode_group_id(episode)
        eval_step = eval_work(episode).step
        bkey = (env_name, eval_step)
        group = self.pending_groups[group_id]
        group.append(episode)
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
            episode = group[0]
            eval_step = eval_work(episode).step
            key = (episode_env_name(episode), eval_step)
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

    def process_group(self, group_id: str) -> None:
        group = self.pending_groups.pop(group_id, [])
        if not group:
            return
        episode = group[0]
        env_name = episode_env_name(episode)
        eval_step = eval_work(episode).step
        self.pending_batches[(env_name, eval_step)].extend(group)

        traces = [trace for episode in group for trace in episode.traces]
        survivors = [trace for trace in traces if not trace.has_error]
        num_errored = len(traces) - len(survivors) + sum(not episode.ok for episode in group if not episode.traces)
        rewards = [trace.reward for trace in survivors]
        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
        task_idx = next((trace.task.data.idx for trace in traces), None)
        get_logger().debug(
            f"Finished group | env={env_name} task_idx={task_idx} "
            f"eval_step={eval_step} | episodes={len(group)} traces={len(traces)} "
            f"(errored={num_errored}) | reward={avg_reward:.4f}"
        )

    def process_batch(self, key: tuple[str, int]) -> EvalBatch:
        env_name, step = key
        episodes = self.pending_batches.pop(key, [])
        return EvalBatch(env_name=env_name, step=step, episodes=EvalEpisodes(episodes))
