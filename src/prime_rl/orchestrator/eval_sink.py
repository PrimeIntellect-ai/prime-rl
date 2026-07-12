"""Graph-native evaluation sink."""

from __future__ import annotations

import uuid
from collections import defaultdict

from prime_rl.orchestrator.envs import EvalEnvs
from prime_rl.orchestrator.metrics import EvalGraphs
from prime_rl.orchestrator.types import AgentGraph, EvalBatch
from prime_rl.utils.logger import get_logger


class EvalSink:
    def __init__(self, *, eval_envs: EvalEnvs) -> None:
        self.eval_envs = eval_envs
        self.pending_groups: dict[uuid.UUID, list[AgentGraph]] = defaultdict(list)
        self.pending_batches: dict[tuple[str, int], list[AgentGraph]] = defaultdict(list)

    def add(self, graph: AgentGraph) -> EvalBatch | None:
        env_name = graph.env_name
        assert graph.eval_step is not None
        key = (env_name, graph.eval_step)
        self.pending_groups[graph.group_id].append(graph)
        if len(self.pending_groups[graph.group_id]) >= self.group_size_for(env_name):
            self.process_group(graph.group_id)
        if len(self.pending_batches[key]) >= self.batch_size_for(env_name):
            return self.process_batch(key)
        return None

    def group_size_for(self, env_name: str) -> int:
        return self.eval_envs.get(env_name).config.group_size

    def batch_size_for(self, env_name: str) -> int:
        env = self.eval_envs.get(env_name)
        return len(env.examples) * env.config.group_size

    def batch_progress(self) -> list[tuple[str, int, int, int, int]]:
        batch_counts = {key: len(bucket) for key, bucket in self.pending_batches.items()}
        buffered: dict[tuple[str, int], int] = {}
        for graphs in self.pending_groups.values():
            if not graphs:
                continue
            graph = graphs[0]
            assert graph.eval_step is not None
            key = (graph.env_name, graph.eval_step)
            buffered[key] = buffered.get(key, 0) + len(graphs)
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
        graph = group[0]
        assert graph.eval_step is not None
        self.pending_batches[(graph.env_name, graph.eval_step)].extend(group)
        survivors = [item for item in group if not item.has_error]
        rewards = [item.reward for item in survivors]
        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
        get_logger().debug(
            f"Finished group | env={graph.env_name} task_idx={graph.task.data.idx} "
            f"eval_step={graph.eval_step} | graphs={len(group)} "
            f"(errored={len(group) - len(survivors)}) | reward={avg_reward:.4f}"
        )

    def process_batch(self, key: tuple[str, int]) -> EvalBatch:
        env_name, step = key
        graphs = self.pending_batches.pop(key, [])
        return EvalBatch(env_name=env_name, step=step, graphs=EvalGraphs(graphs))
