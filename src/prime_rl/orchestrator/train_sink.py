"""Graph-native training sink.

Each completed topology invocation's accepted traces are compiled from their existing
tokens into trainer samples. Graphs are grouped only for algorithmic credit assignment,
then filtered and flattened into the existing ``TrainingSample`` boundary.
"""

from __future__ import annotations

import asyncio
import uuid
from collections import defaultdict

from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.orchestrator.algo import AlgorithmCompatibilityError
from prime_rl.orchestrator.envs import TrainEnvs
from prime_rl.orchestrator.filters import GraphFilter, apply_filters
from prime_rl.orchestrator.metrics import TrainGraphs
from prime_rl.orchestrator.trajectories import trace_to_samples
from prime_rl.orchestrator.types import AgentGraph, TrainBatch
from prime_rl.utils.logger import get_logger


class TrainSink:
    def __init__(
        self,
        config: OrchestratorConfig,
        *,
        tokenizer,
        train_envs: TrainEnvs,
        mm_token_type_ids_mapping: dict[int, int] | None,
        batch_size: int | None,
        token_batch_size: int | None,
        pre_filters: list[GraphFilter],
        post_filters: list[GraphFilter],
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
        self.pre_filters = pre_filters
        self.post_filters = post_filters

        multi_trace = [env.name for env in train_envs if env.algorithm.supports_multiple_traces]
        if multi_trace and token_batch_size is not None:
            raise AlgorithmCompatibilityError(
                f"multi-trace algorithms currently require graph-count batch_size; "
                f"token_batch_size is configured for {multi_trace}"
            )
        if multi_trace and (pre_filters or post_filters):
            raise AlgorithmCompatibilityError(
                f"trace filters are not yet defined for multi-trace algorithms; "
                f"clear pre_batch_filters and post_batch_filters for {multi_trace}"
            )

        self.pending_graphs = TrainGraphs()
        self.pending_groups: dict[uuid.UUID, list[AgentGraph]] = defaultdict(list)
        self.pending_batch: list[AgentGraph] = []
        self.pending_tokens = 0

        self.pre_filter_seen = 0
        self.pre_filter_dropped = 0
        self.pre_filter_dropped_by_name: dict[str, int] = {}

    def group_size_for(self, env_name: str) -> int:
        return self.train_envs.get(env_name).config.group_size

    def in_progress_groups(self) -> list[list[AgentGraph]]:
        return [graphs for graphs in self.pending_groups.values() if graphs]

    def batch_progress(self) -> tuple[int, int, str]:
        if self.batch_size is not None:
            return len(self.pending_batch), self.batch_size, "graphs"
        assert self.token_batch_size is not None
        return self.pending_tokens, self.token_batch_size, "tokens"

    def buffered_count(self) -> int:
        return sum(len(group) for group in self.in_progress_groups())

    def pending_batch_by_env(self) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for graph in self.pending_batch:
            counts[graph.env_name] += 1
        return dict(counts)

    async def add(self, graph: AgentGraph) -> TrainBatch | None:
        await self.process_graph(graph)
        self.pending_graphs.append(graph)
        self.pending_groups[graph.group_id].append(graph)
        if len(self.pending_groups[graph.group_id]) >= self.group_size_for(graph.env_name):
            await self.process_group(graph.group_id)
        ready = (
            len(self.pending_batch) >= self.batch_size
            if self.batch_size is not None
            else self.pending_tokens >= (self.token_batch_size or 0)
        )
        return self.process_batch() if ready else None

    async def process_graph(self, graph: AgentGraph) -> None:
        if graph.error is not None:
            return
        algorithm = self.train_envs.get(graph.env_name).algorithm
        algorithm.validate_graph(graph)
        for trace in algorithm.training_traces(graph):
            trace.samples = await asyncio.to_thread(
                trace_to_samples,
                trace,
                env_name=graph.env_name,
                mm_token_type_ids_mapping=self.mm_token_type_ids_mapping,
            )
            if not trace.samples:
                raise RuntimeError(
                    f"trainable trace {trace.id!r} produced no samples; ensure agent {trace.agent!r} "
                    "uses Prime's train client and returns token ids"
                )
            sampling = trace.sampling
            if sampling is None or sampling.temperature is None:
                raise RuntimeError(f"trainable trace {trace.id!r} has no resolved sampling temperature")
            for sample in trace.samples:
                sample.temperatures = [sampling.temperature] * len(sample.token_ids)
        await algorithm.finalize_graph(graph)

    async def process_group(self, group_id: uuid.UUID) -> None:
        group = self.pending_groups.pop(group_id, [])
        if not group:
            return
        env_name = group[0].env_name
        task_idx = group[0].task.data.idx
        algorithm = self.train_envs.get(env_name).algorithm
        survivors = [graph for graph in group if graph.error is None and algorithm.training_traces(graph)]
        num_errored = len(group) - len(survivors)
        if not survivors:
            get_logger().debug(
                f"Finished group | env={env_name} task_idx={task_idx} | "
                f"graphs={len(group)} (errored={num_errored}) | dropped: all failed"
            )
            return

        await algorithm.finalize_group(survivors)
        if self.pre_filters:
            apply_filters(self.pre_filters, survivors)

        filtered_by_name: dict[str, int] = {}
        num_filtered = 0
        for graph in survivors:
            self.pre_filter_seen += 1
            if graph.is_filtered:
                self.pre_filter_dropped += 1
                num_filtered += 1
                for name, hit in graph.filter_results.items():
                    if hit:
                        self.pre_filter_dropped_by_name[name] = self.pre_filter_dropped_by_name.get(name, 0) + 1
                        filtered_by_name[name] = filtered_by_name.get(name, 0) + 1
                continue
            graph.filter_results = {}
            graph.is_filtered = False
            self.pending_batch.append(graph)
            if self.token_batch_size is not None:
                self.pending_tokens += graph.num_total_tokens

        rewards = [graph.reward for graph in survivors]
        avg_reward = sum(rewards) / len(rewards)
        filter_str = ", ".join(f"{name}={count}" for name, count in filtered_by_name.items()) or "—"
        get_logger().debug(
            f"Finished group | env={env_name} task_idx={task_idx} | "
            f"graphs={len(group)} (errored={num_errored}, filtered={num_filtered}) | "
            f"reward={avg_reward:.4f} | filters: {filter_str}"
        )

    def process_batch(self) -> TrainBatch:
        if self.batch_size is not None:
            cohort = self.pending_batch[: self.batch_size]
            self.pending_batch = self.pending_batch[self.batch_size :]
        else:
            assert self.token_batch_size is not None
            cut = 0
            running = 0
            for cut, graph in enumerate(self.pending_batch, start=1):
                running += graph.num_total_tokens
                if running >= self.token_batch_size:
                    break
            cohort = self.pending_batch[:cut]
            self.pending_batch = self.pending_batch[cut:]
            self.pending_tokens -= running

        if self.post_filters:
            apply_filters(self.post_filters, cohort)
        samples = [
            sample
            for graph in cohort
            if not graph.is_filtered
            for trace in self.train_envs.get(graph.env_name).algorithm.training_traces(graph)
            for sample in trace.samples
        ]
        graphs = self.pending_graphs
        if samples:
            self.pending_graphs = TrainGraphs()
        return TrainBatch(graphs=graphs, samples=samples)

    def reset_pre_filter_stats(self) -> None:
        self.pre_filter_seen = 0
        self.pre_filter_dropped = 0
        self.pre_filter_dropped_by_name.clear()
