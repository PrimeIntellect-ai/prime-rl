from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from prime_rl.orchestrator.algo.base import Algorithm, AlgorithmCompatibilityError

if TYPE_CHECKING:
    from prime_rl.orchestrator.types import AgentGraph, TrainingTrace


def _assign_centered_advantages(traces: list[TrainingTrace]) -> None:
    if not traces:
        return
    rewards = torch.tensor([trace.reward for trace in traces], dtype=torch.float32)
    for trace, advantage in zip(traces, (rewards - rewards.mean()).tolist(), strict=True):
        trace.assign_advantages(advantage)


class ProposerSolverAlgorithm(Algorithm):
    """GRPO within solver panels and across replicated proposers."""

    supports_multiple_traces = True
    topology = "proposer-solver-v1"

    def validate_graph(self, graph: AgentGraph) -> None:
        if graph.topology != self.topology:
            raise AlgorithmCompatibilityError(
                f"algorithm {self.config.type!r} requires topology {self.topology!r}, got {graph.topology!r}"
            )
        unexpected = sorted({trace.agent or "<unnamed>" for trace in graph.traces} - {"proposer", "solver"})
        proposers = graph.by_agent("proposer")
        if unexpected or len(proposers) != 1:
            raise AlgorithmCompatibilityError(
                f"topology {self.topology!r} must return exactly one proposer and only solver children; "
                f"found proposers={len(proposers)}, unexpected_agents={unexpected}"
            )
        non_trainable = [trace.agent for trace in graph.traces if not trace.trainable]
        if non_trainable:
            raise AlgorithmCompatibilityError(
                f"algorithm {self.config.type!r} trains proposer and solver traces; "
                f"found non-trainable agents {non_trainable}"
            )
        proposer = proposers[0]
        misparented = [trace.id for trace in graph.by_agent("solver") if trace.parents != [proposer.id]]
        if proposer.parents or misparented:
            raise AlgorithmCompatibilityError(
                f"topology {self.topology!r} requires a root proposer with every solver as its child; "
                f"proposer_parents={proposer.parents}, misparented_solvers={misparented}"
            )

    async def score_graph(self, graph: AgentGraph) -> None:
        solvers = [trace for trace in self.training_traces(graph) if trace.agent == "solver"]
        _assign_centered_advantages(solvers)

    async def score_group(self, group: list[AgentGraph]) -> None:
        proposers = [trace for graph in group for trace in self.training_traces(graph) if trace.agent == "proposer"]
        _assign_centered_advantages(proposers)
