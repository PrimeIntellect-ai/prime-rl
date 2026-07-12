import asyncio
from types import SimpleNamespace

import pytest
import verifiers.v1 as vf

from prime_rl.configs.algorithm import GRPOAlgoConfig, ProposerSolverAlgoConfig
from prime_rl.orchestrator.algo import AlgorithmCompatibilityError
from prime_rl.orchestrator.algo.grpo import GRPOAlgorithm
from prime_rl.orchestrator.algo.proposer_solver import ProposerSolverAlgorithm
from prime_rl.orchestrator.train_sink import TrainSink
from prime_rl.orchestrator.types import AgentGraph, TrainingTrace


def make_graph(reward: float, temperature: float) -> AgentGraph:
    task = vf.TraceTask(type="Task", data=vf.WireTaskData(idx=0, prompt="q"))
    node = vf.MessageNode(
        message=vf.AssistantMessage(content="a"),
        token_ids=[1, 2],
        mask=[True, True],
        logprobs=[-0.1, -0.2],
        sampled=True,
    )
    solver = TrainingTrace(
        task=task,
        agent="solver",
        trainable=True,
        nodes=[node],
        rewards={"correct": reward},
        sampling=vf.SamplingConfig(temperature=temperature),
    )
    judge = TrainingTrace(
        task=task,
        agent="judge",
        trainable=False,
        sampling=vf.SamplingConfig(temperature=0.1),
    )
    return AgentGraph(topology="llm-judge", task=task, traces=[solver, judge], env_name="test")


def make_proposer_solver_graph(proposer_reward: float, solver_rewards: list[float]) -> AgentGraph:
    task = vf.TraceTask(type="Task", data=vf.WireTaskData(idx=0, prompt="q"))

    def trace(agent: str, reward: float, token_id: int) -> TrainingTrace:
        return TrainingTrace(
            task=task,
            agent=agent,
            nodes=[
                vf.MessageNode(
                    message=vf.AssistantMessage(content="a"),
                    token_ids=[token_id],
                    mask=[True],
                    logprobs=[-0.1],
                    sampled=True,
                )
            ],
            rewards={"reward": reward},
            sampling=vf.SamplingConfig(temperature=0.5),
        )

    proposer = trace("proposer", proposer_reward, 1)
    traces = [proposer]
    for idx, reward in enumerate(solver_rewards):
        solver = trace("solver", reward, idx + 2)
        solver.parents = [proposer.id]
        traces.append(solver)
    return AgentGraph(topology="proposer-solver-v1", task=task, traces=traces, env_name="test")


class FakeEnvs:
    def __init__(self, algorithm, group_size: int):
        self.env = SimpleNamespace(name="test", config=SimpleNamespace(group_size=group_size), algorithm=algorithm)

    def get(self, _name):
        return self.env

    def __iter__(self):
        yield self.env


def test_graph_group_trains_only_the_trainable_trace_with_its_actual_temperature():
    async def run():
        envs = FakeEnvs(GRPOAlgorithm(GRPOAlgoConfig(), policy_pool=None), group_size=2)
        sink = TrainSink(
            SimpleNamespace(),
            tokenizer=None,
            train_envs=envs,
            mm_token_type_ids_mapping=None,
            batch_size=2,
            token_batch_size=None,
            pre_filters=[],
            post_filters=[],
        )
        success = make_graph(1.0, 0.35)
        failure = make_graph(0.0, 0.8)
        failure.group_id = success.group_id

        assert await sink.add(success) is None
        batch = await sink.add(failure)

        assert batch is not None
        assert len(batch.graphs) == 2
        assert [sample.temperatures for sample in batch.samples] == [[0.35, 0.35], [0.8, 0.8]]
        assert [sample.advantages for sample in batch.samples] == [[0.5, 0.5], [-0.5, -0.5]]
        assert all(not trace.samples for graph in batch.graphs for trace in graph.traces if not trace.trainable)

    asyncio.run(run())


def test_graph_requires_exactly_one_trainable_trace():
    graph = make_graph(1.0, 1.0)
    graph.traces[1].trainable = True

    with pytest.raises(ValueError, match="exactly one trainable trace"):
        graph.training_trace


def test_graph_from_wire_preserves_task_specific_fields():
    wire = vf.AgentGraph.load(
        {
            "id": "graph",
            "topology": "swe-style-judge",
            "task": {
                "type": "SWEBenchTask",
                "data": {"idx": 0, "prompt": "fix it", "difficulty": "hard"},
            },
            "traces": [],
        }
    )

    graph = AgentGraph.from_wire(wire)

    assert graph.task.data.difficulty == "hard"


def test_builtin_algorithm_rejects_multiple_trainable_traces():
    graph = make_proposer_solver_graph(1.0, [1.0, 0.0])
    algorithm = GRPOAlgorithm(GRPOAlgoConfig(), policy_pool=None)

    with pytest.raises(ValueError, match="requires exactly one trainable trace"):
        algorithm.validate_graph(graph)


def test_proposer_solver_assigns_solver_and_proposer_grpo_credit():
    async def run():
        algorithm = ProposerSolverAlgorithm(ProposerSolverAlgoConfig(), policy_pool=None)
        sink = TrainSink(
            SimpleNamespace(),
            tokenizer=None,
            train_envs=FakeEnvs(algorithm, group_size=2),
            mm_token_type_ids_mapping=None,
            batch_size=2,
            token_batch_size=None,
            pre_filters=[],
            post_filters=[],
        )
        first = make_proposer_solver_graph(0.2, [1.0, 0.0])
        second = make_proposer_solver_graph(0.8, [1.0, 1.0])
        second.group_id = first.group_id

        assert await sink.add(first) is None
        batch = await sink.add(second)

        assert batch is not None
        assert len(batch.samples) == 6
        assert first.by_agent("proposer")[0].scalar_advantage() == pytest.approx(-0.3)
        assert second.by_agent("proposer")[0].scalar_advantage() == pytest.approx(0.3)
        assert [trace.scalar_advantage() for trace in first.by_agent("solver")] == [0.5, -0.5]
        assert [trace.scalar_advantage() for trace in second.by_agent("solver")] == [0.0, 0.0]
        assert all(sample.temperatures == [0.5] for sample in batch.samples)

    asyncio.run(run())


def test_proposer_solver_rejects_the_wrong_topology():
    graph = make_proposer_solver_graph(1.0, [1.0])
    graph.topology = "other"

    with pytest.raises(ValueError, match="requires topology 'proposer-solver-v1'"):
        ProposerSolverAlgorithm(ProposerSolverAlgoConfig(), policy_pool=None).validate_graph(graph)


def test_proposer_solver_excludes_only_the_errored_solver():
    graph = make_proposer_solver_graph(0.5, [1.0, 0.0])
    graph.by_agent("solver")[0].errors = [vf.Error(type="ProviderError", message="failed")]
    algorithm = ProposerSolverAlgorithm(ProposerSolverAlgoConfig(), policy_pool=None)

    assert not graph.has_error
    assert [trace.agent for trace in algorithm.training_traces(graph)] == ["proposer", "solver"]


def test_multi_trace_algorithm_rejects_deferred_pipeline_features():
    algorithm = ProposerSolverAlgorithm(ProposerSolverAlgoConfig(), policy_pool=None)
    envs = FakeEnvs(algorithm, group_size=2)

    with pytest.raises(AlgorithmCompatibilityError, match="graph-count batch_size"):
        TrainSink(
            SimpleNamespace(),
            tokenizer=None,
            train_envs=envs,
            mm_token_type_ids_mapping=None,
            batch_size=None,
            token_batch_size=100,
            pre_filters=[],
            post_filters=[],
        )
    with pytest.raises(AlgorithmCompatibilityError, match="filters are not yet defined"):
        TrainSink(
            SimpleNamespace(),
            tokenizer=None,
            train_envs=envs,
            mm_token_type_ids_mapping=None,
            batch_size=2,
            token_batch_size=None,
            pre_filters=[object()],
            post_filters=[],
        )
