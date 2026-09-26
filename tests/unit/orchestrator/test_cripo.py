import asyncio
import math
from unittest.mock import AsyncMock, Mock

import pytest
import verifiers.v1 as vf
from verifiers.v1.graph import MessageNode
from verifiers.v1.types import AssistantMessage, ToolMessage, UserMessage

from prime_rl.configs.algorithm import CriPOSAlgoConfig
from prime_rl.orchestrator.algo import build_algorithm
from prime_rl.orchestrator.algo.cripo import criterion_texts, flip_branch_advantages, suppressed_criteria
from prime_rl.orchestrator.algo.grpo import GRPOAlgorithm
from prime_rl.orchestrator.algo.routing import assign_advantages
from prime_rl.orchestrator.trajectories import iter_trainable_branches, trace_to_samples


def _episode(score=1, quality=0, **extra_rewards):
    trace = vf.Trace(
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0)),
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        nodes=[
            MessageNode(parent=None, message=UserMessage(content="Question"), token_ids=[1], mask=[False]),
            MessageNode(
                parent=0,
                message=AssistantMessage(content="Useful behavior"),
                sampled=True,
                token_ids=[2, 3],
                mask=[True, True],
                logprobs=[-9.0, -9.0],
            ),
            MessageNode(
                parent=1, message=ToolMessage(tool_call_id="t", content="Observation"), token_ids=[4], mask=[False]
            ),
            MessageNode(
                parent=2,
                message=AssistantMessage(content="Conclusion"),
                sampled=True,
                token_ids=[5],
                mask=[True],
                logprobs=[-9.0],
            ),
        ],
        rewards={"criterion": vf.Reward(score=score), "quality": vf.Reward(score=quality, weight=3), **extra_rewards},
        info={"criteria": {"criterion": "Explain the useful behavior."}},
        ok=True,
    )
    return vf.Episode(
        env=vf.EnvInfo(id="rubric", name="rubric"), task=trace.task, group=vf.GroupInfo(id="g"), traces=[trace]
    )


@pytest.mark.parametrize(
    ("scores", "advantages", "expected"),
    [
        ([1, 0], [-1, 1], ["criterion"]),
        ([1, 0], [1, -1], []),
        ([1, 1, 0], [-2, -1, 3], ["criterion"]),
        ([1, 1, 0], [-1, 2, -1], []),
        ([1, 0, 0, 0], [0, 0, 0, 0], ["criterion"]),
        ([1, 1, 0, 0], [0, 0, 0, 0], []),
        ([1, 1, 0, 0, 0], [-1, 1, 0, 0, 0], ["criterion"]),
        ([0, 0], [-1, 1], []),
        ([1, 1], [0, 0], []),
    ],
)
def test_suppression_uses_total_aggregate_credit(scores, advantages, expected):
    traces = [_episode(score).traces[0] for score in scores]
    assert suppressed_criteria(traces, advantages, {"criterion": "Text"}) == expected


def test_suppression_uses_raw_scores_and_orders_by_weight_then_name():
    traces = [_episode().traces[0], _episode(0).traces[0]]
    for i, trace in enumerate(traces):
        trace.rewards = {name: vf.Reward(score=1 - i, weight=weight) for name, weight in [("b", 5), ("a", 5), ("c", 1)]}
    assert suppressed_criteria(traces, [-2, 2], {name: name for name in ["c", "b", "a"]}) == ["a", "b", "c"]


@pytest.mark.parametrize(
    "reward",
    [
        None,
        vf.Reward(score=0.5),
        vf.Reward(score=float("nan")),
        vf.Reward(score=1, weight=0),
        vf.Reward(score=1, weight=-1),
        vf.Reward(score=1, weight=float("inf")),
        vf.Reward(score=1, weight=2),
    ],
)
def test_invalid_or_inconsistent_criterion_rewards_fail(reward):
    traces = [_episode().traces[0], _episode(0).traces[0]]
    traces[1].rewards["criterion"] = reward
    with pytest.raises(ValueError, match="cripo_s criterion"):
        suppressed_criteria(traces, [-1, 1], {"criterion": "Text"})


def test_criterion_text_uses_info_then_typed_task_data():
    class RubricData(vf.TaskData):
        rubric: dict[str, str]

    trace = _episode().traces[0]
    trace.task = vf.TraceTask(type="Task", data=RubricData(rubric={"criterion": "Task text"}))
    assert criterion_texts(trace, "rubric") == {"criterion": "Task text"}
    trace.info["rubric"] = {"criterion": "Info text"}
    assert criterion_texts(trace, "rubric") == {"criterion": "Info text"}


@pytest.mark.parametrize("criteria", [None, {}, [], {"criterion": ""}, {"criterion": 1}, {1: "Text"}])
def test_missing_or_malformed_criterion_text_is_not_inferred(criteria):
    trace = _episode().traces[0]
    trace.info["criteria"] = criteria
    with pytest.raises(ValueError, match="criterion_text"):
        criterion_texts(trace, "criteria")


def test_flip_replaces_credit_and_preserves_shared_nodes_and_observations():
    trace = _episode().traces[0]
    trace.nodes.append(
        MessageNode(
            parent=2,
            message=AssistantMessage(content="Other conclusion"),
            sampled=True,
            token_ids=[6],
            mask=[True],
            logprobs=[-0.1],
        )
    )
    assign_advantages(trace, -4.0)
    for branch, mask in iter_trainable_branches(trace):
        flip_branch_advantages(branch, mask, [-0.1] * 5, [-5.0] * 5, [-0.1] * 5, threshold=0.1, advantage=0.1)
    samples = trace_to_samples(trace)
    assert samples[0].advantages == [0, 0.1, 0.1, 0, 0.1]
    assert samples[1].mask == [False, False, False, False, True]
    assert trace.nodes[1].advantages == [0.1, 0.1]
    assert trace.nodes[2].advantages is None
    assert trace.nodes[-1].advantages == [0.1]


@pytest.mark.parametrize(
    ("student", "teacher", "maximum", "expected"),
    [
        (-0.1, -5, -0.1, 0.1),
        (-6, -5, -0.1, -1),
        (-5, -5, -0.1, -1),
        (-0.1, -0.2, -0.1, -1),
        (-0.1, -0.1 + math.log(0.1), -0.1, -1),
    ],
)
def test_flip_requires_both_strict_probability_tests(student, teacher, maximum, expected):
    trace = _episode().traces[0]
    assign_advantages(trace, -1.0)
    branch, mask = next(iter_trainable_branches(trace))
    flip_branch_advantages(branch, mask, [student] * 5, [teacher] * 5, [maximum] * 5, threshold=0.1, advantage=0.1)
    assert trace.nodes[1].advantages == [expected, expected]
    with pytest.raises(ValueError, match="align"):
        flip_branch_advantages(branch, mask, [], [teacher] * 5, [maximum] * 5, threshold=0.1, advantage=0.1)


def _algorithm(**kwargs):
    clients = Mock()
    clients.score = AsyncMock(return_value=[0, -0.1, -0.1, 0, -0.1])
    clients.score_with_max = AsyncMock(return_value=([0, 0, 0, -5, -0.2, 0, -5], [0, 0, 0, -0.1, -0.1, 0, -0.1]))
    algo = build_algorithm(CriPOSAlgoConfig(**kwargs), clients)
    algo.renderer = Mock()
    algo.renderer.render_ids.return_value = [90, 91]
    algo.tokenizer = Mock()
    algo.tokenizer.decode.return_value = "Original response"
    return algo, clients


def test_group_scoring_flips_only_negative_trace_using_fresh_policy_scores():
    episodes = [_episode(), _episode(0, quality=1)]
    algo, clients = _algorithm()
    asyncio.run(algo.score_group(episodes))
    assert trace_to_samples(episodes[0].traces[0])[0].advantages == [0, 0.1, -1, 0, 0.1]
    assert trace_to_samples(episodes[1].traces[0])[0].advantages == [0, 1, 1, 0, 1]
    clients.score.assert_awaited_once_with([1, 2, 3, 4, 5])
    clients.score_with_max.assert_awaited_once_with([90, 91, 1, 2, 3, 4, 5])
    hint = algo.renderer.render_ids.call_args.args[0][0]["content"]
    assert "modifying or deleting" in hint
    assert "Original response" in hint
    assert "Explain the useful behavior." in hint


@pytest.mark.parametrize("flip_zero", [False, True])
def test_zero_advantage_rescue_is_explicitly_opt_in(flip_zero):
    episodes = [_episode()] + [_episode(0, other=vf.Reward(score=1)) for _ in range(3)]
    algo, clients = _algorithm(flip_zero_advantage=flip_zero)
    asyncio.run(algo.score_group(episodes))
    assert clients.score.await_count == int(flip_zero)
    assert episodes[0].traces[0].nodes[1].advantages == ([0.1, 0] if flip_zero else [0, 0])


def test_group_without_suppression_matches_grpo_including_length_penalty():
    episodes = [_episode(1, quality=1), _episode(0)]
    algo, clients = _algorithm(length_penalty={"type": "linear"})
    expected = [episode.model_copy(deep=True) for episode in episodes]
    asyncio.run(GRPOAlgorithm(algo.config, clients).score_group(expected))
    asyncio.run(algo.score_group(episodes))
    assert [trace_to_samples(ep.traces[0])[0].advantages for ep in episodes] == [
        trace_to_samples(ep.traces[0])[0].advantages for ep in expected
    ]
    clients.score.assert_not_awaited()
    clients.score_with_max.assert_not_awaited()


def test_group_combines_highest_weight_criteria_in_one_prompt():
    episodes = [_episode(), _episode(0, quality=10)]
    for i, episode in enumerate(episodes):
        trace = episode.traces[0]
        trace.info["criteria"] = {name: f"Text for {name}" for name in ("low", "high", "medium")}
        trace.rewards.update(
            {name: vf.Reward(score=1 - i, weight=weight) for name, weight in [("low", 1), ("high", 3), ("medium", 2)]}
        )
    algo, clients = _algorithm(max_criteria=2)
    asyncio.run(algo.score_group(episodes))
    hint = algo.renderer.render_ids.call_args.args[0][0]["content"]
    assert "Text for high\n- Text for medium" in hint
    assert "Text for low" not in hint
    clients.score_with_max.assert_awaited_once()


def test_empty_and_ineligible_groups_do_not_request_scores():
    algo, clients = _algorithm()
    episodes = [_episode(), _episode()]
    episodes[0].traces[0].ok = False
    episodes[1].traces[0].agent.trainable = False
    asyncio.run(algo.score_group([]))
    asyncio.run(algo.score_group(episodes))
    clients.score.assert_not_awaited()


def test_inconsistent_rubrics_and_multi_agent_groups_fail_before_scoring():
    episodes = [_episode(), _episode(0, quality=1)]
    algo, clients = _algorithm()
    episodes[1].traces[0].info["criteria"]["criterion"] = "Different criterion"
    with pytest.raises(ValueError, match="same criterion texts"):
        asyncio.run(algo.score_group(episodes))
    episodes[0].traces.append(_episode().traces[0])
    with pytest.raises(ValueError, match="one trainable trace"):
        asyncio.run(algo.score_group(episodes))
    clients.score.assert_not_awaited()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"flip_advantage": 0},
        {"flip_advantage": float("inf")},
        {"flip_threshold": 0},
        {"flip_threshold": 1.1},
        {"flip_threshold": float("nan")},
        {"max_criteria": 0},
    ],
)
def test_invalid_cripo_parameters_fail_validation(kwargs):
    with pytest.raises(ValueError):
        CriPOSAlgoConfig(**kwargs)
