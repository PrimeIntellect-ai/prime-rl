import asyncio

import pytest
import verifiers.v1 as vf

from prime_rl.orchestrator.advantage import advantage_loss, advantage_scope, grpo, load_advantage, run_advantage, sft
from prime_rl.orchestrator.types import Rollout


def _trace(reward: float) -> Rollout:
    nodes = [
        vf.MessageNode(
            message=vf.UserMessage(content="prompt"),
            token_ids=[10],
            mask=[False],
        ),
        vf.MessageNode(
            parent=0,
            message=vf.AssistantMessage(content="answer"),
            sampled=True,
            token_ids=[11, 12],
            mask=[True, True],
            logprobs=[-0.3, -0.4],
        ),
    ]
    return Rollout(task=vf.Task(idx=0, prompt="prompt"), nodes=nodes, rewards={"reward": reward})


def test_grpo_writes_branch_advantages_and_mask():
    traces = [_trace(1.0), _trace(0.0)]

    asyncio.run(grpo(traces))

    assert traces[0].branches[0].advantages == pytest.approx([0.0, 0.5, 0.5])
    assert traces[0].branches[0].mask == [False, True, True]
    assert traces[1].branches[0].advantages == pytest.approx([0.0, -0.5, -0.5])
    assert traces[1].branches[0].mask == [False, True, True]


def test_sft_sets_unit_ce_weights_on_sampled_tokens():
    trace = _trace(3.0)

    asyncio.run(sft([trace]))

    assert trace.branches[0].advantages == pytest.approx([0.0, 1.0, 1.0])
    assert trace.branches[0].mask == [False, True, True]


def test_load_advantage_accepts_registered_and_decorated_imports():
    assert load_advantage("grpo") is grpo

    fn = load_advantage("tests.unit.orchestrator.test_advantage._custom_advantage")

    assert advantage_loss(fn) == "ce"
    assert advantage_scope(fn) == "rollout"


def test_load_advantage_rejects_undecorated_import():
    with pytest.raises(ValueError, match="@vf.advantage"):
        load_advantage("tests.unit.orchestrator.test_advantage._undecorated")


def test_run_advantage_accepts_sync_functions():
    trace = _trace(1.0)

    asyncio.run(run_advantage(_custom_advantage, [trace]))

    assert trace.branches[0].advantages == pytest.approx([2.0, 2.0, 2.0])
    assert trace.branches[0].mask == [True, True, True]


@vf.advantage(loss="ce", scope="rollout")
def _custom_advantage(traces: list[vf.Trace]) -> list[vf.Trace]:
    for trace in traces:
        for branch in trace.branches:
            branch.advantages = [2.0] * len(branch.token_ids)
            branch.mask = [True] * len(branch.token_ids)
    return traces


def _undecorated(traces: list[vf.Trace]) -> list[vf.Trace]:
    return traces
