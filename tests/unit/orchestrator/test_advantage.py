import asyncio
import math

import pytest
import torch
import verifiers.v1 as vf

from prime_rl.configs.algorithm import (
    GRPOAlgorithmConfig,
    MaxRLAlgorithmConfig,
    TokensLengthPenaltyConfig,
    TurnsLengthPenaltyConfig,
)
from prime_rl.orchestrator.algo.advantage import efficiency_shaping
from prime_rl.orchestrator.algo.grpo import GRPOAlgorithm
from prime_rl.orchestrator.algo.max_rl import MaxRLAlgorithm
from prime_rl.orchestrator.trajectories import trace_to_samples
from prime_rl.orchestrator.types import Rollout

# Token-cost penalty over completion tokens only (the harness tool-response
# metric is absent on these rollouts, so the tool weight contributes nothing).
_TOKENS_COMPLETION = TokensLengthPenaltyConfig(completion_weight=1.0, tool_response_weight=0.0)
# Token-cost penalty over tool-response tokens only.
_TOKENS_TOOL_ONLY = TokensLengthPenaltyConfig(completion_weight=0.0, tool_response_weight=1.0)


def _build_rollout(
    reward: float,
    *,
    sampled_lengths: list[int],
    obs_lengths: list[int] | None = None,
    env_name: str = "test",
    metrics: dict | None = None,
) -> Rollout:
    """Build a ``Rollout`` (a ``vf.Trace``) as an alternating message graph.

    ``sampled_lengths`` gives the token count of each model turn (a sampled
    ``AssistantMessage`` node); ``obs_lengths`` (one shorter, if given) gives the
    token count of the non-sampled observation node injected *after* each turn
    (tool output / user feedback). ``samples`` is built via the real
    ``trace_to_samples`` so the rollout matches what ``score_group`` sees.
    """
    obs_lengths = obs_lengths or []
    nodes: list[vf.MessageNode] = []
    parent: int | None = None
    next_token = 0

    def _take(n: int) -> list[int]:
        nonlocal next_token
        ids = list(range(next_token, next_token + n))
        next_token += n
        return ids

    # Leading user prompt (never trainable).
    prompt_ids = _take(1)
    nodes.append(
        vf.MessageNode(
            message=vf.UserMessage(content="q"),
            token_ids=prompt_ids,
            mask=[False] * len(prompt_ids),
            logprobs=[0.0] * len(prompt_ids),
            sampled=False,
            parent=parent,
        )
    )
    parent = len(nodes) - 1

    for i, n_sampled in enumerate(sampled_lengths):
        ids = _take(n_sampled)
        nodes.append(
            vf.MessageNode(
                message=vf.AssistantMessage(content="a"),
                token_ids=ids,
                mask=[True] * n_sampled,
                logprobs=[-0.1] * n_sampled,
                sampled=True,
                parent=parent,
            )
        )
        parent = len(nodes) - 1
        if i < len(obs_lengths):
            obs_ids = _take(obs_lengths[i])
            nodes.append(
                vf.MessageNode(
                    message=vf.ToolMessage(content="t", tool_call_id="x"),
                    token_ids=obs_ids,
                    mask=[False] * obs_lengths[i],
                    logprobs=[0.0] * obs_lengths[i],
                    sampled=False,
                    parent=parent,
                )
            )
            parent = len(nodes) - 1

    rollout = Rollout[vf.Task](
        task=vf.Task(idx=0, prompt=None),
        nodes=nodes,
        rewards={"reward": reward},
        metrics=metrics or {},
    )
    rollout.env_name = env_name
    rollout.samples = trace_to_samples(rollout, env_name=env_name)
    return rollout


def _make_rollout(
    reward: float,
    completion_len: int = 1,
    num_turns: int = 1,
    env_name: str = "test",
    metrics: dict | None = None,
) -> Rollout:
    """Build a ``Rollout`` carrying ``completion_len`` model-sampled tokens split
    across ``num_turns`` sampled turns. Always carries at least one trainable
    token so credit broadcasts somewhere."""
    num_turns = max(num_turns, 1)
    per_turn, rem = divmod(max(completion_len, 1), num_turns)
    sampled_lengths = [per_turn + (rem if i == 0 else 0) for i in range(num_turns)]
    sampled_lengths = [max(n, 1) for n in sampled_lengths]
    return _build_rollout(reward, sampled_lengths=sampled_lengths, env_name=env_name, metrics=metrics)


def _make_group(rewards, completion_lengths=None, num_turns=None) -> list[Rollout]:
    """Build one group of ``Rollout``\\ s from 1D arrays of rewards/lengths/turns —
    exactly what ``score_group`` sees."""
    rollouts = []
    for i, reward in enumerate(rewards):
        cl = int(completion_lengths[i]) if completion_lengths is not None else 1
        nt = int(num_turns[i]) if num_turns is not None else 1
        rollouts.append(_make_rollout(float(reward), cl, nt))
    return rollouts


def _shape(rewards: list[float], costs: list[float]) -> list[float]:
    """Run ``efficiency_shaping`` directly on reward/cost vectors."""
    return efficiency_shaping(
        torch.tensor(rewards, dtype=torch.float32), torch.tensor(costs, dtype=torch.float32)
    ).tolist()


def _scalar(rollout: Rollout) -> float:
    """The per-rollout advantage scalar an algorithm assigned — broadcast over
    the rollout's trainable (mask-True) tokens, so any trainable position holds it."""
    mask = [m for sample in rollout.samples for m in sample.mask]
    return rollout.advantages[mask.index(True)]


def _grpo(group: list[Rollout], length_penalty=None) -> list[float]:
    """Drive ``GRPOAlgorithm.score_group`` and read back each per-rollout scalar."""
    algo = GRPOAlgorithm(GRPOAlgorithmConfig(length_penalty=length_penalty), policy_pool=None, renderer=None)
    asyncio.run(algo.score_group(group))
    return [_scalar(rollout) for rollout in group]


def _max_rl(group: list[Rollout]) -> list[float]:
    """Drive ``MaxRLAlgorithm.score_group`` and read back each per-rollout scalar."""
    algo = MaxRLAlgorithm(MaxRLAlgorithmConfig(), policy_pool=None, renderer=None)
    asyncio.run(algo.score_group(group))
    return [_scalar(rollout) for rollout in group]


# --------------------------------------------------------------------------
# GRPO / MaxRL: group-relative credit, assigned in score_group.
# --------------------------------------------------------------------------


def test_grpo_plain_mean():
    advs = _grpo(_make_group(rewards=[1.0, 0.5, 0.8], completion_lengths=[10, 12, 8]))
    assert len(advs) == 3
    assert sum(advs) == pytest.approx(0.0, abs=1e-6)


def test_grpo_singleton_group_is_zero():
    # A group of size 1 has reward == mean, so its advantage is 0.
    assert _grpo([_build_rollout(0.7, sampled_lengths=[2])]) == pytest.approx([0.0], abs=1e-6)


def test_max_rl_mean_normalized():
    # mean 0.25: the success gets (1 - 0.25)/0.25 = 3, failures (0 - 0.25)/0.25 = -1
    assert _max_rl(_make_group(rewards=[1.0, 0.0, 0.0, 0.0])) == pytest.approx([3.0, -1.0, -1.0, -1.0])
    # no-success groups carry no signal (the paper's K=0 convention) ...
    assert _max_rl(_make_group(rewards=[0.0, 0.0])) == pytest.approx([0.0, 0.0])
    # ... and all-success groups center to zero like GRPO
    assert _max_rl(_make_group(rewards=[1.0, 1.0])) == pytest.approx([0.0, 0.0])


# --------------------------------------------------------------------------
# efficiency_shaping: the bounded correctness-gated reward transform (pure).
# --------------------------------------------------------------------------


def test_efficiency_mixed_group():
    """Mixed group: shaping preserves zero-mean, shorter correct gets higher advantage."""
    # mean_correct_cost = (10+30+20)/3 = 20
    # bonus = clamp(1 - [10,30,20,20]/20, 0, 1) = [0.5, 0, 0, 0]
    # shaped = R * (1 + bonus * correct) = [1.5, 1, 0, 1]; baseline = 0.875
    advs = _shape([1.0, 1.0, 0.0, 1.0], [10, 30, 20, 20])
    assert advs == pytest.approx([0.625, 0.125, -0.875, 0.125], abs=1e-6)
    assert sum(advs) == pytest.approx(0.0, abs=1e-6)


def test_efficiency_all_correct_group():
    """All-correct group: zero-mean, shorter gets higher advantage."""
    # mean_cost = 70/3; bonus = clamp(1 - [10,20,40]/(70/3), 0, 1) = [4/7, 1/7, 0]
    advs = _shape([1.0, 1.0, 1.0], [10, 20, 40])
    shaped = [11.0 / 7, 8.0 / 7, 1.0]
    mean_shaped = sum(shaped) / len(shaped)
    assert advs == pytest.approx([s - mean_shaped for s in shaped], abs=1e-6)
    assert sum(advs) == pytest.approx(0.0, abs=1e-6)
    assert advs[0] > advs[1] > advs[2]


def test_efficiency_all_zero_rewards():
    """When all rewards are 0 there is no correct rollout — no shaping (plain GRPO)."""
    assert _shape([0.0, 0.0, 0.0], [10, 20, 15]) == pytest.approx([0.0, 0.0, 0.0], abs=1e-6)


def test_efficiency_single_correct():
    """A single correct rollout sits at its own mean cost (bonus 0) — plain GRPO."""
    assert _shape([1.0, 0.0, 0.0, 0.0], [100, 50, 200, 150]) == pytest.approx([0.75, -0.25, -0.25, -0.25], abs=1e-6)


def test_efficiency_shorter_correct_higher_advantage():
    """Among correct rollouts, shorter always gets higher advantage."""
    advs = _shape([1.0, 1.0, 1.0, 0.0, 0.0], [50, 100, 200, 80, 120])
    assert advs[0] > advs[1] > advs[2]
    assert all(a > 0 for a in advs[:3])
    assert all(a < 0 for a in advs[3:])


def test_efficiency_zero_mean_preserved():
    assert sum(_shape([1.0, 1.0, 0.0, 1.0], [10, 30, 20, 20])) == pytest.approx(0.0, abs=1e-6)
    assert sum(_shape([1.0, 1.0, 1.0], [10, 20, 40])) == pytest.approx(0.0, abs=1e-6)


def test_efficiency_amplification_bounded():
    """Even with extreme length outliers, amplification is capped at 2x (advantage < 1)."""
    assert _shape([1.0, 1.0, 0.0], [1, 10000, 5000])[0] < 1.0 + 1e-3


def test_efficiency_zero_costs_falls_back_to_plain_grpo():
    """All-zero costs would divide by zero — fall back to plain GRPO, no NaNs."""
    advs = _shape([1.0, 1.0, 0.0], [0.0, 0.0, 0.0])
    assert not any(math.isnan(a) for a in advs)
    assert advs == pytest.approx([1.0 / 3, 1.0 / 3, -2.0 / 3], abs=1e-6)


# --------------------------------------------------------------------------
# GRPO length-penalty cost dispatch (rollout fields -> per-rollout cost).
# --------------------------------------------------------------------------


def test_grpo_tokens_with_tool_response_weight():
    """`tool_response_weight` shifts shaping onto tool-response tokens read from rollout metrics."""

    def group():
        return [
            _build_rollout(1.0, sampled_lengths=[10], metrics={"rlm_total_tool_response_tokens": 200}),
            _build_rollout(1.0, sampled_lengths=[10], metrics={"rlm_total_tool_response_tokens": 0}),
            _build_rollout(1.0, sampled_lengths=[10], metrics={"rlm_total_tool_response_tokens": 100}),
        ]

    # completion tokens identical (10 each) → completion-only shaping is a no-op
    assert _grpo(group(), length_penalty=_TOKENS_COMPLETION) == pytest.approx([0.0, 0.0, 0.0], abs=1e-6)
    # tool-response only: costs are [200, 0, 100], so only the below-mean rollout (idx 1) amplifies
    advs = _grpo(group(), length_penalty=_TOKENS_TOOL_ONLY)
    assert advs[1] > advs[0]
    assert advs[1] > advs[2]
    assert advs[0] == pytest.approx(advs[2], abs=1e-6)
    assert sum(advs) == pytest.approx(0.0, abs=1e-6)


def test_grpo_fractional_weight_with_int_rewards():
    """Fractional weights must not truncate when rollout rewards are emitted as ints."""
    lengths = [7, 11, 13]
    rewards_int = [1, 1, 0]
    fractional = TokensLengthPenaltyConfig(completion_weight=0.3, tool_response_weight=0.0)
    int_group = [_build_rollout(r, sampled_lengths=[length]) for r, length in zip(rewards_int, lengths)]
    float_group = [_build_rollout(float(r), sampled_lengths=[length]) for r, length in zip(rewards_int, lengths)]
    assert _grpo(int_group, length_penalty=fractional) == pytest.approx(
        _grpo(float_group, length_penalty=fractional), abs=1e-6
    )


def test_grpo_tokens_default_weights_match_completion_when_no_metric():
    """Default TokensLengthPenaltyConfig (1,1) reduces to completion-only when rollouts lack the metric."""

    def group():
        return _make_group(rewards=[1.0, 1.0, 0.0, 1.0], completion_lengths=[10, 30, 20, 20])

    assert _grpo(group(), length_penalty=TokensLengthPenaltyConfig()) == pytest.approx(
        _grpo(group(), length_penalty=_TOKENS_COMPLETION), abs=1e-6
    )


def test_grpo_turns_penalty():
    """`TurnsLengthPenaltyConfig` shapes by trajectory turn count rather than token count."""
    # token counts identical, turns differ: mean_correct_turns = (1+3+2)/3 = 2
    # bonus = clamp(1 - [1,3,2,2]/2, 0, 1) = [0.5, 0, 0, 0]
    advs = _grpo(
        _make_group(rewards=[1.0, 1.0, 0.0, 1.0], completion_lengths=[100, 100, 100, 100], num_turns=[1, 3, 2, 2]),
        length_penalty=TurnsLengthPenaltyConfig(),
    )
    assert advs == pytest.approx([0.625, 0.125, -0.875, 0.125], abs=1e-6)


# --------------------------------------------------------------------------
# assign_advantages: scalar broadcast over the rollout's trainable tokens.
# --------------------------------------------------------------------------


def test_assign_advantages_broadcasts_scalar():
    """A scalar broadcasts uniformly over the rollout's trainable (mask-True) tokens."""
    rollout = _build_rollout(0.0, sampled_lengths=[2])
    # one user prompt token (masked) + 2 sampled tokens (trainable)
    rollout.assign_advantages(0.7)
    assert rollout.advantages == [0.0, 0.7, 0.7]


def test_assign_advantages_zeros_non_trainable():
    """Non-trainable (mask=False) positions stay 0.0 under scalar broadcast."""
    # prompt(1, masked) + sampled(1) + obs(1, masked): mask is [F, T, F]
    rollout = _build_rollout(0.0, sampled_lengths=[1], obs_lengths=[1])
    rollout.assign_advantages(0.7)
    assert rollout.advantages == [0.0, 0.7, 0.0]


def test_assign_advantages_rejects_misaligned():
    rollout = _build_rollout(0.0, sampled_lengths=[2])
    # full length is 3 (prompt + 2 sampled); a 1-element list must be rejected
    with pytest.raises(ValueError, match="align"):
        rollout.assign_advantages([0.5])
