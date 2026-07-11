import asyncio

import pytest
import verifiers.v1 as vf

from prime_rl.configs.algorithm import (
    GRPOAlgoConfig,
    LinearLengthPenaltyConfig,
    MaxRLAlgoConfig,
)
from prime_rl.orchestrator.algo.grpo import GRPOAlgorithm
from prime_rl.orchestrator.algo.max_rl import MaxRLAlgorithm
from prime_rl.orchestrator.trajectories import trace_to_samples
from prime_rl.orchestrator.types import Rollout


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

    rollout = Rollout[vf.TaskData](
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt=None)),
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


def _scalar(rollout: Rollout) -> float:
    """The per-rollout advantage scalar an algorithm assigned — broadcast over
    the rollout's trainable (mask-True) tokens, so any trainable position holds it."""
    mask = [m for sample in rollout.samples for m in sample.mask]
    return rollout.advantages[mask.index(True)]


def _grpo(group: list[Rollout], length_penalty=None) -> list[float]:
    """Drive ``GRPOAlgorithm.score_group`` and read back each per-rollout scalar."""
    algo = GRPOAlgorithm(GRPOAlgoConfig(length_penalty=length_penalty), policy_pool=None)
    asyncio.run(algo.score_group(group))
    return [_scalar(rollout) for rollout in group]


def _max_rl(group: list[Rollout]) -> list[float]:
    """Drive ``MaxRLAlgorithm.score_group`` and read back each per-rollout scalar."""
    algo = MaxRLAlgorithm(MaxRLAlgoConfig(), policy_pool=None)
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
# GRPO linear length penalty: pass_rate-scaled penalty before the baseline.
# --------------------------------------------------------------------------


def test_linear_equal_lengths_reduce_to_plain_grpo():
    """Equal completion length and turns → every rollout takes the same penalty
    fraction, so subtracting it leaves the centered advantages unchanged."""
    penalized = _grpo(
        _make_group(rewards=[1.0, 0.0, 1.0], completion_lengths=[10, 10, 10], num_turns=[2, 2, 2]),
        length_penalty=LinearLengthPenaltyConfig(),
    )
    plain = _grpo(_make_group(rewards=[1.0, 0.0, 1.0], completion_lengths=[10, 10, 10], num_turns=[2, 2, 2]))
    assert penalized == pytest.approx(plain, abs=1e-6)


def test_linear_completion_term_penalizes_longer():
    """With only the completion term, longer completions get a larger penalty and a
    lower advantage; advantages stay zero-mean."""
    cfg = LinearLengthPenaltyConfig(num_output_tokens_weight=0.25, num_input_tokens_weight=0.0, num_turns_weight=0.0)
    advs = _grpo(_make_group(rewards=[1.0, 1.0, 1.0], completion_lengths=[10, 20, 30]), length_penalty=cfg)
    assert advs[0] > advs[1] > advs[2]
    assert sum(advs) == pytest.approx(0.0, abs=1e-6)


def test_linear_context_term_penalizes_more_context():
    """The context term penalizes non-completion (prompt / tool-response) tokens: at
    equal completion length, more context tokens yields a lower advantage."""
    cfg = LinearLengthPenaltyConfig(num_output_tokens_weight=0.0, num_input_tokens_weight=0.25, num_turns_weight=0.0)
    group = [
        _build_rollout(1.0, sampled_lengths=[10], obs_lengths=[]),
        _build_rollout(1.0, sampled_lengths=[10], obs_lengths=[100]),
    ]
    asyncio.run(GRPOAlgorithm(GRPOAlgoConfig(length_penalty=cfg), policy_pool=None).score_group(group))
    advs = [_scalar(rollout) for rollout in group]
    assert advs[0] > advs[1]
    assert sum(advs) == pytest.approx(0.0, abs=1e-6)


def test_linear_turns_term_penalizes_more_turns():
    """The turns term penalizes higher turn counts at equal token lengths."""
    cfg = LinearLengthPenaltyConfig(num_output_tokens_weight=0.0, num_input_tokens_weight=0.0, num_turns_weight=0.25)
    advs = _grpo(
        _make_group(rewards=[1.0, 1.0], completion_lengths=[100, 100], num_turns=[1, 4]),
        length_penalty=cfg,
    )
    assert advs[0] > advs[1]
    assert sum(advs) == pytest.approx(0.0, abs=1e-6)


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
