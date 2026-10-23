import asyncio

import pytest
import verifiers.v1 as vf

from prime_rl.configs.algorithm import (
    GRPOAlgoConfig,
    LinearLengthPenaltyConfig,
    LinearSubagentLengthPenaltyConfig,
    MaxRLAlgoConfig,
)
from prime_rl.orchestrator.algo.grpo import GRPOAlgorithm, _length_penalty_tokens
from prime_rl.orchestrator.algo.max_rl import MaxRLAlgorithm
from prime_rl.orchestrator.algo.routing import assign_advantages
from prime_rl.orchestrator.trajectories import trace_to_samples


def _build_episode(
    reward: float,
    *,
    sampled_lengths: list[int],
    obs_lengths: list[int] | None = None,
    env_name: str = "test",
    metrics: dict | None = None,
) -> vf.Episode:
    """Build a training trace as an alternating message graph.

    ``sampled_lengths`` gives the token count of each model turn (a sampled
    ``AssistantMessage`` node); ``obs_lengths`` (one shorter, if given) gives the
    token count of the non-sampled observation node injected *after* each turn
    (tool output / user feedback).
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

    # Trace token counts are usage-based, so carry provider usage on the final turn's call:
    # every model-generated token as completion, the leading prompt + tool observations as the
    # fed-in context (num_input_tokens = num_total_tokens - num_output_tokens).
    output_tokens = sum(sampled_lengths)
    input_tokens = 1 + sum(obs_lengths)
    calls: list[vf.ModelCall] = []

    for i, n_sampled in enumerate(sampled_lengths):
        ids = _take(n_sampled)
        is_last = i == len(sampled_lengths) - 1
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
        if is_last:
            calls.append(
                vf.ModelCall(
                    node=parent,
                    usage=vf.Usage(prompt_tokens=input_tokens, completion_tokens=output_tokens),
                )
            )
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

    trace = vf.Trace[vf.TaskData](
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt=None)),
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        nodes=nodes,
        calls=calls,
        rewards={"reward": vf.Reward(score=reward)},
        metrics=metrics or {},
        ok=True,
    )
    episode = vf.Episode(
        env=vf.EnvInfo(id=env_name, name=env_name),
        task=trace.task,
        group=vf.GroupInfo(id="group"),
        traces=[trace],
        ok=True,
    )
    return episode


def _make_episode(
    reward: float,
    completion_len: int = 1,
    num_turns: int = 1,
    env_name: str = "test",
    metrics: dict | None = None,
) -> vf.Episode:
    """Build a training trace carrying ``completion_len`` model-sampled tokens split
    across ``num_turns`` sampled turns. Always carries at least one trainable
    token so credit broadcasts somewhere."""
    num_turns = max(num_turns, 1)
    per_turn, rem = divmod(max(completion_len, 1), num_turns)
    sampled_lengths = [per_turn + (rem if i == 0 else 0) for i in range(num_turns)]
    sampled_lengths = [max(n, 1) for n in sampled_lengths]
    return _build_episode(reward, sampled_lengths=sampled_lengths, env_name=env_name, metrics=metrics)


def _make_group(rewards, completion_lengths=None, num_turns=None) -> list[vf.Episode]:
    """Build one group of training traces from 1D arrays of rewards/lengths/turns —
    exactly what ``score_group`` sees."""
    episodes = []
    for i, reward in enumerate(rewards):
        cl = int(completion_lengths[i]) if completion_lengths is not None else 1
        nt = int(num_turns[i]) if num_turns is not None else 1
        episodes.append(_make_episode(float(reward), cl, nt))
    return episodes


def _scalar(episode: vf.Episode) -> float:
    """The per-rollout advantage scalar an algorithm assigned — broadcast over
    the rollout's trainable (mask-True) tokens, so any trainable position holds it."""
    for node in episode.traces[0].nodes:
        if node.advantages:
            return node.advantages[0]
    raise AssertionError("episode has no trainable token")


def _grpo(group: list[vf.Episode], length_penalty=None, length_weighted_baseline=False) -> list[float]:
    """Drive ``GRPOAlgorithm.score_group`` and read back each per-rollout scalar."""
    algo = GRPOAlgorithm(
        GRPOAlgoConfig(length_penalty=length_penalty, length_weighted_baseline=length_weighted_baseline), clients=None
    )
    asyncio.run(algo.score_group(group))
    return [_scalar(episode) for episode in group]


def _max_rl(group: list[vf.Episode]) -> list[float]:
    """Drive ``MaxRLAlgorithm.score_group`` and read back each per-rollout scalar."""
    algo = MaxRLAlgorithm(MaxRLAlgoConfig(), clients=None)
    asyncio.run(algo.score_group(group))
    return [_scalar(episode) for episode in group]


# --------------------------------------------------------------------------
# GRPO / MaxRL: group-relative credit, assigned in score_group.
# --------------------------------------------------------------------------


def test_grpo_plain_mean():
    advs = _grpo(_make_group(rewards=[1.0, 0.5, 0.8], completion_lengths=[10, 12, 8]))
    assert len(advs) == 3
    assert sum(advs) == pytest.approx(0.0, abs=1e-6)


def test_grpo_singleton_group_is_zero():
    # A group of size 1 has reward == mean, so its advantage is 0.
    assert _grpo([_build_episode(0.7, sampled_lengths=[2])]) == pytest.approx([0.0], abs=1e-6)


def test_grpo_length_weighted_baseline():
    # L = [10 + 20 (two turns, observation excluded), 10]: b = (30 * 1 + 10 * 0) / 40 = 0.75
    group = [
        _build_episode(1.0, sampled_lengths=[10, 20], obs_lengths=[5]),
        _build_episode(0.0, sampled_lengths=[10]),
    ]
    advs = _grpo(group, length_weighted_baseline=True)
    assert advs == pytest.approx([0.25, -0.75])
    # per-token advantage is zero-mean across the group's trainable tokens
    assert 30 * advs[0] + 10 * advs[1] == pytest.approx(0.0, abs=1e-6)
    # with a length penalty, the weighted baseline applies to the shaped rewards:
    # penalty = mean(r) * 0.5 * L / max(L) = [0.25, 1/12], shaped = [0.75, -1/12], b = (30 * 0.75 - 10 / 12) / 40
    cfg = LinearLengthPenaltyConfig(num_output_tokens_weight=0.5, num_input_tokens_weight=0.0, num_turns_weight=0.0)
    group = [
        _build_episode(1.0, sampled_lengths=[10, 20], obs_lengths=[5]),
        _build_episode(0.0, sampled_lengths=[10]),
    ]
    advs = _grpo(group, length_penalty=cfg, length_weighted_baseline=True)
    assert advs == pytest.approx([0.75 - 65 / 120, -1 / 12 - 65 / 120])


def test_max_rl_mean_normalized():
    # mean 0.25: the success gets (1 - 0.25)/0.25 = 3, failures (0 - 0.25)/0.25 = -1
    assert _max_rl(_make_group(rewards=[1.0, 0.0, 0.0, 0.0])) == pytest.approx([3.0, -1.0, -1.0, -1.0])
    # no-success groups carry no signal (the paper's K=0 convention) ...
    assert _max_rl(_make_group(rewards=[0.0, 0.0])) == pytest.approx([0.0, 0.0])
    # ... and all-success groups center to zero like GRPO
    assert _max_rl(_make_group(rewards=[1.0, 1.0])) == pytest.approx([0.0, 0.0])


def test_grpo_prompt_loss_aggregation_weights_sum_to_one_per_group():
    """Each group's rl weights total 1, spread as 1/T_q over its trainable tokens, whatever its length."""
    algo = GRPOAlgorithm(GRPOAlgoConfig(loss_aggregation="prompt"), clients=None)
    for lengths in ([10, 30], [100, 300]):
        group = _make_group(rewards=[1.0, 0.0], completion_lengths=lengths)
        asyncio.run(algo.score_group(group))
        weights = [w for episode in group for sample in trace_to_samples(episode.traces[0]) for w in sample.rl_weights]
        assert set(weights) == {0.0, 1.0 / sum(lengths)}
        assert sum(weights) == pytest.approx(1.0)


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
        _build_episode(1.0, sampled_lengths=[10], obs_lengths=[]),
        _build_episode(1.0, sampled_lengths=[10], obs_lengths=[100]),
    ]
    asyncio.run(GRPOAlgorithm(GRPOAlgoConfig(length_penalty=cfg), clients=None).score_group(group))
    advs = [_scalar(episode) for episode in group]
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


@pytest.mark.parametrize("subagent_token_weight", [0.0, 0.7, 1.0])
@pytest.mark.parametrize("term", ["num_output_tokens_weight", "num_input_tokens_weight", "num_turns_weight"])
def test_linear_subagent_tokens_follow_semantic_edges(subagent_token_weight, term):
    episode = _build_episode(1.0, sampled_lengths=[1])
    trace = episode.traces[0]
    trace.nodes.clear()
    trace.calls.clear()

    def turn(parent, input_tokens, output_tokens, links=()):
        prompt_id = len(trace.nodes)
        trace.nodes.append(vf.MessageNode(parent=parent, message=vf.UserMessage(content="q")))
        node_id = len(trace.nodes)
        trace.nodes.append(
            vf.MessageNode(
                parent=prompt_id,
                message=vf.AssistantMessage(content="a"),
                sampled=True,
                token_ids=[1] * output_tokens,
                mask=[True] * output_tokens,
                semantic_parents=[vf.ParentLink(node=source, type=kind) for source, kind in links],
            )
        )
        trace.calls.append(
            vf.ModelCall(
                node=node_id,
                usage=vf.Usage(prompt_tokens=input_tokens - 5, cached_input_tokens=5, completion_tokens=output_tokens),
            )
        )
        return node_id

    root = turn(None, 100, 10)
    child = turn(None, 115, 20, [(root, "subagent_call")])
    child_next = turn(child, 142, 30, [(child, "continuation")])
    summary = turn(None, 200, 40, [(child_next, "compaction_attempt")])
    resumed = turn(None, 60, 50, [(summary, "compaction")])
    nested = turn(None, 70, 60, [(resumed, "subagent_call")])
    root_next = turn(
        root,
        125,
        80,
        [(root, "continuation"), (resumed, "subagent_return"), (nested, "agent_message"), (child, "vendor:review")],
    )
    # A cancel closes a causal loop without changing which agent owns either turn.
    trace.nodes[child_next].semantic_parents.append(vf.ParentLink(node=root_next, type="subagent_cancel"))

    # Branches have separate physical roots; continued history is not recounted.
    expected_output = 90 + 200 * subagent_token_weight
    expected_input = 115 + 452 * subagent_token_weight
    assert _length_penalty_tokens(trace, subagent_token_weight) == pytest.approx((expected_output, expected_input))
    restored = vf.WireTrace.model_validate_json(trace.model_dump_json())
    assert _length_penalty_tokens(restored, subagent_token_weight) == pytest.approx((expected_output, expected_input))

    unlabelled = episode.model_copy(deep=True)
    for node in unlabelled.traces[0].nodes:
        node.semantic_parents.clear()
    assert _length_penalty_tokens(unlabelled.traces[0], subagent_token_weight) == pytest.approx((290, 567))

    cfg = LinearSubagentLengthPenaltyConfig(
        num_output_tokens_weight=0.0,
        num_input_tokens_weight=0.0,
        num_turns_weight=0.0,
        subagent_token_weight=subagent_token_weight,
    )
    setattr(cfg, term, 0.25)
    expected = {"num_output_tokens_weight": expected_output / 290, "num_input_tokens_weight": expected_input / 567}
    advantage = 0.125 * (1 - expected.get(term, 1.0))
    assert _grpo([episode, unlabelled], length_penalty=cfg) == pytest.approx([advantage, -advantage], abs=1e-6)
    assert trace.reward == 1.0
    assert trace.num_turns == 7
    assert sum(sum(node.mask) for node in trace.nodes) == 290

    # The plain linear penalty keeps its trace metrics and ignores semantic labels.
    linear = LinearLengthPenaltyConfig.model_validate(
        {k: v for k, v in cfg.model_dump().items() if k not in ("type", "subagent_token_weight")}
    )
    assert _grpo([episode, unlabelled], length_penalty=linear) == pytest.approx([0.0, 0.0], abs=1e-6)
    peer = _build_episode(1.0, sampled_lengths=[10])
    traces = [trace, peer.traces[0]]
    counts = {
        "num_output_tokens_weight": [t.num_output_tokens for t in traces],
        "num_input_tokens_weight": [t.num_total_tokens - t.num_output_tokens for t in traces],
        "num_turns_weight": [t.num_turns for t in traces],
    }[term]
    penalties = [0.25 * count / max(max(counts), 1) for count in counts]
    mean_penalty = sum(penalties) / len(penalties)
    assert _grpo([episode, peer], length_penalty=linear) == pytest.approx(
        [mean_penalty - penalty for penalty in penalties], abs=1e-6
    )


# --------------------------------------------------------------------------
# assign_advantages: scalar broadcast over the rollout's trainable tokens.
# --------------------------------------------------------------------------


def test_assign_advantages_broadcasts_scalar():
    """A scalar broadcasts uniformly over the rollout's trainable (mask-True) tokens."""
    episode = _build_episode(0.0, sampled_lengths=[2])
    trace = episode.traces[0]
    # one user prompt token (masked) + 2 sampled tokens (trainable)
    assign_advantages(trace, 0.7)
    assert trace_to_samples(trace)[0].advantages == [0.0, 0.7, 0.7]


def test_assign_advantages_zeros_non_trainable():
    """Non-trainable (mask=False) positions stay 0.0 under scalar broadcast."""
    # prompt(1, masked) + sampled(1) + obs(1, masked): mask is [F, T, F]
    episode = _build_episode(0.0, sampled_lengths=[1], obs_lengths=[1])
    trace = episode.traces[0]
    assign_advantages(trace, 0.7)
    assert trace_to_samples(trace)[0].advantages == [0.0, 0.7, 0.0]


def test_assign_advantages_rejects_misaligned():
    episode = _build_episode(0.0, sampled_lengths=[2])
    # full length is 3 (prompt + 2 sampled); a 1-element list must be rejected
    with pytest.raises(ValueError, match="align"):
        assign_advantages(episode.traces[0], [0.5])
