import asyncio

import pytest
import verifiers.v1 as vf

from prime_rl.configs.algorithm import (
    CostLengthPenaltyConfig,
    GRPOAlgoConfig,
    LinearLengthPenaltyConfig,
    LinearSubagentLengthPenaltyConfig,
    MaxRLAlgoConfig,
)
from prime_rl.env.heads import episode_head
from prime_rl.orchestrator.algo.grpo import GRPOAlgorithm, length_penalty_tokens
from prime_rl.orchestrator.algo.max_rl import MaxRLAlgorithm
from prime_rl.orchestrator.algo.routing import assign_advantage
from prime_rl.orchestrator.credit import trace_advantage
from tests.unit.orchestrator.test_train_sink import received, train_trace


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
    """The per-rollout advantage an algorithm assigned."""
    advantage = trace_advantage(episode.traces[0])
    assert advantage is not None, "episode has no trainable token"
    return advantage


def _length_penalty_tokens(trace: vf.Trace, subagent_token_weight: float) -> tuple[float, float]:
    """The output and new input tokens GRPO's length penalty weighs."""
    output, input, subagent_output, subagent_input = length_penalty_tokens(trace)
    return output + subagent_token_weight * subagent_output, input + subagent_token_weight * subagent_input


def _grpo(group: list[vf.Episode], length_penalty=None, length_weighted_baseline=False) -> list[float]:
    """Drive ``GRPOAlgorithm.score_group`` over the group's heads and read back each per-rollout scalar."""
    algo = GRPOAlgorithm(
        GRPOAlgoConfig(length_penalty=length_penalty, length_weighted_baseline=length_weighted_baseline), clients=None
    )
    heads = [episode_head(episode) for episode in group]
    asyncio.run(algo.score_group(heads))
    return [_scalar(head) for head in heads]


def _max_rl(group: list[vf.Episode]) -> list[float]:
    """Drive ``MaxRLAlgorithm.score_group`` over the group's heads and read back each per-rollout scalar."""
    algo = MaxRLAlgorithm(MaxRLAlgoConfig(), clients=None)
    heads = [episode_head(episode) for episode in group]
    asyncio.run(algo.score_group(heads))
    return [_scalar(head) for head in heads]


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


def _cost_shaped(group: list[vf.Episode], weight: float, w_out: float, w_in: float) -> list[float]:
    """Each rollout's reward minus its own clipped cost: weight per 1,000 weighted tokens."""
    shaped = []
    for episode in group:
        output, input, subagent_output, subagent_input = length_penalty_tokens(episode.traces[0])
        cost = w_out * (output + subagent_output) + w_in * (input + subagent_input)
        shaped.append(episode.traces[0].reward - min(1.0, weight * cost / 1000))
    return shaped


def test_cost_penalty_charges_each_rollout_its_own_cost():
    def group():
        return [_make_episode(1.0, 10), _make_episode(1.0, 40), _make_episode(0.0, 10), _make_episode(0.0, 40)]

    lengths = [length_penalty_tokens(episode.traces[0]) for episode in group()]
    assert lengths[0][0] < lengths[1][0] and lengths[0] == lengths[2] and lengths[1] == lengths[3]
    cfg = CostLengthPenaltyConfig(weight=5.0, num_output_tokens_weight=1.0, num_input_tokens_weight=0.25)
    shaped = _cost_shaped(group(), 5.0, 1.0, 0.25)
    # solved and failed rollouts pay the same for the same length: no pass-rate scaling, no correctness gating
    assert shaped[0] - shaped[1] == pytest.approx(shaped[2] - shaped[3]) and shaped[3] < shaped[2] < 0
    assert _grpo(group(), length_penalty=cfg) == pytest.approx([s - sum(shaped) / 4 for s in shaped], abs=1e-6)

    # a rollout's shaped reward does not depend on the rest of its group
    solo = [_make_episode(1.0, 10), _make_episode(0.0, 10)]
    shaped_solo = _cost_shaped(solo, 5.0, 1.0, 0.25)
    assert shaped_solo[0] == pytest.approx(shaped[0])

    # the input weight only scales the input term
    no_input = CostLengthPenaltyConfig(weight=5.0, num_input_tokens_weight=0.0)
    shaped_out = _cost_shaped(group(), 5.0, 1.0, 0.0)
    assert _grpo(group(), length_penalty=no_input) == pytest.approx(
        [s - sum(shaped_out) / 4 for s in shaped_out], abs=1e-6
    )

    # the penalty is clipped at 1: past 1000 / weight cost tokens, a success scores 0 and a failure -1
    huge = CostLengthPenaltyConfig(weight=1e6)
    assert _grpo(group(), length_penalty=huge) == pytest.approx([0.5, 0.5, -0.5, -0.5], abs=1e-6)
    # weight 0 is plain GRPO
    off = CostLengthPenaltyConfig(weight=0.0)
    assert _grpo(group(), length_penalty=off) == pytest.approx(_grpo(group()), abs=1e-6)


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
        group = [received(episode) for episode in _make_group(rewards=[1.0, 0.0], completion_lengths=lengths)]
        asyncio.run(algo.score_group(group))
        weights = [w for episode in group for sample in train_trace(episode.traces[0]) for w in sample.rl_weights]
        assert sorted(set(weights)) == pytest.approx([0.0, 1.0 / sum(lengths)])
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


def test_grpo_reconfigure_applies_new_length_penalty():
    """A live update swaps the penalty for groups scored afterwards."""
    cfg = LinearLengthPenaltyConfig(num_output_tokens_weight=0.25, num_input_tokens_weight=0.0, num_turns_weight=0.0)
    algo = GRPOAlgorithm(GRPOAlgoConfig(), clients=None)
    algo.reconfigure(GRPOAlgoConfig(length_penalty=cfg))
    group = [episode_head(episode) for episode in _make_group(rewards=[1.0, 1.0, 1.0], completion_lengths=[10, 20, 30])]

    asyncio.run(algo.score_group(group))

    assert [_scalar(episode) for episode in group] == pytest.approx(
        _grpo(_make_group(rewards=[1.0, 1.0, 1.0], completion_lengths=[10, 20, 30]), length_penalty=cfg)
    )


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
    advs = _grpo(group, length_penalty=cfg)
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


@pytest.mark.parametrize("penalty_type", [LinearLengthPenaltyConfig, LinearSubagentLengthPenaltyConfig])
@pytest.mark.parametrize("term", ["num_output_tokens_weight", "num_input_tokens_weight", "num_turns_weight"])
def test_length_penalty_ignores_discarded_compactions(penalty_type, term):
    episode = _build_episode(1.0, sampled_lengths=[5])
    trace = episode.traces[0]
    trace.calls[0].usage = vf.Usage(prompt_tokens=10, completion_tokens=5)
    trace.nodes.extend(
        [
            vf.MessageNode(parent=1, message=vf.UserMessage(content="summarize")),
            vf.MessageNode(
                parent=2,
                message=vf.AssistantMessage(content="discarded"),
                sampled=True,
                token_ids=[1],
                mask=[True],
                semantic_parents=[vf.ParentLink(node=1, type="compaction_attempt")],
            ),
            vf.MessageNode(
                parent=2,
                message=vf.AssistantMessage(content="summary"),
                sampled=True,
                token_ids=[2] * 10,
                mask=[True] * 10,
                semantic_parents=[vf.ParentLink(node=1, type="compaction_attempt")],
            ),
            vf.MessageNode(parent=None, message=vf.UserMessage(content="summary")),
            vf.MessageNode(
                parent=5,
                message=vf.AssistantMessage(content="answer"),
                sampled=True,
                token_ids=[3] * 15,
                mask=[True] * 15,
                semantic_parents=[vf.ParentLink(node=4, type="compaction")],
            ),
        ]
    )
    trace.calls.extend(
        [
            vf.ModelCall(node=3, usage=vf.Usage(prompt_tokens=1000, completion_tokens=1000)),
            vf.ModelCall(node=4, usage=vf.Usage(prompt_tokens=20, completion_tokens=10)),
            vf.ModelCall(node=6, usage=vf.Usage(prompt_tokens=30, completion_tokens=15)),
        ]
    )
    assert _length_penalty_tokens(trace, 1.0) == pytest.approx((30, 45))
    restored = vf.WireTrace.model_validate_json(trace.model_dump_json())
    assert _length_penalty_tokens(restored, 1.0) == pytest.approx((30, 45))

    # The same trained work and context, with no discarded generation.
    peer = episode.model_copy(deep=True)
    peer.traces[0].nodes[3].sampled = False
    peer.traces[0].nodes[3].mask = [False]
    peer.traces[0].calls[1].usage = vf.Usage(prompt_tokens=0, completion_tokens=0)
    kwargs = dict(num_output_tokens_weight=0.0, num_input_tokens_weight=0.0, num_turns_weight=0.0)
    kwargs[term] = 0.25
    penalty = penalty_type(**kwargs)
    assert _grpo([episode, peer], length_penalty=penalty) == pytest.approx([0.0, 0.0], abs=1e-6)


# --------------------------------------------------------------------------
# assign_advantage: one value broadcast over the rollout's trainable tokens.
# --------------------------------------------------------------------------


def test_assign_advantage_broadcasts_scalar():
    """A scalar broadcasts uniformly over the rollout's trainable (mask-True) tokens."""
    [trace] = received(_build_episode(0.0, sampled_lengths=[2])).traces
    # one user prompt token (masked) + 2 sampled tokens (trainable)
    assign_advantage(trace, 0.7)
    assert train_trace(trace)[0].advantages == pytest.approx([0.0, 0.7, 0.7])


def test_assign_advantage_zeros_non_trainable():
    """Non-trainable (mask=False) positions stay 0.0 under scalar broadcast."""
    # prompt(1, masked) + sampled(1) + obs(1, masked): mask is [F, T, F]
    [trace] = received(_build_episode(0.0, sampled_lengths=[1], obs_lengths=[1])).traces
    assign_advantage(trace, 0.7)
    assert train_trace(trace)[0].advantages == pytest.approx([0.0, 0.7, 0.0])
