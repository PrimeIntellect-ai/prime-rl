import asyncio

import pytest
import verifiers.v1 as vf

from prime_rl.configs.algorithm import (
    ConditionalPrefixConfig,
    CosineLengthPenaltyScheduleConfig,
    FallbackRewardConfig,
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
        if node.advantages is not None and len(node.advantages):
            return float(node.advantages[0])
    raise AssertionError("episode has no trainable token")


def _grpo(
    group: list[vf.Episode],
    length_penalty=None,
    fallback_reward=None,
    length_weighted_baseline=False,
) -> list[float]:
    """Drive ``GRPOAlgorithm.score_group`` and read back each per-rollout scalar."""
    algo = GRPOAlgorithm(
        GRPOAlgoConfig(
            length_penalty=length_penalty,
            fallback_reward=fallback_reward,
            length_weighted_baseline=length_weighted_baseline,
        ),
        clients=None,
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


def test_cosine_schedule_scale():
    """No penalty during warmup, then a cosine cycle from min_scale up to max_scale and back each period."""
    schedule = CosineLengthPenaltyScheduleConfig(period=20, warmup_steps=5)
    assert schedule.scale(0) == 0.0 and schedule.scale(4) == 0.0
    assert schedule.scale(5) == pytest.approx(0.0, abs=1e-9)
    assert schedule.scale(10) == pytest.approx(0.5)
    assert schedule.scale(15) == pytest.approx(1.0)
    assert schedule.scale(25) == pytest.approx(0.0, abs=1e-9)
    shifted = CosineLengthPenaltyScheduleConfig(period=20, phase=0.5, min_scale=0.2, max_scale=0.8)
    assert shifted.scale(0) == pytest.approx(0.8)
    assert shifted.scale(10) == pytest.approx(0.2)
    with pytest.raises(ValueError):
        CosineLengthPenaltyScheduleConfig(period=20, min_scale=0.9, max_scale=0.1)


def test_scheduled_penalty_scales_with_dispatch_step():
    """A scheduled penalty is the unscheduled one times the schedule at the group's dispatch step: none at the
    bottom of a cycle, the full penalty at the top."""
    weights = dict(num_output_tokens_weight=0.25, num_input_tokens_weight=0.0, num_turns_weight=0.0)
    schedule = CosineLengthPenaltyScheduleConfig(period=20)

    def scored(step: int | None, cfg) -> list[float]:
        group = _make_group(rewards=[1.0, 1.0, 1.0], completion_lengths=[10, 20, 30])
        if step is not None:
            for episode in group:
                episode.run = vf.TrainRunInfo(id="run", name="run", work=vf.TrainWorkInfo(step=step))
        return _grpo(group, length_penalty=cfg)

    scheduled = LinearLengthPenaltyConfig(**weights, schedule=schedule)
    full = LinearLengthPenaltyConfig(**weights)
    assert scored(0, scheduled) == pytest.approx(scored(0, None), abs=1e-6)
    assert scored(10, scheduled) == pytest.approx(scored(10, full), abs=1e-6)
    assert scored(None, scheduled) == pytest.approx(scored(None, full), abs=1e-6)
    half = scored(5, scheduled)
    assert half == pytest.approx([0.5 * a for a in scored(5, full)], abs=1e-6)


def test_grpo_reconfigure_applies_new_length_penalty():
    """A live update swaps the penalty for groups scored afterwards."""
    cfg = LinearLengthPenaltyConfig(num_output_tokens_weight=0.25, num_input_tokens_weight=0.0, num_turns_weight=0.0)
    algo = GRPOAlgorithm(GRPOAlgoConfig(), clients=None)
    algo.reconfigure(GRPOAlgoConfig(length_penalty=cfg))
    group = _make_group(rewards=[1.0, 1.0, 1.0], completion_lengths=[10, 20, 30])

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
# assign_advantages: scalar broadcast over the rollout's trainable tokens.
# --------------------------------------------------------------------------


def test_assign_advantages_broadcasts_scalar():
    """A scalar broadcasts uniformly over the rollout's trainable (mask-True) tokens."""
    episode = _build_episode(0.0, sampled_lengths=[2])
    trace = episode.traces[0]
    # one user prompt token (masked) + 2 sampled tokens (trainable)
    assign_advantages(trace, 0.7)
    assert trace_to_samples(trace)[0].advantages == pytest.approx([0.0, 0.7, 0.7])


def test_assign_advantages_zeros_non_trainable():
    """Non-trainable (mask=False) positions stay 0.0 under scalar broadcast."""
    # prompt(1, masked) + sampled(1) + obs(1, masked): mask is [F, T, F]
    episode = _build_episode(0.0, sampled_lengths=[1], obs_lengths=[1])
    trace = episode.traces[0]
    assign_advantages(trace, 0.7)
    assert trace_to_samples(trace)[0].advantages == pytest.approx([0.0, 0.7, 0.0])


def test_assign_advantages_rejects_misaligned():
    episode = _build_episode(0.0, sampled_lengths=[2])
    # full length is 3 (prompt + 2 sampled); a 1-element list must be rejected
    with pytest.raises(ValueError, match="align"):
        assign_advantages(episode.traces[0], [0.5])


# --------------------------------------------------------------------------
# GRPO fallback_reward: per-group fallback when no rollout reaches threshold.
# --------------------------------------------------------------------------


def test_grpo_fallback_used_when_no_solve():
    """All-fail groups score on weight * fallback metric and mark fallback_reward_active=1."""
    group = [
        _build_episode(0.0, sampled_lengths=[2], metrics={"criteria_score": 0.2}),
        _build_episode(0.0, sampled_lengths=[2], metrics={"criteria_score": 0.8}),
        _build_episode(0.0, sampled_lengths=[2], metrics={"criteria_score": None}),
        _build_episode(0.0, sampled_lengths=[2], metrics={}),
    ]
    # Scaled rewards: [0.1, 0.4, 0.0, 0.0], mean = 0.125
    fb = FallbackRewardConfig(metric="criteria_score", weight=0.5)
    assert _grpo(group, fallback_reward=fb) == pytest.approx([-0.025, 0.275, -0.125, -0.125])
    assert [ep.traces[0].reward for ep in group] == pytest.approx([0.1, 0.4, 0.0, 0.0])
    assert [ep.traces[0].metrics["binary_reward"] for ep in group] == [0.0, 0.0, 0.0, 0.0]
    assert [ep.traces[0].metrics["fallback_reward_active"] for ep in group] == [1.0, 1.0, 1.0, 1.0]


def test_grpo_fallback_skipped_when_group_has_solve():
    """Any rollout reaching threshold keeps the group on pure primary reward."""
    group = [
        _build_episode(1.0, sampled_lengths=[2], metrics={"criteria_score": 1.0}),
        _build_episode(0.0, sampled_lengths=[2], metrics={"criteria_score": 0.9}),
        _build_episode(0.0, sampled_lengths=[2], metrics={"criteria_score": 0.1}),
        _build_episode(0.0, sampled_lengths=[2], metrics={"criteria_score": 0.0}),
    ]
    fb = FallbackRewardConfig(metric="criteria_score", weight=0.5)
    assert _grpo(group, fallback_reward=fb) == pytest.approx([0.75, -0.25, -0.25, -0.25])
    assert [ep.traces[0].reward for ep in group] == [1.0, 0.0, 0.0, 0.0]
    assert [ep.traces[0].metrics["binary_reward"] for ep in group] == [1.0, 0.0, 0.0, 0.0]
    assert [ep.traces[0].metrics["fallback_reward_active"] for ep in group] == [0.0, 0.0, 0.0, 0.0]


def test_grpo_fallback_all_pass_stays_zero():
    """All-pass groups do not trigger fallback and produce zero advantage."""
    group = [
        _build_episode(1.0, sampled_lengths=[2], metrics={"criteria_score": 0.8}),
        _build_episode(1.0, sampled_lengths=[2], metrics={"criteria_score": 1.0}),
    ]
    fb = FallbackRewardConfig(metric="criteria_score", weight=0.5)
    assert _grpo(group, fallback_reward=fb) == pytest.approx([0.0, 0.0])
    assert [ep.traces[0].metrics["fallback_reward_active"] for ep in group] == [0.0, 0.0]


def test_grpo_fallback_reconfigure_live():
    """reconfigure updates fallback_reward live."""
    algo = GRPOAlgorithm(GRPOAlgoConfig(), clients=None)
    g1 = [
        _build_episode(0.0, sampled_lengths=[2], metrics={"criteria_score": 0.0}),
        _build_episode(0.0, sampled_lengths=[2], metrics={"criteria_score": 1.0}),
    ]
    asyncio.run(algo.score_group(g1))
    assert [_scalar(ep) for ep in g1] == pytest.approx([0.0, 0.0])

    algo.reconfigure(GRPOAlgoConfig(fallback_reward=FallbackRewardConfig(metric="criteria_score", weight=0.5)))
    g2 = [
        _build_episode(0.0, sampled_lengths=[2], metrics={"criteria_score": 0.0}),
        _build_episode(0.0, sampled_lengths=[2], metrics={"criteria_score": 1.0}),
    ]
    asyncio.run(algo.score_group(g2))
    assert [_scalar(ep) for ep in g2] == pytest.approx([-0.25, 0.25])


def test_grpo_length_penalty_records_metrics():
    """Length penalty logs both its schedule scale and per-trace penalty in trace.metrics."""
    group = _make_group(rewards=[1.0, 1.0], completion_lengths=[4, 8])
    for ep in group:
        ep.run = vf.TrainRunInfo(id="run", name="run", work=vf.TrainWorkInfo(step=100))
    lp = LinearLengthPenaltyConfig(
        num_output_tokens_weight=0.5,
        num_input_tokens_weight=0.0,
        num_turns_weight=0.0,
        schedule=CosineLengthPenaltyScheduleConfig(period=100, warmup_steps=50),
    )
    _grpo(group, length_penalty=lp)
    assert [ep.traces[0].metrics["length_penalty_scale"] for ep in group] == pytest.approx([1.0, 1.0])
    assert [ep.traces[0].metrics["length_penalty"] for ep in group] == pytest.approx([0.25, 0.5])


def test_grpo_fallback_convex_threshold_and_snr_gate():
    """Convex floor transform zeros low-score boilerplate and SNR gate skips hopeless/flat groups."""
    fb = FallbackRewardConfig(
        metric="criteria_score", weight=0.5, floor=0.35, exponent=2.0, min_max_score=0.45, min_spread=0.10
    )
    # Hopeless group (max < 0.45) -> zero advantage
    hopeless = [
        _build_episode(0.0, sampled_lengths=[2], metrics={"criteria_score": 0.10}),
        _build_episode(0.0, sampled_lengths=[2], metrics={"criteria_score": 0.40}),
    ]
    assert _grpo(hopeless, fallback_reward=fb) == pytest.approx([0.0, 0.0])

    # Flat group (spread < 0.10) -> zero advantage
    flat = [
        _build_episode(0.0, sampled_lengths=[2], metrics={"criteria_score": 0.55}),
        _build_episode(0.0, sampled_lengths=[2], metrics={"criteria_score": 0.63}),
    ]
    assert _grpo(flat, fallback_reward=fb) == pytest.approx([0.0, 0.0])

    # Near-miss group: 0.20 & 0.35 map to 0.0; 1.00 maps to 0.5 * 1.0^2 = 0.5
    near_miss = [
        _build_episode(0.0, sampled_lengths=[2], metrics={"criteria_score": 0.20}),
        _build_episode(0.0, sampled_lengths=[2], metrics={"criteria_score": 0.35}),
        _build_episode(0.0, sampled_lengths=[2], metrics={"criteria_score": 1.00}),
    ]
    advs = _grpo(near_miss, fallback_reward=fb)
    mean_r = 0.5 / 3.0
    assert advs == pytest.approx([-mean_r, -mean_r, 0.5 - mean_r])


def test_grpo_fallback_disable_length_penalty_only_in_fallback():
    """disable_length_penalty skips LP in all-fail fallback groups while keeping it in binary groups."""
    fb = FallbackRewardConfig(metric="criteria_score", weight=0.5, disable_length_penalty=True)
    lp = LinearLengthPenaltyConfig(num_output_tokens_weight=0.5, num_input_tokens_weight=0.0, num_turns_weight=0.0)

    # All-fail fallback: longer rollout has higher criteria_score and must NOT be penalized by LP
    all_fail = [
        _build_episode(0.0, sampled_lengths=[2], metrics={"criteria_score": 0.2}),
        _build_episode(0.0, sampled_lengths=[8], metrics={"criteria_score": 0.8}),
    ]
    adv_fb = _grpo(all_fail, length_penalty=lp, fallback_reward=fb)
    assert adv_fb == pytest.approx([-0.15, 0.15])
    assert "length_penalty" not in all_fail[0].traces[0].metrics

    # Binary group: LP still applies so shorter pass beats longer pass
    binary = [
        _build_episode(1.0, sampled_lengths=[2], metrics={"criteria_score": 1.0}),
        _build_episode(1.0, sampled_lengths=[8], metrics={"criteria_score": 1.0}),
    ]
    adv_bin = _grpo(binary, length_penalty=lp, fallback_reward=fb)
    assert adv_bin[0] > adv_bin[1]
    assert "length_penalty" in binary[0].traces[0].metrics


def test_grpo_fallback_self_annealing_ema():
    """Fallback weight shrinks via (1 - p_bin_ema) ** anneal_power as binary groups arrive."""
    fb = FallbackRewardConfig(metric="criteria_score", weight=0.5, anneal_power=2.0, anneal_ema_decay=0.5)
    algo = GRPOAlgorithm(GRPOAlgoConfig(fallback_reward=fb), clients=None)

    # First group is binary-active -> ema = 1.0
    g_bin = [
        _build_episode(1.0, sampled_lengths=[2], metrics={"criteria_score": 1.0}),
        _build_episode(0.0, sampled_lengths=[2], metrics={"criteria_score": 0.0}),
    ]
    asyncio.run(algo.score_group(g_bin))
    assert g_bin[0].traces[0].metrics["fallback_binary_group_ema"] == pytest.approx(1.0)

    # Next group is all-fail -> ema = 0.5 * 1.0 + 0.5 * 0.0 = 0.5 -> w_eff = 0.5 * (1 - 0.5)^2 = 0.125
    g_fb = [
        _build_episode(0.0, sampled_lengths=[2], metrics={"criteria_score": 0.0}),
        _build_episode(0.0, sampled_lengths=[2], metrics={"criteria_score": 1.0}),
    ]
    asyncio.run(algo.score_group(g_fb))
    assert g_fb[0].traces[0].metrics["fallback_binary_group_ema"] == pytest.approx(0.5)
    assert g_fb[0].traces[0].metrics["fallback_reward_weight"] == pytest.approx(0.125)
    assert [_scalar(ep) for ep in g_fb] == pytest.approx([-0.0625, 0.0625])


def test_grpo_zero_on_timeout():
    """Timed-out traces get reward=0.0 (and criteria_score=0.0 in fallback) when zero_on_timeout=True."""
    fb = FallbackRewardConfig(metric="criteria_score", weight=0.5)
    algo = GRPOAlgorithm(GRPOAlgoConfig(fallback_reward=fb, zero_on_timeout=True), clients=None)

    g = [
        _build_episode(0.0, sampled_lengths=[2], metrics={"criteria_score": 0.8}),
        _build_episode(0.0, sampled_lengths=[8], metrics={"criteria_score": 0.9}),
    ]
    g[1].traces[0].is_timeout = True
    assert g[1].traces[0].is_timeout
    asyncio.run(algo.score_group(g))
    assert [ep.traces[0].reward for ep in g] == pytest.approx([0.4, 0.0])
    assert [_scalar(ep) for ep in g] == pytest.approx([0.2, -0.2])


class _FakePrefixRenderer:
    def __init__(self, prefix_ids: list[int] | None = None):
        self.prefix_ids = [900, 901, 902] if prefix_ids is None else prefix_ids
        self.calls: list[list[dict]] = []

    def render_ids(self, messages: list[dict], *, add_generation_prompt: bool = False) -> list[int]:
        assert not add_generation_prompt
        self.calls.append(messages)
        return list(self.prefix_ids)


class _FakeScoringClients:
    def __init__(self, *, fail: bool = False):
        self.fail = fail
        self.scored: list[list[int]] = []

    async def score(self, token_ids: list[int]) -> list[float]:
        self.scored.append(list(token_ids))
        if self.fail:
            raise RuntimeError("boom")
        return [-0.01 * (i + 1) for i in range(len(token_ids))]


def test_grpo_conditional_prefix_positive_only_and_alignment():
    """Only positive-advantage traces with a non-empty info_key are prefixed and rescored,
    keeping trace_to_samples, prepare_sample, and staleness alignment intact."""
    import numpy as np

    from prime_rl.orchestrator.batch import prepare_sample
    from prime_rl.orchestrator.train_sink import mask_stale_tokens, token_staleness

    cfg = GRPOAlgoConfig(conditional_prefix=ConditionalPrefixConfig(info_key="ctx_prefix", positive_only=True))
    clients = _FakeScoringClients()
    algo = GRPOAlgorithm(cfg, clients=clients)
    algo.renderer = _FakePrefixRenderer([900, 901, 902])

    # ep0: A < 0 with prefix -> must stay un-prefixed and not scored
    # ep1: A > 0 with prefix -> must be prefixed and rescored
    # ep2: A > 0 without prefix -> must stay un-prefixed and not scored
    ep0 = _build_episode(0.0, sampled_lengths=[2, 2], obs_lengths=[1])
    ep0.traces[0].info["ctx_prefix"] = "prefix for negative"
    ep1 = _build_episode(1.0, sampled_lengths=[2, 2], obs_lengths=[1])
    ep1.traces[0].info["ctx_prefix"] = "prefix for positive"
    ep2 = _build_episode(1.0, sampled_lengths=[2, 2], obs_lengths=[1])

    orig_len = len(ep1.traces[0].branches[0].token_ids)
    orig_ids = ep1.traces[0].branches[0].token_ids.tolist()

    asyncio.run(algo.score_group([ep0, ep1, ep2]))

    assert len(clients.scored) == 1
    assert clients.scored[0] == [900, 901, 902, *orig_ids]

    # ep0 and ep2 unchanged
    assert len(ep0.traces[0].branches[0].token_ids) == orig_len
    assert len(ep2.traces[0].branches[0].token_ids) == orig_len
    assert ep0.traces[0].metrics["conditional_prefix/applied"] == 0.0
    assert ep0.traces[0].metrics["conditional_prefix/has_prefix"] == 1.0
    assert "conditional_prefix/adv_bare" in ep0.traces[0].metrics
    assert ep2.traces[0].metrics["conditional_prefix/applied"] == 0.0
    assert ep2.traces[0].metrics["conditional_prefix/has_prefix"] == 0.0

    # ep1 prefixed + rescored + aligned through trace_to_samples, prepare_sample, and staleness
    b1 = ep1.traces[0].branches[0]
    assert b1.token_ids.tolist() == [900, 901, 902, *orig_ids]
    assert b1.sampled_mask[:3].tolist() == [False, False, False]
    assert int(b1.sampled_mask.sum()) == 4
    assert ep1.traces[0].metrics["conditional_prefix/applied"] == 1.0
    assert "conditional_prefix/adv_prefixed" in ep1.traces[0].metrics

    # Rescored logprobs match the fake scorer positions (root has 3 prefix + 1 prompt = 4 tokens,
    # turn 0 sampled at positions 4,5; obs at 6; turn 1 sampled at positions 7,8).
    expected_turn0_lp = [-0.05, -0.06]
    expected_turn1_lp = [-0.08, -0.09]
    assert ep1.traces[0].nodes[1].logprobs.tolist() == pytest.approx(expected_turn0_lp)
    assert ep1.traces[0].nodes[3].logprobs.tolist() == pytest.approx(expected_turn1_lp)

    samples = trace_to_samples(ep1.traces[0], env_name="test")
    assert len(samples) == 1
    sample = samples[0]
    sample.temperatures = np.ones(len(sample.token_ids), dtype=np.float32)
    staleness = token_staleness(b1, fallback=1, trained_version=3)
    assert len(staleness) == len(sample.token_ids)
    assert (staleness[:4] == -1).all()
    assert (staleness[4:6] == 2).all()
    sample.staleness = staleness
    mb = prepare_sample(sample, seq_len=64)
    assert len(mb.input_ids) == orig_len + 3
    masked = mask_stale_tokens(sample, b1, fallback=1, min_version=2)
    assert masked == 4


def test_grpo_conditional_prefix_max_fraction_and_guards():
    """max_positive_fraction caps prefixed positives; max_total_tokens and scorer errors fail open."""
    # Cap: 2 positive candidates with max_positive_fraction=0.5 -> only 1 (highest advantage) prefixed
    fb = FallbackRewardConfig(metric="criteria_score", weight=1.0)
    cfg = GRPOAlgoConfig(
        fallback_reward=fb,
        conditional_prefix=ConditionalPrefixConfig(info_key="ctx", positive_only=True, max_positive_fraction=0.5),
    )
    clients = _FakeScoringClients()
    algo = GRPOAlgorithm(cfg, clients=clients)
    algo.renderer = _FakePrefixRenderer([900, 901])

    ep0 = _build_episode(0.0, sampled_lengths=[2], metrics={"criteria_score": 0.0})
    ep1 = _build_episode(0.0, sampled_lengths=[2], metrics={"criteria_score": 0.6})
    ep2 = _build_episode(0.0, sampled_lengths=[2], metrics={"criteria_score": 0.9})
    for ep in (ep0, ep1, ep2):
        ep.traces[0].info["ctx"] = "p"
    asyncio.run(algo.score_group([ep0, ep1, ep2]))
    assert [ep.traces[0].metrics["conditional_prefix/applied"] for ep in (ep0, ep1, ep2)] == [0.0, 0.0, 1.0]

    # Length cap guard -> skipped gracefully
    cfg_len = GRPOAlgoConfig(
        conditional_prefix=ConditionalPrefixConfig(info_key="ctx", positive_only=True, max_total_tokens=4)
    )
    algo_len = GRPOAlgorithm(cfg_len, clients=_FakeScoringClients())
    algo_len.renderer = _FakePrefixRenderer([900, 901, 902])
    g_len = [_build_episode(0.0, sampled_lengths=[2]), _build_episode(1.0, sampled_lengths=[2])]
    g_len[1].traces[0].info["ctx"] = "too long"
    asyncio.run(algo_len.score_group(g_len))
    assert g_len[1].traces[0].metrics["conditional_prefix/applied"] == 0.0
    assert g_len[1].traces[0].metrics["conditional_prefix/skipped_length"] == 1.0

    # Scorer error guard -> fails open to bare trace
    cfg_err = GRPOAlgoConfig(conditional_prefix=ConditionalPrefixConfig(info_key="ctx", positive_only=True))
    algo_err = GRPOAlgorithm(cfg_err, clients=_FakeScoringClients(fail=True))
    algo_err.renderer = _FakePrefixRenderer([900, 901])
    g_err = [_build_episode(0.0, sampled_lengths=[2]), _build_episode(1.0, sampled_lengths=[2])]
    g_err[1].traces[0].info["ctx"] = "will fail score"
    orig = g_err[1].traces[0].branches[0].token_ids.tolist()
    asyncio.run(algo_err.score_group(g_err))
    assert g_err[1].traces[0].metrics["conditional_prefix/applied"] == 0.0
    assert g_err[1].traces[0].metrics["conditional_prefix/skipped_error"] == 1.0
    assert g_err[1].traces[0].branches[0].token_ids.tolist() == orig


def test_grpo_conditional_prefix_zero_group_promotion_ranking_ema_and_revert():
    """Zero-advantage all-fail groups promote top-k prefixed traces by rank_metric, respect
    the global EMA cap, never promote all-pass groups, and revert advantage to 0 on score error."""
    cfg = GRPOAlgoConfig(
        conditional_prefix=ConditionalPrefixConfig(
            info_key="ctx",
            positive_only=True,
            zero_group_advantage=0.25,
            zero_group_max_traces=2,
            zero_group_rank_metric="quality",
            max_global_prefixed_positive_fraction=0.65,
            ema_decay=0.5,
        )
    )
    clients = _FakeScoringClients()
    algo = GRPOAlgorithm(cfg, clients=clients)
    algo.renderer = _FakePrefixRenderer([900, 901])

    # Group 1: 0/4 all-fail -> should promote top-2 by quality (ep2=0.8, ep1=0.6), skipping timeout ep3=0.95
    ep0 = _build_episode(0.0, sampled_lengths=[2], metrics={"quality": 0.2})
    ep1 = _build_episode(0.0, sampled_lengths=[2], metrics={"quality": 0.6})
    ep2 = _build_episode(0.0, sampled_lengths=[2], metrics={"quality": 0.8})
    ep3 = _build_episode(0.0, sampled_lengths=[2], metrics={"quality": 0.95})
    ep3.traces[0].is_timeout = True
    for ep in (ep0, ep1, ep2, ep3):
        ep.traces[0].info["ctx"] = "prefix text"

    asyncio.run(algo.score_group([ep0, ep1, ep2, ep3]))
    assert [_scalar(ep) for ep in (ep0, ep1, ep2, ep3)] == pytest.approx([0.0, 0.25, 0.25, 0.0])
    assert [ep.traces[0].metrics["conditional_prefix/zero_group_promoted"] for ep in (ep0, ep1, ep2, ep3)] == [
        0.0,
        1.0,
        1.0,
        0.0,
    ]
    # EMA is now 1.0 (> 0.65 cap) -> next 0/2 group must be paused!
    assert ep1.traces[0].metrics["conditional_prefix/prefixed_pos_ema"] == pytest.approx(1.0)

    g_paused = [
        _build_episode(0.0, sampled_lengths=[2], metrics={"quality": 0.9}),
        _build_episode(0.0, sampled_lengths=[2], metrics={"quality": 0.8}),
    ]
    for ep in g_paused:
        ep.traces[0].info["ctx"] = "prefix text"
    asyncio.run(algo.score_group(g_paused))
    assert [_scalar(ep) for ep in g_paused] == pytest.approx([0.0, 0.0])

    # Clean mixed group (1/2 pass with no prefix) lowers EMA: 0.5*1.0 + 0.5*0.0 = 0.5 <= 0.65!
    g_clean = [_build_episode(0.0, sampled_lengths=[2]), _build_episode(1.0, sampled_lengths=[2])]
    asyncio.run(algo.score_group(g_clean))
    assert g_clean[1].traces[0].metrics["conditional_prefix/prefixed_pos_ema"] == pytest.approx(0.5)

    # All-pass group (2/2 reward=1.0) must NEVER be promoted as a zero-group!
    g_allpass = [_build_episode(1.0, sampled_lengths=[2]), _build_episode(1.0, sampled_lengths=[2])]
    for ep in g_allpass:
        ep.traces[0].info["ctx"] = "prefix text"
    asyncio.run(algo.score_group(g_allpass))
    assert [_scalar(ep) for ep in g_allpass] == pytest.approx([0.0, 0.0])

    # Scoring failure on a zero-group-promoted trace must revert advantage to 0.0 (never reinforce bare!)
    algo_fail = GRPOAlgorithm(cfg, clients=_FakeScoringClients(fail=True))
    algo_fail.renderer = _FakePrefixRenderer([900, 901])
    g_fail = [_build_episode(0.0, sampled_lengths=[2], metrics={"quality": 0.9})]
    g_fail[0].traces[0].info["ctx"] = "prefix text"
    asyncio.run(algo_fail.score_group(g_fail))
    assert _scalar(g_fail[0]) == pytest.approx(0.0)
    assert g_fail[0].traces[0].metrics["conditional_prefix/skipped_error"] == 1.0
