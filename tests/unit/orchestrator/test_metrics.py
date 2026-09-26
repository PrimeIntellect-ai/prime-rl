import math
from itertools import count
from types import SimpleNamespace

import pytest
import verifiers.v1 as vf

from prime_rl.orchestrator.metrics import Episodes, Stat
from prime_rl.orchestrator.types import Group
from prime_rl.orchestrator.utils import compute_pass_metrics

_ids = count()


def mk(
    reward: float = 0.0,
    *,
    episode_id: str = "",
    agent_name: str = "agent",
    num_total_tokens: int = 10,
    num_input_tokens: int = 4,
    num_output_tokens: int = 6,
    num_turns: int = 1,
    num_branches: int = 1,
    is_truncated: bool = False,
    is_completed: bool = True,
    has_error: bool = False,
    error_type: str = "error",
    stop_condition: str | None = None,
    metrics: dict | None = None,
    rewards: dict | None = None,
    env_name: str = "env",
    group_id: str = "g0",
    trainable: bool = True,
    is_trainable: bool = True,
    is_admitted: bool = True,
    sampled: bool = True,
    setup: float = 0.0,
    agent: float = 0.0,
    agent_model: float = 0.0,
    agent_harness: float = 0.0,
    finalize: float = 0.0,
    scoring: float = 0.0,
):
    """Build one episode around a trace-shaped metrics fixture."""
    trace = SimpleNamespace(
        id=f"t{next(_ids)}",
        reward=reward,
        rewards=rewards or {},
        num_total_tokens=num_total_tokens,
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        num_turns=num_turns,
        num_branches=num_branches,
        is_truncated=is_truncated,
        is_completed=is_completed,
        has_error=has_error,
        last_error=SimpleNamespace(type=error_type) if has_error else None,
        stop_condition=stop_condition,
        metrics=metrics or {},
        agent=SimpleNamespace(trainable=trainable, name=agent_name),
        nodes=[SimpleNamespace(advantages=[1.0] if is_trainable else [0.0])],
        timing=SimpleNamespace(
            setup=SimpleNamespace(duration=setup),
            agent=SimpleNamespace(
                duration=agent,
                model=SimpleNamespace(duration=agent_model),
                harness=SimpleNamespace(duration=agent_harness),
            ),
            finalize=SimpleNamespace(duration=finalize),
            scoring=SimpleNamespace(duration=scoring),
        ),
    )
    episode = SimpleNamespace(
        id=episode_id or f"e{next(_ids)}",
        traces=[trace],
        ok=not has_error,
        errors=[],
        last_error=None,
        env=SimpleNamespace(id=env_name, name=env_name),
        group=SimpleNamespace(id=group_id),
    )
    episode._sampled = {trace.id} if sampled else set()
    episode._admitted = is_admitted
    return episode


def combine(*episodes):
    """Combine trace fixtures into one multi-trace episode."""
    first = episodes[0]
    first.traces = [trace for episode in episodes for trace in episode.traces]
    first._sampled = {trace_id for episode in episodes for trace_id in episode._sampled}
    first._admitted = all(episode._admitted for episode in episodes)
    return first


def groups(episodes) -> list[Group]:
    """One group per (env, group id); sampled traces carry a placeholder payload."""
    by_key: dict[tuple[str, str], list] = {}
    for episode in episodes:
        by_key.setdefault((episode.env.name, episode.group.id), []).append(episode)
    return [
        Group(
            env,
            gid,
            1,
            members,
            admitted=all(episode._admitted for episode in members),
            samples={trace_id: [] for episode in members for trace_id in episode._sampled},
        )
        for (env, gid), members in by_key.items()
    ]


def view(episodes) -> Episodes:
    return Episodes(groups(episodes))


def train_metrics(episodes, subset: str = "all") -> dict:
    pool = view(episodes)
    return (pool if subset == "all" else pool.clean.sampled).train_metrics("train/agg", subset=subset)


def eval_metrics(episodes, subset: str = "all", k: int = 2) -> dict:
    pool = view(episodes)
    return (pool if subset == "all" else pool.clean).eval_metrics("eval/x", subset=subset, k=k)


def test_stat():
    s = Stat([1.0, 2.0, 3.0])
    assert (s.mean(), s.max(), s.min()) == (2.0, 3.0, 1.0)
    assert (s.percentile(10), s.percentile(90)) == pytest.approx((1.2, 2.8))  # linear-interpolated percentiles
    assert s.to_dict("p") == pytest.approx({"p/mean": 2.0, "p/max": 3.0, "p/min": 1.0, "p/p10": 1.2, "p/p90": 2.8})
    assert Stat([]).percentile(90) == 0.0 and Stat([]).to_dict("p") == {}


def test_views_compose_and_narrow_the_episodes():
    pool = view(
        [
            mk(env_name="a"),
            mk(env_name="a", has_error=True),
            mk(env_name="b", is_admitted=False),
            mk(env_name="b", sampled=False),
        ]
    )
    assert len(pool) == 4 and [episode.env.name for episode in pool] == ["a", "a", "b", "b"]
    assert len(pool.clean) == 3 and len(pool.sampled) == 3 and len(pool.clean.sampled) == 2
    assert len(pool.admitted) == 2
    assert all(not episode.traces[0].has_error and episode in pool.episodes for episode in pool.clean)
    by_env = pool.by_env()
    assert set(by_env) == {"a", "b"} and len(by_env["a"]) == 2 and isinstance(by_env["a"], Episodes)
    assert len(pool.clean.by_env()["a"]) == 1


def test_trace_less_episodes_count_in_the_root_view_only():
    blank = SimpleNamespace(
        id="blank",
        traces=[],
        ok=False,
        errors=[vf.Error(type="Boom", message="boom")],
        last_error=vf.Error(type="Boom", message="boom"),
        env=SimpleNamespace(id="env", name="env"),
        group=SimpleNamespace(id="g0"),
        _sampled=set(),
        _admitted=False,
    )
    pool = view([mk(reward=1.0), blank])
    assert len(pool) == 2 and pool.num_traces == 1
    assert len(pool.clean) == 1
    out = pool.train_metrics("train/agg", subset="all")
    assert out["train/agg/all/has_error/mean"] == 0.5
    assert out["train/agg/all/error/Boom"] == 1


def test_distributions():
    pool = view(
        [
            mk(reward=1.0, num_total_tokens=10, num_input_tokens=4),
            mk(reward=0.0, num_total_tokens=20, num_input_tokens=6),
        ]
    )
    assert pool.episode_stat(lambda trace: trace.num_input_tokens).mean() == 5.0  # fluent Stat access
    out = pool.train_metrics("train/agg", subset="all")
    assert out["train/agg/all/agent/reward/mean"] == 0.5
    assert "train/agg/all/reward/mean" not in out  # trace-level metrics are agent-only
    assert out["train/agg/all/num_total_tokens/mean"] == 15.0
    assert out["train/agg/all/num_total_tokens/max"] == 20.0  # single-trace episodes: one value per rollout
    assert out["train/agg/all/num_input_tokens/mean"] == 5.0
    assert out["train/agg/all/num_output_tokens/mean"] == 6.0


def test_episode_and_agent_levels():
    # Two proposer-solver episodes: one proposer + two solvers each (the solver fan-out).
    episodes = [
        combine(
            mk(reward=1.0, num_turns=1, agent_name="proposer", episode_id="e1"),
            mk(reward=0.0, num_turns=2, agent_name="solver", episode_id="e1"),
            mk(reward=1.0, num_turns=4, agent_name="solver", episode_id="e1"),
        ),
        combine(
            mk(reward=0.0, num_turns=3, agent_name="proposer", episode_id="e2"),
            mk(reward=1.0, num_turns=6, agent_name="solver", episode_id="e2"),
            mk(reward=0.0, num_turns=8, agent_name="solver", episode_id="e2"),
        ),
    ]
    pool = view(episodes)
    assert pool.num_turns.mean() == 12.0  # episode-level sums: 1+2+4 and 3+6+8
    assert pool.episode_stat(lambda trace: trace.num_total_tokens).values == [30.0, 30.0]
    out = pool.train_metrics("train/agg", subset="all")
    assert out["train/agg/all/num_turns/mean"] == 12.0
    assert out["train/agg/all/proposer/num_turns/mean"] == 2.0  # (1 + 3) / 2
    assert out["train/agg/all/solver/num_turns/mean"] == 5.0  # flat over the 4 solver traces
    assert out["train/agg/all/solver/num_turns/max"] == 8.0  # a real trace, not an episode mean
    assert out["train/agg/all/solver/reward/mean"] == 0.5
    assert out["train/agg/all/proposer/is_truncated/mean"] == 0.0
    assert "train/agg/all/proposer/is_truncated/p90" not in out  # rates emit /mean only
    assert "train/agg/all/reward/mean" not in out  # reward never pools across agents


def test_agent_metrics_are_flat_over_traces():
    """Inside a seat the trace is the unit of aggregation, so an uneven fan-out (one solver trace
    from this episode, three from that) never reweights anything: every agent-level metric is the
    plain figure over that agent's rollouts."""
    rollouts = [
        mk(agent_name="solver", episode_id="e1", is_truncated=True, reward=1.0),
        *[mk(agent_name="solver", episode_id="e2", is_truncated=False, reward=0.0) for _ in range(3)],
    ]
    out = train_metrics(rollouts)
    assert out["train/agg/all/solver/is_truncated/mean"] == 0.25  # 1 of 4 traces, not (1.0 + 0.0) / 2
    assert out["train/agg/all/solver/is_completed/mean"] == 1.0
    assert out["train/agg/all/solver/is_trainable/mean"] == 1.0  # sibling rates agree
    assert out["train/agg/all/solver/reward/mean"] == 0.25  # 1 of 4 traces scored, not (1.0 + 0.0) / 2


def test_boolean_rates_and_error_breakdown_all_only():
    rollouts = [mk(is_truncated=True), mk(has_error=True, error_type="ProviderError"), mk(is_admitted=False)]
    out = train_metrics(rollouts)
    assert out["train/agg/all/agent/is_truncated/mean"] == 1 / 3
    assert out["train/agg/all/agent/is_completed/mean"] == 1.0
    assert out["train/agg/all/agent/has_error/mean"] == 1 / 3
    assert out["train/agg/all/has_error/mean"] == 1 / 3
    assert out["train/agg/all/error/ProviderError"] == 1  # error-type breakdown by count, per episode
    # has_error + the error-type counts are structurally empty on effective, so emitted on `all` only
    eff = train_metrics(rollouts, "effective")
    assert not any(k.endswith("/has_error/mean") or "/error/" in k for k in eff)


def test_solve_rates():
    rates = {"A": [1.0, 1.0], "B": [0.0, 0.0], "C": [1.0, 0.0], "D": [1.0, 0.0]}  # all / none / some / some
    out = train_metrics([mk(reward=r, group_id=g) for g, rs in rates.items() for r in rs])
    assert (
        out["train/agg/all/agent/solved_all"],
        out["train/agg/all/agent/solved_none"],
        out["train/agg/all/agent/solved_some"],
    ) == (0.25, 0.25, 0.5)


def test_stop_condition_breakdown():
    truncated = [mk(is_truncated=True, stop_condition=c) for c in ("length", "max_turns", "prompt_too_long")]
    out = train_metrics(truncated + [mk(stop_condition=None)])
    assert out["train/agg/all/agent/stop_condition/generation_truncated"] == 0.5  # truncated & not prompt_too_long
    assert out["train/agg/all/agent/stop_condition/length"] == 1 / 3  # over the 3 recorded conditions
    assert out["train/agg/all/agent/stop_condition/prompt_too_long"] == 1 / 3


def test_nested_metrics_and_rewards():
    rollouts = [
        mk(metrics={"acc": 1.0}, rewards={"correct": vf.Reward(score=1.0), "format": vf.Reward(score=0.0)}),
        mk(metrics={"acc": 3.0, "fmt": 5.0}, rewards={"correct": vf.Reward(score=0.0), "format": vf.Reward(score=1.0)}),
        # scoring failed after seeding: unscored (None) entries count as 0.0 on `all`
        mk(has_error=True, metrics={"acc": None}, rewards={"correct": None, "format": None}),
    ]
    out = train_metrics(rollouts)
    assert out["train/agg/all/agent/metrics/acc/mean"] == pytest.approx(4 / 3)
    assert out["train/agg/all/agent/metrics/fmt/mean"] == 5.0  # single reporter
    assert out["train/agg/all/agent/rewards/format/mean"] == pytest.approx(1 / 3)
    # effective drops the errored rollout, so its seeds don't dilute the effective means
    eff = train_metrics(rollouts, "effective")
    assert eff["train/agg/effective/agent/metrics/acc/mean"] == 2.0
    assert eff["train/agg/effective/agent/rewards/format/mean"] == 0.5
    # cross-env agg: another env's unscored trace carries different keys, so it can't dilute these
    other = mk(env_name="other", has_error=True, rewards={"solved": None})
    agg = train_metrics(rollouts + [other])
    assert agg["train/agg/all/agent/rewards/format/mean"] == pytest.approx(1 / 3)
    assert agg["train/agg/all/agent/rewards/solved/mean"] == 0.0


def test_nested_timing():
    out = train_metrics([mk(setup=1.0, agent=2.0, agent_model=1.5, agent_harness=0.5, finalize=0.5, scoring=0.5)])
    assert out["train/agg/all/agent/timing/setup/mean"] == 1.0
    assert out["train/agg/all/agent/timing/total/mean"] == 4.0  # total sums all four phases
    assert out["train/agg/all/agent/timing/agent/model/mean"] == 1.5
    assert out["train/agg/all/agent/timing/agent/harness/mean"] == 0.5


def test_train_only_metrics_absent_from_eval():
    rollouts = [
        mk(is_trainable=True, is_admitted=False, group_id="g0"),
        mk(is_trainable=False, group_id="g1"),
    ]
    out = train_metrics(rollouts)
    assert out["train/agg/all/agent/is_trainable/mean"] == 0.5
    assert out["train/agg/all/agent/is_admitted/mean"] == 0.5
    assert "train/agg/all/is_trainable/mean" not in out  # pipeline verdicts are per-trace
    eval_out = eval_metrics(rollouts)
    assert not any("is_trainable" in key or "is_admitted" in key for key in eval_out)


def test_eval_avg_at_k_and_pass_k():
    binary = [mk(reward=1.0, group_id="g0"), mk(reward=0.0, group_id="g0")]
    eff = eval_metrics(binary, "effective")
    assert eff["eval/x/effective/agent/avg@2"] == 0.5  # k is the configured episode group size
    assert "eval/x/effective/avg@2" not in eff  # scores are per-agent, never pooled
    assert eff["eval/x/effective/agent/pass@1"] == 0.5 and eff["eval/x/effective/agent/pass^2"] == 0.0
    all_out = eval_metrics(binary)
    assert all_out["eval/x/all/agent/avg@2"] == 0.5
    assert not any("pass@" in k or "pass^" in k for k in all_out)  # pass@k effective-only
    non_binary = [mk(reward=0.5, group_id="g0"), mk(reward=1.0, group_id="g0")]
    assert not any("pass@" in k for k in eval_metrics(non_binary, "effective"))

    multi_agent = [
        combine(mk(agent_name="proposer"), mk(agent_name="solver"), mk(agent_name="solver")),
        combine(mk(agent_name="proposer"), mk(agent_name="solver"), mk(agent_name="solver")),
    ]
    multi_agent_out = eval_metrics(multi_agent)
    assert "eval/x/all/proposer/avg@2" in multi_agent_out
    assert "eval/x/all/solver/avg@2" in multi_agent_out
    assert not any("avg@4" in key for key in multi_agent_out)


def test_compute_pass_metrics_matches_closed_form():
    out = compute_pass_metrics([1.0, 1.0, 0.0, 0.0])  # n=4, c=2
    assert out["pass@1"] == 1.0 - math.comb(2, 1) / math.comb(4, 1)
    assert out["pass@2"] == 1.0 - math.comb(2, 2) / math.comb(4, 2)
    assert out["pass^2"] == math.comb(2, 2) / math.comb(4, 2)
    assert set(out) == {"pass@1", "pass@2", "pass@4", "pass^1", "pass^2", "pass^4"}
