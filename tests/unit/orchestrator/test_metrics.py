import math
from dataclasses import replace
from itertools import count
from types import SimpleNamespace
from uuid import uuid4

import pytest
import verifiers.v1 as vf

from prime_rl.orchestrator.eval_sink import eval_step_of
from prime_rl.orchestrator.metrics import EvalRollouts, Stat, TrainRollouts
from prime_rl.orchestrator.types import (
    Episode,
    InflightEpisode,
    Rollout,
    TrainRollout,
    rollouts_of,
    run_of,
)
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
    trainable: bool = True,
    is_trainable: bool = True,
    is_filtered: bool = False,
    detections: dict | None = None,
    setup: float = 0.0,
    agent: float = 0.0,
    agent_model: float = 0.0,
    agent_harness: float = 0.0,
    finalize: float = 0.0,
    scoring: float = 0.0,
):
    """Duck-typed stand-in for ``Rollout``, exposing only the Trace properties the metrics read.
    Without an ``episode_id`` each rollout is its own single-trace episode (the unique ``id``
    is the grouping fallback)."""
    return SimpleNamespace(
        id=f"t{next(_ids)}",
        episode_id=episode_id,
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
        env_name=env_name,
        agent=SimpleNamespace(trainable=trainable, name=agent_name),
        is_trainable=is_trainable,
        is_filtered=is_filtered,
        detections=detections or {},
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


def ep(*rollouts, env_name: str = "env", errors=(), group_id="g0", cls=Episode):
    """One episode over these traces, named and grouped the way the dispatcher stamps it.
    ``model_construct`` skips validation so the duck-typed stand-ins above can stand in for real
    ones."""
    return cls.model_construct(
        id=f"e{next(_ids)}",
        traces=list(rollouts),
        env=vf.EnvInfo(name=env_name),
        group=vf.GroupInfo(id=group_id),
        errors=list(errors),
    )


def solo(rollouts, group_ids=None, cls=Episode):
    """Each rollout as its own single-trace episode — the single-agent shape. ``group_ids`` gives
    the example each answers; by default they all answer the same one."""
    ids = group_ids or ["g0"] * len(rollouts)
    return [ep(r, group_id=g, cls=cls) for r, g in zip(rollouts, ids, strict=True)]


def train_wandb(rollouts, subset: str = "all", group_ids=None) -> dict:
    return TrainRollouts(solo(rollouts, group_ids)).metrics.to_wandb(prefix="train/agg", subset=subset)


def test_stat():
    s = Stat([1.0, 2.0, 3.0])
    assert (s.mean(), s.max(), s.min()) == (2.0, 3.0, 1.0)
    assert (s.p10(), s.p90()) == pytest.approx((1.2, 2.8))  # linear-interpolated percentiles
    assert s.to_dict("p") == pytest.approx({"p/mean": 2.0, "p/max": 3.0, "p/min": 1.0, "p/p10": 1.2, "p/p90": 2.8})
    assert Stat([]).p90() == 0.0 and Stat([]).to_dict("p") == {}


def test_container_effective_by_env_and_listlike():
    rc = TrainRollouts(
        [
            ep(mk(env_name="a"), env_name="a"),
            ep(mk(env_name="a", has_error=True), env_name="a"),
            ep(mk(env_name="b", is_filtered=True), env_name="b"),
            ep(mk(env_name="b"), env_name="b"),
        ]
    )
    assert len(rc) == 4 and [r.env_name for r in rc] == ["a", "a", "b", "b"]  # sized + iterable
    eff = rc.effective
    assert isinstance(eff, TrainRollouts) and len(eff) == 2  # same type, errored + filtered dropped
    assert all(not r.has_error and not r.is_filtered and r in rc.rollouts for r in eff)  # view of references
    by_env = rc.by_env()
    assert set(by_env) == {"a", "b"} and len(by_env["a"]) == 2 and isinstance(by_env["a"], TrainRollouts)
    rc.append(ep(mk()))
    assert len(rc) == 5  # sized by rollout, appended by episode


def test_to_wandb_distributions():
    m = TrainRollouts(
        solo(
            [
                mk(reward=1.0, num_total_tokens=10, num_input_tokens=4),
                mk(reward=0.0, num_total_tokens=20, num_input_tokens=6),
            ]
        )
    ).metrics
    assert m.num_input_tokens.mean() == 5.0  # fluent Stat access
    out = m.to_wandb(prefix="train/agg", subset="all")
    assert out["train/agg/all/agent/reward/mean"] == 0.5
    assert "train/agg/all/reward/mean" not in out  # trace-level metrics are agent-only
    assert out["train/agg/all/num_total_tokens/mean"] == 15.0
    assert out["train/agg/all/num_total_tokens/max"] == 20.0  # single-trace episodes: one value per rollout
    assert out["train/agg/all/num_input_tokens/mean"] == 5.0
    assert out["train/agg/all/num_output_tokens/mean"] == 6.0


def test_episode_and_agent_levels():
    # Two proposer-solver episodes: one proposer + two solvers each (the solver fan-out).
    m = TrainRollouts(
        [
            ep(
                mk(reward=1.0, num_turns=1, agent_name="proposer"),
                mk(reward=0.0, num_turns=2, agent_name="solver"),
                mk(reward=1.0, num_turns=4, agent_name="solver"),
            ),
            ep(
                mk(reward=0.0, num_turns=3, agent_name="proposer"),
                mk(reward=1.0, num_turns=6, agent_name="solver"),
                mk(reward=0.0, num_turns=8, agent_name="solver"),
            ),
        ]
    ).metrics
    assert m.num_turns.mean() == 12.0  # episode-level sums: 1+2+4 and 3+6+8
    assert m.num_total_tokens.values == [30.0, 30.0]  # summed across the episode's traces
    out = m.to_wandb(prefix="train/agg", subset="all")
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
    episodes = [
        ep(mk(agent_name="solver", is_truncated=True, reward=1.0)),
        ep(*[mk(agent_name="solver", is_truncated=False, reward=0.0) for _ in range(3)]),
    ]
    out = TrainRollouts(episodes).metrics.to_wandb(prefix="train/agg", subset="all")
    assert out["train/agg/all/solver/is_truncated/mean"] == 0.25  # 1 of 4 traces, not (1.0 + 0.0) / 2
    assert out["train/agg/all/solver/is_completed/mean"] == 1.0
    assert out["train/agg/all/solver/is_trainable/mean"] == 1.0  # sibling rates agree
    assert out["train/agg/all/solver/reward/mean"] == 0.25  # 1 of 4 traces scored, not (1.0 + 0.0) / 2


def test_boolean_rates_and_error_breakdown_all_only():
    rc = TrainRollouts(
        solo([mk(is_truncated=True), mk(has_error=True, error_type="ProviderError"), mk(is_filtered=True)])
    )
    out = rc.metrics.to_wandb(prefix="train/agg", subset="all")
    assert out["train/agg/all/agent/is_truncated/mean"] == 1 / 3
    assert out["train/agg/all/agent/is_completed/mean"] == 1.0
    assert out["train/agg/all/agent/has_error/mean"] == 1 / 3
    assert out["train/agg/all/agent/error/ProviderError"] == 1  # error-type breakdown by count
    assert not any("no_response" in k for k in out)  # removed metric
    # has_error + the error-type counts are structurally empty on effective, so emitted on `all` only
    eff = rc.effective.metrics.to_wandb(prefix="train/agg", subset="effective")
    assert not any(k.endswith("/has_error/mean") or "/error/" in k for k in eff)


def test_solve_rates():
    groups = {"A": [1.0, 1.0], "B": [0.0, 0.0], "C": [1.0, 0.0], "D": [1.0, 0.0]}  # all / none / some / some
    out = train_wandb(
        [mk(reward=r) for _, rs in groups.items() for r in rs], group_ids=[g for g, rs in groups.items() for _ in rs]
    )
    rates = (
        out["train/agg/all/agent/solved_all"],
        out["train/agg/all/agent/solved_none"],
        out["train/agg/all/agent/solved_some"],
    )
    assert rates == (0.25, 0.25, 0.5)


def test_stop_condition_breakdown():
    truncated = [mk(is_truncated=True, stop_condition=c) for c in ("length", "max_turns", "prompt_too_long")]
    out = train_wandb(truncated + [mk(stop_condition=None)])
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
    rc = TrainRollouts(solo(rollouts))
    m = rc.metrics
    agent = m.by_agent()["agent"]
    assert agent.metrics["acc"].mean() == pytest.approx(4 / 3)  # nested group access
    assert agent.rewards["correct"].mean() == pytest.approx(1 / 3)
    out = m.to_wandb(prefix="train/agg", subset="all")
    assert out["train/agg/all/agent/metrics/acc/mean"] == pytest.approx(4 / 3)
    assert out["train/agg/all/agent/metrics/fmt/mean"] == 5.0  # single reporter
    assert out["train/agg/all/agent/rewards/format/mean"] == pytest.approx(1 / 3)
    # effective drops the errored rollout, so its seeds don't dilute the effective means
    eff = rc.effective.metrics.to_wandb(prefix="train/agg", subset="effective")
    assert eff["train/agg/effective/agent/metrics/acc/mean"] == 2.0
    assert eff["train/agg/effective/agent/rewards/format/mean"] == 0.5
    # cross-env agg: another env's unscored trace carries different keys, so it can't dilute these
    other = mk(env_name="other", has_error=True, rewards={"solved": None})
    agg = TrainRollouts(solo(rollouts) + [ep(other, env_name="other")]).metrics.to_wandb(
        prefix="train/agg", subset="all"
    )
    assert agg["train/agg/all/agent/rewards/format/mean"] == pytest.approx(1 / 3)
    assert agg["train/agg/all/agent/rewards/solved/mean"] == 0.0


def test_nested_timing():
    m = TrainRollouts(
        solo([mk(setup=1.0, agent=2.0, agent_model=1.5, agent_harness=0.5, finalize=0.5, scoring=0.5)])
    ).metrics
    timing = m.by_agent()["agent"].timing
    assert timing.setup.mean() == 1.0 and timing.total.mean() == 4.0  # total sums all four phases
    assert timing.agent_model.mean() == 1.5 and timing.agent_harness.mean() == 0.5
    out = m.to_wandb(prefix="train/agg", subset="all")
    assert out["train/agg/all/agent/timing/setup/mean"] == 1.0
    assert out["train/agg/all/agent/timing/total/mean"] == 4.0
    assert out["train/agg/all/agent/timing/agent/model/mean"] == 1.5
    assert out["train/agg/all/agent/timing/agent/harness/mean"] == 0.5


def test_train_only_metrics_absent_from_eval():
    rollouts = [
        mk(is_trainable=True, is_filtered=True, detections={"gibberish": True}),
        mk(is_trainable=False, detections={"gibberish": False}),
    ]
    out = train_wandb(rollouts)
    assert out["train/agg/all/agent/is_trainable/mean"] == 0.5
    assert out["train/agg/all/agent/is_filtered/mean"] == 0.5
    assert out["train/agg/all/agent/detected/gibberish/mean"] == 0.5
    assert "train/agg/all/is_trainable/mean" not in out  # pipeline verdicts are per-trace
    eval_out = EvalRollouts(solo(rollouts)).metrics.to_wandb(prefix="eval/x", subset="all")
    assert not any("is_trainable" in k or "is_filtered" in k or "/detected/" in k for k in eval_out)


def test_eval_avg_at_k_and_pass_k():
    binary = EvalRollouts(solo([mk(reward=1.0), mk(reward=0.0)]))
    eff = binary.effective.metrics.to_wandb(prefix="eval/x", subset="effective")
    assert eff["eval/x/effective/agent/avg@2"] == 0.5  # mean reward under avg@<k> (k from the groups)
    assert "eval/x/effective/avg@2" not in eff  # scores are per-agent, never pooled
    assert eff["eval/x/effective/agent/pass@1"] == 0.5 and eff["eval/x/effective/agent/pass^2"] == 0.0
    all_out = binary.metrics.to_wandb(prefix="eval/x", subset="all")
    assert all_out["eval/x/all/agent/avg@2"] == 0.5
    assert not any("pass@" in k or "pass^" in k for k in all_out)  # pass@k effective-only
    non_binary = EvalRollouts(solo([mk(reward=0.5), mk(reward=1.0)]))
    assert not any("pass@" in k for k in non_binary.effective.metrics.to_wandb(prefix="eval/x", subset="effective"))


def test_compute_pass_metrics_matches_closed_form():
    out = compute_pass_metrics([1.0, 1.0, 0.0, 0.0])  # n=4, c=2
    assert out["pass@1"] == 1.0 - math.comb(2, 1) / math.comb(4, 1)
    assert out["pass@2"] == 1.0 - math.comb(2, 2) / math.comb(4, 2)
    assert out["pass^2"] == math.comb(2, 2) / math.comb(4, 2)
    assert set(out) == {"pass@1", "pass@2", "pass@4", "pass^1", "pass^2", "pass^4"}


def test_traceless_episode_keeps_its_reason():
    """The failure needs no type of its own: ``failed`` is just "no traces", and the reason is the
    error vf (or the dispatcher) already put on the episode."""
    episode = ep(errors=[vf.Error(type="Cancelled", message="Off-policy cancel")])
    assert not episode.traces and rollouts_of(episode) == []
    assert episode.last_error is not None and episode.last_error.type == "Cancelled"
    assert ep(mk()).traces
    # A traceless episode still counts as an episode, but contributes no rollouts.
    pool = TrainRollouts([ep(mk()), episode])
    assert len(pool) == 1 and len(pool.episodes) == 2
    assert len(pool.effective.episodes) == 1  # it survives nothing, so the subset drops it


def test_training_state_is_train_only():
    """Only a train rollout carries trainer-bound state; an eval trace has no field for it.
    Credit is not among them — it lives on the graph's nodes, which every trace has."""
    assert {"samples", "is_filtered", "detections"} <= set(TrainRollout.model_fields)
    assert "advantages" not in TrainRollout.model_fields  # derived from the nodes
    # An eval trace is the env's own, unextended — where a trace sits is the episode's to say.
    assert Rollout is vf.Trace
    assert not {"samples", "is_filtered", "detections", "group_id", "episode_id"} & set(Rollout.model_fields)


def test_inflight_episode_stamps_what_lands():
    """The dispatch and the landed episode are one pair: `stamp` is the only place the facts of a
    dispatch become facts of an episode, and the run it writes is what tells the rest of the
    orchestrator which path the episode is on."""
    wire = vf.WireEpisode.model_construct(id="e", traces=[])
    inflight = InflightEpisode(
        kind="train", env_name="rt", group_id=uuid4(), policy_version=3, episodes_owed=1, off_policy_steps=2
    )
    train = inflight.stamp(wire, run_id="r", policy_version=7, eval_step=None)
    run = run_of(train)
    assert (run.kind, run.policy_version, run.off_policy_steps) == ("train", 7, 2)  # group's wins
    assert run.step is None  # the batch window it lands in is not known yet
    assert train.env.name == "rt" and train.group is not None

    evaluation = replace(inflight, kind="eval").stamp(
        vf.WireEpisode.model_construct(id="e", traces=[]), run_id="r", policy_version=7, eval_step=12
    )
    # An online eval belongs to the same training run — same record, told apart by kind.
    assert (run_of(evaluation).kind, run_of(evaluation).step) == ("eval", 12)
    assert run_of(evaluation).off_policy_steps == 0  # only what trains can go stale
    with pytest.raises(AssertionError):  # an eval episode without its step is not representable
        replace(inflight, kind="eval").stamp(wire, run_id="r", policy_version=7, eval_step=None)


def test_eval_sink_reads_the_epoch_a_stamped_episode_landed_in():
    """The eval sink places an episode by the step on its run. Regression: online eval moved onto
    the training run (kind="eval"), and nothing unit-tested the sink, so the sink kept asserting on
    the old record and only a live run caught it."""
    inflight = InflightEpisode(
        kind="eval", env_name="rt", group_id=uuid4(), policy_version=3, episodes_owed=1, eval_step=12
    )
    landed = inflight.stamp(
        vf.WireEpisode.model_construct(id="e", traces=[]), run_id="r", policy_version=3, eval_step=12
    )
    assert eval_step_of(landed) == 12

    trained_on = replace(inflight, kind="train").stamp(
        vf.WireEpisode.model_construct(id="e2", traces=[]), run_id="r", policy_version=3, eval_step=None
    )
    with pytest.raises(AssertionError):  # an episode to train on is not placed by an eval epoch
        eval_step_of(trained_on)


def test_empty_is_not_the_same_as_failed():
    """An episode can error and still have traces — that failure rides the traces and must not be
    counted as an episode that produced nothing."""
    errored = ep(mk(has_error=True), errors=[vf.Error(type="EnvError", message="boom")])
    assert errored.traces and not errored.ok  # failed, but something came back
    # the failure rides the trace, where the seat's rate sees it
    assert rollouts_of(errored)[0].has_error
