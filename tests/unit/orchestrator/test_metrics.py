import math
from types import SimpleNamespace

import pytest

from prime_rl.orchestrator.metrics import EvalRollouts, Stat, TrainRollouts
from prime_rl.orchestrator.utils import compute_pass_metrics


def mk(
    reward: float = 0.0,
    *,
    num_total_tokens: int = 10,
    num_input_tokens: int = 4,
    num_output_tokens: int = 6,
    num_turns: int = 1,
    num_branches: int = 1,
    is_truncated: bool = False,
    has_response: bool = True,
    has_error: bool = False,
    stop_condition: str | None = None,
    metrics: dict | None = None,
    env_name: str = "env",
    group_id: str = "g0",
    is_trainable: bool = True,
    is_filtered: bool = False,
    filter_results: dict | None = None,
    is_completed: bool = True,
    rewards: dict | None = None,
    setup: float = 0.0,
    generation: float = 0.0,
    finalize: float = 0.0,
    scoring: float = 0.0,
):
    """A duck-typed stand-in for ``Rollout`` exposing only the Trace properties the metrics read."""
    return SimpleNamespace(
        reward=reward,
        rewards=rewards or {},
        num_total_tokens=num_total_tokens,
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        num_turns=num_turns,
        num_branches=num_branches,
        is_truncated=is_truncated,
        is_completed=is_completed,
        has_response=has_response,
        has_error=has_error,
        stop_condition=stop_condition,
        metrics=metrics or {},
        env_name=env_name,
        group_id=group_id,
        is_trainable=is_trainable,
        is_filtered=is_filtered,
        filter_results=filter_results or {},
        timing=SimpleNamespace(
            setup=SimpleNamespace(duration=setup),
            generation=SimpleNamespace(duration=generation),
            finalize=SimpleNamespace(duration=finalize),
            scoring=SimpleNamespace(duration=scoring),
        ),
    )


def test_stat():
    s = Stat([1.0, 2.0, 3.0])
    assert (s.mean(), s.max(), s.min()) == (2.0, 3.0, 1.0)
    assert (s.p10(), s.p90()) == pytest.approx((1.2, 2.8))  # linear-interpolated percentiles
    assert s.to_dict("p") == pytest.approx({"p/mean": 2.0, "p/max": 3.0, "p/min": 1.0, "p/p10": 1.2, "p/p90": 2.8})
    assert Stat([]).mean() == 0.0
    assert Stat([]).p90() == 0.0
    assert Stat([]).to_dict("p") == {}


def test_container_effective_by_env_and_listlike():
    rollouts = [
        mk(env_name="a"),
        mk(env_name="a", has_error=True),
        mk(env_name="b", is_filtered=True),
        mk(env_name="b"),
    ]
    rc = TrainRollouts(rollouts)
    assert len(rc) == 4
    assert [r.env_name for r in rc] == ["a", "a", "b", "b"]  # iterable
    eff = rc.effective
    assert isinstance(eff, TrainRollouts) and len(eff) == 2  # same type, errored + filtered dropped
    assert all(not r.has_error and not r.is_filtered for r in eff)
    assert all(r in rc.rollouts for r in eff)  # view of references, not copies
    by_env = rc.by_env()
    assert set(by_env) == {"a", "b"} and len(by_env["a"]) == 2 and isinstance(by_env["a"], TrainRollouts)
    rc.append(mk())
    assert len(rc) == 5


def test_fluent_stat_access():
    m = TrainRollouts([mk(num_input_tokens=4, reward=1.0), mk(num_input_tokens=6, reward=0.0)]).metrics
    assert m.num_input_tokens.mean() == 5.0
    assert m.num_input_tokens.max() == 6.0
    assert m.reward.mean() == 0.5


def test_train_to_wandb_keys_and_flat_stats():
    out = TrainRollouts([mk(reward=1.0, num_total_tokens=10), mk(reward=0.0, num_total_tokens=20)]).metrics.to_wandb(
        prefix="train/agg", subset="all"
    )
    assert out["train/agg/all/reward/mean"] == 0.5
    assert out["train/agg/all/num_total_tokens/mean"] == 15.0
    assert out["train/agg/all/num_total_tokens/max"] == 20.0  # flat over rollouts
    assert out["train/agg/all/num_input_tokens/mean"] == 4.0
    assert out["train/agg/all/num_output_tokens/mean"] == 6.0


def test_has_error_on_all_only():
    rc = TrainRollouts([mk(), mk(has_error=True), mk(is_filtered=True)])
    assert rc.metrics.to_wandb(prefix="train/agg", subset="all")["train/agg/all/has_error/mean"] == 1 / 3
    assert not any(
        k.endswith("/has_error/mean") for k in rc.effective.metrics.to_wandb(prefix="train/agg", subset="effective")
    )


def test_rates_use_mean_suffix():
    out = TrainRollouts([mk(is_truncated=True), mk(is_truncated=False)]).metrics.to_wandb(prefix="x", subset="all")
    assert out["x/all/is_truncated/mean"] == 0.5
    assert out["x/all/is_completed/mean"] == 1.0
    assert not any("no_response" in k for k in out)


def test_solve_rates():
    rollouts = [
        mk(reward=1.0, group_id="A"),
        mk(reward=1.0, group_id="A"),  # solved_all
        mk(reward=0.0, group_id="B"),
        mk(reward=0.0, group_id="B"),  # solved_none
        mk(reward=1.0, group_id="C"),
        mk(reward=0.0, group_id="C"),  # solved_some
        mk(reward=1.0, group_id="D"),
        mk(reward=0.0, group_id="D"),  # solved_some
    ]
    out = TrainRollouts(rollouts).metrics.to_wandb(prefix="train/agg", subset="all")
    assert out["train/agg/all/solved_all"] == 0.25
    assert out["train/agg/all/solved_none"] == 0.25
    assert out["train/agg/all/solved_some"] == 0.5


def test_stop_condition_breakdown():
    rollouts = [
        mk(is_truncated=True, stop_condition="length"),
        mk(is_truncated=True, stop_condition="max_turns"),
        mk(is_truncated=True, stop_condition="prompt_too_long"),
        mk(is_truncated=False, stop_condition=None),
    ]
    out = TrainRollouts(rollouts).metrics.to_wandb(prefix="train/agg", subset="all")
    assert out["train/agg/all/stop_condition/generation_truncated"] == 0.5  # truncated & not prompt_too_long, over all
    assert out["train/agg/all/stop_condition/length"] == 1 / 3  # over the 3 non-None conditions
    assert out["train/agg/all/stop_condition/prompt_too_long"] == 1 / 3


def test_custom_metrics_and_reward_components():
    rollouts = [
        mk(metrics={"acc": 1.0}, rewards={"correct": 1.0, "format": 0.0}),
        mk(metrics={"acc": 3.0, "fmt": 5.0}, rewards={"correct": 0.0, "format": 1.0}),
    ]
    out = TrainRollouts(rollouts).metrics.to_wandb(prefix="train/agg", subset="all")
    assert out["train/agg/all/metrics/acc/mean"] == 2.0  # over both reporters
    assert out["train/agg/all/metrics/fmt/mean"] == 5.0  # single reporter
    assert out["train/agg/all/rewards/correct/mean"] == 0.5
    assert out["train/agg/all/rewards/format/mean"] == 0.5


def test_timing_total_sums_all_phases():
    m = TrainRollouts([mk(setup=1.0, generation=2.0, finalize=0.5, scoring=0.5)]).metrics
    assert m.timing.setup.mean() == 1.0  # nested fluent access
    assert m.timing.total.mean() == 4.0
    out = m.to_wandb(prefix="train/agg", subset="all")
    assert out["train/agg/all/timing/setup/mean"] == 1.0
    assert out["train/agg/all/timing/finalize/mean"] == 0.5
    assert out["train/agg/all/timing/total/mean"] == 4.0


def test_is_trainable_is_train_only_rate():
    rollouts = [mk(is_trainable=True), mk(is_trainable=False), mk(is_trainable=True), mk(is_trainable=True)]
    train_out = TrainRollouts(rollouts).metrics.to_wandb(prefix="train/agg", subset="all")
    assert train_out["train/agg/all/is_trainable/mean"] == 0.75
    eval_out = EvalRollouts(rollouts).metrics.to_wandb(prefix="eval/x", subset="all")
    assert not any("is_trainable" in k for k in eval_out)


def test_filters_are_train_only():
    rollouts = [mk(is_filtered=True, filter_results={"gibberish": True}), mk(filter_results={"gibberish": False})]
    train_out = TrainRollouts(rollouts).metrics.to_wandb(prefix="train/agg", subset="all")
    assert train_out["train/agg/all/is_filtered/mean"] == 0.5
    assert train_out["train/agg/all/filters/gibberish/mean"] == 0.5
    eval_out = EvalRollouts(rollouts).metrics.to_wandb(prefix="eval/x", subset="all")
    assert not any("is_filtered" in k or "/filters/" in k for k in eval_out)


def test_eval_avg_at_k_and_pass_k():
    binary = EvalRollouts([mk(reward=1.0, group_id="g0"), mk(reward=0.0, group_id="g0")])
    eff = binary.effective.metrics.to_wandb(prefix="eval/x", subset="effective")
    assert eff["eval/x/effective/avg@2"] == 0.5  # mean reward under the avg@k key (not reward/...)
    assert not any(k.startswith("eval/x/effective/reward") for k in eff)
    assert eff["eval/x/effective/pass@1"] == 0.5
    assert eff["eval/x/effective/pass^2"] == 0.0

    all_out = binary.metrics.to_wandb(prefix="eval/x", subset="all")
    assert all_out["eval/x/all/avg@2"] == 0.5
    assert not any("pass@" in k or "pass^" in k for k in all_out)  # pass@k effective-only

    non_binary = EvalRollouts([mk(reward=0.5, group_id="g0"), mk(reward=1.0, group_id="g0")])
    assert not any("pass@" in k for k in non_binary.effective.metrics.to_wandb(prefix="eval/x", subset="effective"))


def test_compute_pass_metrics_matches_closed_form():
    rewards = [1.0, 1.0, 0.0, 0.0]  # n=4, c=2
    out = compute_pass_metrics(rewards)
    assert out["pass@1"] == 1.0 - math.comb(2, 1) / math.comb(4, 1)
    assert out["pass@2"] == 1.0 - math.comb(2, 2) / math.comb(4, 2)
    assert out["pass^2"] == math.comb(2, 2) / math.comb(4, 2)
    assert set(out) == {"pass@1", "pass@2", "pass@4", "pass^1", "pass^2", "pass^4"}
