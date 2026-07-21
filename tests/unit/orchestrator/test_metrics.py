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
    is_filtered: bool = False,
    filter_results: dict | None = None,
    setup: float = 0.0,
    generation: float = 0.0,
    generation_model: float = 0.0,
    generation_harness: float = 0.0,
    finalize: float = 0.0,
    scoring: float = 0.0,
):
    """Duck-typed stand-in for ``Rollout``, exposing only the Trace properties the metrics read."""
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
        has_error=has_error,
        error=SimpleNamespace(type=error_type) if has_error else None,
        stop_condition=stop_condition,
        metrics=metrics or {},
        env_name=env_name,
        group_id=group_id,
        trainable=trainable,
        is_trainable=is_trainable,
        is_filtered=is_filtered,
        filter_results=filter_results or {},
        timing=SimpleNamespace(
            setup=SimpleNamespace(duration=setup),
            generation=SimpleNamespace(
                duration=generation,
                model=SimpleNamespace(duration=generation_model),
                harness=SimpleNamespace(duration=generation_harness),
            ),
            finalize=SimpleNamespace(duration=finalize),
            scoring=SimpleNamespace(duration=scoring),
        ),
    )


def train_wandb(rollouts, subset: str = "all") -> dict:
    return TrainRollouts(rollouts).metrics.to_wandb(prefix="train/agg", subset=subset)


def test_stat():
    s = Stat([1.0, 2.0, 3.0])
    assert (s.mean(), s.max(), s.min()) == (2.0, 3.0, 1.0)
    assert (s.p10(), s.p90()) == pytest.approx((1.2, 2.8))  # linear-interpolated percentiles
    assert s.to_dict("p") == pytest.approx({"p/mean": 2.0, "p/max": 3.0, "p/min": 1.0, "p/p10": 1.2, "p/p90": 2.8})
    assert Stat([]).p90() == 0.0 and Stat([]).to_dict("p") == {}


def test_container_effective_by_env_and_listlike():
    rc = TrainRollouts(
        [mk(env_name="a"), mk(env_name="a", has_error=True), mk(env_name="b", is_filtered=True), mk(env_name="b")]
    )
    assert len(rc) == 4 and [r.env_name for r in rc] == ["a", "a", "b", "b"]  # sized + iterable
    eff = rc.effective
    assert isinstance(eff, TrainRollouts) and len(eff) == 2  # same type, errored + filtered dropped
    assert all(not r.has_error and not r.is_filtered and r in rc.rollouts for r in eff)  # view of references
    by_env = rc.by_env()
    assert set(by_env) == {"a", "b"} and len(by_env["a"]) == 2 and isinstance(by_env["a"], TrainRollouts)
    rc.append(mk())
    assert len(rc) == 5


def test_to_wandb_distributions():
    m = TrainRollouts(
        [
            mk(reward=1.0, num_total_tokens=10, num_input_tokens=4),
            mk(reward=0.0, num_total_tokens=20, num_input_tokens=6),
        ]
    ).metrics
    assert m.num_input_tokens.mean() == 5.0  # fluent Stat access
    out = m.to_wandb(prefix="train/agg", subset="all")
    assert out["train/agg/all/reward/mean"] == 0.5
    assert out["train/agg/all/num_total_tokens/mean"] == 15.0
    assert out["train/agg/all/num_total_tokens/max"] == 20.0  # flat over rollouts, not per-group
    assert out["train/agg/all/num_input_tokens/mean"] == 5.0
    assert out["train/agg/all/num_output_tokens/mean"] == 6.0


def test_boolean_rates_and_error_breakdown_all_only():
    rc = TrainRollouts([mk(is_truncated=True), mk(has_error=True, error_type="ProviderError"), mk(is_filtered=True)])
    out = rc.metrics.to_wandb(prefix="train/agg", subset="all")
    assert out["train/agg/all/is_truncated/mean"] == 1 / 3
    assert out["train/agg/all/is_completed/mean"] == 1.0
    assert out["train/agg/all/has_error/mean"] == 1 / 3
    assert out["train/agg/all/error/ProviderError"] == 1  # error-type breakdown by count
    assert not any("no_response" in k for k in out)  # removed metric
    # has_error + the error-type counts are structurally empty on effective, so emitted on `all` only
    eff = rc.effective.metrics.to_wandb(prefix="train/agg", subset="effective")
    assert not any(k.endswith("/has_error/mean") or "/error/" in k for k in eff)


def test_solve_rates():
    groups = {"A": [1.0, 1.0], "B": [0.0, 0.0], "C": [1.0, 0.0], "D": [1.0, 0.0]}  # all / none / some / some
    out = train_wandb([mk(reward=r, group_id=g) for g, rs in groups.items() for r in rs])
    rates = (out["train/agg/all/solved_all"], out["train/agg/all/solved_none"], out["train/agg/all/solved_some"])
    assert rates == (0.25, 0.25, 0.5)


def test_stop_condition_breakdown():
    truncated = [mk(is_truncated=True, stop_condition=c) for c in ("length", "max_turns", "prompt_too_long")]
    out = train_wandb(truncated + [mk(stop_condition=None)])
    assert out["train/agg/all/stop_condition/generation_truncated"] == 0.5  # truncated & not prompt_too_long, over all
    assert out["train/agg/all/stop_condition/length"] == 1 / 3  # over the 3 recorded conditions
    assert out["train/agg/all/stop_condition/prompt_too_long"] == 1 / 3


def test_nested_metrics_and_rewards():
    rollouts = [
        mk(metrics={"acc": 1.0}, rewards={"correct": 1.0, "format": 0.0}),
        mk(metrics={"acc": 3.0, "fmt": 5.0}, rewards={"correct": 0.0, "format": 1.0}),
    ]
    m = TrainRollouts(rollouts).metrics
    assert m.metrics["acc"].mean() == 2.0 and m.rewards["correct"].mean() == 0.5  # nested group access
    out = m.to_wandb(prefix="train/agg", subset="all")
    assert out["train/agg/all/metrics/acc/mean"] == 2.0  # averaged over reporters
    assert out["train/agg/all/metrics/fmt/mean"] == 5.0  # single reporter
    assert out["train/agg/all/rewards/format/mean"] == 0.5


def test_nested_timing():
    m = TrainRollouts(
        [mk(setup=1.0, generation=2.0, generation_model=1.5, generation_harness=0.5, finalize=0.5, scoring=0.5)]
    ).metrics
    assert m.timing.setup.mean() == 1.0 and m.timing.total.mean() == 4.0  # total sums all four phases
    assert m.timing.generation_model.mean() == 1.5 and m.timing.generation_harness.mean() == 0.5
    out = m.to_wandb(prefix="train/agg", subset="all")
    assert out["train/agg/all/timing/setup/mean"] == 1.0
    assert out["train/agg/all/timing/total/mean"] == 4.0
    assert out["train/agg/all/timing/generation/model/mean"] == 1.5
    assert out["train/agg/all/timing/generation/harness/mean"] == 0.5


def test_train_only_metrics_absent_from_eval():
    rollouts = [
        mk(is_trainable=True, is_filtered=True, filter_results={"gibberish": True}),
        mk(is_trainable=False, filter_results={"gibberish": False}),
    ]
    out = train_wandb(rollouts)
    assert out["train/agg/all/is_trainable/mean"] == 0.5
    assert out["train/agg/all/is_filtered/mean"] == 0.5
    assert out["train/agg/all/filters/gibberish/mean"] == 0.5
    eval_out = EvalRollouts(rollouts).metrics.to_wandb(prefix="eval/x", subset="all")
    assert not any("is_trainable" in k or "is_filtered" in k or "/filters/" in k for k in eval_out)


def test_eval_avg_at_k_and_pass_k():
    binary = EvalRollouts([mk(reward=1.0, group_id="g0"), mk(reward=0.0, group_id="g0")])
    eff = binary.effective.metrics.to_wandb(prefix="eval/x", subset="effective")
    assert eff["eval/x/effective/avg@2"] == 0.5  # mean reward under avg@<k> (k derived from the groups)
    assert not any(k.startswith("eval/x/effective/reward") for k in eff)
    assert eff["eval/x/effective/pass@1"] == 0.5 and eff["eval/x/effective/pass^2"] == 0.0
    all_out = binary.metrics.to_wandb(prefix="eval/x", subset="all")
    assert all_out["eval/x/all/avg@2"] == 0.5
    assert not any("pass@" in k or "pass^" in k for k in all_out)  # pass@k effective-only
    non_binary = EvalRollouts([mk(reward=0.5, group_id="g0"), mk(reward=1.0, group_id="g0")])
    assert not any("pass@" in k for k in non_binary.effective.metrics.to_wandb(prefix="eval/x", subset="effective"))


def test_compute_pass_metrics_matches_closed_form():
    out = compute_pass_metrics([1.0, 1.0, 0.0, 0.0])  # n=4, c=2
    assert out["pass@1"] == 1.0 - math.comb(2, 1) / math.comb(4, 1)
    assert out["pass@2"] == 1.0 - math.comb(2, 2) / math.comb(4, 2)
    assert out["pass^2"] == math.comb(2, 2) / math.comb(4, 2)
    assert set(out) == {"pass@1", "pass@2", "pass@4", "pass^1", "pass^2", "pass^4"}
