import math
from types import SimpleNamespace

from prime_rl.orchestrator.eval_utils import compute_pass_metrics
from prime_rl.orchestrator.rollout_metrics import compute_rollout_metrics


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
    is_filtered: bool = False,
    filter_results: dict | None = None,
    generation: float = 0.0,
    scoring: float = 0.0,
):
    """A duck-typed stand-in for ``Rollout`` exposing only the Trace properties the metric
    function reads (keeps the test fast and free of the message-graph machinery)."""
    return SimpleNamespace(
        reward=reward,
        num_total_tokens=num_total_tokens,
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        num_turns=num_turns,
        num_branches=num_branches,
        is_truncated=is_truncated,
        has_response=has_response,
        has_error=has_error,
        stop_condition=stop_condition,
        metrics=metrics or {},
        env_name=env_name,
        group_id=group_id,
        is_filtered=is_filtered,
        filter_results=filter_results or {},
        timing=SimpleNamespace(
            generation=SimpleNamespace(duration=generation),
            scoring=SimpleNamespace(duration=scoring),
        ),
    )


def test_empty_returns_empty():
    assert compute_rollout_metrics([], prefix="train/agg", subset="all", env_group_size={}) == {}


def test_key_prefixes_and_flat_stats():
    rollouts = [mk(reward=1.0, num_total_tokens=10), mk(reward=0.0, num_total_tokens=20)]
    out = compute_rollout_metrics(rollouts, prefix="train/agg", subset="all", env_group_size={"env": 2})
    assert out["train/agg/all/reward/mean"] == 0.5
    # max/min are flat over rollouts (not over per-group means)
    assert out["train/agg/all/num_total_tokens/mean"] == 15.0
    assert out["train/agg/all/num_total_tokens/max"] == 20.0
    assert out["train/agg/all/num_total_tokens/min"] == 10.0
    assert out["train/agg/all/num_input_tokens/mean"] == 4.0
    assert out["train/agg/all/num_output_tokens/mean"] == 6.0


def test_all_carries_error_rate_effective_does_not():
    pool_all = [mk(), mk(has_error=True), mk(is_filtered=True)]
    out_all = compute_rollout_metrics(pool_all, prefix="train/agg", subset="all", env_group_size={"env": 3})
    assert out_all["train/agg/all/error_rate"] == 1 / 3

    effective = [r for r in pool_all if not r.has_error and not r.is_filtered]
    out_eff = compute_rollout_metrics(effective, prefix="train/agg", subset="effective", env_group_size={"env": 3})
    assert not any(k.endswith("/error_rate") for k in out_eff)


def test_rates_use_mean_suffix():
    rollouts = [mk(is_truncated=True), mk(is_truncated=False)]
    out = compute_rollout_metrics(rollouts, prefix="eval/x", subset="all", env_group_size={"env": 2})
    assert out["eval/x/all/is_truncated/mean"] == 0.5
    assert not any("no_response" in k for k in out)


def test_solve_rates_per_env_group_size():
    # group A (env=a, size 2): both solved -> solve_all; group B (env=b, size 4): none -> solve_none
    rollouts = [
        mk(reward=1.0, env_name="a", group_id="A"),
        mk(reward=1.0, env_name="a", group_id="A"),
        mk(reward=0.0, env_name="b", group_id="B"),
        mk(reward=0.0, env_name="b", group_id="B"),
    ]
    out = compute_rollout_metrics(rollouts, prefix="train/agg", subset="all", env_group_size={"a": 2, "b": 4})
    assert out["train/agg/all/solve_all"] == 0.5
    assert out["train/agg/all/solve_none"] == 0.5
    assert out["train/agg/all/effective_batch_size"] == 0.0


def test_stop_condition_breakdown():
    rollouts = [
        mk(is_truncated=True, stop_condition="length"),
        mk(is_truncated=True, stop_condition="max_turns"),
        mk(is_truncated=True, stop_condition="prompt_too_long"),
        mk(is_truncated=False, stop_condition=None),
    ]
    out = compute_rollout_metrics(rollouts, prefix="train/agg", subset="all", env_group_size={"env": 4})
    # generation_truncated: truncated AND not prompt_too_long, over all rollouts -> 2/4
    assert out["train/agg/all/stop_condition/generation_truncated"] == 0.5
    # per-condition rate normalized over non-None conditions (3 of them)
    assert out["train/agg/all/stop_condition/length"] == 1 / 3
    assert out["train/agg/all/stop_condition/prompt_too_long"] == 1 / 3


def test_custom_metrics_averaged_over_reporters():
    rollouts = [mk(metrics={"acc": 1.0}), mk(metrics={"acc": 3.0, "fmt": 5.0})]
    out = compute_rollout_metrics(rollouts, prefix="train/agg", subset="all", env_group_size={"env": 2})
    assert out["train/agg/all/metrics/acc/mean"] == 2.0  # over both reporters
    assert out["train/agg/all/metrics/fmt/mean"] == 5.0  # over the single reporter


def test_filters_only_when_included():
    rollouts = [mk(is_filtered=True, filter_results={"gibberish": True}), mk(filter_results={"gibberish": False})]
    without = compute_rollout_metrics(rollouts, prefix="train/agg", subset="all", env_group_size={"env": 2})
    assert "train/agg/all/is_filtered/mean" not in without
    assert not any("/filters/" in k for k in without)

    with_filters = compute_rollout_metrics(
        rollouts, prefix="train/agg", subset="all", env_group_size={"env": 2}, include_filters=True
    )
    # is_filtered is a top-level rollout metric; per-filter detection stays under filters/
    assert with_filters["train/agg/all/is_filtered/mean"] == 0.5
    assert with_filters["train/agg/all/filters/gibberish/mean"] == 0.5


def test_pass_metrics_only_for_binary_rewards():
    # one example, rewards [1, 0]: n=2, c=1
    binary = [mk(reward=1.0, group_id="g0"), mk(reward=0.0, group_id="g0")]
    out = compute_rollout_metrics(
        binary, prefix="eval/x", subset="effective", env_group_size={"env": 2}, include_pass_at_k=True
    )
    assert out["eval/x/effective/pass@1"] == 0.5  # 1 - C(1,1)/C(2,1)
    assert out["eval/x/effective/pass@2"] == 1.0  # 1 - C(1,2)/C(2,2)
    assert out["eval/x/effective/pass^1"] == 0.5  # C(1,1)/C(2,1)
    assert out["eval/x/effective/pass^2"] == 0.0  # C(0... )/C(2,2) -> C(1,2)=0

    non_binary = [mk(reward=0.5, group_id="g0"), mk(reward=1.0, group_id="g0")]
    out_nb = compute_rollout_metrics(
        non_binary, prefix="eval/x", subset="effective", env_group_size={"env": 2}, include_pass_at_k=True
    )
    assert not any("pass@" in k or "pass^" in k for k in out_nb)


def test_compute_pass_metrics_matches_closed_form():
    rewards = [1.0, 1.0, 0.0, 0.0]  # n=4, c=2
    out = compute_pass_metrics(rewards)
    assert out["pass@1"] == 1.0 - math.comb(2, 1) / math.comb(4, 1)
    assert out["pass@2"] == 1.0 - math.comb(2, 2) / math.comb(4, 2)
    assert out["pass^2"] == math.comb(2, 2) / math.comb(4, 2)
    assert set(out) == {"pass@1", "pass@2", "pass@4", "pass^1", "pass^2", "pass^4"}
