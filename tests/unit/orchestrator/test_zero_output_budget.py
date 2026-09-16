from types import SimpleNamespace

import pytest

from prime_rl.orchestrator.train_sink import ZeroOutputBudget, zero_output_units


def trace(tokens: int = 100, error: bool = False):
    return SimpleNamespace(has_error=error, num_total_tokens=tokens)


def episode(*traces):
    return SimpleNamespace(traces=list(traces), ok=not any(t.has_error for t in traces))


def units(group, survivors, *, n_owed, n_errored, batch_mode=True):
    return zero_output_units(group, survivors, n_owed=n_owed, n_errored=n_errored, batch_mode=batch_mode, seq_len=1000)


def test_all_equal_reward_group_counts_its_survivors():
    # Eight clean rollouts with identical rewards: trainable survivors, but every advantage is 0.
    group = [episode(trace()) for _ in range(8)]
    survivors = [ep.traces[0] for ep in group]
    assert units(group, survivors, n_owed=8, n_errored=0) == 8


def test_errored_group_does_not_advance_the_budget():
    # An outage: every rollout failed, no trainable survivors.
    group = [episode(trace(tokens=0, error=True)) for _ in range(8)]
    assert units(group, [], n_owed=8, n_errored=8) == 0
    # Same for token-based batching, which used to fall back to seq_len * n_owed.
    assert units(group, [], n_owed=8, n_errored=8, batch_mode=False) == 0


def test_dispatch_failures_do_not_advance_the_budget():
    # Nothing arrived at all: the whole group failed to dispatch.
    assert units([], [], n_owed=8, n_errored=8) == 0
    assert units([], [], n_owed=8, n_errored=8, batch_mode=False) == 0


def test_mixed_group_counts_only_clean_units():
    group = [episode(trace()) for _ in range(5)] + [episode(trace(error=True)) for _ in range(3)]
    # Five clean rollouts with identical rewards survive; the three errors are excluded.
    survivors = [ep.traces[0] for ep in group[:5]]
    assert units(group, survivors, n_owed=8, n_errored=3) == 5
    # If nothing is trainable, the clean arrived traces still count, the errored ones do not.
    assert units(group, [], n_owed=8, n_errored=3) == 5


def test_stale_drop_still_counts_as_a_pipeline_decision():
    # Three rollouts arrived clean, five were cancelled as stale: the owed clean budget counts.
    arrived = [episode(trace()) for _ in range(3)]
    assert units(arrived, [], n_owed=8, n_errored=0) == 3
    assert units([], [], n_owed=8, n_errored=0) == 8
    assert units([], [], n_owed=8, n_errored=0, batch_mode=False) == 8 * 1000


def test_budget_warns_per_window_and_aborts_at_the_limit():
    budget = ZeroOutputBudget(target=128, max_windows=3)
    budget.record(128)
    budget.record(128)
    assert budget.reported_windows == 2
    with pytest.raises(RuntimeError, match="3 consecutive zero-output batch equivalents"):
        budget.record(128)


def test_budget_reset_and_disabled_limit():
    budget = ZeroOutputBudget(target=128, max_windows=1)
    budget.record(100)
    budget.reset()
    budget.record(100)  # 200 units total would have aborted without the reset
    assert budget.units == 100 and budget.reported_windows == 0

    unlimited = ZeroOutputBudget(target=128, max_windows=None)
    for _ in range(50):
        unlimited.record(128)  # warns, never raises
    assert unlimited.reported_windows == 50
