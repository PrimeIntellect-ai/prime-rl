"""Future-task credit, with no model or task-data dependencies."""

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


module_path = Path(__file__).resolve().parents[1] / "src/prime_rl/orchestrator/algo/rime_credit.py"
spec = importlib.util.spec_from_file_location("writer_suffix_credit_under_test", module_path)
credit = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = credit
spec.loader.exec_module(credit)


def test_suffix_excludes_current_and_includes_delayed_outcomes():
    task = np.array([[100, 1, 2, 4], [-100, 3, 5, -2]])
    result = credit.assign_suffix_credit(task, np.zeros_like(task), np.ones_like(task))
    np.testing.assert_array_equal(result.writer_reward, [[7, 6, 4, 0], [6, 3, -2, 0]])
    np.testing.assert_array_equal(result.actor, np.zeros_like(task))
    np.testing.assert_array_equal(result.writer_mask, [[True, True, True, False]] * 2)
    changed_current = task.copy()
    changed_current[:, 0] += 10000
    np.testing.assert_array_equal(
        result.writer_reward,
        credit.assign_suffix_credit(changed_current, np.zeros_like(task), np.ones_like(task)).writer_reward,
    )


def test_format_diagnostics_do_not_change_credit():
    task = np.array([[1, 2, 3], [5, 1, 8]])
    ones = np.ones_like(task)
    invalid = ones.copy()
    invalid[0, 0] = 0
    valid = credit.assign_suffix_credit(task, np.zeros_like(task), ones)
    bad = credit.assign_suffix_credit(task, np.zeros_like(task), invalid)
    np.testing.assert_array_equal(valid.writer_reward, bad.writer_reward)
    np.testing.assert_array_equal(valid.writer, bad.writer)


def test_normalization_is_per_position_and_terminal_zero():
    task = np.array([[0, 1, 2], [0, 3, 4], [0, 5, 6], [0, 7, 8]])
    result = credit.assign_suffix_credit(task, np.zeros_like(task), np.ones_like(task))
    np.testing.assert_allclose(result.writer[:, :-1].mean(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(result.writer[:, :-1].std(axis=0), 1)
    np.testing.assert_array_equal(result.writer[:, -1], 0)


def test_constant_rewards_give_finite_zero_advantages():
    task = np.ones((4, 10))
    result = credit.assign_suffix_credit(task, np.zeros_like(task), np.ones_like(task))
    assert np.isfinite(result.writer).all()
    np.testing.assert_array_equal(result.writer, 0)


@pytest.mark.parametrize("kind", ["exploration", "gate", "shape", "nonfinite", "singleton"])
def test_invalid_credit_inputs_fail_closed(kind):
    task = np.ones((4, 10))
    exploration, gates = np.zeros_like(task), np.ones_like(task)
    if kind == "exploration":
        exploration[0, 0] = 1
    elif kind == "gate":
        gates[0, 0] = 0.5
    elif kind == "shape":
        gates = gates[:, :-1]
    elif kind == "nonfinite":
        task[0, 0] = np.nan
    else:
        task, exploration, gates = task[:1], exploration[:1], gates[:1]
    with pytest.raises(ValueError):
        credit.assign_suffix_credit(task, exploration, gates)
