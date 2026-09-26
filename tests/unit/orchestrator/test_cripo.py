from types import SimpleNamespace

import pytest

from prime_rl.orchestrator.algo.cripo import _add_branch_bonus, criterion_advantages


def _trace(**rewards):
    return SimpleNamespace(
        rewards={name: SimpleNamespace(value=value) for name, value in rewards.items()},
        info={},
    )


def test_criterion_advantages_keep_rubric_components_separate():
    traces = [
        _trace(correctness=1.0, style=0.0),
        _trace(correctness=0.0, style=1.0),
        _trace(correctness=0.0, style=0.0),
    ]

    advantages = criterion_advantages(traces)
    assert advantages["correctness"] == pytest.approx([2 / 3, -1 / 3, -1 / 3])
    assert advantages["style"] == pytest.approx([-1 / 3, 2 / 3, -1 / 3])


def test_branch_bonus_does_not_touch_shared_context_tokens():
    node = SimpleNamespace(token_ids=[10, 11, 12], mask=[False, True, True], advantages=[0.5, 0.5])
    branch = SimpleNamespace(nodes=[node])

    _add_branch_bonus(branch, [False, True, False], [False, True, True], 0.25)

    assert node.advantages == [0.75, 0.5]
